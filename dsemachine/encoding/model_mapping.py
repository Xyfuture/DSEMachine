from __future__ import annotations

# 本文件负责把已校验的模型配置、batch/sequence 维度和并行策略，
# 转换为单层代表 chip 视角下的 workload IR。
# 当前 model mapping 只划分到 chip 级：位置为 pim_chiplets 的 BMMOp、
# AttentionCoreOp、FFNCoreOp 和 VectorOp 表示该 chip 内所有 PIM chiplets
# 共同承担的 op 尺寸，不是单个 chiplet 的尺寸。
# chip 内 D2D 通信，以及这些 op 在 chiplets 级的进一步划分，本轮都不建模。
# 这里只描述算子结构、chip 级局部维度和 chip 间通信数据量，不读取 model card，
# 也不展开全模型所有层。

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from sympy import Expr, ceiling

from dsemachine.config.model_config import ModelConfigBase, load_model_config


SymbolicDim: TypeAlias = int | Expr

IO_DIE = "io_die"
PIM_CHIPLETS = "pim_chiplets"
_ALLOWED_LOCATIONS = {IO_DIE, PIM_CHIPLETS}


def _validate_symbolic_dim(name: str, value: object) -> SymbolicDim:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or sympy.Expr, got bool")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value
    if isinstance(value, Expr):
        return value
    raise TypeError(f"{name} must be an int or sympy.Expr, got {type(value).__name__}")


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _validate_op_kind(op_kind: object) -> str:
    if not isinstance(op_kind, str) or not op_kind:
        raise TypeError("op_kind must be a non-empty str")
    return op_kind


def _validate_location(location: object) -> str:
    if not isinstance(location, str):
        raise TypeError(f"location must be a str, got {type(location).__name__}")
    if location not in _ALLOWED_LOCATIONS:
        raise ValueError(f"Unsupported location: {location}")
    return location


def ceil_div_expr(dividend: SymbolicDim, divisor: int) -> SymbolicDim:
    """对符号或整数做向上取整除法，统一返回本项目使用的维度表达。"""
    _validate_symbolic_dim("dividend", dividend)
    divisor = _validate_positive_int("divisor", divisor)
    if isinstance(dividend, int):
        return -(-dividend // divisor)
    return ceiling(dividend / divisor)


def mul_expr(*values: SymbolicDim) -> SymbolicDim:
    """将多个维度表达相乘，保留 sympy 表达式而不转成字符串。"""
    result: SymbolicDim = 1
    for value in values:
        checked = _validate_symbolic_dim("value", value)
        result *= checked
    return result


def _exact_div_int(numerator: int, denominator: int, name: str) -> int:
    numerator = _validate_positive_int(f"{name} numerator", numerator)
    denominator = _validate_positive_int(f"{name} denominator", denominator)
    if numerator % denominator != 0:
        raise ValueError(f"{name} must be exactly divisible, got {numerator} / {denominator}")
    return numerator // denominator


def _require_model_int_attr(model_config: ModelConfigBase, name: str) -> int:
    value = getattr(model_config, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NotImplementedError(f"Current model mapping requires positive int model_config.{name}")
    return value


class AttnParallelStrategy(Enum):
    DP = "dp"
    TP = "tp"
    CP = "cp"


class FFNParallelStrategy(Enum):
    EP = "ep"
    TP = "tp"


@dataclass(frozen=True)
class BatchedMatmulOp:
    """表示一个 BMM。

    b/m/n/k 分别对应 batched matmul 的 B、M、N、K 维度。
    位置为 pim_chiplets 时，尺寸是一个 chip 内所有 PIM chiplets 共同承担的尺寸。
    BMM 的运算量可直接按输出元素个数理解，因此记录 output_elements = B * M * N。
    dtype_bytes 表示输出元素按多少 byte 计。
    """

    op_kind: str
    location: str
    b: SymbolicDim
    m: SymbolicDim
    n: SymbolicDim
    k: SymbolicDim
    dtype_bytes: int
    output_elements: SymbolicDim = field(init=False)

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        for name in ("b", "m", "n", "k"):
            _validate_symbolic_dim(name, getattr(self, name))
        _validate_positive_int("dtype_bytes", self.dtype_bytes)
        object.__setattr__(self, "output_elements", mul_expr(self.b, self.m, self.n))


@dataclass(frozen=True)
class VectorOp:
    """表示 M 个 N 维向量上的同类操作。

    这里只记录 m/n 原始语义，不保存派生运算量。
    位置为 pim_chiplets 时，尺寸是该 chip 内所有 PIM chiplets 共同承担的尺寸。
    外部如果需要 vector 输出元素数，可以用 m * n 推导。
    """

    op_kind: str
    location: str
    m: SymbolicDim
    n: SymbolicDim
    dtype_bytes: int

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        _validate_symbolic_dim("m", self.m)
        _validate_symbolic_dim("n", self.n)
        _validate_positive_int("dtype_bytes", self.dtype_bytes)


@dataclass(frozen=True)
class CommOp:
    """表示一次 chip 间通信。

    op_kind 只使用 chip_input/chip_output；data_bytes 是本次通信的数据量，单位为 byte。
    chip 的输入/输出入口都抽象为 io die，因此 location 统一记录为 io_die。
    """

    op_kind: str
    location: str
    data_bytes: SymbolicDim

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        if self.op_kind not in {"chip_input", "chip_output"}:
            raise ValueError(f"Unsupported comm op_kind: {self.op_kind}")
        _validate_location(self.location)
        if self.location != IO_DIE:
            raise ValueError("CommOp.location must be io_die for chip-level mapping")
        _validate_symbolic_dim("data_bytes", self.data_bytes)


CoreSubOp: TypeAlias = BatchedMatmulOp | VectorOp


@dataclass(frozen=True)
class AttentionCoreOp:
    """表示 attention core 内部按顺序执行的 BMM/Vector 子操作。

    GQA 是 qk -> softmax -> sv；MLA 会展开为 qk_nope、qk_rope、
    score add、softmax、sv_latent、vo_absorb、head reduce 等多段。
    location=pim_chiplets 表示单个 chip 内所有 PIM chiplets 共同执行。
    """

    op_kind: str
    location: str
    sub_ops: tuple[CoreSubOp, ...]

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        if not self.sub_ops:
            raise ValueError("sub_ops must not be empty")
        for sub_op in self.sub_ops:
            if not isinstance(sub_op, (BatchedMatmulOp, VectorOp)):
                raise TypeError("attention sub_ops must be BatchedMatmulOp or VectorOp")


@dataclass(frozen=True)
class FFNCoreOp:
    """表示 FFN core 内部按顺序执行的 BMM/Vector 子操作。

    location=pim_chiplets 表示单个 chip 内所有 PIM chiplets 共同执行。
    """

    op_kind: str
    location: str
    sub_ops: tuple[CoreSubOp, ...]

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        if not self.sub_ops:
            raise ValueError("sub_ops must not be empty")
        for sub_op in self.sub_ops:
            if not isinstance(sub_op, (BatchedMatmulOp, VectorOp)):
                raise TypeError("ffn sub_ops must be BatchedMatmulOp or VectorOp")


@dataclass(frozen=True)
class DSAOp:
    """表示 DeepSeek Sparse Attention 的 Lightning Indexer。

    这里只建模 qI/kI/wI projection、index score、ReLU 和 weighted reduce；
    不展开 top-k selector、sparse gather 或跨 chip redistribution。
    """

    op_kind: str
    location: str
    sub_ops: tuple[CoreSubOp | CommOp, ...]

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        if not self.sub_ops:
            raise ValueError("sub_ops must not be empty")
        for sub_op in self.sub_ops:
            if not isinstance(sub_op, (BatchedMatmulOp, VectorOp, CommOp)):
                raise TypeError("dsa sub_ops must be BatchedMatmulOp, VectorOp, or CommOp")


LayerOp: TypeAlias = BatchedMatmulOp | AttentionCoreOp | FFNCoreOp | VectorOp | CommOp | DSAOp


@dataclass(frozen=True)
class ModelMappingRequest:
    """model mapping 的输入请求。当前是单层、单个代表 chip 的视角。"""

    model_name: str
    num_of_chip: int
    attn_parallel_strategy: AttnParallelStrategy
    ffn_parallel_strategy: FFNParallelStrategy
    batch_size: SymbolicDim
    seq_len: SymbolicDim
    dtype_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise TypeError("model_name must be a non-empty str")
        _validate_positive_int("num_of_chip", self.num_of_chip)
        if not isinstance(self.attn_parallel_strategy, AttnParallelStrategy):
            raise TypeError("attn_parallel_strategy must be an AttnParallelStrategy")
        if not isinstance(self.ffn_parallel_strategy, FFNParallelStrategy):
            raise TypeError("ffn_parallel_strategy must be an FFNParallelStrategy")
        _validate_symbolic_dim("batch_size", self.batch_size)
        _validate_symbolic_dim("seq_len", self.seq_len)
        _validate_positive_int("dtype_bytes", self.dtype_bytes)


@dataclass(frozen=True)
class ModelMappingIR:
    """单层 workload IR。

    ops 表示单个代表 chip 的本地操作列表，不是全模型全层展开。
    """

    model_name: str
    num_of_chip: int
    representative_chip_id: int
    attn_parallel_strategy: AttnParallelStrategy
    ffn_parallel_strategy: FFNParallelStrategy
    model_attn_type: str
    model_dsa: bool
    model_ffn_type: str
    model_use_qk_norm: bool
    batch_size: SymbolicDim
    seq_len: SymbolicDim
    dtype_bytes: int
    model_config: ModelConfigBase
    ops: tuple[LayerOp, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.representative_chip_id != 0:
            raise ValueError("representative_chip_id must be 0 in the current view")
        _validate_positive_int("num_of_chip", self.num_of_chip)
        if self.model_attn_type not in {"gqa", "mla"}:
            raise ValueError(f"Unsupported model_attn_type: {self.model_attn_type}")
        if not isinstance(self.model_dsa, bool):
            raise TypeError(f"model_dsa must be a bool, got {type(self.model_dsa).__name__}")
        if self.model_ffn_type not in {"dense", "moe"}:
            raise ValueError(f"Unsupported model_ffn_type: {self.model_ffn_type}")
        if not isinstance(self.model_use_qk_norm, bool):
            raise TypeError(
                "model_use_qk_norm must be a bool, "
                f"got {type(self.model_use_qk_norm).__name__}"
            )
        _validate_symbolic_dim("batch_size", self.batch_size)
        _validate_symbolic_dim("seq_len", self.seq_len)
        _validate_positive_int("dtype_bytes", self.dtype_bytes)
        if not isinstance(self.model_config, ModelConfigBase):
            raise TypeError("model_config must be a ModelConfigBase")
        if not isinstance(self.ops, tuple):
            raise TypeError("ops must be a tuple")


def _build_comm_op(op_kind: str, data_bytes: SymbolicDim) -> CommOp:
    return CommOp(op_kind=op_kind, location=IO_DIE, data_bytes=data_bytes)


def _chip_input(data_bytes: SymbolicDim) -> CommOp:
    return _build_comm_op("chip_input", data_bytes)


def _chip_output(data_bytes: SymbolicDim) -> CommOp:
    return _build_comm_op("chip_output", data_bytes)


def _build_bmm_op(
    op_kind: str,
    *,
    b: SymbolicDim,
    m: SymbolicDim,
    n: SymbolicDim,
    k: SymbolicDim,
    dtype_bytes: int,
) -> BatchedMatmulOp:
    return BatchedMatmulOp(
        op_kind=op_kind,
        location=PIM_CHIPLETS,
        b=b,
        m=m,
        n=n,
        k=k,
        dtype_bytes=dtype_bytes,
    )


def _build_vector_op(
    op_kind: str,
    location: str,
    *,
    m: SymbolicDim,
    n: SymbolicDim,
    dtype_bytes: int,
) -> VectorOp:
    return VectorOp(
        op_kind=op_kind,
        location=location,
        m=m,
        n=n,
        dtype_bytes=dtype_bytes,
    )


def _local_bs_for_strategy(bs: SymbolicDim, num_of_chip: int, strategy: AttnParallelStrategy) -> SymbolicDim:
    if strategy is AttnParallelStrategy.TP:
        return bs
    return ceil_div_expr(bs, num_of_chip)


def _build_gqa_attention_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> tuple[LayerOp, ...]:
    bs = request.batch_size
    seq_len = request.seq_len
    dtype_bytes = request.dtype_bytes
    num_of_chip = request.num_of_chip
    attn_strategy = request.attn_parallel_strategy

    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    num_attention_heads = _require_model_int_attr(model_config, "num_attention_heads")
    num_key_value_heads = _require_model_int_attr(model_config, "num_key_value_heads")
    head_dim = _require_model_int_attr(model_config, "head_dim")
    grouped_heads = _exact_div_int(
        num_attention_heads,
        num_key_value_heads,
        "num_attention_heads / num_key_value_heads",
    )

    local_bs = ceil_div_expr(bs, num_of_chip)
    local_seq_len = ceil_div_expr(seq_len, num_of_chip)
    local_kv_heads = ceil_div_expr(num_key_value_heads, num_of_chip)
    local_hidden_size = ceil_div_expr(hidden_size, num_of_chip)
    attention_output_width = num_attention_heads * head_dim
    qkv_proj_width = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    local_qkv_proj_width = mul_expr(local_kv_heads, grouped_heads + 2, head_dim)
    pre_m = _local_bs_for_strategy(bs, num_of_chip, attn_strategy)

    ops: list[LayerOp] = [
        _chip_input(mul_expr(dtype_bytes, pre_m, hidden_size)),
        _build_vector_op("rms_norm", IO_DIE, m=pre_m, n=hidden_size, dtype_bytes=dtype_bytes),
    ]

    if attn_strategy is AttnParallelStrategy.TP:
        ops.append(
            _build_bmm_op(
                "qkv_proj",
                b=1,
                m=bs,
                n=local_qkv_proj_width,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            )
        )
    else:
        ops.append(
            _build_bmm_op(
                "qkv_proj",
                b=1,
                m=local_bs,
                n=qkv_proj_width,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            )
        )

    if attn_strategy is AttnParallelStrategy.CP:
        ops.extend(
            [
                _chip_output(mul_expr(dtype_bytes, local_bs, qkv_proj_width)),
                _chip_input(mul_expr(dtype_bytes, bs, qkv_proj_width)),
            ]
        )

    if attn_strategy is AttnParallelStrategy.DP:
        qk_norm_m = mul_expr(local_bs, num_attention_heads + num_key_value_heads)
    elif attn_strategy is AttnParallelStrategy.CP:
        qk_norm_m = mul_expr(bs, num_attention_heads + num_key_value_heads)
    else:
        qk_norm_m = mul_expr(bs, local_kv_heads, grouped_heads + 1)

    if model_config.use_qk_norm:
        ops.append(
            _build_vector_op(
                "qk_norm",
                IO_DIE,
                m=qk_norm_m,
                n=head_dim,
                dtype_bytes=dtype_bytes,
            )
        )

    ops.append(
        _build_vector_op(
            "rope",
            IO_DIE,
            m=qk_norm_m,
            n=head_dim,
            dtype_bytes=dtype_bytes,
        )
    )

    if attn_strategy is AttnParallelStrategy.DP:
        attention_to_o_proj_comm: list[LayerOp] = []
        qk_bmm = _build_bmm_op(
            "qk",
            b=mul_expr(local_bs, num_key_value_heads),
            m=grouped_heads,
            n=seq_len,
            k=head_dim,
            dtype_bytes=dtype_bytes,
        )
        softmax = _build_vector_op(
            "softmax",
            PIM_CHIPLETS,
            m=mul_expr(local_bs, num_attention_heads),
            n=seq_len,
            dtype_bytes=dtype_bytes,
        )
        sv_bmm = _build_bmm_op(
            "sv",
            b=mul_expr(local_bs, num_key_value_heads),
            m=grouped_heads,
            n=head_dim,
            k=seq_len,
            dtype_bytes=dtype_bytes,
        )
        o_proj_bmm = _build_bmm_op(
            "o_proj",
            b=1,
            m=local_bs,
            n=hidden_size,
            k=attention_output_width,
            dtype_bytes=dtype_bytes,
        )
    elif attn_strategy is AttnParallelStrategy.CP:
        attention_to_o_proj_comm = [
            _chip_output(mul_expr(dtype_bytes, bs, num_attention_heads, head_dim)),
            _chip_input(mul_expr(dtype_bytes, local_bs, attention_output_width)),
        ]
        qk_bmm = _build_bmm_op(
            "qk",
            b=mul_expr(bs, num_key_value_heads),
            m=grouped_heads,
            n=local_seq_len,
            k=head_dim,
            dtype_bytes=dtype_bytes,
        )
        softmax = _build_vector_op(
            "softmax",
            PIM_CHIPLETS,
            m=mul_expr(bs, num_attention_heads),
            n=local_seq_len,
            dtype_bytes=dtype_bytes,
        )
        sv_bmm = _build_bmm_op(
            "sv",
            b=mul_expr(bs, num_key_value_heads),
            m=grouped_heads,
            n=head_dim,
            k=local_seq_len,
            dtype_bytes=dtype_bytes,
        )
        o_proj_bmm = _build_bmm_op(
            "o_proj",
            b=1,
            m=local_bs,
            n=hidden_size,
            k=attention_output_width,
            dtype_bytes=dtype_bytes,
        )
    else:
        attention_to_o_proj_comm = [
            _chip_output(mul_expr(dtype_bytes, bs, local_kv_heads, grouped_heads, head_dim)),
            _chip_input(mul_expr(dtype_bytes, bs, attention_output_width)),
        ]
        qk_bmm = _build_bmm_op(
            "qk",
            b=mul_expr(bs, local_kv_heads),
            m=grouped_heads,
            n=seq_len,
            k=head_dim,
            dtype_bytes=dtype_bytes,
        )
        softmax = _build_vector_op(
            "softmax",
            PIM_CHIPLETS,
            m=mul_expr(bs, local_kv_heads, grouped_heads),
            n=seq_len,
            dtype_bytes=dtype_bytes,
        )
        sv_bmm = _build_bmm_op(
            "sv",
            b=mul_expr(bs, local_kv_heads),
            m=grouped_heads,
            n=head_dim,
            k=seq_len,
            dtype_bytes=dtype_bytes,
        )
        o_proj_bmm = _build_bmm_op(
            "o_proj",
            b=1,
            m=bs,
            n=local_hidden_size,
            k=attention_output_width,
            dtype_bytes=dtype_bytes,
        )

    ops.extend(
        [
            AttentionCoreOp(
                op_kind="attention_core",
                location=PIM_CHIPLETS,
                sub_ops=(qk_bmm, softmax, sv_bmm),
            ),
            *attention_to_o_proj_comm,
        ]
    )

    ops.append(o_proj_bmm)

    if attn_strategy is AttnParallelStrategy.TP:
        ops.extend(
            [
                _chip_output(mul_expr(dtype_bytes, bs, local_hidden_size)),
                _chip_input(mul_expr(dtype_bytes, local_bs, hidden_size)),
            ]
        )

    ops.extend(
        [
            _build_vector_op("residual", IO_DIE, m=local_bs, n=hidden_size, dtype_bytes=dtype_bytes),
            _build_vector_op("rms_norm", IO_DIE, m=local_bs, n=hidden_size, dtype_bytes=dtype_bytes),
            _chip_output(mul_expr(dtype_bytes, local_bs, hidden_size)),
        ]
    )
    return tuple(ops)


def _build_dense_ffn_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
    *,
    tail_uses_local_bs: bool = False,
) -> tuple[LayerOp, ...]:
    bs = request.batch_size
    dtype_bytes = request.dtype_bytes
    num_of_chip = request.num_of_chip
    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    intermediate_size = _require_model_int_attr(model_config, "intermediate_size")

    local_bs = ceil_div_expr(bs, num_of_chip)
    tail_m = local_bs if tail_uses_local_bs else bs
    local_intermediate_size = ceil_div_expr(intermediate_size, num_of_chip)
    local_two_intermediate_size = ceil_div_expr(2 * intermediate_size, num_of_chip)

    return (
        _chip_input(mul_expr(dtype_bytes, bs, hidden_size)),
        FFNCoreOp(
            op_kind="ffn_core",
            location=PIM_CHIPLETS,
            sub_ops=(
                _build_bmm_op(
                    "up_gate",
                    b=1,
                    m=bs,
                    n=local_two_intermediate_size,
                    k=hidden_size,
                    dtype_bytes=dtype_bytes,
                ),
                _build_vector_op(
                    "silu",
                    PIM_CHIPLETS,
                    m=bs,
                    n=local_intermediate_size,
                    dtype_bytes=dtype_bytes,
                ),
                _build_bmm_op(
                    "down",
                    b=1,
                    m=bs,
                    n=hidden_size,
                    k=local_intermediate_size,
                    dtype_bytes=dtype_bytes,
                ),
            ),
        ),
        _chip_output(mul_expr(dtype_bytes, bs, hidden_size)),
        _chip_input(mul_expr(dtype_bytes, tail_m, hidden_size)),
        _build_vector_op("residual", IO_DIE, m=tail_m, n=hidden_size, dtype_bytes=dtype_bytes),
        _chip_output(mul_expr(dtype_bytes, tail_m, hidden_size)),
    )


def _build_moe_ffn_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
    *,
    tail_uses_local_bs: bool = False,
) -> tuple[LayerOp, ...]:
    # 当前 MoE 只建模 routed experts 已经选定后的核心计算；
    # shared expert、router、expert top-k selection、dispatch/combine 都暂时忽略。
    bs = request.batch_size
    dtype_bytes = request.dtype_bytes
    num_of_chip = request.num_of_chip
    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    moe_intermediate_size = _require_model_int_attr(model_config, "moe_intermediate_size")
    num_experts_per_tok = _require_model_int_attr(model_config, "num_experts_per_tok")

    local_bs = ceil_div_expr(bs, num_of_chip)
    tail_m = local_bs if tail_uses_local_bs else bs
    local_num_experts_per_tok = ceil_div_expr(num_experts_per_tok, num_of_chip)
    local_moe_intermediate_size = ceil_div_expr(moe_intermediate_size, num_of_chip)
    local_two_moe_intermediate_size = ceil_div_expr(2 * moe_intermediate_size, num_of_chip)

    if request.ffn_parallel_strategy is FFNParallelStrategy.TP:
        sub_ops = (
            _build_bmm_op(
                "up_gate",
                b=num_experts_per_tok,
                m=bs,
                n=local_two_moe_intermediate_size,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            _build_vector_op(
                "silu",
                PIM_CHIPLETS,
                m=mul_expr(num_experts_per_tok, bs),
                n=local_moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
            _build_bmm_op(
                "down",
                b=num_experts_per_tok,
                m=bs,
                n=hidden_size,
                k=local_moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
        )
    else:
        sub_ops = (
            _build_bmm_op(
                "up_gate",
                b=local_num_experts_per_tok,
                m=bs,
                n=2 * moe_intermediate_size,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            _build_vector_op(
                "silu",
                PIM_CHIPLETS,
                m=mul_expr(local_num_experts_per_tok, bs),
                n=moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
            _build_bmm_op(
                "down",
                b=local_num_experts_per_tok,
                m=bs,
                n=hidden_size,
                k=moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
        )

    return (
        _chip_input(mul_expr(dtype_bytes, bs, hidden_size)),
        FFNCoreOp(op_kind="ffn_core", location=PIM_CHIPLETS, sub_ops=sub_ops),
        _chip_output(mul_expr(dtype_bytes, bs, hidden_size)),
        _chip_input(mul_expr(dtype_bytes, tail_m, hidden_size)),
        _build_vector_op("residual", IO_DIE, m=tail_m, n=hidden_size, dtype_bytes=dtype_bytes),
        _chip_output(mul_expr(dtype_bytes, tail_m, hidden_size)),
    )


# MLA cache 只保存 latent C cache 和共享 k_rope cache，不显式保存完整 K/V。
# QK 和 SV/O 都按 mat absorb 之后的等效矩阵建模。
def _build_mla_attention_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> tuple[LayerOp, ...]:
    bs = request.batch_size
    dtype_bytes = request.dtype_bytes
    num_of_chip = request.num_of_chip
    attn_strategy = request.attn_parallel_strategy

    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    num_attention_heads = _require_model_int_attr(model_config, "num_attention_heads")
    q_lora_rank = _require_model_int_attr(model_config, "q_lora_rank")
    kv_lora_rank = _require_model_int_attr(model_config, "kv_lora_rank")
    qk_rope_head_dim = _require_model_int_attr(model_config, "qk_rope_head_dim")

    local_bs = ceil_div_expr(bs, num_of_chip)
    local_heads = ceil_div_expr(num_attention_heads, num_of_chip)
    latent_width = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    local_latent_width = ceil_div_expr(latent_width, num_of_chip)
    pre_m = _local_bs_for_strategy(bs, num_of_chip, attn_strategy)

    ops: list[LayerOp] = [
        _chip_input(mul_expr(dtype_bytes, pre_m, hidden_size)),
        _build_vector_op("rms_norm", IO_DIE, m=pre_m, n=hidden_size, dtype_bytes=dtype_bytes),
    ]

    if attn_strategy is AttnParallelStrategy.TP:
        ops.append(
            _build_bmm_op(
                "mla_latent_down_proj",
                b=1,
                m=bs,
                n=local_latent_width,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            )
        )
    else:
        ops.append(
            _build_bmm_op(
                "mla_latent_down_proj",
                b=1,
                m=local_bs,
                n=latent_width,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            )
        )

    if attn_strategy is AttnParallelStrategy.CP:
        ops.extend(
            [
                _chip_output(mul_expr(dtype_bytes, local_bs, latent_width)),
                _chip_input(mul_expr(dtype_bytes, bs, latent_width)),
            ]
        )
    elif attn_strategy is AttnParallelStrategy.TP:
        ops.extend(
            [
                _chip_output(mul_expr(dtype_bytes, bs, local_latent_width)),
                _chip_input(mul_expr(dtype_bytes, bs, latent_width)),
            ]
        )

    norm_m = local_bs if attn_strategy is AttnParallelStrategy.DP else bs
    ops.extend(
        [
            _build_vector_op("q_a_layernorm", IO_DIE, m=norm_m, n=q_lora_rank, dtype_bytes=dtype_bytes),
            _build_vector_op("kv_a_layernorm", IO_DIE, m=norm_m, n=kv_lora_rank, dtype_bytes=dtype_bytes),
            _build_vector_op("rope", IO_DIE, m=norm_m, n=qk_rope_head_dim, dtype_bytes=dtype_bytes),
        ]
    )

    if attn_strategy is AttnParallelStrategy.DP:
        q_rope_proj_m = local_bs
        q_rope_proj_n = num_attention_heads * qk_rope_head_dim
        q_rope_vector_m = mul_expr(local_bs, num_attention_heads)
    elif attn_strategy is AttnParallelStrategy.CP:
        q_rope_proj_m = bs
        q_rope_proj_n = num_attention_heads * qk_rope_head_dim
        q_rope_vector_m = mul_expr(bs, num_attention_heads)
    else:
        q_rope_proj_m = bs
        q_rope_proj_n = mul_expr(local_heads, qk_rope_head_dim)
        q_rope_vector_m = mul_expr(bs, local_heads)

    ops.extend(
        [
            _build_bmm_op(
                "q_rope_proj",
                b=1,
                m=q_rope_proj_m,
                n=q_rope_proj_n,
                k=q_lora_rank,
                dtype_bytes=dtype_bytes,
            ),
            _build_vector_op("rope", PIM_CHIPLETS, m=q_rope_vector_m, n=qk_rope_head_dim, dtype_bytes=dtype_bytes),
        ]
    )

    if model_config.dsa:
        ops.append(_build_dsa_indexer_op(request, model_config))

    ops.append(_build_mla_attention_core_op(request, model_config))

    if attn_strategy is AttnParallelStrategy.CP:
        ops.append(_chip_output(mul_expr(dtype_bytes, bs, hidden_size)))
        ops.append(_chip_input(mul_expr(dtype_bytes, local_bs, hidden_size)))
    elif attn_strategy is AttnParallelStrategy.TP:
        ops.append(_chip_output(mul_expr(dtype_bytes, bs, hidden_size)))
        ops.append(_chip_input(mul_expr(dtype_bytes, local_bs, hidden_size)))

    ops.extend(
        [
            _build_vector_op("residual", IO_DIE, m=local_bs, n=hidden_size, dtype_bytes=dtype_bytes),
            _build_vector_op("rms_norm", IO_DIE, m=local_bs, n=hidden_size, dtype_bytes=dtype_bytes),
            _chip_output(mul_expr(dtype_bytes, local_bs, hidden_size)),
        ]
    )
    return tuple(ops)


def _build_dsa_indexer_op(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> DSAOp:
    bs = request.batch_size
    seq_len = request.seq_len
    dtype_bytes = request.dtype_bytes
    num_of_chip = request.num_of_chip
    attn_strategy = request.attn_parallel_strategy

    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    dsa_len = _require_model_int_attr(model_config, "dsa_len")
    indexer_num_heads = _require_model_int_attr(model_config, "indexer_num_heads")
    indexer_head_dim = _require_model_int_attr(model_config, "indexer_head_dim")

    local_bs = ceil_div_expr(bs, num_of_chip)
    local_seq_len = ceil_div_expr(seq_len, num_of_chip)
    local_indexer_heads = ceil_div_expr(indexer_num_heads, num_of_chip)
    full_qkw_width = indexer_num_heads * indexer_head_dim + indexer_head_dim + indexer_num_heads
    local_qkw_width = mul_expr(local_indexer_heads, indexer_head_dim) + indexer_head_dim + local_indexer_heads

    if attn_strategy is AttnParallelStrategy.TP:
        qkw_m = bs
        qkw_n = local_qkw_width
        qk_b = bs
        qk_m = local_indexer_heads
        qk_n = seq_len
        relu_m = mul_expr(bs, local_indexer_heads)
        reduce_m = bs
    elif attn_strategy is AttnParallelStrategy.CP:
        qkw_m = local_bs
        qkw_n = full_qkw_width
        qk_b = bs
        qk_m = indexer_num_heads
        qk_n = local_seq_len
        relu_m = mul_expr(bs, indexer_num_heads)
        reduce_m = bs
    else:
        qkw_m = local_bs
        qkw_n = full_qkw_width
        qk_b = local_bs
        qk_m = indexer_num_heads
        qk_n = seq_len
        relu_m = mul_expr(local_bs, indexer_num_heads)
        reduce_m = local_bs

    sub_ops: list[CoreSubOp | CommOp] = [
        _build_bmm_op(
            "indexer_qkw_projection",
            b=1,
            m=qkw_m,
            n=qkw_n,
            k=hidden_size,
            dtype_bytes=dtype_bytes,
        ),
    ]

    if attn_strategy is AttnParallelStrategy.CP:
        sub_ops.extend(
            [
                _chip_output(mul_expr(dtype_bytes, local_bs, full_qkw_width)),
                _chip_input(mul_expr(dtype_bytes, bs, full_qkw_width)),
            ]
        )

    sub_ops.extend(
        [
            _build_bmm_op(
                "indexer_qk_score",
                b=qk_b,
                m=qk_m,
                n=qk_n,
                k=indexer_head_dim,
                dtype_bytes=dtype_bytes,
            ),
            _build_vector_op("relu", PIM_CHIPLETS, m=relu_m, n=qk_n, dtype_bytes=dtype_bytes),
            _build_vector_op(
                "indexer_weighted_head_reduce",
                PIM_CHIPLETS,
                m=reduce_m,
                n=qk_n,
                dtype_bytes=dtype_bytes,
            ),
        ]
    )

    # DP/CP 下不生成 score 通信：这里是有意忽略 seqlen 个 score 的通信量，
    # 不是表示全局 top-k selection 完全没有代价。TP 仍保留 score 回传与 selected view 下发。
    if attn_strategy is AttnParallelStrategy.TP:
        sub_ops.extend(
            [
                _chip_output(mul_expr(dtype_bytes, bs, seq_len)),
                _chip_input(mul_expr(dtype_bytes, bs, dsa_len)),
            ]
        )

    return DSAOp(op_kind="lightning_indexer", location=PIM_CHIPLETS, sub_ops=tuple(sub_ops))


def _history_len_for_mla_attention(request: ModelMappingRequest, model_config: ModelConfigBase) -> SymbolicDim:
    if model_config.dsa:
        history_len = _require_model_int_attr(model_config, "dsa_len")
    else:
        history_len = request.seq_len
    if request.attn_parallel_strategy is AttnParallelStrategy.CP:
        # CP 只处理本地历史分片；DSA 时这里得到 local_dsa_len。
        # 其建模假设是 top-k token 在各 chip 中近似均匀分布。
        return ceil_div_expr(history_len, request.num_of_chip)
    return history_len


def _build_mla_attention_core_op(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> AttentionCoreOp:
    bs = request.batch_size
    dtype_bytes = request.dtype_bytes
    num_of_chip = request.num_of_chip

    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    num_attention_heads = _require_model_int_attr(model_config, "num_attention_heads")
    q_lora_rank = _require_model_int_attr(model_config, "q_lora_rank")
    kv_lora_rank = _require_model_int_attr(model_config, "kv_lora_rank")
    qk_rope_head_dim = _require_model_int_attr(model_config, "qk_rope_head_dim")

    local_bs = ceil_div_expr(bs, num_of_chip)
    local_heads = ceil_div_expr(num_attention_heads, num_of_chip)
    history_len = _history_len_for_mla_attention(request, model_config)

    if request.attn_parallel_strategy is AttnParallelStrategy.DP:
        b = mul_expr(local_bs, num_attention_heads)
        vector_m = b
        head_reduce_m = local_bs
        softmax_kind = "softmax"
        sv_kind = "sv_latent"
        head_reduce_kind = "head_reduce_sum"
    elif request.attn_parallel_strategy is AttnParallelStrategy.CP:
        b = mul_expr(bs, num_attention_heads)
        vector_m = b
        head_reduce_m = bs
        softmax_kind = "flash_attention_local_softmax_stats"
        sv_kind = "sv_latent_partial"
        head_reduce_kind = "head_reduce_sum"
    else:
        b = mul_expr(bs, local_heads)
        vector_m = b
        head_reduce_m = bs
        softmax_kind = "softmax"
        sv_kind = "sv_latent"
        head_reduce_kind = "local_head_reduce_sum"

    # MLA mat absorb 将 nope QK、RoPE QK、SV latent 与 VO absorb 放进同一个 attention core。
    sub_ops: tuple[CoreSubOp, ...] = (
        _build_bmm_op(
            "qk_nope_absorb_q",
            b=b,
            m=1,
            n=kv_lora_rank,
            k=q_lora_rank,
            dtype_bytes=dtype_bytes,
        ),
        _build_bmm_op(
            "qk_nope",
            b=b,
            m=1,
            n=history_len,
            k=kv_lora_rank,
            dtype_bytes=dtype_bytes,
        ),
        _build_bmm_op(
            "qk_rope",
            b=b,
            m=1,
            n=history_len,
            k=qk_rope_head_dim,
            dtype_bytes=dtype_bytes,
        ),
        _build_vector_op("qk_score_add", PIM_CHIPLETS, m=vector_m, n=history_len, dtype_bytes=dtype_bytes),
        _build_vector_op(softmax_kind, PIM_CHIPLETS, m=vector_m, n=history_len, dtype_bytes=dtype_bytes),
        _build_bmm_op(
            sv_kind,
            b=b,
            m=1,
            n=kv_lora_rank,
            k=history_len,
            dtype_bytes=dtype_bytes,
        ),
        _build_bmm_op(
            "vo_absorb",
            b=b,
            m=1,
            n=hidden_size,
            k=kv_lora_rank,
            dtype_bytes=dtype_bytes,
        ),
        _build_vector_op(head_reduce_kind, PIM_CHIPLETS, m=head_reduce_m, n=hidden_size, dtype_bytes=dtype_bytes),
    )
    return AttentionCoreOp(op_kind="mla_attention_core", location=PIM_CHIPLETS, sub_ops=sub_ops)


def build_model_mapping_ir(request: ModelMappingRequest) -> ModelMappingIR:
    model_config = load_model_config(request.model_name)
    if request.model_name == "glm-5.2":
        raise NotImplementedError("GLM 5.2 model mapping is not supported yet")

    if model_config.attn_type == "gqa":
        ops = list(_build_gqa_attention_ops(request, model_config))
        if model_config.ffn_type == "dense":
            ops.extend(_build_dense_ffn_ops(request, model_config, tail_uses_local_bs=True))
        elif model_config.ffn_type == "moe":
            ops.extend(_build_moe_ffn_ops(request, model_config, tail_uses_local_bs=True))
        else:
            raise NotImplementedError(f"Unsupported ffn_type: {model_config.ffn_type}")
    elif model_config.attn_type == "mla" and request.model_name in {"deepseek-v3", "deepseek-v3.2"}:
        ops = list(_build_mla_attention_ops(request, model_config))
        ops.extend(_build_moe_ffn_ops(request, model_config, tail_uses_local_bs=True))
    else:
        raise NotImplementedError(f"Unsupported attention type: {model_config.attn_type}")

    return ModelMappingIR(
        model_name=request.model_name,
        num_of_chip=request.num_of_chip,
        representative_chip_id=0,
        attn_parallel_strategy=request.attn_parallel_strategy,
        ffn_parallel_strategy=request.ffn_parallel_strategy,
        model_attn_type=model_config.attn_type,
        model_dsa=model_config.dsa,
        model_ffn_type=model_config.ffn_type,
        model_use_qk_norm=model_config.use_qk_norm,
        batch_size=request.batch_size,
        seq_len=request.seq_len,
        dtype_bytes=request.dtype_bytes,
        model_config=model_config,
        ops=tuple(ops),
    )


__all__ = [
    "SymbolicDim",
    "AttnParallelStrategy",
    "FFNParallelStrategy",
    "BatchedMatmulOp",
    "AttentionCoreOp",
    "FFNCoreOp",
    "DSAOp",
    "VectorOp",
    "CommOp",
    "LayerOp",
    "ModelMappingRequest",
    "ModelMappingIR",
    "build_model_mapping_ir",
]

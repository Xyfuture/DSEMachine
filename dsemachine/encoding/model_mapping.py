from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from sympy import Expr, ceiling

from dsemachine.config.model_config import ModelConfigBase, load_model_config


SymbolicDim: TypeAlias = int | Expr
LayerOp: TypeAlias = "BatchedMatmulOp | AttentionCoreOp | FFNCoreOp | VectorOp | CommOp"

_ALLOWED_LOCATIONS = {"io_die", "pim_chiplet"}


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
    这里的运算量用输出张量元素个数表示，因此 output_elements = B * M * N。
    dtype_bytes 表示该算子的输出元素按多少 byte 计。
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

    注意这里的 m/n 含义与 BMM 不同：
    - m: 一共要做多少个 vector op
    - n: 每个 vector 的维度
    运算量同样用输出元素个数表示，因此 output_elements = M * N。
    """

    op_kind: str
    location: str
    m: SymbolicDim
    n: SymbolicDim
    dtype_bytes: int
    output_elements: SymbolicDim = field(init=False)

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        _validate_symbolic_dim("m", self.m)
        _validate_symbolic_dim("n", self.n)
        _validate_positive_int("dtype_bytes", self.dtype_bytes)
        object.__setattr__(self, "output_elements", mul_expr(self.m, self.n))


@dataclass(frozen=True)
class CommOp:
    """表示一次单向通信。

    data_bytes 是本次通信的数据量，单位为 byte。
    location 记录发起侧：
    - io_input / io_output / io2pim 记为 io_die
    - pim2io 记为 pim_chiplet
    """

    op_kind: str
    location: str
    data_bytes: SymbolicDim

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        _validate_symbolic_dim("data_bytes", self.data_bytes)


@dataclass(frozen=True)
class AttentionCoreOp:
    """表示 attention core 三段操作：qk -> softmax -> sv。

    三个子操作都发生在 pim_chiplet 上，其中：
    - qk_matmul: Q * K
    - softmax: 对 attention score 做 softmax
    - sv_matmul: softmax score * V
    """

    op_kind: str
    location: str
    qk_matmul: BatchedMatmulOp
    softmax: VectorOp
    sv_matmul: BatchedMatmulOp

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        if not isinstance(self.qk_matmul, BatchedMatmulOp):
            raise TypeError("qk_matmul must be a BatchedMatmulOp")
        if not isinstance(self.softmax, VectorOp):
            raise TypeError("softmax must be a VectorOp")
        if not isinstance(self.sv_matmul, BatchedMatmulOp):
            raise TypeError("sv_matmul must be a BatchedMatmulOp")


@dataclass(frozen=True)
class FFNCoreOp:
    """表示 FFN core 三段操作：up&gate -> activation -> down。"""

    op_kind: str
    location: str
    up_gate_matmul: BatchedMatmulOp
    activation: VectorOp
    down_matmul: BatchedMatmulOp

    def __post_init__(self) -> None:
        _validate_op_kind(self.op_kind)
        _validate_location(self.location)
        if not isinstance(self.up_gate_matmul, BatchedMatmulOp):
            raise TypeError("up_gate_matmul must be a BatchedMatmulOp")
        if not isinstance(self.activation, VectorOp):
            raise TypeError("activation must be a VectorOp")
        if not isinstance(self.down_matmul, BatchedMatmulOp):
            raise TypeError("down_matmul must be a BatchedMatmulOp")


@dataclass(frozen=True)
class ModelMappingRequest:
    """model mapping 的输入请求。

    当前仍是单层、单个代表 chip 的视角：
    - batch_size / seq_len 支持整数和 sympy 符号
    - dtype_bytes 由调用方显式给出
    """

    model_name: str
    num_pim_chiplets: int
    attn_parallel_strategy: AttnParallelStrategy
    ffn_parallel_strategy: FFNParallelStrategy
    batch_size: SymbolicDim
    seq_len: SymbolicDim
    dtype_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise TypeError("model_name must be a non-empty str")
        _validate_positive_int("num_pim_chiplets", self.num_pim_chiplets)
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

    ops 表示单个代表 chip 需要执行的本地操作列表，
    当前不是全模型全层展开，也不是所有 chip 的全集合。
    """

    model_name: str
    num_pim_chiplets: int
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
        _validate_positive_int("num_pim_chiplets", self.num_pim_chiplets)
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


def _build_comm_op(op_kind: str, location: str, data_bytes: SymbolicDim) -> CommOp:
    return CommOp(op_kind=op_kind, location=location, data_bytes=data_bytes)


def _build_bmm_op(
    op_kind: str,
    location: str,
    *,
    b: SymbolicDim,
    m: SymbolicDim,
    n: SymbolicDim,
    k: SymbolicDim,
    dtype_bytes: int,
) -> BatchedMatmulOp:
    return BatchedMatmulOp(
        op_kind=op_kind,
        location=location,
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


def _build_gqa_attention_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> tuple[LayerOp, ...]:
    bs = request.batch_size
    seq_len = request.seq_len
    dtype_bytes = request.dtype_bytes
    num_chiplets = request.num_pim_chiplets

    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    num_attention_heads = _require_model_int_attr(model_config, "num_attention_heads")
    num_key_value_heads = _require_model_int_attr(model_config, "num_key_value_heads")
    head_dim = _require_model_int_attr(model_config, "head_dim")
    grouped_heads = _exact_div_int(
        num_attention_heads,
        num_key_value_heads,
        "num_attention_heads / num_key_value_heads",
    )

    local_bs = ceil_div_expr(bs, num_chiplets)
    local_seq_len = ceil_div_expr(seq_len, num_chiplets)
    local_kv_heads = ceil_div_expr(num_key_value_heads, num_chiplets)
    local_attention_heads = ceil_div_expr(num_attention_heads, num_chiplets)
    local_hidden_size = ceil_div_expr(hidden_size, num_chiplets)
    qkv_proj_width = hidden_size + 2 * num_key_value_heads * head_dim
    local_qkv_proj_width = ceil_div_expr(qkv_proj_width, num_chiplets)

    ops: list[LayerOp] = [
        _build_comm_op(
            "io_input",
            "io_die",
            mul_expr(dtype_bytes, bs, hidden_size),
        ),
        _build_vector_op(
            "rms_norm",
            "io_die",
            m=bs,
            n=hidden_size,
            dtype_bytes=dtype_bytes,
        ),
    ]

    if request.attn_parallel_strategy is AttnParallelStrategy.TP:
        step3_bytes = mul_expr(dtype_bytes, bs, hidden_size)
        qkv_proj_bmm = _build_bmm_op(
            "qkv_proj",
            "pim_chiplet",
            b=1,
            m=bs,
            n=local_qkv_proj_width,
            k=hidden_size,
            dtype_bytes=dtype_bytes,
        )
        step5_bytes = mul_expr(dtype_bytes, bs, local_qkv_proj_width)
    else:
        step3_bytes = mul_expr(dtype_bytes, local_bs, hidden_size)
        qkv_proj_bmm = _build_bmm_op(
            "qkv_proj",
            "pim_chiplet",
            b=1,
            m=local_bs,
            n=qkv_proj_width,
            k=hidden_size,
            dtype_bytes=dtype_bytes,
        )
        step5_bytes = mul_expr(dtype_bytes, local_bs, qkv_proj_width)

    ops.extend(
        [
            _build_comm_op("io2pim", "io_die", step3_bytes),
            qkv_proj_bmm,
            _build_comm_op("pim2io", "pim_chiplet", step5_bytes),
        ]
    )

    if model_config.use_qk_norm:
        ops.append(
            _build_vector_op(
                "qk_norm",
                "io_die",
                m=mul_expr(bs, num_attention_heads + num_key_value_heads),
                n=head_dim,
                dtype_bytes=dtype_bytes,
            )
        )

    ops.append(
        _build_vector_op(
            "rope",
            "io_die",
            m=mul_expr(bs, num_attention_heads + num_key_value_heads),
            n=head_dim,
            dtype_bytes=dtype_bytes,
        )
    )

    if request.attn_parallel_strategy is AttnParallelStrategy.DP:
        step7_bytes = mul_expr(dtype_bytes, local_bs, hidden_size)
        qk_bmm = _build_bmm_op(
            "qk",
            "pim_chiplet",
            b=mul_expr(local_bs, num_key_value_heads),
            m=grouped_heads,
            n=seq_len,
            k=head_dim,
            dtype_bytes=dtype_bytes,
        )
        softmax = _build_vector_op(
            "softmax",
            "pim_chiplet",
            m=mul_expr(local_bs, num_attention_heads),
            n=seq_len,
            dtype_bytes=dtype_bytes,
        )
        sv_bmm = _build_bmm_op(
            "sv",
            "pim_chiplet",
            b=mul_expr(local_bs, num_key_value_heads),
            m=grouped_heads,
            n=head_dim,
            k=seq_len,
            dtype_bytes=dtype_bytes,
        )
        step9_bytes = mul_expr(dtype_bytes, local_bs, num_attention_heads, head_dim)
        step10_bytes = mul_expr(dtype_bytes, local_bs, hidden_size)
        o_proj_bmm = _build_bmm_op(
            "o_proj",
            "pim_chiplet",
            b=1,
            m=local_bs,
            n=hidden_size,
            k=hidden_size,
            dtype_bytes=dtype_bytes,
        )
        step12_bytes = mul_expr(dtype_bytes, local_bs, hidden_size)
    elif request.attn_parallel_strategy is AttnParallelStrategy.CP:
        step7_bytes = mul_expr(dtype_bytes, bs, hidden_size)
        qk_bmm = _build_bmm_op(
            "qk",
            "pim_chiplet",
            b=mul_expr(bs, num_key_value_heads),
            m=grouped_heads,
            n=local_seq_len,
            k=head_dim,
            dtype_bytes=dtype_bytes,
        )
        softmax = _build_vector_op(
            "softmax",
            "pim_chiplet",
            m=mul_expr(bs, num_attention_heads),
            n=local_seq_len,
            dtype_bytes=dtype_bytes,
        )
        sv_bmm = _build_bmm_op(
            "sv",
            "pim_chiplet",
            b=mul_expr(bs, num_key_value_heads),
            m=grouped_heads,
            n=head_dim,
            k=local_seq_len,
            dtype_bytes=dtype_bytes,
        )
        step9_bytes = mul_expr(dtype_bytes, bs, num_attention_heads, head_dim)
        step10_bytes = mul_expr(dtype_bytes, local_bs, hidden_size)
        o_proj_bmm = _build_bmm_op(
            "o_proj",
            "pim_chiplet",
            b=1,
            m=local_bs,
            n=hidden_size,
            k=hidden_size,
            dtype_bytes=dtype_bytes,
        )
        step12_bytes = mul_expr(dtype_bytes, local_bs, hidden_size)
    else:
        step7_bytes = mul_expr(dtype_bytes, bs, local_attention_heads, head_dim)
        qk_bmm = _build_bmm_op(
            "qk",
            "pim_chiplet",
            b=mul_expr(bs, local_kv_heads),
            m=grouped_heads,
            n=seq_len,
            k=head_dim,
            dtype_bytes=dtype_bytes,
        )
        softmax = _build_vector_op(
            "softmax",
            "pim_chiplet",
            m=mul_expr(bs, local_kv_heads, grouped_heads),
            n=seq_len,
            dtype_bytes=dtype_bytes,
        )
        sv_bmm = _build_bmm_op(
            "sv",
            "pim_chiplet",
            b=mul_expr(bs, local_kv_heads),
            m=grouped_heads,
            n=head_dim,
            k=seq_len,
            dtype_bytes=dtype_bytes,
        )
        step9_bytes = mul_expr(dtype_bytes, bs, local_kv_heads, grouped_heads, head_dim)
        step10_bytes = mul_expr(dtype_bytes, bs, local_hidden_size)
        o_proj_bmm = _build_bmm_op(
            "o_proj",
            "pim_chiplet",
            b=1,
            m=bs,
            n=hidden_size,
            k=local_hidden_size,
            dtype_bytes=dtype_bytes,
        )
        step12_bytes = mul_expr(dtype_bytes, bs, hidden_size)

    ops.extend(
        [
            _build_comm_op("io2pim", "io_die", step7_bytes),
            AttentionCoreOp(
                op_kind="attention_core",
                location="pim_chiplet",
                qk_matmul=qk_bmm,
                softmax=softmax,
                sv_matmul=sv_bmm,
            ),
            _build_comm_op("pim2io", "pim_chiplet", step9_bytes),
            _build_comm_op("io2pim", "io_die", step10_bytes),
            o_proj_bmm,
            _build_comm_op("pim2io", "pim_chiplet", step12_bytes),
            _build_vector_op(
                "residual",
                "io_die",
                m=bs,
                n=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            _build_vector_op(
                "rms_norm",
                "io_die",
                m=bs,
                n=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            _build_comm_op("io2pim", "io_die", mul_expr(dtype_bytes, bs, hidden_size)),
        ]
    )
    return tuple(ops)


def _build_dense_ffn_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> tuple[LayerOp, ...]:
    bs = request.batch_size
    dtype_bytes = request.dtype_bytes
    num_chiplets = request.num_pim_chiplets
    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    intermediate_size = _require_model_int_attr(model_config, "intermediate_size")

    local_intermediate_size = ceil_div_expr(intermediate_size, num_chiplets)
    local_two_intermediate_size = ceil_div_expr(2 * intermediate_size, num_chiplets)

    return (
        FFNCoreOp(
            op_kind="ffn_core",
            location="pim_chiplet",
            up_gate_matmul=_build_bmm_op(
                "up_gate",
                "pim_chiplet",
                b=1,
                m=bs,
                n=local_two_intermediate_size,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            activation=_build_vector_op(
                "silu",
                "pim_chiplet",
                m=bs,
                n=local_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
            down_matmul=_build_bmm_op(
                "down",
                "pim_chiplet",
                b=1,
                m=bs,
                n=hidden_size,
                k=local_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
        ),
        _build_comm_op("pim2io", "pim_chiplet", mul_expr(dtype_bytes, bs, hidden_size)),
        _build_vector_op("residual", "io_die", m=bs, n=hidden_size, dtype_bytes=dtype_bytes),
        _build_comm_op("io_output", "io_die", mul_expr(dtype_bytes, bs, hidden_size)),
    )


def _build_moe_ffn_ops(
    request: ModelMappingRequest,
    model_config: ModelConfigBase,
) -> tuple[LayerOp, ...]:
    bs = request.batch_size
    dtype_bytes = request.dtype_bytes
    num_chiplets = request.num_pim_chiplets
    hidden_size = _require_model_int_attr(model_config, "hidden_size")
    moe_intermediate_size = _require_model_int_attr(model_config, "moe_intermediate_size")
    num_experts_per_tok = _require_model_int_attr(model_config, "num_experts_per_tok")

    local_num_experts_per_tok = ceil_div_expr(num_experts_per_tok, num_chiplets)
    local_moe_intermediate_size = ceil_div_expr(moe_intermediate_size, num_chiplets)
    local_two_moe_intermediate_size = ceil_div_expr(2 * moe_intermediate_size, num_chiplets)

    if request.ffn_parallel_strategy is FFNParallelStrategy.TP:
        ffn_core = FFNCoreOp(
            op_kind="ffn_core",
            location="pim_chiplet",
            up_gate_matmul=_build_bmm_op(
                "up_gate",
                "pim_chiplet",
                b=num_experts_per_tok,
                m=bs,
                n=local_two_moe_intermediate_size,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            activation=_build_vector_op(
                "silu",
                "pim_chiplet",
                m=mul_expr(num_experts_per_tok, bs),
                n=local_moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
            down_matmul=_build_bmm_op(
                "down",
                "pim_chiplet",
                b=num_experts_per_tok,
                m=bs,
                n=hidden_size,
                k=local_moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
        )
    else:
        ffn_core = FFNCoreOp(
            op_kind="ffn_core",
            location="pim_chiplet",
            up_gate_matmul=_build_bmm_op(
                "up_gate",
                "pim_chiplet",
                b=local_num_experts_per_tok,
                m=bs,
                n=2 * moe_intermediate_size,
                k=hidden_size,
                dtype_bytes=dtype_bytes,
            ),
            activation=_build_vector_op(
                "silu",
                "pim_chiplet",
                m=mul_expr(local_num_experts_per_tok, bs),
                n=moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
            down_matmul=_build_bmm_op(
                "down",
                "pim_chiplet",
                b=local_num_experts_per_tok,
                m=bs,
                n=hidden_size,
                k=moe_intermediate_size,
                dtype_bytes=dtype_bytes,
            ),
        )

    return (
        ffn_core,
        _build_comm_op("pim2io", "pim_chiplet", mul_expr(dtype_bytes, bs, hidden_size)),
        _build_vector_op("residual", "io_die", m=bs, n=hidden_size, dtype_bytes=dtype_bytes),
        _build_comm_op("io_output", "io_die", mul_expr(dtype_bytes, bs, hidden_size)),
    )


def build_model_mapping_ir(request: ModelMappingRequest) -> ModelMappingIR:
    model_config = load_model_config(request.model_name)
    if model_config.attn_type != "gqa":
        raise NotImplementedError("Only gqa is supported for now")
    if model_config.dsa:
        raise NotImplementedError("DSA is not supported for now")

    ops = list(_build_gqa_attention_ops(request, model_config))
    if model_config.ffn_type == "dense":
        ops.extend(_build_dense_ffn_ops(request, model_config))
    elif model_config.ffn_type == "moe":
        ops.extend(_build_moe_ffn_ops(request, model_config))
    else:
        raise NotImplementedError(f"Unsupported ffn_type: {model_config.ffn_type}")

    return ModelMappingIR(
        model_name=request.model_name,
        num_pim_chiplets=request.num_pim_chiplets,
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
    "VectorOp",
    "CommOp",
    "LayerOp",
    "ModelMappingRequest",
    "ModelMappingIR",
    "build_model_mapping_ir",
]

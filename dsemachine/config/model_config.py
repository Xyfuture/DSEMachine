from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfigBase:
    attn_type: str = field(kw_only=True)
    dsa: bool = field(kw_only=True)
    ffn_type: str = field(kw_only=True)
    use_qk_norm: bool = field(kw_only=True)

    @property
    def attention_type(self) -> str:
        return self.attn_type


_MODEL_CARD_DIR = Path(__file__).resolve().parent.parent / "model_cards"
_SUPPORTED_MODEL_CARD_FILENAMES = {
    "llama-405B": "llama-405B.json",
    "qwen3-coder-480B": "qwen3-coder-480B.json",
    "deepseek-v3": "deepseek-v3.json",
    "deepseek-v3.2": "deepseek-v3.2.json",
    "glm-5.2": "glm-5.2.json",
}


class _ConfigReader:
    def __init__(self, data: object) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"config must be a dict, got {type(data).__name__}")
        self._data = data

    def _get(self, name: str) -> Any:
        if name not in self._data:
            raise KeyError(f"config missing required field: {name}")
        return self._data[name]

    def int(self, name: str) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value

    def non_negative_int(self, name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        return value

    def float(self, name: str) -> float:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a float, got {type(value).__name__}")
        return float(value)

    def bool(self, name: str) -> bool:
        value = self._get(name)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        return value

    def str(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a str, got {type(value).__name__}")
        return value

    def int_list(self, name: str) -> list[int]:
        value = self._get(name)
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list, got {type(value).__name__}")

        result: list[int] = []
        seen: set[int] = set()
        for item in value:
            checked = self.non_negative_int(f"{name} entry", item)
            if checked in seen:
                raise ValueError(f"{name} must not contain duplicates, got {checked}")
            seen.add(checked)
            result.append(checked)
        return result


def _require_hidden_act(reader: _ConfigReader) -> str:
    hidden_act = reader.str("hidden_act")
    if hidden_act != "silu":
        raise ValueError(f"Unsupported hidden_act: {hidden_act}")
    return hidden_act


def _require_exact_str(reader: _ConfigReader, name: str, expected: str) -> str:
    value = reader.str(name)
    if value != expected:
        raise ValueError(f"Unsupported {name}: {value}")
    return value


def _require_exact_bool(reader: _ConfigReader, name: str, expected: bool) -> bool:
    value = reader.bool(name)
    if value is not expected:
        raise ValueError(f"Unsupported {name}: {value}")
    return value


def _validate_attention_layout(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
) -> None:
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads, "
            f"got hidden_size={hidden_size}, num_attention_heads={num_attention_heads}"
        )
    if num_key_value_heads is not None and num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads, "
            f"got num_attention_heads={num_attention_heads}, "
            f"num_key_value_heads={num_key_value_heads}"
        )


def _validate_experts_per_tok(
    num_experts_per_tok: int,
    total_experts: int,
    total_experts_name: str,
) -> None:
    if num_experts_per_tok > total_experts:
        raise ValueError(
            f"num_experts_per_tok must be <= {total_experts_name}, "
            f"got {num_experts_per_tok} > {total_experts}"
        )


def _validate_mlp_only_layers(
    mlp_only_layers: list[int],
    num_hidden_layers: int,
) -> None:
    for layer_idx in mlp_only_layers:
        if layer_idx >= num_hidden_layers:
            raise ValueError(
                "mlp_only_layers entries must be within [0, num_hidden_layers), "
                f"got {layer_idx} for num_hidden_layers={num_hidden_layers}"
            )


@dataclass
class LlamaModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    intermediate_size: int
    hidden_act: str
    head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    mlp_bias: bool
    rope_theta: float
    num_hidden_layers: int
    attn_type: str = field(default="gqa", kw_only=True)
    dsa: bool = field(default=False, kw_only=True)
    ffn_type: str = field(default="dense", kw_only=True)
    use_qk_norm: bool = field(default=False, kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LlamaModelConfig":
        reader = _ConfigReader(data)
        hidden_size = reader.int("hidden_size")
        num_attention_heads = reader.int("num_attention_heads")
        num_key_value_heads = reader.int("num_key_value_heads")
        _validate_attention_layout(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
        )
        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=reader.int("max_position_embeddings"),
            intermediate_size=reader.int("intermediate_size"),
            hidden_act=_require_hidden_act(reader),
            head_dim=reader.int("head_dim"),
            rms_norm_eps=reader.float("rms_norm_eps"),
            attention_bias=reader.bool("attention_bias"),
            mlp_bias=reader.bool("mlp_bias"),
            rope_theta=reader.float("rope_theta"),
            num_hidden_layers=reader.int("num_hidden_layers"),
            attn_type=_require_exact_str(reader, "attn_type", "gqa"),
            dsa=_require_exact_bool(reader, "dsa", False),
            ffn_type=_require_exact_str(reader, "ffn_type", "dense"),
            use_qk_norm=_require_exact_bool(reader, "use_qk_norm", False),
        )


@dataclass
class Qwen3MoEModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    decoder_sparse_step: int
    mlp_only_layers: list[int]
    hidden_act: str
    head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    rope_theta: float
    shared_expert_intermediate_size: int | None
    attn_type: str = field(default="gqa", kw_only=True)
    dsa: bool = field(default=False, kw_only=True)
    ffn_type: str = field(default="moe", kw_only=True)
    use_qk_norm: bool = field(default=True, kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3MoEModelConfig":
        reader = _ConfigReader(data)
        hidden_size = reader.int("hidden_size")
        num_attention_heads = reader.int("num_attention_heads")
        num_key_value_heads = reader.int("num_key_value_heads")
        _validate_attention_layout(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
        )

        num_experts = reader.int("num_experts")
        num_experts_per_tok = reader.int("num_experts_per_tok")
        _validate_experts_per_tok(num_experts_per_tok, num_experts, "num_experts")

        num_hidden_layers = reader.int("num_hidden_layers")
        mlp_only_layers = reader.int_list("mlp_only_layers")
        _validate_mlp_only_layers(mlp_only_layers, num_hidden_layers)

        shared_expert_intermediate_size = reader._get("shared_expert_intermediate_size")
        if shared_expert_intermediate_size is not None:
            raise NotImplementedError("shared expert is not implemented")

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=reader.int("max_position_embeddings"),
            intermediate_size=reader.int("intermediate_size"),
            moe_intermediate_size=reader.int("moe_intermediate_size"),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_hidden_layers=num_hidden_layers,
            decoder_sparse_step=reader.int("decoder_sparse_step"),
            mlp_only_layers=mlp_only_layers,
            hidden_act=_require_hidden_act(reader),
            head_dim=reader.int("head_dim"),
            rms_norm_eps=reader.float("rms_norm_eps"),
            attention_bias=reader.bool("attention_bias"),
            rope_theta=reader.float("rope_theta"),
            shared_expert_intermediate_size=None,
            attn_type=_require_exact_str(reader, "attn_type", "gqa"),
            dsa=_require_exact_bool(reader, "dsa", False),
            ffn_type=_require_exact_str(reader, "ffn_type", "moe"),
            use_qk_norm=_require_exact_bool(reader, "use_qk_norm", True),
        )


@dataclass
class DeepseekV3ModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts_per_tok: int
    n_routed_experts: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    rope_theta: float
    hidden_act: str
    num_nextn_predict_layers: int
    attn_type: str = field(default="mla", kw_only=True)
    dsa: bool = field(default=False, kw_only=True)
    ffn_type: str = field(default="moe", kw_only=True)
    use_qk_norm: bool = field(default=False, kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeepseekV3ModelConfig":
        reader = _ConfigReader(data)
        hidden_size = reader.int("hidden_size")
        num_attention_heads = reader.int("num_attention_heads")
        _validate_attention_layout(hidden_size, num_attention_heads)

        n_routed_experts = reader.int("n_routed_experts")
        num_experts_per_tok = reader.int("num_experts_per_tok")
        _validate_experts_per_tok(
            num_experts_per_tok,
            n_routed_experts,
            "n_routed_experts",
        )

        num_nextn_predict_layers = reader.int("num_nextn_predict_layers")
        if num_nextn_predict_layers != 1:
            raise ValueError(
                "DeepseekV3 only supports num_nextn_predict_layers == 1, "
                f"got {num_nextn_predict_layers}"
            )

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=reader.int("max_position_embeddings"),
            intermediate_size=reader.int("intermediate_size"),
            moe_intermediate_size=reader.int("moe_intermediate_size"),
            num_hidden_layers=reader.int("num_hidden_layers"),
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            q_lora_rank=reader.int("q_lora_rank"),
            kv_lora_rank=reader.int("kv_lora_rank"),
            qk_nope_head_dim=reader.int("qk_nope_head_dim"),
            qk_rope_head_dim=reader.int("qk_rope_head_dim"),
            v_head_dim=reader.int("v_head_dim"),
            rms_norm_eps=reader.float("rms_norm_eps"),
            attention_bias=reader.bool("attention_bias"),
            rope_theta=reader.float("rope_theta"),
            hidden_act=_require_hidden_act(reader),
            num_nextn_predict_layers=num_nextn_predict_layers,
            attn_type=_require_exact_str(reader, "attn_type", "mla"),
            dsa=_require_exact_bool(reader, "dsa", False),
            ffn_type=_require_exact_str(reader, "ffn_type", "moe"),
            use_qk_norm=_require_exact_bool(reader, "use_qk_norm", False),
        )


@dataclass
class DeepseekV32ModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts_per_tok: int
    n_routed_experts: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    rope_theta: float
    hidden_act: str
    num_nextn_predict_layers: int
    lightning_index_dim: int
    indexer_num_heads: int
    indexer_head_dim: int
    dsa_len: int
    topk_sharing: bool
    attn_type: str = field(default="mla", kw_only=True)
    dsa: bool = field(default=True, kw_only=True)
    ffn_type: str = field(default="moe", kw_only=True)
    use_qk_norm: bool = field(default=False, kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeepseekV32ModelConfig":
        reader = _ConfigReader(data)
        hidden_size = reader.int("hidden_size")
        num_attention_heads = reader.int("num_attention_heads")
        _validate_attention_layout(hidden_size, num_attention_heads)

        n_routed_experts = reader.int("n_routed_experts")
        num_experts_per_tok = reader.int("num_experts_per_tok")
        _validate_experts_per_tok(
            num_experts_per_tok,
            n_routed_experts,
            "n_routed_experts",
        )

        num_nextn_predict_layers = reader.int("num_nextn_predict_layers")
        if num_nextn_predict_layers != 1:
            raise ValueError(
                "DeepseekV3 only supports num_nextn_predict_layers == 1, "
                f"got {num_nextn_predict_layers}"
            )

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=reader.int("max_position_embeddings"),
            intermediate_size=reader.int("intermediate_size"),
            moe_intermediate_size=reader.int("moe_intermediate_size"),
            num_hidden_layers=reader.int("num_hidden_layers"),
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            q_lora_rank=reader.int("q_lora_rank"),
            kv_lora_rank=reader.int("kv_lora_rank"),
            qk_nope_head_dim=reader.int("qk_nope_head_dim"),
            qk_rope_head_dim=reader.int("qk_rope_head_dim"),
            v_head_dim=reader.int("v_head_dim"),
            rms_norm_eps=reader.float("rms_norm_eps"),
            attention_bias=reader.bool("attention_bias"),
            rope_theta=reader.float("rope_theta"),
            hidden_act=_require_hidden_act(reader),
            num_nextn_predict_layers=num_nextn_predict_layers,
            lightning_index_dim=reader.int("lightning_index_dim"),
            indexer_num_heads=reader.int("indexer_num_heads"),
            indexer_head_dim=reader.int("inderxer_head_dim"),
            dsa_len=reader.int("dsa_len"),
            topk_sharing=_require_exact_bool(reader, "topk_sharing", False),
            attn_type=_require_exact_str(reader, "attn_type", "mla"),
            dsa=_require_exact_bool(reader, "dsa", True),
            ffn_type=_require_exact_str(reader, "ffn_type", "moe"),
            use_qk_norm=_require_exact_bool(reader, "use_qk_norm", False),
        )


@dataclass
class Glm52ModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts_per_tok: int
    n_routed_experts: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    rope_theta: float
    hidden_act: str
    num_nextn_predict_layers: int
    lightning_index_dim: int
    dsa_len: int
    topk_sharing: bool
    index_topk_freq: int
    attn_type: str = field(default="mla", kw_only=True)
    dsa: bool = field(default=True, kw_only=True)
    ffn_type: str = field(default="moe", kw_only=True)
    use_qk_norm: bool = field(default=False, kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Glm52ModelConfig":
        reader = _ConfigReader(data)
        hidden_size = reader.int("hidden_size")
        num_attention_heads = reader.int("num_attention_heads")
        _validate_attention_layout(hidden_size, num_attention_heads)

        n_routed_experts = reader.int("n_routed_experts")
        num_experts_per_tok = reader.int("num_experts_per_tok")
        _validate_experts_per_tok(
            num_experts_per_tok,
            n_routed_experts,
            "n_routed_experts",
        )

        num_nextn_predict_layers = reader.int("num_nextn_predict_layers")
        if num_nextn_predict_layers != 1:
            raise ValueError(
                "DeepseekV3 only supports num_nextn_predict_layers == 1, "
                f"got {num_nextn_predict_layers}"
            )

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=reader.int("max_position_embeddings"),
            intermediate_size=reader.int("intermediate_size"),
            moe_intermediate_size=reader.int("moe_intermediate_size"),
            num_hidden_layers=reader.int("num_hidden_layers"),
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            q_lora_rank=reader.int("q_lora_rank"),
            kv_lora_rank=reader.int("kv_lora_rank"),
            qk_nope_head_dim=reader.int("qk_nope_head_dim"),
            qk_rope_head_dim=reader.int("qk_rope_head_dim"),
            v_head_dim=reader.int("v_head_dim"),
            rms_norm_eps=reader.float("rms_norm_eps"),
            attention_bias=reader.bool("attention_bias"),
            rope_theta=reader.float("rope_theta"),
            hidden_act=_require_hidden_act(reader),
            num_nextn_predict_layers=num_nextn_predict_layers,
            lightning_index_dim=reader.int("lightning_index_dim"),
            dsa_len=reader.int("dsa_len"),
            topk_sharing=_require_exact_bool(reader, "topk_sharing", True),
            index_topk_freq=reader.int("index_topk_freq"),
            attn_type=_require_exact_str(reader, "attn_type", "mla"),
            dsa=_require_exact_bool(reader, "dsa", True),
            ffn_type=_require_exact_str(reader, "ffn_type", "moe"),
            use_qk_norm=_require_exact_bool(reader, "use_qk_norm", False),
        )


def load_model_config(
    model_name: str,
) -> LlamaModelConfig | Qwen3MoEModelConfig | DeepseekV3ModelConfig | DeepseekV32ModelConfig | Glm52ModelConfig:
    if model_name not in _SUPPORTED_MODEL_CARD_FILENAMES:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_CARD_FILENAMES))
        raise ValueError(
            f"Unsupported model_name: {model_name}. Supported values: {supported}"
        )

    model_card_path = _MODEL_CARD_DIR / _SUPPORTED_MODEL_CARD_FILENAMES[model_name]
    if not model_card_path.is_file():
        raise FileNotFoundError(f"Model card not found: {model_card_path}")

    with model_card_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if model_name == "llama-405B":
        return LlamaModelConfig.from_dict(data)
    if model_name == "qwen3-coder-480B":
        return Qwen3MoEModelConfig.from_dict(data)
    if model_name == "deepseek-v3":
        return DeepseekV3ModelConfig.from_dict(data)
    if model_name == "deepseek-v3.2":
        return DeepseekV32ModelConfig.from_dict(data)
    return Glm52ModelConfig.from_dict(data)


__all__ = [
    "ModelConfigBase",
    "LlamaModelConfig",
    "Qwen3MoEModelConfig",
    "DeepseekV3ModelConfig",
    "DeepseekV32ModelConfig",
    "Glm52ModelConfig",
    "load_model_config",
]

"""
Model attribute configurations for REAM/REAP MoE compression.

This module contains MODEL_ATTRS - a mapping of model class names to their
MoE-specific attributes needed for compression operations.

Each model configuration specifies:
- moe_block: Name of the MoE block attribute in decoder layers
- gate_proj/up_proj/down_proj: Projection layer names within experts
- experts: Name of the experts container (ModuleList or fused tensor)
- fused: Whether experts use fused (gate_up_proj) vs separate projections
- router: Name of the router/gate module
- num_experts: Config attribute name for total expert count
- num_experts_per_tok: Config attribute name for top-k routing
"""

from typing import Any, Dict

# MODEL_ATTRS maps model class names to their MoE architecture details.
# These attributes are used throughout the compression pipeline to
# correctly navigate and modify model structures.
MODEL_ATTRS: Dict[str, Dict[str, Any]] = {
    # Qwen3 MoE models
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # Llama4 MoE - uses fused experts (gate_up_proj combined)
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # Mixtral MoE - uses w1/w2/w3 naming convention
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # DeepSeek V2 MoE
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # DeepSeek V3 MoE (also used by INTELLECT-3, Kimi-K2)
    "DeepseekV3ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # Kimi-K2-Thinking - DeepSeek V3 based architecture
    "KimiK2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # Ernie 4.5 MoE (Baidu)
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },

    # GLM-4 MoE (Zhipu AI)
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # GLM-4.7-Flash (glm4_moe_lite architecture)
    # Layer 0 is dense, layers 1-46 are MoE with fused experts
    "Glm4MoeLiteForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # GLM-5 (GlmMoeDsaForCausalLM) - Hybrid MoE with routed + shared experts
    # Uses FUSED experts (gate_up_proj combined) + always-active shared_experts
    "GlmMoeDsaForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # Solar Open MoE
    "SolarOpenForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # Vaetki MoE
    "VaetkiForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # MiMo V2 Flash (Xiaomi) - 309B MoE model
    "MiMoV2FlashForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # LongCat (Meituan) - 560B MoE model with identity "zero experts"
    # Uses LongcatTopkRouter where router.classifier is the gate Linear
    "LongcatCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "router",
        "router_weight_attr": "classifier.weight",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "moe_topk",
    },
    "LongcatForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "router",
        "router_weight_attr": "classifier.weight",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "moe_topk",
    },
    "LongcatFlashNgramForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "router",
        "router_weight_attr": "classifier.weight",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "moe_topk",
    },

    # MiniMax M2.5 - Uses w1/w2/w3 projections
    "MiniMaxM2ForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    # gpt-oss models (OpenAI-like architecture)
    "gpt-oss-20b": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
}


def get_model_attrs(model_class_name: str) -> Dict[str, Any] | None:
    """
    Get model attributes for a given model class name.

    Args:
        model_class_name: Name of the model class (e.g., "MixtralForCausalLM")

    Returns:
        Dictionary of model attributes if found, None otherwise
    """
    return MODEL_ATTRS.get(model_class_name)


def list_supported_models() -> list[str]:
    """Return a list of all supported model class names."""
    return sorted(MODEL_ATTRS.keys())

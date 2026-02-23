"""
Observer hook configurations for REAM/REAP MoE compression.

This module contains OBSERVER_CONFIG_REGISTRY - a mapping of model class names
to their observer hook configuration dataclasses.

Observer configs specify how to locate and hook into MoE layers during the
forward pass to collect activation statistics for compression.

Each configuration specifies:
- module_class_name_to_hook_regex: Regex pattern matching MoE block class names
- num_experts_attr_name: Attribute path to get expert count (supports dot notation)
- top_k_attr_name: Attribute path to get top-k routing value
- fused_experts: Whether the model uses fused expert implementations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ObserverHookConfig:
    """
    Base configuration for observer hooks.

    Attributes:
        module_class_name_to_hook_regex: Regex pattern matching MoE block class names to hook
        num_experts_attr_name: Attribute path for num_experts (e.g., "num_experts" or "config.n_routed_experts")
        top_k_attr_name: Attribute path for top_k routing (e.g., "top_k" or "config.num_experts_per_tok")
        fused_experts: Whether experts use fused (gate_up_proj) vs separate projections
        renormalize_router_weights: Whether to renormalize router weights after pruning
        record_pruning_metrics_only: If True, only collect metrics needed for pruning (not merging)
        track_category_expert_frequency: Track per-category expert activation frequencies
    """

    module_class_name_to_hook_regex: Optional[str] = None
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = False
    track_category_expert_frequency: bool = True


# Model-specific observer configurations
OBSERVER_CONFIG_REGISTRY: Dict[str, type[ObserverHookConfig]] = {
    # Qwen3 MoE
    "Qwen3MoeForCausalLM": type(
        "Qwen3MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Qwen3MoeSparseMoeBlock",
        },
    ),
    "NonUniformQwen3MoeForCausalLM": type(
        "NonUniformQwen3MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Qwen3MoeSparseMoeBlock",
        },
    ),

    # Llama4 MoE - uses fused experts
    "Llama4ForCausalLM": type(
        "Llama4MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Llama4TextMoe",
            "fused_experts": True,
        },
    ),

    # Mixtral MoE
    "MixtralForCausalLM": type(
        "MixtralMoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "MixtralSparseMoeBlock",
        },
    ),

    # DeepSeek V2 MoE
    "DeepseekV2ForCausalLM": type(
        "DeepSeekMoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "DeepseekV2MoE",
            "num_experts_attr_name": "experts_per_rank",  # only for ep=1!
            "top_k_attr_name": "num_experts_per_tok",
        },
    ),

    # DeepSeek V3 MoE (also used by INTELLECT-3, Kimi-K2)
    "DeepseekV3ForCausalLM": type(
        "DeepSeekV3MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "DeepseekV3MoE",
            "num_experts_attr_name": "n_routed_experts",
            "top_k_attr_name": "num_experts_per_tok",
        },
    ),

    # Kimi-K2-Thinking - DeepSeek V3 based
    "KimiK2ForCausalLM": type(
        "KimiK2MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "DeepseekV3MoE",
            "num_experts_attr_name": "n_routed_experts",
            "top_k_attr_name": "num_experts_per_tok",
        },
    ),

    # Kimi-VL - Vision-language model with DeepSeek V3-based language model
    "KimiVLForConditionalGeneration": type(
        "KimiVLMoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "DeepseekV3MoE",
            "num_experts_attr_name": "n_routed_experts",
            "top_k_attr_name": "num_experts_per_tok",
        },
    ),

    # Ernie 4.5 MoE
    "Ernie4_5_MoEForCausalLM": type(
        "Ernie4_5MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Ernie4_5_MoeMLP",
            "num_experts_attr_name": "num_local_experts",
            "top_k_attr_name": "k",
        },
    ),
    "Ernie4_5_MoeForCausalLM": type(
        "Ernie4_5_MoeObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Ernie4_5_MoeMLP",
            "num_experts_attr_name": "num_local_experts",
            "top_k_attr_name": "k",
        },
    ),

    # GLM-4 MoE
    "Glm4MoeForCausalLM": type(
        "Glm44MoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Glm4MoeMoE",
            "num_experts_attr_name": "config.n_routed_experts",
            "top_k_attr_name": "config.num_experts_per_tok",
        },
    ),

    # GLM-4.7-Flash (glm4_moe_lite)
    "Glm4MoeLiteForCausalLM": type(
        "Glm4MoeLiteObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "Glm4MoeLiteMoE",
            "num_experts_attr_name": "config.n_routed_experts",
            "top_k_attr_name": "config.num_experts_per_tok",
            "fused_experts": True,
        },
    ),

    # GLM-5 (GlmMoeDsaForCausalLM)
    "GlmMoeDsaForCausalLM": type(
        "GlmMoeDsaObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "GlmMoeDsaMoE",
            "num_experts_attr_name": "config.n_routed_experts",
            "top_k_attr_name": "config.num_experts_per_tok",
        },
    ),

    # Solar Open MoE
    "SolarOpenForCausalLM": type(
        "SolarOpenObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "SolarOpenMoE",
            "num_experts_attr_name": "config.n_routed_experts",
            "top_k_attr_name": "config.num_experts_per_tok",
        },
    ),

    # Vaetki MoE
    "VaetkiForCausalLM": type(
        "VaetkiObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "VaetkiForCausalLM",
            "num_experts_attr_name": "n_routed_experts",
            "top_k_attr_name": "num_experts_per_tok",
        },
    ),

    # MiMo V2 Flash
    "MiMoV2FlashForCausalLM": type(
        "MiMoV2FlashObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "MiMoV2FlashMoE",
            "num_experts_attr_name": "num_experts",
            "top_k_attr_name": "num_experts_per_tok",
        },
    ),

    # LongCat (Meituan) - uses router.classifier for gate weights
    "LongcatCausalLM": type(
        "LongcatMoEObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "LongcatMoE",
            "num_experts_attr_name": "config.n_routed_experts",
            "top_k_attr_name": "router.top_k",
        },
    ),
    "LongcatForCausalLM": type(
        "LongcatObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "LongcatMoE",
            "num_experts_attr_name": "config.n_routed_experts",
            "top_k_attr_name": "router.top_k",
        },
    ),

    # MiniMax M2.5
    "MiniMaxM2ForCausalLM": type(
        "MiniMaxM2ObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "MiniMaxM2SparseMoeBlock",
            "num_experts_attr_name": "experts.num_experts",
            "top_k_attr_name": "top_k",
        },
    ),

    # gpt-oss models
    "gpt-oss-20b": type(
        "GptOssObserverConfig",
        (ObserverHookConfig,),
        {
            "module_class_name_to_hook_regex": "MoE",
        },
    ),
}


def get_observer_config(model_class_name: str) -> type[ObserverHookConfig] | None:
    """
    Get observer config class for a given model class name.

    Args:
        model_class_name: Name of the model class (e.g., "MixtralForCausalLM")

    Returns:
        Observer config class if found, None otherwise
    """
    return OBSERVER_CONFIG_REGISTRY.get(model_class_name)


def create_observer_config(model_class_name: str) -> ObserverHookConfig | None:
    """
    Create an observer config instance for a given model class name.

    Args:
        model_class_name: Name of the model class (e.g., "MixtralForCausalLM")

    Returns:
        Observer config instance if found, None otherwise
    """
    config_class = get_observer_config(model_class_name)
    if config_class is None:
        return None
    return config_class()


def list_supported_observer_models() -> list[str]:
    """Return a list of all supported model class names for observation."""
    return sorted(OBSERVER_CONFIG_REGISTRY.keys())

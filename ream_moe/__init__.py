"""
REAM/REAP-style Mixture-of-Experts compression framework.

This package provides tools for compressing MoE (Mixture-of-Experts) models
using expert pruning and merging algorithms based on activation statistics.

Supported model families:
- Qwen (Qwen3Moe, NonUniformQwen3Moe)
- Llama4 (Llama4ForCausalLM)
- Mixtral (MixtralForCausalLM)
- DeepSeek (DeepseekV2ForCausalLM, DeepseekV3ForCausalLM)
- Kimi (KimiK2ForCausalLM)
- GLM (Glm4MoeForCausalLM, Glm4MoeLiteForCausalLM, GlmMoeDsaForCausalLM)
- Ernie (Ernie4_5_MoEForCausalLM, Ernie4_5_MoeForCausalLM)
- Solar (SolarOpenForCausalLM)
- Vaetki (VaetkiForCausalLM)
- MiMo (MiMoV2FlashForCausalLM)
- LongCat (LongcatCausalLM, LongcatForCausalLM)
- MiniMax (MiniMaxM2ForCausalLM)

Basic usage:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ream_moe import observe_model, prune_model, merge_model, setup_logging
    from ream_moe.pruning import PruningConfig
    from ream_moe.merging import MergeConfig

    # Optional: suppress INFO logs
    setup_logging(level="WARNING")

    # Load model
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B-MoE")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-MoE")

    # Collect activation statistics
    observer_data = observe_model(
        model,
        calibration_input_ids,
        calibration_attention_mask,
    )

    # Option 1: Prune experts (remove low-saliency experts)
    prune_config = PruningConfig(compression_ratio=0.25)
    retained_counts = prune_model(model, observer_data, prune_config)

    # Option 2: Merge experts (combine similar experts)
    merge_config = MergeConfig(target_ratio=0.75)
    retained_counts = merge_model(model, observer_data, merge_config)

    # Save compressed model
    model.save_pretrained("./compressed_model")
    tokenizer.save_pretrained("./compressed_model")
    ```
"""

import logging

# Observer - collect activation statistics
from ream_moe.observer import (
    MoEObserver,
    ObserverConfig,
    observe_model,
)

# Pruning - remove experts
from ream_moe.prune import (
    PruningConfig,
    prune_model,
    prune_layer,
    compute_experts_to_prune,
)

# Merging - combine experts
from ream_moe.merge import (
    MergeConfig,
    merge_model,
    merge_layer,
)

# Model configurations
from ream_moe.model_attr_configs import (
    MODEL_ATTRS,
    get_model_attrs,
    list_supported_models,
)

# Observer configurations
from ream_moe.observer_configs import (
    OBSERVER_CONFIG_REGISTRY,
    ObserverHookConfig,
    get_observer_config,
    create_observer_config,
    list_supported_observer_models,
)

# Model utilities
from ream_moe.model_utils import (
    get_moe_block,
    get_num_experts,
    get_top_k,
    list_moe_layers,
    ensure_model_registered,
    verify_model_config,
    print_verification_result,
)

# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    "setup_logging",
    "MoEObserver",
    "ObserverConfig",
    "observe_model",
    "PruningConfig",
    "prune_model",
    "prune_layer",
    "compute_experts_to_prune",
    "MergeConfig",
    "merge_model",
    "merge_layer",
    "MODEL_ATTRS",
    "get_model_attrs",
    "list_supported_models",
    "OBSERVER_CONFIG_REGISTRY",
    "ObserverHookConfig",
    "get_observer_config",
    "create_observer_config",
    "list_supported_observer_models",
    "get_moe_block",
    "get_num_experts",
    "get_top_k",
    "list_moe_layers",
    "ensure_model_registered",
    "verify_model_config",
    "print_verification_result",
]


def setup_logging(level: str = "WARNING") -> None:
    """
    Configure logging for the ream_moe package.

    By default, suppresses INFO messages to reduce noise. Only WARNING and
    ERROR messages will be shown.

    Args:
        level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL"

    Example:
        >>> from ream_moe import setup_logging
        >>> setup_logging(level="WARNING")  # Suppress INFO logs
    """
    # Convert string to logging level
    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Suppress INFO logs from noisy libraries
    noisy_loggers = [
        "",
        "ream_moe",
        "datasets",
        "transformers",
        "torch",
        "urllib3",
        "filelock",
        "huggingface_hub",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(numeric_level)

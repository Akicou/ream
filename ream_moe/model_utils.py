"""
Model utility functions for REAM/REAP MoE compression.

This module provides helper functions for:
- Getting MoE blocks from models
- Auto-registering unknown models
- Verifying model configurations
- Navigation utilities for model structures
"""

from __future__ import annotations

import logging
import re
from functools import reduce
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ream_moe.model_attr_configs import MODEL_ATTRS, get_model_attrs

logger = logging.getLogger(__name__)


def get_moe_block(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Get the MoE block for a specific decoder layer.

    Args:
        model: The model containing the MoE layers
        layer_idx: Index of the layer to get

    Returns:
        The MoE block module

    Raises:
        ValueError: If model class is not registered or MoE block cannot be found
    """
    model_class = model.__class__.__name__
    attrs = get_model_attrs(model_class)

    if attrs is None:
        raise ValueError(
            f"Model class '{model_class}' not registered in MODEL_ATTRS. "
            f"Supported classes: {sorted(MODEL_ATTRS.keys())}"
        )

    moe_attr_name = attrs.get("moe_block", "mlp")

    # Navigate to the layer's MoE block
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        # Vision-language models with nested language model (e.g., Kimi-VL)
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            layers = model.language_model.model.layers
        elif hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h  # GPT-2 style
    else:
        raise ValueError(f"Cannot find layers in model {model_class}")

    if layer_idx >= len(layers):
        raise ValueError(f"Layer index {layer_idx} out of range (max: {len(layers) - 1})")

    layer = layers[layer_idx]

    # Handle nested attributes (e.g., "block_sparse_moe")
    moe_block = reduce(getattr, moe_attr_name.split("."), layer)

    return moe_block


def get_num_experts(model: nn.Module, layer_idx: int = 0) -> int:
    """
    Get the number of experts in a model's MoE layer.

    Args:
        model: The model to query
        layer_idx: Layer index to check (default: 0)

    Returns:
        Number of experts in the MoE layer

    Raises:
        ValueError: If expert count cannot be determined
    """
    model_class = model.__class__.__name__
    attrs = get_model_attrs(model_class)

    if attrs is None:
        # Try to infer from model structure
        moe_block = get_moe_block(model, layer_idx)
        if hasattr(moe_block, "experts"):
            experts = moe_block.experts
            if isinstance(experts, nn.ModuleList):
                return len(experts)
            elif isinstance(experts, nn.Module) and hasattr(experts, "gate_up_proj"):
                # Fused experts - check tensor size
                return experts.gate_up_proj.shape[0]
        raise ValueError(f"Cannot determine num_experts for unregistered model {model_class}")

    # Try to get from config first
    num_experts_attr = attrs.get("num_experts", "num_experts")

    if num_experts_attr.startswith("config."):
        # Get from model.config
        config_key = num_experts_attr.split(".", 1)[1]
        if hasattr(model, "config") and hasattr(model.config, config_key):
            return getattr(model.config, config_key)

    # Try to get from MoE block attribute
    moe_block = get_moe_block(model, layer_idx)

    # Handle nested attributes
    try:
        num_experts = reduce(getattr, num_experts_attr.split("."), moe_block)
        if isinstance(num_experts, int):
            return num_experts
    except AttributeError:
        pass

    # Fallback: count experts
    if hasattr(moe_block, "experts"):
        experts = moe_block.experts
        if isinstance(experts, nn.ModuleList):
            return len(experts)
        elif isinstance(experts, nn.Module):
            # Fused experts
            if hasattr(experts, "gate_up_proj"):
                return experts.gate_up_proj.shape[0]
            elif hasattr(experts, "__len__"):
                return len(experts)

    raise ValueError(f"Cannot determine num_experts for model {model_class} at layer {layer_idx}")


def get_top_k(model: nn.Module, layer_idx: int = 0) -> int:
    """
    Get the top-k routing value for a model's MoE layer.

    Args:
        model: The model to query
        layer_idx: Layer index to check (default: 0)

    Returns:
        Top-k value (number of experts activated per token)
    """
    model_class = model.__class__.__name__
    attrs = get_model_attrs(model_class)

    if attrs is None:
        raise ValueError(f"Model class '{model_class}' not registered in MODEL_ATTRS")

    top_k_attr = attrs.get("num_experts_per_tok", "top_k")

    # Try to get from config first
    if top_k_attr.startswith("config."):
        config_key = top_k_attr.split(".", 1)[1]
        if hasattr(model, "config") and hasattr(model.config, config_key):
            return getattr(model.config, config_key)

    # Try to get from MoE block attribute
    moe_block = get_moe_block(model, layer_idx)

    try:
        top_k = reduce(getattr, top_k_attr.split("."), moe_block)
        if isinstance(top_k, int):
            return top_k
    except AttributeError:
        pass

    # Fallback: try common patterns
    for attr_name in ["top_k", "num_experts_per_tok", "k", "num_selected_experts"]:
        if hasattr(moe_block, attr_name):
            val = getattr(moe_block, attr_name)
            if isinstance(val, int):
                return val

    raise ValueError(f"Cannot determine top_k for model {model_class} at layer {layer_idx}")


def list_moe_layers(model: nn.Module) -> List[int]:
    """
    List all layer indices that contain MoE blocks.

    Args:
        model: The model to scan

    Returns:
        List of layer indices containing MoE blocks
    """
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        # Vision-language models with nested language model (e.g., Kimi-VL)
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            layers = model.language_model.model.layers
        elif hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return []

    moe_layers = []
    model_class = model.__class__.__name__
    attrs = get_model_attrs(model_class)

    moe_attr_name = attrs.get("moe_block", "mlp") if attrs else "mlp"

    for idx, layer in enumerate(layers):
        if moe_attr_name in ["mlp"] and hasattr(layer, "mlp"):
            moe_block = layer.mlp
        elif moe_attr_name in ["block_sparse_moe"] and hasattr(layer, "block_sparse_moe"):
            moe_block = layer.block_sparse_moe
        elif moe_attr_name in ["feed_forward"] and hasattr(layer, "feed_forward"):
            moe_block = layer.feed_forward
        elif moe_attr_name in ["moe"] and hasattr(layer, "moe"):
            moe_block = layer.moe
        else:
            # Try nested access
            try:
                moe_block = reduce(getattr, moe_attr_name.split("."), layer)
            except AttributeError:
                continue

        # Check if this is actually an MoE block
        if hasattr(moe_block, "experts"):
            moe_layers.append(idx)

    return moe_layers


def _infer_model_attrs(model: nn.Module) -> Dict[str, Any]:
    """
    Infer MODEL_ATTRS configuration by inspecting a loaded model.

    This analyzes the model structure to determine:
    - MoE block attribute name
    - Projection names (gate, up, down)
    - Whether experts are fused
    - Router attribute name
    - Config keys for num_experts and num_experts_per_tok

    Args:
        model: The loaded model to inspect

    Returns:
        Dictionary of inferred model attributes
    """
    attrs = {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    }

    # Try to find a layer with MoE
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        # Vision-language models with nested language model (e.g., Kimi-VL)
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            layers = model.language_model.model.layers
        elif hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None or len(layers) == 0:
        return attrs

    for layer in layers:
        # Find MoE block attribute
        for moe_attr in ["mlp", "block_sparse_moe", "moe", "feed_forward", "ffn"]:
            if hasattr(layer, moe_attr):
                moe = getattr(layer, moe_attr)
                # Check if it has experts (MoE indicator)
                if hasattr(moe, "experts"):
                    attrs["moe_block"] = moe_attr

                    # Check for fused experts
                    if hasattr(moe.experts, "gate_up_proj"):
                        attrs["fused"] = True
                        attrs["gate_proj"] = "gate_up_proj"
                        attrs["up_proj"] = "gate_up_proj"
                    elif hasattr(moe.experts, "__len__") and len(moe.experts) > 0:
                        expert = list(moe.experts)[0]
                        # Find projection names
                        for proj_name in ["gate_proj", "w3", "wi_0"]:
                            if hasattr(expert, proj_name):
                                attrs["gate_proj"] = proj_name
                                break
                        for proj_name in ["up_proj", "w1", "wi_1", "fc1"]:
                            if hasattr(expert, proj_name):
                                attrs["up_proj"] = proj_name
                                break
                        for proj_name in ["down_proj", "w2", "wo", "fc2"]:
                            if hasattr(expert, proj_name):
                                attrs["down_proj"] = proj_name
                                break

                    # Find router attribute
                    for router_name in ["gate", "router", "gating"]:
                        if hasattr(moe, router_name):
                            attrs["router"] = router_name
                            # Check if router uses classifier pattern (like LongCat)
                            router_module = getattr(moe, router_name)
                            if hasattr(router_module, "classifier") and isinstance(
                                router_module.classifier, nn.Linear
                            ):
                                attrs["router_weight_attr"] = "classifier.weight"
                            break

                    break
        if attrs["moe_block"] != "mlp" or hasattr(
            getattr(layer, "mlp", None), "experts"
        ):
            break

    # Infer num_experts config key from model.config
    if hasattr(model, "config"):
        config = model.config
        for key in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
            if hasattr(config, key):
                attrs["num_experts"] = key
                break
        for key in ["num_experts_per_tok", "top_k", "moe_k", "num_selected_experts"]:
            if hasattr(config, key):
                attrs["num_experts_per_tok"] = key
                break

    return attrs


def ensure_model_registered(model: nn.Module) -> bool:
    """
    Ensure a model is registered in MODEL_ATTRS.

    If the model class is not in MODEL_ATTRS, this function will:
    1. Analyze the model structure
    2. Generate appropriate MODEL_ATTRS configuration
    3. Inject it into MODEL_ATTRS at runtime

    Args:
        model: The loaded model to check/register

    Returns:
        True if model was already registered or successfully auto-registered,
        False if registration failed.
    """
    from ream_moe.model_attr_configs import MODEL_ATTRS

    model_class = model.__class__.__name__

    if model_class in MODEL_ATTRS:
        logger.debug(f"Model {model_class} already in MODEL_ATTRS")
        return True

    logger.info(f"Model {model_class} not in MODEL_ATTRS, attempting auto-registration...")

    try:
        # Infer attributes from the loaded model
        inferred_attrs = _infer_model_attrs(model)

        # Register the model
        MODEL_ATTRS[model_class] = inferred_attrs

        logger.info(f"Auto-registered MODEL_ATTRS for {model_class}: {inferred_attrs}")
        return True

    except Exception as e:
        logger.error(f"Failed to auto-register MODEL_ATTRS for {model_class}: {e}")
        return False


def verify_model_config(model_name: str, model: nn.Module | None = None) -> Dict[str, Any]:
    """
    Verify that all model configurations are correct for REAM/REAP compression.

    Args:
        model_name: Name of the model to verify
        model: Optional pre-loaded model instance. If None, will try to load.

    Returns:
        Dictionary with verification results:
        {
            "valid": bool,
            "model_class": str,
            "model_attrs": dict | None,
            "errors": list[str],
            "warnings": list[str],
            "details": dict,
        }
    """
    from transformers import AutoConfig

    errors = []
    warnings = []
    details = {}

    logger.info(f"Verifying model configuration for: {model_name}")

    # Step 1: Get model class name
    try:
        if model is None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_class = config.architectures[0] if config.architectures else None
            details["config_class"] = config.__class__.__name__
        else:
            model_class = model.__class__.__name__

        if not model_class:
            errors.append("Could not determine model class from config.architectures")
            return _format_verification_result(
                False, None, None, errors, warnings, details
            )

        details["model_class"] = model_class
        logger.info(f"Model class: {model_class}")

    except Exception as e:
        errors.append(f"Failed to get model config: {e}")
        return _format_verification_result(False, None, None, errors, warnings, details)

    # Step 2: Check MODEL_ATTRS
    from ream_moe.model_attr_configs import MODEL_ATTRS

    model_attrs = MODEL_ATTRS.get(model_class)
    if model_attrs is None:
        errors.append(
            f"Model class '{model_class}' not found in MODEL_ATTRS. "
            f"Supported classes: {sorted(MODEL_ATTRS.keys())}"
        )
    else:
        logger.info(f"✅ MODEL_ATTRS found for {model_class}")
        details["model_attrs"] = model_attrs

        # Verify required MODEL_ATTRS fields
        required_fields = ["moe_block", "gate_proj", "up_proj", "down_proj", "experts", "router"]
        missing_fields = [f for f in required_fields if f not in model_attrs]
        if missing_fields:
            errors.append(f"MODEL_ATTRS missing required fields: {missing_fields}")
        else:
            logger.info(f"✅ All required MODEL_ATTRS fields present")

    # Step 3: If model provided, verify structure matches MODEL_ATTRS
    if model is not None and model_attrs:
        try:
            structure_errors = _verify_model_structure(model, model_class, model_attrs)
            if structure_errors:
                errors.extend(structure_errors)
            else:
                logger.info(f"✅ Model structure matches MODEL_ATTRS")
        except Exception as e:
            errors.append(f"Failed to verify model structure: {e}")

    valid = len(errors) == 0
    return _format_verification_result(
        valid, model_class, model_attrs, errors, warnings, details
    )


def _verify_model_structure(
    model: nn.Module, model_class: str, model_attrs: Dict[str, Any]
) -> List[str]:
    """Verify that the actual model structure matches MODEL_ATTRS."""
    errors = []

    # Find a decoder layer to inspect
    layers = None
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model.model, "decoder"):
            if hasattr(model.model.decoder, "layers"):
                layers = model.model.decoder.layers
    elif hasattr(model, "language_model"):
        # Vision-language models with nested language model (e.g., Kimi-VL)
        if hasattr(model.language_model, "model"):
            if hasattr(model.language_model.model, "layers"):
                layers = model.language_model.model.layers
        elif hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None or len(layers) == 0:
        return ["Could not find any decoder layers in the model"]

    # Find first actual MoE layer (some models have first k layers as dense)
    moe_block_path = model_attrs.get("moe_block")
    if not moe_block_path:
        return ["MODEL_ATTRS missing 'moe_block' path"]

    moe_block = None
    moe_layer_idx = None

    for layer_idx in range(min(10, len(layers))):  # Check first 10 layers
        layer = layers[layer_idx]

        # Navigate to MoE block
        current = layer
        found = True
        for attr in moe_block_path.split("."):
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                found = False
                break

        if found and current is not None:
            # Check if this is actually an MoE block (has experts)
            experts_path = model_attrs.get("experts", "experts")
            parts = experts_path.split(".")
            temp = current
            for part in parts:
                if hasattr(temp, part):
                    temp = getattr(temp, part)
                else:
                    found = False
                    break

            if found and temp is not None:
                moe_block = current
                moe_layer_idx = layer_idx
                logger.info(f"✅ Found MoE block in layer {layer_idx}: {moe_block.__class__.__name__}")
                break

    if moe_block is None:
        errors.append(f"Could not find MoE block with 'experts' in first 10 layers")
        return errors

    # Check experts
    experts_path = model_attrs.get("experts")
    if experts_path:
        parts = experts_path.split(".")
        current = moe_block
        for i, part in enumerate(parts[:-1]) if len(parts) > 1 else []:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                errors.append(f"MoE block missing attribute '{part}' in experts path")
                return errors

        if hasattr(current, parts[-1]):
            experts = getattr(current, parts[-1])

            # Handle fused experts (Qwen3, Llama4, etc.)
            if model_attrs.get("fused", False):
                # For fused experts, check for gate_up_proj tensor
                if hasattr(experts, "gate_up_proj"):
                    num_experts = experts.gate_up_proj.shape[0]
                    logger.info(f"✅ Found {num_experts} fused experts (via gate_up_proj)")
                elif hasattr(experts, "__len__"):
                    try:
                        num_experts = len(experts)
                        logger.info(f"✅ Found {num_experts} experts")
                    except TypeError:
                        errors.append(f"Fused experts at '{experts_path}' has no gate_up_proj and is not subscriptable")
                else:
                    errors.append(f"Fused experts at '{experts_path}' has no gate_up_proj attribute")
            else:
                # Regular (non-fused) experts - should be ModuleList
                try:
                    num_experts = len(experts)
                    logger.info(f"✅ Found {num_experts} experts")
                except TypeError:
                    errors.append(f"Experts at '{experts_path}' is not a list/array")
        else:
            errors.append(f"MoE block missing 'experts' attribute at '{experts_path}'")

    # Check router
    router_path = model_attrs.get("router")
    if router_path and hasattr(moe_block, router_path):
        logger.info(f"✅ Found router: {getattr(moe_block, router_path).__class__.__name__}")
    elif router_path:
        errors.append(f"MoE block missing 'router' attribute at '{router_path}'")

    return errors


def _format_verification_result(
    valid: bool,
    model_class: str | None,
    model_attrs: Dict[str, Any] | None,
    errors: List[str],
    warnings: List[str],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    """Format verification results into a structured dictionary."""
    return {
        "valid": valid,
        "model_class": model_class,
        "model_attrs": model_attrs,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def print_verification_result(result: Dict[str, Any]) -> None:
    """Print verification results in a formatted way."""
    print("\n" + "=" * 70)
    print("REAM/REAP Model Configuration Verification")
    print("=" * 70)

    if result["model_class"]:
        print(f"\n📦 Model Class: {result['model_class']}")

    if result["valid"]:
        print("\n✅ Configuration is VALID for REAM/REAP compression!")
    else:
        print("\n❌ Configuration has ERRORS - compression will likely FAIL!")

    if result["model_attrs"]:
        print(f"\n🔧 MODEL_ATTRS:")
        for key, value in result["model_attrs"].items():
            print(f"   {key}: {value}")

    if result["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(result['warnings'])}):")
        for warning in result["warnings"]:
            print(f"   - {warning}")

    if result["errors"]:
        print(f"\n❌ ERRORS ({len(result['errors'])}):")
        for error in result["errors"]:
            print(f"   - {error}")

    print("\n" + "=" * 70)

"""
Pruning module for removing experts from MoE models.

This module provides functions to prune experts based on various saliency metrics:
- frequency: Experts activated least frequently
- saliency_scores: REAP-weighted expert importance
- under_average: Prune all experts below average activation

The pruned model is modified in-place and can be saved directly.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ream_moe.model_attr_configs import get_model_attrs
from ream_moe.model_utils import get_moe_block, ensure_model_registered
from ream_moe.observer import LayerObserverState

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass
class PruningConfig:
    """Configuration for expert pruning."""

    prune_method: str = "saliency_scores"  # "frequency", "saliency_scores", "under_average"
    n_experts_to_prune: int | None = None  # Number to prune (None = auto from ratio)
    compression_ratio: float = 0.25  # Fraction of experts to prune (0.25 = prune 25%)
    preserve_super_experts: bool = False  # Don't prune experts with max activation > threshold
    super_expert_quantile: float = 99.5  # Quantile threshold for super experts


def _get_num_experts_from_moe_block(
    moe_block: nn.Module,
    attrs: Dict[str, Any],
) -> int:
    """
    Count experts from the layer module itself, not from model.config.

    During pruning the global config may be updated only after all layers are
    processed. Reading model.config inside each layer can cascade the first
    layer's reduced count into later, still-unpruned layers.
    """
    experts_attr = attrs.get("experts", "experts")
    current: Any = moe_block
    for part in experts_attr.split("."):
        current = getattr(current, part)
    experts = current

    if attrs.get("fused", False) and hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[0])

    try:
        return int(len(experts))
    except TypeError:
        pass

    if hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[0])
    if hasattr(experts, "num_experts"):
        return int(experts.num_experts)
    if hasattr(moe_block, "num_experts"):
        return int(moe_block.num_experts)

    router_attr = attrs.get("router", "gate")
    if hasattr(moe_block, router_attr):
        router = getattr(moe_block, router_attr)
        router_weight_attr = attrs.get("router_weight_attr")
        if router_weight_attr:
            inner: Any = router
            for part in router_weight_attr.split("."):
                inner = getattr(inner, part)
            if isinstance(inner, torch.Tensor):
                return int(inner.shape[0])
        elif hasattr(router, "weight"):
            return int(router.weight.shape[0])

    raise ValueError(f"Cannot determine number of experts from {moe_block.__class__.__name__}")


def prune_layer(
    model: nn.Module,
    layer_idx: int,
    experts_to_prune: torch.Tensor,
) -> int:
    """
    Prune specified experts from a single MoE layer.

    Args:
        model: The model containing the MoE layer
        layer_idx: Index of the layer to prune
        experts_to_prune: Indices of experts to remove [n_prune]

    Returns:
        Number of experts remaining after pruning
    """
    model_class = model.__class__.__name__
    attrs = get_model_attrs(model_class)

    if attrs is None:
        raise ValueError(f"Model {model_class} not registered in MODEL_ATTRS")

    moe_block = get_moe_block(model, layer_idx)
    num_experts = _get_num_experts_from_moe_block(moe_block, attrs)

    # Get retained expert indices
    experts_to_prune_set = set(experts_to_prune.tolist())
    retained_indices = [i for i in range(num_experts) if i not in experts_to_prune_set]

    if not retained_indices:
        logger.warning(f"Layer {layer_idx}: Cannot prune all experts! Keeping at least one.")
        retained_indices = [int(experts_to_prune[0].item())]

    logger.info(
        f"Layer {layer_idx}: Pruning {len(experts_to_prune)}/{num_experts} experts, "
        f"retaining {len(retained_indices)}"
    )

    # Prune based on whether experts are fused or separate. Detect fused
    # tensors from the loaded module as well as MODEL_ATTRS to support models
    # with multiple published checkpoint layouts.
    experts_attr = attrs.get("experts", "experts")
    experts_obj = getattr(moe_block, experts_attr)
    if attrs.get("fused", False) or hasattr(experts_obj, "gate_up_proj"):
        _prune_fused_experts(moe_block, retained_indices, attrs)
    else:
        _prune_separate_experts(moe_block, retained_indices, attrs)

    return len(retained_indices)


def _prune_fused_experts(
    moe_block: nn.Module,
    retained_indices: List[int],
    attrs: Dict[str, Any],
) -> None:
    """
    Prune fused experts (gate_up_proj + down_proj pattern).

    Used by models like Llama4, Glm4MoeLite.
    """
    experts = moe_block.experts
    idx_tensor = torch.as_tensor(retained_indices, device=experts.gate_up_proj.device)

    # Prune fused weights
    experts.gate_up_proj.data = experts.gate_up_proj[idx_tensor]
    experts.down_proj.data = experts.down_proj[idx_tensor]

    # Update num_experts if present
    if hasattr(experts, "num_experts"):
        experts.num_experts = len(retained_indices)

    # Prune router
    _prune_router(moe_block, retained_indices, attrs)


def _prune_separate_experts(
    moe_block: nn.Module,
    retained_indices: List[int],
    attrs: Dict[str, Any],
) -> None:
    """
    Prune separate experts (individual Linear modules in ModuleList).

    Used by most MoE models (Mixtral, Qwen, DeepSeek, etc.).
    """
    experts_attr = attrs.get("experts", "experts")
    all_experts = getattr(moe_block, experts_attr)
    retained_experts = [all_experts[i] for i in retained_indices]

    # Create new ModuleList with retained experts
    new_experts = nn.ModuleList(retained_experts)
    setattr(moe_block, experts_attr, new_experts)

    # Handle e_score_correction_bias (Ernie, some GLM models)
    if hasattr(moe_block, "moe_statics") and hasattr(
        moe_block.moe_statics, "e_score_correction_bias"
    ):
        moe_block.moe_statics.e_score_correction_bias.data = (
            moe_block.moe_statics.e_score_correction_bias.data[:, retained_indices]
        )

    # Prune router
    _prune_router(moe_block, retained_indices, attrs)


def _prune_router(
    moe_block: nn.Module,
    retained_indices: List[int],
    attrs: Dict[str, Any],
) -> None:
    """Prune router/gate weights to match retained experts."""
    router_attr = attrs.get("router", "gate")
    router_weight_attr = attrs.get("router_weight_attr")

    if router_weight_attr and "." in router_weight_attr:
        # Handle nested attribute like "classifier.weight" (LongCat)
        parts = router_weight_attr.split(".")
        router = getattr(moe_block, router_attr)
        inner_module = router
        for part in parts[:-1]:
            inner_module = getattr(inner_module, part)

        weight_attr = parts[-1]
        weight = getattr(inner_module, weight_attr)
        idx_tensor = torch.as_tensor(retained_indices, device=weight.device)

        # Update weight. Preserve Parameter registration when this is an nn.Linear.
        if isinstance(weight, nn.Parameter):
            weight.data = weight.data[idx_tensor]
        else:
            setattr(inner_module, weight_attr, weight[idx_tensor])

        # Update bias if present
        bias_attr = weight_attr.replace("weight", "bias")
        if hasattr(inner_module, bias_attr) and getattr(inner_module, bias_attr) is not None:
            bias = getattr(inner_module, bias_attr)
            if isinstance(bias, nn.Parameter):
                bias.data = bias.data[idx_tensor]
            else:
                setattr(inner_module, bias_attr, bias[idx_tensor])

        # Update out_features
        if hasattr(inner_module, "out_features"):
            inner_module.out_features = len(retained_indices)

        # Update n_routed_experts if present (LongCat)
        if hasattr(router, "n_routed_experts"):
            router.n_routed_experts = len(retained_indices)

    else:
        # Standard router with direct weight attribute
        router = getattr(moe_block, router_attr)
        idx_tensor = torch.as_tensor(retained_indices, device=router.weight.device)

        router.weight.data = router.weight.data[idx_tensor]

        if getattr(router, "bias", None) is not None:
            router.bias.data = router.bias.data[idx_tensor]

        router.out_features = len(retained_indices)

        if hasattr(router, "num_experts"):  # transformers >= 4.54+
            router.num_experts = len(retained_indices)

    # Remap DeepSeek-V4 hash router token-id -> expert-id tables.
    # Pruned expert ids are mapped to the first retained expert so indices stay
    # in range; retained ids are remapped to their new compact positions.
    if hasattr(router, "tid2eid"):
        old_num_experts = max(
            int(router.tid2eid.max().item()) + 1,
            max(retained_indices) + 1 if retained_indices else 0,
        )
        mapping = torch.zeros(old_num_experts, device=router.tid2eid.device, dtype=router.tid2eid.dtype)
        for new_idx, old_idx in enumerate(retained_indices):
            if old_idx < mapping.numel():
                mapping[old_idx] = new_idx
        router.tid2eid.data = mapping[router.tid2eid.clamp(max=mapping.numel() - 1)]

    # Handle e_score_correction_bias if present
    if hasattr(router, "e_score_correction_bias"):
        idx_tensor = torch.as_tensor(
            retained_indices, device=router.e_score_correction_bias.device
        )
        router.e_score_correction_bias.data = router.e_score_correction_bias.data[idx_tensor]


def _update_model_config(
    model: nn.Module,
    num_retained_experts: int,
    attrs: Dict[str, Any],
) -> None:
    """Update model config to reflect new expert count."""
    num_experts_attr = attrs.get("num_experts", "num_experts")

    if num_experts_attr.startswith("config."):
        config_key = num_experts_attr.split(".", 1)[1]
        if hasattr(model.config, config_key):
            setattr(model.config, config_key, num_retained_experts)
    else:
        # Try to update config directly using the attribute name
        if hasattr(model.config, num_experts_attr):
            setattr(model.config, num_experts_attr, num_retained_experts)

    # Handle special cases
    model_class = model.__class__.__name__

    if model_class == "Ernie4_5_MoeForCausalLM":
        if hasattr(model.config, "moe_capacity"):
            model.config.moe_capacity = [
                num_retained_experts,
                num_retained_experts,
                num_retained_experts,
            ]


def compute_experts_to_prune(
    observer_data: Dict[int, Dict[str, torch.Tensor]],
    config: PruningConfig,
) -> Dict[int, torch.Tensor]:
    """
    Compute which experts to prune per layer based on saliency metrics.

    Args:
        observer_data: Collected observer statistics per layer
        config: Pruning configuration

    Returns:
        Dictionary mapping layer_idx -> tensor of expert indices to prune
    """
    experts_to_prune = {}

    for layer_idx, layer_data in observer_data.items():
        expert_frequency = layer_data.get("expert_frequency")
        if expert_frequency is None:
            logger.warning(f"Layer {layer_idx}: No expert_frequency data, skipping")
            continue

        num_experts = len(expert_frequency)

        # Determine number to prune
        if config.prune_method == "under_average":
            # Under-average pruning: prune all below average
            if isinstance(expert_frequency, torch.Tensor):
                freq_values = expert_frequency.float()
            else:
                freq_values = torch.tensor(expert_frequency, dtype=torch.float32)

            total_activations = freq_values.sum().item()
            average_threshold = math.ceil(total_activations / num_experts)

            below_average = freq_values < average_threshold

            # Ensure we don't prune ALL experts
            if below_average.all():
                # Keep the one with highest frequency
                _, best = torch.topk(freq_values, 1)
                below_average[best] = False

            experts_to_prune[layer_idx] = torch.where(below_average)[0]

        else:
            # Saliency-based pruning
            if config.n_experts_to_prune is not None:
                n_prune = config.n_experts_to_prune
            else:
                n_prune = int(num_experts * config.compression_ratio)

            n_prune = max(0, min(n_prune, num_experts - 1))

            if n_prune == 0:
                logger.info(f"Layer {layer_idx}: No experts to prune")
                experts_to_prune[layer_idx] = torch.tensor([], dtype=torch.long)
                continue

            # Get saliency metric
            metric_key = config.prune_method
            if metric_key == "frequency":
                metric_key = "expert_frequency"

            saliency = layer_data.get(metric_key)

            if saliency is None:
                logger.warning(
                    f"Layer {layer_idx}: Metric {config.prune_method} not found, "
                    f"using expert_frequency"
                )
                saliency = layer_data.get("expert_frequency", expert_frequency)

            # Prune experts with lowest saliency. Infinite saliency is used to
            # mark preserved/super experts, so never select those unless there
            # are literally no finite candidates left.
            if isinstance(saliency, torch.Tensor):
                saliency_for_pruning = saliency.clone()
                finite_candidates = torch.isfinite(saliency_for_pruning)
                n_prune = min(n_prune, int(finite_candidates.sum().item()))
                if n_prune == 0:
                    experts_to_prune[layer_idx] = torch.tensor([], dtype=torch.long)
                    continue
                saliency_for_pruning[~finite_candidates] = float("inf")
                _, pruned = torch.topk(saliency_for_pruning, n_prune, largest=False)
            else:
                finite_items = [
                    (i, value) for i, value in enumerate(saliency)
                    if math.isfinite(float(value))
                ]
                n_prune = min(n_prune, len(finite_items))
                pruned = torch.tensor(
                    [i for i, _ in sorted(finite_items, key=lambda item: item[1])[:n_prune]],
                    dtype=torch.long,
                )

            experts_to_prune[layer_idx] = pruned

    return experts_to_prune


def prune_model(
    model: nn.Module,
    observer_data: Dict[int, Dict[str, torch.Tensor]],
    config: PruningConfig | None = None,
) -> Dict[int, int]:
    """
    Prune experts from a MoE model based on observer statistics.

    Args:
        model: The model to prune (modified in-place)
        observer_data: Collected observer statistics per layer
        config: Pruning configuration

    Returns:
        Dictionary mapping layer_idx -> number of experts retained
    """
    config = config or PruningConfig()

    # Ensure model is registered
    ensure_model_registered(model)

    # Get super expert indices if preserving
    if config.preserve_super_experts:
        super_expert_indices = _get_super_expert_indices(
            observer_data, config.super_expert_quantile
        )
        _mark_super_experts_preserved(observer_data, super_expert_indices)

    # Compute experts to prune per layer
    experts_to_prune = compute_experts_to_prune(observer_data, config)

    # Actually prune each layer
    retained_counts = {}

    for layer_idx, pruned_indices in tqdm(
        experts_to_prune.items(), desc="Pruning layers"
    ):
        if len(pruned_indices) == 0:
            # No pruning for this layer
            model_class = model.__class__.__name__
            attrs = get_model_attrs(model_class)
            if attrs is None:
                raise ValueError(f"Model {model_class} not registered in MODEL_ATTRS")
            moe_block = get_moe_block(model, layer_idx)
            retained_counts[layer_idx] = _get_num_experts_from_moe_block(moe_block, attrs)
            continue

        retained = prune_layer(model, layer_idx, pruned_indices)
        retained_counts[layer_idx] = retained

    # Log summary
    original_avg = sum(
        len(d.get("expert_frequency", [])) for d in observer_data.values()
    ) / len(observer_data) if observer_data else 0
    pruned_avg = sum(retained_counts.values()) / len(retained_counts) if retained_counts else 0
    compression = (1 - pruned_avg / original_avg) * 100 if original_avg > 0 else 0

    logger.info(
        f"Pruning complete: {original_avg:.1f} -> {pruned_avg:.1f} "
        f"experts per layer ({compression:.1f}% compression)"
    )

    # Update model.config once, after every layer has been physically pruned.
    # Updating inside prune_layer corrupts later layers because get_num_experts()
    # reads the global config before inspecting the actual layer module.
    if retained_counts:
        model_class = model.__class__.__name__
        attrs = get_model_attrs(model_class)
        if attrs is not None:
            unique_counts = set(retained_counts.values())
            final_expert_count = next(iter(retained_counts.values()))
            if len(unique_counts) == 1:
                _update_model_config(model, final_expert_count, attrs)
            else:
                logger.warning(
                    f"Layers have different retained expert counts after pruning: {unique_counts}. "
                    "Skipping scalar model.config expert-count update."
                )

    return retained_counts


def _get_super_expert_indices(
    observer_data: Dict[int, Dict[str, torch.Tensor]],
    quantile: float = 99.5,
) -> torch.Tensor:
    """
    Identify "super experts" with unusually high maximum activation.

    These experts are often important to preserve.

    Args:
        observer_data: Observer statistics per layer
        quantile: Quantile threshold for super experts

    Returns:
        Tensor of [layer_idx, expert_idx] pairs
    """
    flat_scores = []
    index_pairs: List[tuple[int, int]] = []

    for layer_idx, layer_data in observer_data.items():
        if "max_activations" in layer_data:
            max_per_expert = layer_data["max_activations"].flatten()
        elif "expert_outputs" in layer_data:
            # Compute from expert outputs
            expert_outputs = layer_data["expert_outputs"]  # [num_experts, tokens, hidden]
            max_per_expert = expert_outputs.abs().amax(dim=(1, 2))
        else:
            continue

        flat_scores.append(max_per_expert.float())
        index_pairs.extend((layer_idx, expert_idx) for expert_idx in range(max_per_expert.numel()))

    if not flat_scores:
        return torch.empty(0, 2, dtype=torch.long)

    all_max = torch.cat(flat_scores)
    threshold = torch.quantile(all_max, quantile / 100)

    selected = torch.nonzero(all_max > threshold, as_tuple=False).flatten().tolist()
    if not selected:
        return torch.empty(0, 2, dtype=torch.long)

    return torch.tensor([index_pairs[i] for i in selected], dtype=torch.long)


def _mark_super_experts_preserved(
    observer_data: Dict[int, Dict[str, torch.Tensor]],
    super_expert_indices: torch.Tensor,
) -> None:
    """Set saliency to infinity for super experts so they won't be pruned."""
    if super_expert_indices.numel() == 0:
        return

    for layer_idx, expert_idx in super_expert_indices:
        layer = layer_idx.item()
        expert = expert_idx.item()

        if layer in observer_data:
            for key in ["saliency_scores", "expert_frequency"]:
                if key in observer_data[layer]:
                    data = observer_data[layer][key]
                    if expert < len(data):
                        if isinstance(data, torch.Tensor):
                            observer_data[layer][key] = data.clone()
                            observer_data[layer][key][expert] = float("inf")
                        else:
                            data = list(data)
                            data[expert] = float("inf")
                            observer_data[layer][key] = data

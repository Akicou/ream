"""
Merging module for combining experts in MoE models.

This module implements the REAM/REAP expert merging algorithm:
1. Compute saliency scores for each expert
2. Select centroid experts (highest saliency)
3. Group remaining experts around centroids using similarity
4. Merge each group using permutation-aware averaging (Hungarian algorithm)
5. Adjust router weights to only output centroids

The result is a compressed model with fewer experts that preserves
most of the original model's capability.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ream_moe.model_attr_configs import get_model_attrs
from ream_moe.model_utils import get_moe_block, get_num_experts
from ream_moe.observer import LayerObserverState

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for expert merging."""

    target_ratio: float = 0.75  # Keep this fraction of experts (0.75 = 75%)
    group_size: int = 16  # Max experts per group (excluding centroid)
    use_gated_similarity: bool = True  # Use router+hidden similarity for grouping
    saliency_metric: str = "saliency_scores"  # Metric to use for centroid selection


def merge_layer(
    model: nn.Module,
    layer_idx: int,
    observer_stats: Dict[str, torch.Tensor],
    config: MergeConfig,
) -> int:
    """
    Merge experts in a single MoE layer using REAM/REAP algorithm.

    Args:
        model: The model containing the MoE layer
        layer_idx: Index of the layer to merge
        observer_stats: Collected observer statistics for this layer
        config: Merge configuration

    Returns:
        Number of experts after merging
    """
    model_class = model.__class__.__name__
    attrs = get_model_attrs(model_class)

    if attrs is None:
        raise ValueError(f"Model {model_class} not registered in MODEL_ATTRS")

    moe_block = get_moe_block(model, layer_idx)
    num_experts = get_num_experts(model, layer_idx)

    router_logits = observer_stats.get("router_logits")  # [T, N]
    expert_outputs = observer_stats.get("expert_outputs")  # [N, T, D]

    if router_logits is None or expert_outputs is None:
        raise ValueError(f"Layer {layer_idx}: Missing required observer data")

    # Step 1: Compute saliency scores
    saliency = _compute_saliency_scores(
        router_logits, expert_outputs, observer_stats, config.saliency_metric
    )  # [N]

    # Step 2: Select centroids
    target_experts = max(1, int(num_experts * config.target_ratio))
    centroid_indices = torch.argsort(saliency, descending=True)[:target_experts]

    logger.info(
        f"Layer {layer_idx}: Merging {num_experts} -> {target_experts} experts "
        f"({100 * (1 - config.target_ratio):.0f}% compression)"
    )

    # Step 3: Group experts around centroids
    groups = _group_experts_around_centroids(
        router_logits, expert_outputs, saliency, centroid_indices, config
    )

    # Step 4: Merge each group
    merged_weights = _merge_groups(
        moe_block, groups, saliency, attrs, observer_stats
    )

    # Step 5: Update model with merged weights
    _update_merged_weights(moe_block, merged_weights, groups, attrs)

    return len(groups)


def _compute_saliency_scores(
    router_logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    observer_stats: Dict[str, torch.Tensor],
    metric: str,
) -> torch.Tensor:
    """
    Compute saliency/importance scores for each expert.

    Args:
        router_logits: [num_tokens, num_experts]
        expert_outputs: [num_experts, num_tokens, hidden_dim]
        observer_stats: Additional observer statistics
        metric: Which metric to use ("saliency_scores", "expert_frequency", etc.)

    Returns:
        Saliency scores [num_experts]
    """
    num_experts = router_logits.shape[-1]

    # Use pre-computed metric if available
    if metric in observer_stats:
        precomputed = observer_stats[metric]
        if isinstance(precomputed, torch.Tensor) and precomputed.shape[0] == num_experts:
            return precomputed

    # Compute REAP saliency
    T, N = router_logits.shape
    probs = torch.softmax(router_logits, dim=-1)
    top_k = probs.shape[-1]
    topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)

    saliency = torch.zeros(N, device=router_logits.device)

    for i in range(N):
        token_idx, within_topk_idx = torch.where(topk_idx == i)
        if token_idx.numel() == 0:
            continue

        h_i = expert_outputs[i, token_idx]
        p_i = topk_vals[token_idx, within_topk_idx]
        saliency[i] = (h_i.norm(dim=-1) * p_i).mean()

    return saliency


def _group_experts_around_centroids(
    router_logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    saliency: torch.Tensor,
    centroid_indices: torch.Tensor,
    config: MergeConfig,
) -> List[List[int]]:
    """
    Group experts around centroids using similarity-based clustering.

    Implements pseudo-pruning: most low-saliency experts remain singletons;
    a small number near each centroid form compact clusters.

    Args:
        router_logits: [num_tokens, num_experts]
        expert_outputs: [num_experts, num_tokens, hidden_dim]
        saliency: [num_experts]
        centroid_indices: Indices of centroid experts
        config: Merge configuration

    Returns:
        List of groups, where each group is a list of expert indices
        (first element is the centroid/retained expert)
    """
    device = router_logits.device
    T, N = router_logits.shape
    used = torch.zeros(N, dtype=torch.bool, device=device)

    probs = torch.softmax(router_logits, dim=-1)

    # Compute expert representations
    gated = probs.T.unsqueeze(-1) * expert_outputs
    expert_repr_hidden = gated.mean(dim=1)  # [N, D]
    expert_repr_router = router_logits.T.mean(dim=1)  # [N]

    def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
        return (a_norm * b_norm).sum(dim=-1)

    groups: List[List[int]] = []

    for c in centroid_indices:
        c_idx = int(c.item())
        if used[c_idx]:
            continue

        group = [c_idx]
        used[c_idx] = True

        # Find unused candidates
        unused_idx = torch.where(~used)[0]
        if unused_idx.numel() == 0:
            groups.append(group)
            break

        # Compute similarities
        sim_hidden = cosine_sim(
            expert_repr_hidden[unused_idx],
            expert_repr_hidden[c_idx].expand_as(expert_repr_hidden[unused_idx]),
        )

        sim_router = cosine_sim(
            expert_repr_router[unused_idx].unsqueeze(-1),
            expert_repr_router[c_idx].unsqueeze(0).unsqueeze(-1),
        )

        if config.use_gated_similarity:
            sim = 0.5 * (sim_hidden + sim_router)
        else:
            sim = sim_hidden

        # Sort by similarity and take top group_size-1
        _, order = torch.sort(sim, descending=True)
        ordered_unused = unused_idx[order]

        max_group = config.group_size
        for idx in ordered_unused[: max_group - 1]:
            idx_int = int(idx.item())
            group.append(idx_int)
            used[idx_int] = True

        groups.append(group)

    # Force remaining unused experts into groups to meet exact target count
    remaining = torch.where(~used)[0]
    if len(remaining) > 0:
        # Distribute remaining experts to the most similar groups
        for r in remaining:
            r_idx = int(r.item())
            # Find the most similar group (by centroid)
            best_group_idx = 0
            best_sim = float('-inf')
            for i, group in enumerate(groups):
                c_idx = group[0]
                # Compute similarity using router logits
                sim = cosine_sim(
                    expert_repr_router[r_idx].unsqueeze(-1),
                    expert_repr_router[c_idx].unsqueeze(-1),
                )
                if sim > best_sim:
                    best_sim = sim
                    best_group_idx = i
            # Add to best group (even if similarity is low)
            groups[best_group_idx].append(r_idx)
            used[r_idx] = True

    return groups


def _merge_groups(
    moe_block: nn.Module,
    groups: List[List[int]],
    saliency: torch.Tensor,
    attrs: Dict[str, Any],
    observer_stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Merge each group of experts using permutation-aware averaging.

    Args:
        moe_block: The MoE block containing experts
        groups: List of expert groups to merge
        saliency: Saliency scores per expert
        attrs: Model attributes
        observer_stats: Observer statistics

    Returns:
        Merged expert weights tensor
    """
    all_weights = _get_expert_weights(moe_block, attrs)
    device = all_weights.device
    merged_weights: List[torch.Tensor] = []

    for group in groups:
        if len(group) == 1:
            # Singleton: keep original
            merged_weights.append(all_weights[group[0]].detach().clone())
            continue

        # Merge group
        group_tensor = all_weights[group]  # [G, ...]
        G = len(group)

        # Reshape: [G, neurons, rest]
        group_flat = group_tensor.view(G, group_tensor.shape[1], -1)

        # Use first expert as reference for permutation
        ref = group_flat[0]  # [neurons, rest]
        weights_accum = torch.zeros_like(ref)

        # Get normalized saliency weights
        s_vals = saliency[torch.tensor(group, device=device)]
        s_norm = s_vals / (s_vals.sum() + 1e-8)

        # Start with reference
        weights_accum += s_norm[0] * ref

        # Merge other experts with permutation alignment
        for g_idx in range(1, G):
            candidate = group_flat[g_idx]  # [neurons, rest]

            # Hungarian algorithm for optimal permutation
            cost = torch.cdist(ref, candidate)  # [neurons, neurons]
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            perm = torch.as_tensor(col_ind, device=device, dtype=torch.long)

            # Apply permutation and add weighted sum
            permuted = candidate[perm]
            weights_accum += s_norm[g_idx] * permuted

        # Reshape back to original shape
        merged = weights_accum.view_as(group_tensor[0])
        merged_weights.append(merged)

    return torch.stack(merged_weights, dim=0)


def _get_expert_weights(
    moe_block: nn.Module,
    attrs: Dict[str, Any],
) -> torch.Tensor:
    """
    Get all expert weights stacked into a single tensor.

    For fused experts: returns [num_experts, 2*intermediate + hidden_dim]
    For separate experts: returns concatenated projection weights

    Args:
        moe_block: The MoE block
        attrs: Model attributes

    Returns:
        Stacked expert weights [num_experts, ...]
    """
    experts = moe_block.experts

    if attrs.get("fused", False):
        # Fused: gate_up_proj and down_proj
        gate_up = experts.gate_up_proj  # [E, 2*I, H]
        down = experts.down_proj  # [E, H, I]

        num_experts = gate_up.shape[0]
        intermediate_size = down.shape[2]
        hidden_dim = gate_up.shape[2]

        # Stack as [gate, up, down] flattened
        weights = []
        for i in range(num_experts):
            gate_up_flat = gate_up[i].view(-1)  # [2*I*H]
            down_flat = down[i].view(-1)  # [H*I]
            weights.append(torch.cat([gate_up_flat, down_flat]))

        return torch.stack(weights, dim=0).unsqueeze(1)  # [E, 1, D]
    else:
        # Separate: concatenate gate_proj, up_proj, down_proj weights
        gate_proj = attrs.get("gate_proj", "gate_proj")
        up_proj = attrs.get("up_proj", "up_proj")
        down_proj = attrs.get("down_proj", "down_proj")

        num_experts = len(experts)
        weights = []

        for i in range(num_experts):
            expert = experts[i]
            gate = getattr(expert, gate_proj).weight.flatten()
            up = getattr(expert, up_proj).weight.flatten()
            down = getattr(expert, down_proj).weight.flatten()
            weights.append(torch.cat([gate, up, down]))

        return torch.stack(weights, dim=0).unsqueeze(1)  # [E, 1, D]


def _update_merged_weights(
    moe_block: nn.Module,
    merged_weights: torch.Tensor,
    groups: List[List[int]],
    attrs: Dict[str, Any],
) -> None:
    """
    Write merged weights back to the model and update router.

    Args:
        moe_block: The MoE block to update
        merged_weights: Merged expert weights [len(groups), ...]
        groups: Expert groups (used to know original shapes)
        attrs: Model attributes
    """
    experts = moe_block.experts
    num_retained = len(groups)

    if attrs.get("fused", False):
        # Update fused experts
        intermediate_size = experts.down_proj.shape[2]
        hidden_dim = experts.gate_up_proj.shape[2]

        new_gate_up = []
        new_down = []

        for group_idx in range(num_retained):
            flat = merged_weights[group_idx].squeeze().view(-1)

            # Split back into gate_up and down
            gate_up_size = 2 * intermediate_size * hidden_dim
            gate_up_flat = flat[:gate_up_size]
            down_flat = flat[gate_up_size:]

            gate_up = gate_up_flat.view(2 * intermediate_size, hidden_dim)
            down = down_flat.view(hidden_dim, intermediate_size)

            new_gate_up.append(gate_up)
            new_down.append(down)

        # Create new tensors
        experts.gate_up_proj.data = torch.stack(new_gate_up, dim=0)
        experts.down_proj.data = torch.stack(new_down, dim=0)

        if hasattr(experts, "num_experts"):
            experts.num_experts = num_retained

    else:
        # Update separate experts
        gate_proj = attrs.get("gate_proj", "gate_proj")
        up_proj = attrs.get("up_proj", "up_proj")
        down_proj = attrs.get("down_proj", "down_proj")

        # Get original shapes from first expert
        first_expert = experts[groups[0][0]]
        gate_shape = getattr(first_expert, gate_proj).weight.shape
        up_shape = getattr(first_expert, up_proj).weight.shape
        down_shape = getattr(first_expert, down_proj).weight.shape

        # Compute sizes
        gate_size = gate_shape[0] * gate_shape[1]
        up_size = up_shape[0] * up_shape[1]
        down_size = down_shape[0] * down_shape[1]

        # Create new ModuleList - reuse existing experts and update weights
        new_experts = nn.ModuleList()

        for group_idx in range(num_retained):
            flat = merged_weights[group_idx].squeeze()

            # Split into projections
            gate_flat = flat[:gate_size]
            up_flat = flat[gate_size : gate_size + up_size]
            down_flat = flat[gate_size + up_size : gate_size + up_size + down_size]

            # Clone the first expert from this group to preserve structure
            expert_idx = groups[0][0] if groups and group_idx < len(groups) else group_idx
            expert = deepcopy(experts[expert_idx])

            # Update weights directly without creating new instances
            getattr(expert, gate_proj).weight.data = gate_flat.view(*gate_shape)
            getattr(expert, up_proj).weight.data = up_flat.view(*up_shape)
            getattr(expert, down_proj).weight.data = down_flat.view(*down_shape)

            new_experts.append(expert)

        # Replace experts
        experts_attr = attrs.get("experts", "experts")
        setattr(moe_block, experts_attr, new_experts)

    # Update router
    _update_router_for_merge(moe_block, groups, attrs)


def _update_router_for_merge(
    moe_block: nn.Module,
    groups: List[List[int]],
    attrs: Dict[str, Any],
) -> None:
    """
    Update router weights to only output centroids (first expert in each group).

    Args:
        moe_block: The MoE block
        groups: Expert groups (first element of each is the centroid)
        attrs: Model attributes
    """
    router_attr = attrs.get("router", "gate")
    router_weight_attr = attrs.get("router_weight_attr")

    centroid_indices = [g[0] for g in groups]
    idx_tensor = torch.as_tensor(
        centroid_indices, device=getattr(moe_block, router_attr).weight.device
    )

    if router_weight_attr and "." in router_weight_attr:
        # Handle nested router (e.g., LongCat's router.classifier)
        parts = router_weight_attr.split(".")
        router = getattr(moe_block, router_attr)
        inner = router
        for part in parts[:-1]:
            inner = getattr(inner, part)

        weight_attr = parts[-1]
        setattr(inner, weight_attr, getattr(inner, weight_attr)[idx_tensor])

        # Update bias if present
        bias_attr = weight_attr.replace("weight", "bias")
        if hasattr(inner, bias_attr) and getattr(inner, bias_attr) is not None:
            setattr(inner, bias_attr, getattr(inner, bias_attr)[idx_tensor])

        # Update e_score_correction_bias if present (DeepSeek V3)
        if hasattr(inner, "e_score_correction_bias"):
            inner.e_score_correction_bias.data = inner.e_score_correction_bias.data[idx_tensor]

        # Update top_k if needed (can't select more experts than available)
        if hasattr(inner, "top_k"):
            inner.top_k = min(inner.top_k, len(centroid_indices))

        # Update out_features
        if hasattr(inner, "out_features"):
            inner.out_features = len(centroid_indices)

    else:
        # Standard router
        router = getattr(moe_block, router_attr)
        router.weight.data = router.weight.data[idx_tensor]

        if getattr(router, "bias", None) is not None:
            router.bias.data = router.bias.data[idx_tensor]

        # Handle e_score_correction_bias (DeepSeek V3 / MoEGate)
        if hasattr(router, "e_score_correction_bias"):
            router.e_score_correction_bias.data = router.e_score_correction_bias.data[idx_tensor]

        # Update top_k if needed (can't select more experts than available)
        if hasattr(router, "top_k"):
            router.top_k = min(router.top_k, len(centroid_indices))

        router.out_features = len(centroid_indices)

        if hasattr(router, "num_experts"):
            router.num_experts = len(centroid_indices)

        # Handle n_routed_experts attribute (DeepSeek V3)
        if hasattr(router, "n_routed_experts"):
            router.n_routed_experts = len(centroid_indices)


def merge_model(
    model: nn.Module,
    observer_data: Dict[int, Dict[str, torch.Tensor]],
    config: MergeConfig | None = None,
) -> Dict[int, int]:
    """
    Merge experts across all MoE layers in a model.

    Args:
        model: The model to merge (modified in-place)
        observer_data: Collected observer statistics per layer
        config: Merge configuration

    Returns:
        Dictionary mapping layer_idx -> number of experts after merging
    """
    config = config or MergeConfig()

    retained_counts = {}

    for layer_idx, layer_stats in tqdm(observer_data.items(), desc="Merging layers"):
        try:
            retained = merge_layer(model, layer_idx, layer_stats, config)
            retained_counts[layer_idx] = retained
        except Exception as e:
            logger.error(f"Layer {layer_idx}: Failed to merge - {e}")
            raise

    # Update model config with new expert count
    if retained_counts:
        # Verify all layers have the same expert count
        unique_counts = set(retained_counts.values())
        if len(unique_counts) > 1:
            logger.warning(
                f"Layers have different expert counts after merging: {retained_counts}. "
                f"Using the most common count."
            )
            # Use the most common expert count
            from collections import Counter
            final_expert_count = Counter(retained_counts.values()).most_common(1)[0][0]
        else:
            final_expert_count = list(unique_counts)[0]

        logger.info(f"Setting expert count to {final_expert_count} (across {len(retained_counts)} MoE layers)")

        # Handle Kimi-VL's nested text_config
        model_class = model.__class__.__name__
        if model_class == "KimiVLForConditionalGeneration":
            if hasattr(model.config, "text_config"):
                # Update n_routed_experts
                for attr_name in ["n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"]:
                    if hasattr(model.config.text_config, attr_name):
                        logger.info(f"Updating model.config.text_config.{attr_name} = {final_expert_count}")
                        setattr(model.config.text_config, attr_name, final_expert_count)
                        break
                # Update num_experts_per_tok if needed (can't select more than available)
                if hasattr(model.config.text_config, "num_experts_per_tok"):
                    old_topk = model.config.text_config.num_experts_per_tok
                    if old_topk > final_expert_count:
                        logger.info(f"Updating model.config.text_config.num_experts_per_tok = {final_expert_count} (was {old_topk})")
                        setattr(model.config.text_config, "num_experts_per_tok", final_expert_count)

        # Try to update various possible config attributes
        for attr_name in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
            if hasattr(model.config, attr_name):
                logger.info(f"Updating model.config.{attr_name} = {final_expert_count}")
                setattr(model.config, attr_name, final_expert_count)

        # Update num_experts_per_tok for non-nested configs
        if not hasattr(model.config, "text_config") and hasattr(model.config, "num_experts_per_tok"):
            old_topk = model.config.num_experts_per_tok
            if old_topk > final_expert_count:
                logger.info(f"Updating model.config.num_experts_per_tok = {final_expert_count} (was {old_topk})")
                setattr(model.config, "num_experts_per_tok", final_expert_count)

    # Log summary
    if retained_counts:
        original_avg = sum(len(s.get("router_logits", [])[0]) if "router_logits" in s else 0 for s in observer_data.values()) / len(observer_data)
        merged_avg = sum(retained_counts.values()) / len(retained_counts)
        compression = (1 - merged_avg / original_avg) * 100 if original_avg > 0 else 0

        logger.info(
            f"Merging complete: {original_avg:.1f} -> {merged_avg:.1f} "
            f"experts per layer ({compression:.0f}% compression)"
        )

    return retained_counts

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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ream_moe.model_attr_configs import get_model_attrs
from ream_moe.model_utils import get_moe_block, get_num_experts, get_top_k
from ream_moe.observer import LayerObserverState

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for expert merging."""

    target_ratio: float = 0.75  # Keep this fraction of experts (0.75 = 75%)
    group_size: int = 16  # Max experts per group (excluding centroid)
    use_gated_similarity: bool = True  # Use router+hidden similarity for grouping
    saliency_metric: str = "saliency_scores"  # Metric to use for centroid selection
    use_cpu_for_weights: bool = False  # Process expert weights on CPU to save GPU memory
    skip_permutation: bool = False  # Skip Hungarian algorithm for faster merging (simple averaging)


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

    # Step 1: Compute saliency scores using the model's actual top-k routing value
    try:
        layer_top_k = get_top_k(model, layer_idx)
    except Exception:
        layer_top_k = None  # will fall back to num_experts (less accurate)

    saliency = _compute_saliency_scores(
        router_logits, expert_outputs, observer_stats, config.saliency_metric,
        top_k=layer_top_k,
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
        moe_block, groups, saliency, attrs, observer_stats,
        use_cpu_for_weights=config.use_cpu_for_weights,
        skip_permutation=config.skip_permutation
    )

    # Step 5: Update model with merged weights
    _update_merged_weights(moe_block, merged_weights, groups, attrs)

    return len(groups)


def _compute_saliency_scores(
    router_logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    observer_stats: Dict[str, torch.Tensor],
    metric: str,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute saliency/importance scores for each expert.

    Args:
        router_logits: [num_tokens, num_experts]
        expert_outputs: [num_experts, num_tokens, hidden_dim]
        observer_stats: Additional observer statistics
        metric: Which metric to use ("saliency_scores", "expert_frequency", etc.)
        top_k: Model's actual routing top-k. Only tokens where expert i is in the
               top-k contribute to its saliency score. If None, all tokens are used
               (inaccurate — inflates saliency for rarely-routed experts).

    Returns:
        Saliency scores [num_experts]
    """
    num_experts = router_logits.shape[-1]

    # Use pre-computed metric if available (e.g. from observer's saliency_scores)
    if metric in observer_stats:
        precomputed = observer_stats[metric]
        if isinstance(precomputed, torch.Tensor) and precomputed.shape[0] == num_experts:
            return precomputed

    # Compute REAP saliency from scratch using the correct routing top-k
    T, N = router_logits.shape
    probs = torch.softmax(router_logits, dim=-1)
    actual_top_k = top_k if top_k is not None else N
    actual_top_k = min(actual_top_k, N)
    topk_vals, topk_idx = torch.topk(probs, k=actual_top_k, dim=-1)

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
    expert_repr_hidden = gated.mean(dim=1)  # [N, D] — gate-weighted mean hidden state
    # Keep full routing distribution [N, T] so cosine similarity captures routing pattern,
    # not just a collapsed scalar (which was near-meaningless before).
    expert_repr_router = router_logits.T  # [N, T]

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

        # Cosine similarity over full routing distribution [T] — compares which tokens
        # each expert is activated for, not just a single scalar mean value.
        sim_router = cosine_sim(
            expert_repr_router[unused_idx],   # [n_unused, T]
            expert_repr_router[c_idx],        # [T] — broadcasts to [n_unused, T]
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

    # Remaining unused experts become singletons
    remaining = torch.where(~used)[0]
    for r in remaining:
        groups.append([int(r.item())])

    return groups


def _merge_groups(
    moe_block: nn.Module,
    groups: List[List[int]],
    saliency: torch.Tensor,
    attrs: Dict[str, Any],
    observer_stats: Dict[str, torch.Tensor],
    use_cpu_for_weights: bool = False,
    skip_permutation: bool = False,
) -> torch.Tensor:
    """
    Merge each group of experts using permutation-aware averaging.

    Args:
        moe_block: The MoE block containing experts
        groups: List of expert groups to merge
        saliency: Saliency scores per expert
        attrs: Model attributes
        observer_stats: Observer statistics
        use_cpu_for_weights: If True, process weights on CPU to save GPU memory
        skip_permutation: If True, use simple averaging instead of Hungarian (faster)

    Returns:
        Merged expert weights tensor
    """
    # all_weights: [E, I, 3H]  — intermediate axis (I) is the neuron/permutation axis
    all_weights = _get_expert_weights(moe_block, attrs, use_cpu=use_cpu_for_weights)
    device = all_weights.device
    merged_list: List[torch.Tensor] = []

    for group in groups:
        if len(group) == 1:
            # Singleton: keep original weights unchanged
            merged_list.append(all_weights[group[0]].detach().clone())
            continue

        G = len(group)
        group_tensor = all_weights[group]  # [G, I, 3H]

        # Saliency-normalised weights for this group
        s_vals = saliency.to(device)[torch.tensor(group, device=device)]
        s_norm = s_vals / (s_vals.sum() + 1e-8)

        if skip_permutation:
            # Fast path: saliency-weighted average without neuron permutation.
            # ~10-100× faster but skips alignment, so merged neurons may cancel.
            merged = torch.sum(group_tensor * s_norm.view(-1, 1, 1), dim=0)  # [I, 3H]
        else:
            # Permutation-aware averaging with Hungarian algorithm.
            # For each non-centroid expert, find the neuron permutation that best
            # aligns it to the centroid, then accumulate the weighted average.
            ref = group_tensor[0]                  # [I, 3H] — centroid as reference
            weights_accum = s_norm[0] * ref.clone()

            for g_idx in range(1, G):
                candidate = group_tensor[g_idx]    # [I, 3H]

                # Pairwise Euclidean distance between neuron vectors [I, 3H] → cost [I, I]
                # BFloat16 is unsupported by torch.cdist on CPU; upcast to float32.
                if not device.type.startswith("cuda") and (
                    ref.dtype == torch.bfloat16 or candidate.dtype == torch.bfloat16
                ):
                    cost = torch.cdist(ref.float(), candidate.float())
                else:
                    cost = torch.cdist(ref, candidate)

                _row, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
                perm = torch.as_tensor(col_ind, device=device, dtype=torch.long)

                permuted = candidate[perm]         # [I, 3H] — neurons reordered to match ref
                weights_accum = weights_accum + s_norm[g_idx] * permuted

            merged = weights_accum  # [I, 3H]

        merged_list.append(merged)

    return torch.stack(merged_list, dim=0)  # [num_groups, I, 3H]


def _get_expert_weights(
    moe_block: nn.Module,
    attrs: Dict[str, Any],
    use_cpu: bool = False,
) -> torch.Tensor:
    """
    Get all expert weights shaped as ``[E, intermediate, 3 * hidden_dim]``.

    Dimension 1 is the intermediate/neuron axis used for Hungarian permutation
    alignment.  The last dimension concatenates the three projections so that
    each row represents one "neuron":

        [:, :, :H]    — gate projection   (gate_proj / w3)
        [:, :, H:2H]  — up   projection   (up_proj   / w1)
        [:, :, 2H:]   — down projection^T (down_proj^T / w2^T)

    Transposing the down projection puts its columns (intermediate neurons)
    along the same axis as the rows of gate/up, enabling a single permutation
    that consistently reorders neurons across all three matrices.

    For fused experts (gate_up_proj tensor of shape [E, 2*I, H]):
        gate portion = gate_up_proj[:, :I, :]   shape [E, I, H]
        up   portion = gate_up_proj[:, I:, :]   shape [E, I, H]
        down^T       = down_proj.permute(0,2,1) shape [E, I, H]
    """
    experts = moe_block.experts

    def safe_cpu(t: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU if requested, handling meta/offloaded tensors."""
        if not use_cpu or t.device.type == "cpu":
            return t
        try:
            return t.to("cpu")
        except NotImplementedError:
            return t.data.to("cpu")

    if attrs.get("fused", False):
        gate_up = safe_cpu(experts.gate_up_proj)  # [E, 2I, H]
        down    = safe_cpu(experts.down_proj)      # [E, H, I]

        _E, two_I, H = gate_up.shape
        I = two_I // 2

        gate   = gate_up[:, :I, :]          # [E, I, H]
        up     = gate_up[:, I:, :]          # [E, I, H]
        down_t = down.permute(0, 2, 1)      # [E, I, H]

        return torch.cat([gate, up, down_t], dim=-1)  # [E, I, 3H]
    else:
        gate_attr = attrs.get("gate_proj", "gate_proj")
        up_attr   = attrs.get("up_proj",   "up_proj")
        down_attr = attrs.get("down_proj", "down_proj")

        gates: List[torch.Tensor] = []
        ups:   List[torch.Tensor] = []
        downs: List[torch.Tensor] = []

        for expert in experts:
            gates.append(safe_cpu(getattr(expert, gate_attr).weight))    # [I, H]
            ups.append(  safe_cpu(getattr(expert, up_attr).weight))      # [I, H]
            downs.append(safe_cpu(getattr(expert, down_attr).weight.T))  # [I, H]

        gate_stack  = torch.stack(gates)   # [E, I, H]
        up_stack    = torch.stack(ups)     # [E, I, H]
        down_t_stack = torch.stack(downs)  # [E, I, H]

        return torch.cat([gate_stack, up_stack, down_t_stack], dim=-1)  # [E, I, 3H]


def _update_merged_weights(
    moe_block: nn.Module,
    merged_weights: torch.Tensor,  # [num_groups, I, 3H]
    groups: List[List[int]],
    attrs: Dict[str, Any],
) -> None:
    """
    Write merged expert weights back to the model and update the router.

    ``merged_weights`` has shape ``[num_groups, I, 3H]`` produced by
    ``_merge_groups``.  The last dimension is unpacked as::

        [:, :, :H]    → gate projection
        [:, :, H:2H]  → up   projection
        [:, :, 2H:]   → down projection^T  (transpose back before writing)

    For fused experts the three matrices are repacked into ``gate_up_proj``
    and ``down_proj``.  For separate experts the centroid module of each group
    is reused (weights updated in-place) to avoid model-specific constructor
    arguments that differ across architectures (w1/w3, gate_proj, etc.).
    """
    experts = moe_block.experts
    num_retained = len(groups)

    if attrs.get("fused", False):
        # gate_up_proj: [E, 2I, H],  down_proj: [E, H, I]
        H           = experts.gate_up_proj.shape[2]
        target_dev  = experts.gate_up_proj.device

        new_gate_up: List[torch.Tensor] = []
        new_down:    List[torch.Tensor] = []

        for group_idx in range(num_retained):
            m      = merged_weights[group_idx].to(target_dev)  # [I, 3H]
            gate   = m[:, :H].contiguous()                     # [I, H]
            up     = m[:, H:2 * H].contiguous()                # [I, H]
            down_t = m[:, 2 * H:].contiguous()                 # [I, H]  (was down^T)

            new_gate_up.append(torch.cat([gate, up], dim=0))   # [2I, H]
            new_down.append(down_t.T.contiguous())              # [H, I]

        experts.gate_up_proj.data = torch.stack(new_gate_up)   # [num_retained, 2I, H]
        experts.down_proj.data    = torch.stack(new_down)       # [num_retained, H, I]

        if hasattr(experts, "num_experts"):
            experts.num_experts = num_retained

    else:
        # Non-fused: reuse the centroid expert module from each group and update
        # its weights in-place.  This avoids model-specific constructor arguments
        # (gate_proj vs w3 vs wi_0, etc.) that differ across architectures.
        gate_attr = attrs.get("gate_proj", "gate_proj")
        up_attr   = attrs.get("up_proj",   "up_proj")
        down_attr = attrs.get("down_proj", "down_proj")

        new_experts = nn.ModuleList()

        for group_idx, group in enumerate(groups):
            m   = merged_weights[group_idx]  # [I, 3H]
            H   = m.shape[1] // 3

            gate_w = m[:, :H].contiguous()          # [I, H]
            up_w   = m[:, H:2 * H].contiguous()     # [I, H]
            down_w = m[:, 2 * H:].T.contiguous()    # [H, I]  (transpose back)

            # Centroid expert module from this group
            centroid = experts[group[0]]
            tgt = getattr(centroid, gate_attr).weight.device

            getattr(centroid, gate_attr).weight.data = gate_w.to(tgt)
            getattr(centroid, up_attr  ).weight.data = up_w.to(tgt)
            getattr(centroid, down_attr).weight.data = down_w.to(tgt)

            new_experts.append(centroid)

        experts_attr = attrs.get("experts", "experts")
        setattr(moe_block, experts_attr, new_experts)

    # Shrink router to output only the centroid experts
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

        # Update out_features
        if hasattr(inner, "out_features"):
            inner.out_features = len(centroid_indices)

    else:
        # Standard router
        router = getattr(moe_block, router_attr)
        router.weight.data = router.weight.data[idx_tensor]

        if getattr(router, "bias", None) is not None:
            router.bias.data = router.bias.data[idx_tensor]

        router.out_features = len(centroid_indices)

        if hasattr(router, "num_experts"):
            router.num_experts = len(centroid_indices)


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
        unique_counts = set(retained_counts.values())
        final_expert_count = list(retained_counts.values())[0]

        if len(unique_counts) > 1:
            logger.warning(
                f"Layers have different retained expert counts after merging: {unique_counts}. "
                f"model.config will be updated to the first layer's count ({final_expert_count}), "
                f"which may not reflect per-layer differences (e.g. NonUniform models)."
            )

        for attr_name in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
            if hasattr(model.config, attr_name):
                logger.info(f"Updating model.config.{attr_name} = {final_expert_count}")
                setattr(model.config, attr_name, final_expert_count)

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

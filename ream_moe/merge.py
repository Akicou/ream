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
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ream_moe.model_attr_configs import get_model_attrs
from ream_moe.model_utils import get_moe_block, get_top_k
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

    router_logits = observer_stats.get("router_logits")  # [T, N]
    expert_outputs = observer_stats.get("expert_outputs")  # [N, T, D]

    if router_logits is None or expert_outputs is None:
        raise ValueError(f"Layer {layer_idx}: Missing required observer data")

    num_experts = router_logits.shape[-1]
    if expert_outputs.shape[0] != num_experts:
        raise ValueError(
            f"Layer {layer_idx}: router/expert shape mismatch "
            f"({num_experts} router experts vs {expert_outputs.shape[0]} expert outputs)"
        )

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
    target_experts = min(num_experts, max(1, math.ceil(num_experts * config.target_ratio)))
    centroid_indices = torch.argsort(saliency, descending=True)[:target_experts]
    actual_compression = 100 * (1 - target_experts / num_experts)

    logger.info(
        f"Layer {layer_idx}: Merging {num_experts} -> {target_experts} experts "
        f"({actual_compression:.1f}% compression)"
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
    _update_merged_weights(moe_block, merged_weights, groups, attrs, saliency)

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

    # Compute REAP saliency from scratch using the correct routed experts.
    # If the observer captured actual selections (e.g. DeepSeek-V4 hash MoE),
    # use them; otherwise derive top-k from router probabilities.
    T, N = router_logits.shape
    selected_experts = observer_stats.get("selected_experts")
    routing_weights = observer_stats.get("routing_weights")

    if isinstance(selected_experts, torch.Tensor):
        topk_idx = selected_experts.to(router_logits.device)
        if isinstance(routing_weights, torch.Tensor):
            topk_vals = routing_weights.to(router_logits.device)
        else:
            probs = torch.softmax(router_logits, dim=-1)
            topk_vals = torch.gather(probs, dim=-1, index=topk_idx)
    else:
        probs = torch.softmax(router_logits, dim=-1)
        actual_top_k = top_k if top_k is not None else N
        actual_top_k = min(actual_top_k, N)
        topk_vals, topk_idx = torch.topk(probs, k=actual_top_k, dim=-1)

    # Memory-efficient vectorized approach: iterate but with batched operations
    saliency = torch.zeros(N, device=router_logits.device)
    
    # Process in chunks to balance memory and speed
    chunk_size = max(1, N // 8)  # Process 1/8 of experts at a time
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        
        # For this chunk of experts, find all positions where they appear in top-k
        chunk_experts = torch.arange(start_idx, end_idx, device=topk_idx.device)
        
        # Create mask: [T, top_k, chunk_size]
        mask = (topk_idx.unsqueeze(-1) == chunk_experts.unsqueeze(0).unsqueeze(0))  # [T, top_k, chunk_size]
        
        # Get corresponding values
        chunk_saliency = torch.zeros(end_idx - start_idx, device=router_logits.device)
        
        for local_idx, global_idx in enumerate(chunk_experts.tolist()):
            expert_mask = mask[..., local_idx]  # [T, top_k]
            if expert_mask.any():
                # Get token positions and within-topk positions
                token_idx, within_topk_idx = torch.where(expert_mask)
                h_i = expert_outputs[global_idx, token_idx]
                p_i = topk_vals[token_idx, within_topk_idx]
                chunk_saliency[local_idx] = (h_i.norm(dim=-1) * p_i).mean()
        
        saliency[start_idx:end_idx] = chunk_saliency

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
    centroid_indices = centroid_indices.to(device=device, dtype=torch.long)
    centroid_mask = torch.zeros(N, dtype=torch.bool, device=device)
    centroid_mask[centroid_indices] = True

    probs = torch.softmax(router_logits, dim=-1)

    # Compute expert representations
    gated = probs.T.unsqueeze(-1) * expert_outputs
    expert_repr_hidden = gated.mean(dim=1)  # [N, D] — gate-weighted mean hidden state
    # Keep full routing distribution [N, T] so cosine similarity captures routing pattern,
    # not just a collapsed scalar (which was near-meaningless before).
    expert_repr_router = router_logits.T  # [N, T]

    # Pre-normalize all expert representations once (OPTIMIZATION)
    eps = 1e-8
    expert_repr_hidden_norm = expert_repr_hidden / (expert_repr_hidden.norm(dim=-1, keepdim=True) + eps)
    expert_repr_router_norm = expert_repr_router / (expert_repr_router.norm(dim=-1, keepdim=True) + eps)

    groups: List[List[int]] = []

    for c in centroid_indices:
        c_idx = int(c.item())
        if used[c_idx]:
            continue

        group = [c_idx]
        used[c_idx] = True

        # Find unused non-centroid candidates. Centroids are protected so high-
        # saliency retained experts cannot be swallowed by earlier groups.
        unused_idx = torch.where((~used) & (~centroid_mask))[0]
        if unused_idx.numel() == 0:
            groups.append(group)
            continue

        # Compute similarities using pre-normalized representations (OPTIMIZATION)
        c_hidden_norm = expert_repr_hidden_norm[c_idx]  # [D]
        c_router_norm = expert_repr_router_norm[c_idx]  # [T]
        
        # Dot product with pre-normalized vectors = cosine similarity
        sim_hidden = (expert_repr_hidden_norm[unused_idx] * c_hidden_norm).sum(dim=-1)
        sim_router = (expert_repr_router_norm[unused_idx] * c_router_norm).sum(dim=-1)

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


def _solve_hungarian_permutation(
    ref: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Solve Hungarian algorithm to find optimal neuron permutation.
    
    Args:
        ref: Reference expert weights [I, 3H]
        candidate: Candidate expert weights to permute [I, 3H]
        device: Target device for the permutation tensor
        
    Returns:
        Permutation indices [I]
    """
    # Pairwise Euclidean distance between neuron vectors [I, 3H] → cost [I, I]
    # BFloat16 is unsupported by torch.cdist on CPU; upcast to float32.
    if not device.type.startswith("cuda") and (
        ref.dtype == torch.bfloat16 or candidate.dtype == torch.bfloat16
    ):
        cost = torch.cdist(ref.float(), candidate.float())
    else:
        cost = torch.cdist(ref, candidate)
    
    _row, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(col_ind, device=device, dtype=torch.long)


def _normalise_group_saliency(
    saliency: torch.Tensor,
    group: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return stable merge weights for a group.

    If calibration never routed any expert in the group, all saliency values can
    be zero. In that case a naive normalization would produce an all-zero merged
    expert, so we preserve the centroid (group[0]) instead.
    """
    idx = torch.as_tensor(group, device=device, dtype=torch.long)
    vals = saliency.to(device=device)[idx].float()

    pos_inf = torch.isinf(vals) & (vals > 0)
    if pos_inf.any():
        weights = pos_inf.float() / pos_inf.float().sum()
        return weights.to(dtype=dtype)

    vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = vals.sum()

    if torch.isfinite(total).item() and total.item() > 1e-8:
        return (vals / total).to(dtype=dtype)

    weights = torch.zeros_like(vals)
    weights[0] = 1.0
    return weights.to(dtype=dtype)


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
            # Singleton: keep original weights unchanged (no clone needed for singletons)
            merged_list.append(all_weights[group[0]])
            continue

        G = len(group)
        group_tensor = all_weights[group]  # [G, I, 3H]

        # Saliency-normalised weights for this group. Falls back to the
        # centroid if calibration provides no signal for this group.
        s_norm = _normalise_group_saliency(saliency, group, device, group_tensor.dtype)

        if skip_permutation:
            # Fast path: saliency-weighted average without neuron permutation.
            # ~10-100× faster but skips alignment, so merged neurons may cancel.
            merged = torch.sum(group_tensor * s_norm.view(-1, 1, 1), dim=0)  # [I, 3H]
        else:
            # Permutation-aware averaging with Hungarian algorithm.
            # For each non-centroid expert, find the neuron permutation that best
            # aligns it to the centroid, then accumulate the weighted average.
            ref = group_tensor[0]                  # [I, 3H] — centroid as reference
            weights_accum = s_norm[0] * ref

            for g_idx in range(1, G):
                candidate = group_tensor[g_idx]    # [I, 3H]
                
                # Use extracted Hungarian function (better code organization)
                perm = _solve_hungarian_permutation(ref, candidate, device)

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

    if attrs.get("fused", False) or hasattr(experts, "gate_up_proj"):
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
    saliency: torch.Tensor,
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

    if attrs.get("fused", False) or hasattr(experts, "gate_up_proj"):
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

    # Shrink router to one output per merged expert group
    _update_router_for_merge(moe_block, groups, attrs, saliency)


def _update_router_for_merge(
    moe_block: nn.Module,
    groups: List[List[int]],
    attrs: Dict[str, Any],
    saliency: torch.Tensor,
) -> None:
    """
    Update router weights to output one merged route per expert group.

    Args:
        moe_block: The MoE block
        groups: Expert groups (first element of each is the centroid)
        attrs: Model attributes
        saliency: Per-expert saliency scores used for weighted router merging
    """
    router_attr = attrs.get("router", "gate")
    router_weight_attr = attrs.get("router_weight_attr")
    router = getattr(moe_block, router_attr)

    def merge_expert_rows(tensor: torch.Tensor) -> torch.Tensor:
        """Merge row-0 expert axis using the same saliency weights as experts."""
        merged_rows: List[torch.Tensor] = []
        for group in groups:
            idx = torch.as_tensor(group, device=tensor.device, dtype=torch.long)
            selected = tensor[idx]
            weights = _normalise_group_saliency(saliency, group, tensor.device, selected.dtype)
            view_shape = (len(group),) + (1,) * (selected.ndim - 1)
            merged_rows.append((selected * weights.view(view_shape)).sum(dim=0))
        return torch.stack(merged_rows, dim=0)

    def merge_expert_columns(tensor: torch.Tensor) -> torch.Tensor:
        """Merge last expert axis, used by some correction-bias tensors."""
        merged_cols: List[torch.Tensor] = []
        for group in groups:
            idx = torch.as_tensor(group, device=tensor.device, dtype=torch.long)
            selected = torch.index_select(tensor, dim=-1, index=idx)
            weights = _normalise_group_saliency(saliency, group, tensor.device, selected.dtype)
            view_shape = (1,) * (selected.ndim - 1) + (len(group),)
            merged_cols.append((selected * weights.view(view_shape)).sum(dim=-1))
        return torch.stack(merged_cols, dim=-1)

    if router_weight_attr and "." in router_weight_attr:
        # Handle nested router (e.g., LongCat's router.classifier)
        parts = router_weight_attr.split(".")
        inner = router
        for part in parts[:-1]:
            inner = getattr(inner, part)

        weight_attr = parts[-1]
        weight = getattr(inner, weight_attr)
        new_weight = merge_expert_rows(weight.data if isinstance(weight, nn.Parameter) else weight)
        if isinstance(weight, nn.Parameter):
            weight.data = new_weight
        else:
            setattr(inner, weight_attr, new_weight)

        # Update bias if present
        bias_attr = weight_attr.replace("weight", "bias")
        if hasattr(inner, bias_attr) and getattr(inner, bias_attr) is not None:
            bias = getattr(inner, bias_attr)
            new_bias = merge_expert_rows(bias.data if isinstance(bias, nn.Parameter) else bias)
            if isinstance(bias, nn.Parameter):
                bias.data = new_bias
            else:
                setattr(inner, bias_attr, new_bias)

        # Update out_features
        if hasattr(inner, "out_features"):
            inner.out_features = len(groups)

        if hasattr(router, "n_routed_experts"):
            router.n_routed_experts = len(groups)

    else:
        # Standard router. Average grouped router rows instead of copying only
        # centroid rows, so tokens that routed to merged-away experts can still
        # reach the merged expert.
        router.weight.data = merge_expert_rows(router.weight.data)

        if getattr(router, "bias", None) is not None:
            router.bias.data = merge_expert_rows(router.bias.data)

        router.out_features = len(groups)

        if hasattr(router, "num_experts"):
            router.num_experts = len(groups)

    for attr_path in attrs.get("router_expert_attrs", []):
        parts = [part for part in str(attr_path).split(".") if part]
        if not parts:
            continue
        parent = router
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent = None
                break
            parent = getattr(parent, part)
        if parent is None or not hasattr(parent, parts[-1]):
            continue
        value = getattr(parent, parts[-1])
        data = value.data if isinstance(value, nn.Parameter) else value
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            continue
        if data.shape[0] >= max(max(g) for g in groups) + 1:
            merged_value = merge_expert_rows(data)
        elif data.shape[-1] >= max(max(g) for g in groups) + 1:
            merged_value = merge_expert_columns(data)
        else:
            continue
        if isinstance(value, nn.Parameter):
            value.data = merged_value
        else:
            setattr(parent, parts[-1], merged_value)

    # Remap DeepSeek-V4 hash router token-id -> expert-id tables from old
    # expert ids to new merged group ids.
    if hasattr(router, "tid2eid"):
        old_num_experts = max(max(g) for g in groups) + 1
        if hasattr(router, "num_experts"):
            old_num_experts = max(old_num_experts, int(router.num_experts))
        mapping = torch.zeros(old_num_experts, device=router.tid2eid.device, dtype=router.tid2eid.dtype)
        for group_idx, group in enumerate(groups):
            for old_idx in group:
                if old_idx < mapping.numel():
                    mapping[old_idx] = group_idx
        router.tid2eid.data = mapping[router.tid2eid.clamp(max=mapping.numel() - 1)]

    # Handle router-side correction biases if present.
    if hasattr(router, "e_score_correction_bias"):
        bias = router.e_score_correction_bias
        if bias.ndim == 1:
            bias.data = merge_expert_rows(bias.data)
        elif bias.shape[-1] >= max(max(g) for g in groups) + 1:
            bias.data = merge_expert_columns(bias.data)

    if hasattr(moe_block, "moe_statics") and hasattr(moe_block.moe_statics, "e_score_correction_bias"):
        bias = moe_block.moe_statics.e_score_correction_bias
        if bias.shape[-1] >= max(max(g) for g in groups) + 1:
            bias.data = merge_expert_columns(bias.data)


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
                "Skipping scalar model.config expert-count update."
            )
        else:
            attrs = get_model_attrs(model.__class__.__name__) or {}

            def get_config_path(path: str) -> Any:
                if path.startswith("config."):
                    path = path.split(".", 1)[1]
                current: Any = model.config
                for part in [part for part in path.split(".") if part]:
                    if isinstance(current, dict):
                        if part not in current:
                            return None
                        current = current[part]
                    elif hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        return None
                return current

            def set_config_path(path: str, value: int) -> bool:
                if path.startswith("config."):
                    path = path.split(".", 1)[1]
                parts = [part for part in path.split(".") if part]
                if not parts:
                    return False
                current: Any = model.config
                for part in parts[:-1]:
                    if isinstance(current, dict):
                        if part not in current:
                            return False
                        current = current[part]
                    elif hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        return False
                if isinstance(current, dict) and parts[-1] in current:
                    current[parts[-1]] = value
                    return True
                if hasattr(current, parts[-1]):
                    setattr(current, parts[-1], value)
                    return True
                return False

            for attr_name in [
                str(attrs.get("num_experts", "")),
                "num_experts",
                "num_local_experts",
                "n_routed_experts",
                "moe_num_experts",
                "text_config.num_experts",
            ]:
                if set_config_path(attr_name, final_expert_count):
                    logger.info(f"Updating model.config.{attr_name} = {final_expert_count}")

            for attr_name in [
                str(attrs.get("num_experts_per_tok", "")),
                "num_experts_per_tok",
                "moe_k",
                "moe_topk",
                "top_k",
                "topk",
                "top_k_experts",
                "num_selected_experts",
                "text_config.top_k_experts",
            ]:
                current = get_config_path(attr_name)
                if isinstance(current, int) and current > final_expert_count:
                    set_config_path(attr_name, final_expert_count)

    # Log summary
    if retained_counts:
        original_avg = sum(s["router_logits"].shape[-1] if "router_logits" in s else 0 for s in observer_data.values()) / len(observer_data)
        merged_avg = sum(retained_counts.values()) / len(retained_counts)
        compression = (1 - merged_avg / original_avg) * 100 if original_avg > 0 else 0

        logger.info(
            f"Merging complete: {original_avg:.1f} -> {merged_avg:.1f} "
            f"experts per layer ({compression:.0f}% compression)"
        )

    return retained_counts

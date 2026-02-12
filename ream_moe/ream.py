from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .adapters.base import MoEAdapter, CalibrationBatch, MoELayerHandle


@dataclass
class REAMConfig:
    """
    Configuration for the REAM compressor.
    """

    # Fraction of experts to keep per MoE layer, e.g. 0.75 -> keep 75%.
    target_ratio: float = 0.75
    # Max tokens to collect per layer from calibration data (avoid OOM).
    max_tokens_per_layer: int = 2048 * 512
    # Group size (G) for clustering experts around centroids.
    group_size: int = 16
    # Whether to use gated similarity (router + hidden) when grouping.
    use_gated_similarity: bool = True
    # If True, run REAM sequentially layer-by-layer using the compressed model.
    sequential_merging: bool = True


class REAMCompressor:
    """
    Generic REAM-style expert merging for MoE models.

    This implementation is backend-agnostic and talks to models only via
    the `MoEAdapter` interface.
    """

    def __init__(self, adapter: MoEAdapter, cfg: REAMConfig | None = None):
        self.adapter = adapter
        self.cfg = cfg or REAMConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, calib_batches: Iterable[CalibrationBatch]) -> None:
        """
        Compress all MoE layers in-place using REAM.
        """

        if self.cfg.sequential_merging:
            # Sequential: propagate calibration through the current model
            # and compress one layer at a time.
            for layer in self.adapter.moe_layers():
                stats = self.adapter.forward_collect_calibration(
                    calib_batches,
                    max_tokens=self.cfg.max_tokens_per_layer,
                )[layer]
                self._compress_single_layer(layer, stats)
        else:
            # Non-sequential: run calibration once and compress all layers
            # based on activations of the original model.
            layer_stats = self.adapter.forward_collect_calibration(
                calib_batches,
                max_tokens=self.cfg.max_tokens_per_layer,
            )
            for layer, stats in layer_stats.items():
                self._compress_single_layer(layer, stats)

        self.adapter.rebuild_caches()

    # ------------------------------------------------------------------
    # Single-layer REAM
    # ------------------------------------------------------------------

    def _compress_single_layer(
        self,
        layer: MoELayerHandle,
        stats: Dict[str, Tensor],
    ) -> None:
        router_logits: Tensor = stats["router_logits"]  # [T, N]
        expert_outputs: Tensor = stats["expert_outputs"]  # [N, T, D]

        num_tokens, num_experts = router_logits.shape
        top_k = self.adapter.top_k(layer)
        target_experts = max(1, int(num_experts * self.cfg.target_ratio))

        # 1. Compute REAP-style saliency per expert.
        saliency = self._compute_reap_scores(router_logits, expert_outputs, top_k)

        # 2. Pick centroids: experts with highest saliency.
        centroid_indices = torch.argsort(saliency, descending=True)[:target_experts]

        # 3–4. Group experts around centroids with gated similarity and pseudo-pruning.
        groups = self._group_experts(
            router_logits,
            expert_outputs,
            saliency,
            centroid_indices,
        )

        # 5. Merge each group via permutation-aware averaging.
        new_expert_weights = self._merge_groups_with_alignment(layer, groups, saliency)

        # 6. Adjust router (gate) weights, dropping non-centroids.
        self.adapter.set_expert_weights(layer, new_expert_weights)
        self._adjust_gate_weights(layer, groups)

    # ------------------------------------------------------------------
    # REAP saliency
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reap_scores(
        router_logits: Tensor,   # [T, N]
        expert_outputs: Tensor,  # [N, T, D]
        top_k: int,
    ) -> Tensor:
        """
        REAP-inspired importance:

            S[i] = mean_{tokens routed to i} ||h_i(x)|| * p_i(x)
        """

        T, N = router_logits.shape
        assert expert_outputs.shape[0] == N, "expert axis mismatch"

        probs = torch.softmax(router_logits, dim=-1)   # [T, N]
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)  # [T, top_k]

        saliency = torch.zeros(N, device=router_logits.device)
        for i in range(N):
            token_idx, within_topk_idx = torch.where(topk_idx == i)
            if token_idx.numel() == 0:
                continue

            h_i = expert_outputs[i, token_idx]           # [n_i, D]
            p_i = topk_vals[token_idx, within_topk_idx]  # [n_i]

            saliency[i] = (h_i.norm(dim=-1) * p_i).mean()

        return saliency

    # ------------------------------------------------------------------
    # Grouping: gated similarity + pseudo-pruning
    # ------------------------------------------------------------------

    def _group_experts(
        self,
        router_logits: Tensor,   # [T, N]
        expert_outputs: Tensor,  # [N, T, D]
        saliency: Tensor,        # [N]
        centroid_indices: Tensor,  # [M]
    ) -> List[List[int]]:
        """
        Build groups centered on high-saliency centroids.

        Pseudo-pruning: most low-saliency experts remain singleton; a small
        number near each centroid form compact clusters.
        """

        device = router_logits.device
        T, N = router_logits.shape
        used = torch.zeros(N, dtype=torch.bool, device=device)

        probs = torch.softmax(router_logits, dim=-1)  # [T, N]
        # gated outputs: [N, T, D]
        gated = probs.T.unsqueeze(-1) * expert_outputs
        expert_repr_hidden = gated.mean(dim=1)            # [N, D]
        expert_repr_router = router_logits.T.mean(dim=1)  # [N]

        def cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
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

            # Candidates not yet used.
            unused_idx = torch.where(~used)[0]
            if unused_idx.numel() == 0:
                groups.append(group)
                break

            sim_hidden = cosine_sim(
                expert_repr_hidden[unused_idx],
                expert_repr_hidden[c_idx].expand_as(expert_repr_hidden[unused_idx]),
            )

            sim_router = cosine_sim(
                expert_repr_router[unused_idx].unsqueeze(-1),
                expert_repr_router[c_idx].unsqueeze(0).unsqueeze(-1),
            )

            if self.cfg.use_gated_similarity:
                sim = 0.5 * (sim_hidden + sim_router)
            else:
                sim = sim_hidden

            _, order = torch.sort(sim, descending=True)
            ordered_unused = unused_idx[order]

            max_group = self.cfg.group_size
            for idx in ordered_unused[: max_group - 1]:
                idx_int = int(idx.item())
                group.append(idx_int)
                used[idx_int] = True

            groups.append(group)

        # Any remaining unused experts become singletons.
        remaining = torch.where(~used)[0]
        for r in remaining:
            groups.append([int(r.item())])

        return groups

    # ------------------------------------------------------------------
    # Weight merging with permutation alignment
    # ------------------------------------------------------------------

    def _merge_groups_with_alignment(
        self,
        layer: MoELayerHandle,
        groups: List[List[int]],
        saliency: Tensor,
    ) -> Tensor:
        """
        For each group of expert indices, align and merge their weights with
        REAP-weighted averaging and neuron permutation alignment.
        """

        all_weights = self.adapter.get_expert_weights(layer)  # [N, ...]
        device = all_weights.device

        merged_weights: List[Tensor] = []

        for group in groups:
            if len(group) == 1:
                merged_weights.append(all_weights[group[0]].detach().clone())
                continue

            group_tensor = all_weights[group]  # [G, ...]
            G = len(group)

            # First non-expert axis is treated as the "neuron" axis.
            group_flat = group_tensor.view(G, group_tensor.shape[1], -1)

            ref = group_flat[0]  # [neurons, rest]
            weights_accum = torch.zeros_like(ref)
            s_vals = saliency[torch.tensor(group, device=device)]
            s_norm = s_vals / (s_vals.sum() + 1e-8)

            weights_accum += s_norm[0] * ref

            for g_idx in range(1, G):
                candidate = group_flat[g_idx]  # [neurons, rest]
                cost = torch.cdist(ref, candidate)  # [neurons, neurons]
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
                perm = torch.as_tensor(col_ind, device=device, dtype=torch.long)
                permuted = candidate[perm]
                weights_accum += s_norm[g_idx] * permuted

            merged = weights_accum.view_as(group_tensor[0])
            merged_weights.append(merged)

        new_weights = torch.stack(merged_weights, dim=0)
        return new_weights

    # ------------------------------------------------------------------
    # Gate weight adjustment
    # ------------------------------------------------------------------

    def _adjust_gate_weights(
        self,
        layer: MoELayerHandle,
        groups: List[List[int]],
    ) -> None:
        """
        REAM/REAP-style gate adjustment:

        - Keep only router weights for centroid experts (groups[i][0]).
        - Drop all non-centroid expert logits from the router.
        """

        router = self.adapter.get_router_weights(layer)
        expert_axis = self.adapter.router_expert_axis(layer)

        centroid_indices = [g[0] for g in groups]
        idx_tensor = torch.as_tensor(
            centroid_indices, device=router.device, dtype=torch.long
        )

        router = torch.index_select(router, dim=expert_axis, index=idx_tensor)
        self.adapter.set_router_weights(layer, router)


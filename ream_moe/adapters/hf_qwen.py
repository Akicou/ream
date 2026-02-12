from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import (
    CalibrationBatch,
    ExpertHandle,
    MoEAdapter,
    MoELayerHandle,
)


class HFQwenMoEAdapter(MoEAdapter):
    """
    Example HuggingFace-style adapter for Qwen-like MoE models.

    This is intentionally a skeleton:
      - You must point `_find_moe_layers` at the actual MoE layer class(es)
        in your model implementation.
      - You must implement `forward_collect_calibration` with appropriate
        forward hooks or custom MoE wrappers.
      - You may need to adapt how expert/router weights are accessed.
    """

    def __init__(self, model: PreTrainedModel, device: str = "cuda"):
        super().__init__(model, device)
        self._moe_layers_cache: List[MoELayerHandle] = self._find_moe_layers()

    # ------------------------------------------------------------------
    # MoE layer discovery
    # ------------------------------------------------------------------

    def _find_moe_layers(self) -> List[MoELayerHandle]:
        layers: List[MoELayerHandle] = []
        for module in self.model.modules():
            # TODO: Replace this check with the actual MoE layer class
            # e.g., if isinstance(module, QwenMoeLayer): ...
            if module.__class__.__name__.lower().startswith("moe"):
                layers.append(module)
        return layers

    def moe_layers(self) -> List[MoELayerHandle]:
        return self._moe_layers_cache

    def experts_in_layer(self, layer: MoELayerHandle) -> List[ExpertHandle]:
        # Example: assume `layer.experts` is a ModuleList of expert FFNs.
        return list(layer.experts)

    def top_k(self, layer: MoELayerHandle) -> int:
        # Example: attribute on the MoE layer.
        return int(getattr(layer, "top_k", 8))

    # ------------------------------------------------------------------
    # Calibration forward pass
    # ------------------------------------------------------------------

    def forward_collect_calibration(
        self,
        batches: Iterable[CalibrationBatch],
        max_tokens: int = 2048 * 512,
    ) -> Dict[MoELayerHandle, Dict[str, Tensor]]:
        """
        Collect router logits and expert outputs per MoE layer.

        Pseudocode outline:
          - install forward hooks on each MoE layer to capture:
              router_logits: [num_tokens_layer, num_experts]
              expert_outputs: [num_experts, num_tokens_layer, d_hidden]
          - run the model over calibration batches until `max_tokens` reached
          - stack results per layer

        This needs to be implemented based on the concrete MoE layer API in
        your chosen model family.
        """
        raise NotImplementedError(
            "forward_collect_calibration must be implemented for your concrete MoE layer."
        )

    # ------------------------------------------------------------------
    # Expert weights
    # ------------------------------------------------------------------

    def get_expert_weights(self, layer: MoELayerHandle) -> Tensor:
        """
        Stack each expert's parameters into a [num_experts, ...] tensor.

        Example for GLU-FFN:
          - W_in, W_gate, W_out all of shape [d_hidden, d_model]
          - concatenate along the first dimension -> [3 * d_hidden, d_model]
        """
        weights = []

        for expert in layer.experts:
            # These attribute names are guesses; adapt to your model.
            w_in = expert.w_in.weight.data
            w_gate = expert.w_gate.weight.data
            w_out = expert.w_out.weight.data
            concat = torch.cat([w_in, w_gate, w_out], dim=0)
            weights.append(concat)

        return torch.stack(weights, dim=0)

    def set_expert_weights(self, layer: MoELayerHandle, new_weights: Tensor) -> None:
        """
        Write merged weights back into the first N experts.

        You may also want to actually truncate the expert list to match
        `new_weights.shape[0]`.
        """
        num_new = new_weights.shape[0]
        assert num_new <= len(layer.experts), "Cannot grow number of experts."

        for i in range(num_new):
            w = new_weights[i]
            expert = layer.experts[i]

            d_hidden = expert.w_in.weight.shape[0]
            # Split concatenated weights back into GLU pieces.
            w_in, w_gate, w_out = torch.split(w, d_hidden, dim=0)

            expert.w_in.weight.data.copy_(w_in)
            expert.w_gate.weight.data.copy_(w_gate)
            expert.w_out.weight.data.copy_(w_out)

        # Optional: actually truncate `layer.experts` to `num_new`.
        # This depends on how your MoE implementation handles expert lists.

    # ------------------------------------------------------------------
    # Router (gate) weights
    # ------------------------------------------------------------------

    def get_router_weights(self, layer: MoELayerHandle) -> Tensor:
        """
        Assume router is a Linear(d_model, num_experts) or similar.
        """
        router = layer.router  # type: ignore[attr-defined]
        return router.weight.data

    def set_router_weights(self, layer: MoELayerHandle, new_router_weights: Tensor) -> None:
        router = layer.router  # type: ignore[attr-defined]
        router.weight.data.copy_(new_router_weights)

    def router_expert_axis(self, layer: MoELayerHandle) -> int:
        """
        For Linear(d_model, num_experts) in PyTorch:
          - weight is [out_features, in_features] = [num_experts, d_model]
          => expert axis is 0.
        """
        return 0

    # ------------------------------------------------------------------
    # Post-merge
    # ------------------------------------------------------------------

    def rebuild_caches(self) -> None:
        """
        Ensure model is on the correct device and any internal caches are
        refreshed after weight surgery.
        """
        self.model.to(self.device)


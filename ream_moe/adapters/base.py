from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol

import torch
from torch import Tensor


@dataclass
class CalibrationBatch:
    """
    Generic container for calibration batches.

    Adapters are free to extend this via inheritance or by attaching
    additional attributes, but REAM only depends on `input_ids` and
    `attention_mask`.
    """

    input_ids: Tensor      # (batch, seq_len)
    attention_mask: Tensor # (batch, seq_len)


class MoELayerHandle(Protocol):
    """Opaque identifier for a single MoE layer in the model."""

    ...


class ExpertHandle(Protocol):
    """Opaque identifier for an expert within a specific MoE layer."""

    ...


class MoEAdapter(ABC):
    """
    Abstract adapter: implement this per-model-family (Qwen, gpt-oss, Kimi, etc.).

    The REAM compressor only interacts with models via this interface, so
    adding new model families is as simple as writing a new adapter.
    """

    def __init__(self, model: Any, device: torch.device | str = "cuda"):
        self.model = model
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Model / layer enumeration
    # ------------------------------------------------------------------

    @abstractmethod
    def moe_layers(self) -> List[MoELayerHandle]:
        """Return handles for all MoE layers in the order of execution."""

    @abstractmethod
    def experts_in_layer(self, layer: MoELayerHandle) -> List[ExpertHandle]:
        """Return expert handles for this layer in fixed order [0..N-1]."""

    @abstractmethod
    def top_k(self, layer: MoELayerHandle) -> int:
        """Return TopK used by this MoE layer."""

    # ------------------------------------------------------------------
    # Forward pass collection
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_collect_calibration(
        self,
        batches: Iterable[CalibrationBatch],
        max_tokens: int = 2048 * 512,
    ) -> dict[MoELayerHandle, dict[str, Tensor]]:
        """
        Run the model on calibration data and collect, per MoE layer:

        Returns:
            stats[layer] = {
                "router_logits": Tensor[num_tokens, num_experts],
                "expert_outputs": Tensor[num_experts, num_tokens, d_hidden],
            }

        Exact shapes can vary, but they must be consistent across calls,
        and the first dimension of `expert_outputs` must be the expert axis.

        Typical implementation:
            - register forward hooks on MoE layers
            - run the model for up to `max_tokens`
            - collate the tensors per layer
        """

    # ------------------------------------------------------------------
    # Expert and router (gate) weights access
    # ------------------------------------------------------------------

    @abstractmethod
    def get_expert_weights(self, layer: MoELayerHandle) -> Tensor:
        """
        Return a weight tensor of shape [num_experts, ...] where the last
        dimensions contain concatenated parameters for that expert's FFN/GLU.
        """

    @abstractmethod
    def set_expert_weights(self, layer: MoELayerHandle, new_weights: Tensor) -> None:
        """
        Write back merged expert weights to the model.

        `new_weights` must have the same inner param shape as returned by
        `get_expert_weights`, but with possibly reduced `num_experts`.
        """

    @abstractmethod
    def get_router_weights(self, layer: MoELayerHandle) -> Tensor:
        """
        Return router (gate) weights with the expert axis explicit.

        Example:
            - Linear(d_model, num_experts) -> weight [num_experts, d_model],
              expert axis = 0.
        """

    @abstractmethod
    def set_router_weights(self, layer: MoELayerHandle, new_router_weights: Tensor) -> None:
        """Write updated router weights after expert merging."""

    @abstractmethod
    def router_expert_axis(self, layer: MoELayerHandle) -> int:
        """
        Return which dimension of router weights corresponds to experts.

        For example, if router weights are [num_experts, d_model], return 0.
        If [d_model, num_experts], return -1.
        """

    # ------------------------------------------------------------------
    # Utility / post-merge
    # ------------------------------------------------------------------

    @abstractmethod
    def rebuild_caches(self) -> None:
        """
        Optional: called once after all layers are merged.

        Use this to:
            - rebuild any internal lookup tables
            - move model to correct device
            - re-tie weights, etc.
        """

        ...


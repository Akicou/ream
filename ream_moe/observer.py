"""
Observer module for collecting activation statistics during forward pass.

This module provides the MoEObserver class which hooks into MoE layers
to collect:
- Router logits (for computing routing probabilities)
- Expert outputs (hidden states)
- Expert activation frequencies
- Saliency scores (REAP: norm of expert outputs weighted by routing probs)

These statistics are used by the compressor to determine which experts
to merge or prune.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ream_moe.model_attr_configs import MODEL_ATTRS, get_model_attrs
from ream_moe.model_utils import get_moe_block, get_top_k, ensure_model_registered

logger = logging.getLogger(__name__)


@dataclass
class LayerObserverState:
    """
    Collected statistics for a single MoE layer.

    Attributes:
        router_logits: Router logits for each token [num_tokens, num_experts]
        expert_outputs: Expert output hidden states [num_experts, num_tokens, hidden_dim]
        expert_frequency: Count of how often each expert was activated [num_experts]
        saliency_scores: REAP saliency scores per expert [num_experts]
    """

    router_logits: List[torch.Tensor] = field(default_factory=list)
    expert_outputs: List[torch.Tensor] = field(default_factory=list)
    expert_frequency: Optional[torch.Tensor] = None
    saliency_scores: Optional[torch.Tensor] = None

    def finalize(self, device: torch.device, store_on_cpu: bool = False) -> None:
        """
        Convert collected lists to tensors and compute final statistics.

        Args:
            device: Device for forward pass
            store_on_cpu: If True, keep tensors on CPU to save GPU memory
        """
        # Determine storage device (CPU if requested to save GPU memory)
        storage_device = torch.device('cpu') if store_on_cpu else device

        if self.router_logits:
            self.router_logits = torch.cat(self.router_logits, dim=0).to(storage_device)

        if self.expert_outputs:
            # Concatenate along token dimension
            # expert_outputs is a list of tensors, each with shape [num_experts, num_tokens, hidden_dim]
            self.expert_outputs = torch.cat(self.expert_outputs, dim=1).to(storage_device)

        # Compute final statistics (keep on storage device)
        if self.router_logits is not None:
            num_experts = self.router_logits.shape[-1]
            probs = torch.softmax(self.router_logits, dim=-1)

            # Get top-k selections
            top_k = probs.shape[-1]  # Default to all if not specified
            _, topk_idx = torch.topk(probs, k=min(top_k, num_experts), dim=-1)

            # Count expert activations
            flat_idx = topk_idx.view(-1)
            self.expert_frequency = torch.bincount(
                flat_idx, minlength=num_experts
            ).to(storage_device)


@dataclass
class ObserverConfig:
    """Configuration for the observer."""

    max_tokens_per_layer: int = 2048 * 512  # Maximum tokens to collect per layer
    renormalize_router_weights: bool = False  # Renormalize router after top-k
    device: str = "cuda"  # Device for forward pass
    store_on_cpu: bool = False  # Store collected statistics on CPU to save GPU memory


class MoEObserver:
    """
    Observer for collecting MoE activation statistics during forward pass.

    Usage:
        observer = MoEObserver(model, config=ObserverConfig())
        observer.hook_model()

        # Run forward pass on calibration data
        for batch in calibration_data:
            model(batch.input_ids, batch.attention_mask)

        observer.unhook_model()
        stats = observer.get_collected_stats()
    """

    def __init__(self, model: nn.Module, config: ObserverConfig | None = None):
        """
        Initialize the observer.

        Args:
            model: The model to observe
            config: Observer configuration
        """
        self.model = model
        self.config = config or ObserverConfig()
        self.hooks: List[Callable] = []
        self.layer_states: Dict[int, LayerObserverState] = {}

        # Ensure model is registered
        ensure_model_registered(model)

        # Get model attributes
        self.model_attrs = get_model_attrs(model.__class__.__name__)
        if self.model_attrs is None:
            raise ValueError(
                f"Model {model.__class__.__name__} not registered in MODEL_ATTRS"
            )

        # Find MoE layers
        from ream_moe.model_utils import list_moe_layers
        self.moe_layer_indices = list_moe_layers(model)

        if not self.moe_layer_indices:
            logger.warning(f"No MoE layers found in model {model.__class__.__name__}")

    def hook_model(self) -> None:
        """Register forward hooks on all MoE layers."""
        for layer_idx in self.moe_layer_indices:
            moe_block = get_moe_block(self.model, layer_idx)
            self.layer_states[layer_idx] = LayerObserverState()

            # Create hook function for this layer
            def make_hook(idx: int):
                def hook(module, args, output):
                    return self._forward_hook(idx, module, args, output)
                return hook

            handle = moe_block.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

        logger.info(
            f"Registered hooks on {len(self.hooks)} MoE layers for model "
            f"{self.model.__class__.__name__}"
        )

    def unhook_model(self) -> None:
        """Remove all forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("Removed all observer hooks")

    def _forward_hook(
        self,
        layer_idx: int,
        module: nn.Module,
        args: tuple,
        output: Any,
    ) -> None:
        """
        Forward hook for collecting statistics from a single MoE layer.

        Args:
            layer_idx: Index of the current layer
            module: The MoE block module
            args: Input arguments (input_ids, attention_mask, etc.)
            output: Output from the MoE block
        """
        state = self.layer_states[layer_idx]

        # Get input hidden states
        input_hidden = args[0]  # [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = input_hidden.shape
        num_tokens = batch_size * seq_len

        # Check if we've collected enough tokens
        tokens_collected = sum(
            t.shape[0] for t in state.router_logits
        ) if state.router_logits else 0

        if tokens_collected >= self.config.max_tokens_per_layer:
            return  # Skip this layer, already have enough data

        # Get num_experts and top_k
        num_experts = _get_num_experts_from_module(module, self.model_attrs)
        top_k = _get_top_k_from_module(module, self.model_attrs)

        # Flatten input for processing
        flat_input = input_hidden.view(-1, hidden_dim)  # [num_tokens, hidden_dim]

        # Get router logits
        router_logits = self._extract_router_logits(
            module, output, flat_input, num_experts
        )  # [num_tokens, num_experts]

        # Get selected experts
        probs = torch.softmax(router_logits, dim=-1)
        if self.config.renormalize_router_weights:
            topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
            probs = probs / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        _, selected_experts = torch.topk(probs, k=top_k, dim=-1)  # [num_tokens, top_k]

        # Limit tokens to collect
        remaining_tokens = self.config.max_tokens_per_layer - tokens_collected
        if num_tokens > remaining_tokens:
            # Randomly sample tokens
            indices = torch.randperm(num_tokens, device=input_hidden.device)[:remaining_tokens]
            flat_input = flat_input[indices]
            router_logits = router_logits[indices]
            selected_experts = selected_experts[indices]
            num_tokens = remaining_tokens

        # Collect router logits
        state.router_logits.append(router_logits.cpu())

        # Compute expert outputs
        expert_outputs_list = []

        if self.model_attrs.get("fused", False):
            # Fused experts - compute all at once
            expert_outputs = self._compute_fused_expert_outputs(
                module, flat_input, num_experts
            )  # [num_experts, num_tokens, hidden_dim]
        else:
            # Non-fused experts - compute each separately
            expert_outputs = self._compute_separate_expert_outputs(
                module, flat_input, num_experts
            )  # [num_experts, num_tokens, hidden_dim]

        expert_outputs_list.append(expert_outputs.cpu())

        state.expert_outputs.extend(expert_outputs_list)

    def _extract_router_logits(
        self,
        module: nn.Module,
        output: Any,
        flat_input: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """
        Extract router logits from the MoE block.

        Handles multiple patterns for where router logits might be stored.
        """
        # Pattern 1: Check module's _last_router_logits (auto-patched models)
        if hasattr(module, "_last_router_logits") and module._last_router_logits is not None:
            logits = module._last_router_logits
            module._last_router_logits = None
            return logits

        # Pattern 2: Check output tuple
        if isinstance(output, tuple) and len(output) >= 2:
            logits = output[-1]
            if isinstance(logits, torch.Tensor) and logits.ndim == 2:
                return logits

        # Pattern 3: Find router/gate module and compute
        router_attr = self.model_attrs.get("router", "gate")
        if hasattr(module, router_attr):
            router = getattr(module, router_attr)

            # Handle nested router (e.g., router.classifier for LongCat)
            router_weight_attr = self.model_attrs.get("router_weight_attr")
            if router_weight_attr and "." in router_weight_attr:
                parts = router_weight_attr.split(".")
                weight = reduce(getattr, parts, router)
            elif hasattr(router, "weight"):
                weight = router.weight
            elif hasattr(router, "classifier") and hasattr(router.classifier, "weight"):
                weight = router.classifier.weight
            else:
                # Fallback: create zeros
                return torch.zeros(
                    flat_input.shape[0], num_experts,
                    device=flat_input.device, dtype=flat_input.dtype
                )

            # Compute logits
            logits = F.linear(flat_input.to(weight.dtype), weight)
            return logits

        # Fallback: create placeholder
        return torch.zeros(
            flat_input.shape[0], num_experts,
            device=flat_input.device, dtype=flat_input.dtype
        )

    def _compute_fused_expert_outputs(
        self, module: nn.Module, flat_input: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        """
        Compute outputs for fused experts (gate_up_proj + down_proj pattern).

        Returns:
            Expert outputs [num_experts, num_tokens, hidden_dim]
        """
        experts = module.experts
        gate_up_proj = experts.gate_up_proj  # [num_experts, 2*intermediate, hidden_dim]
        down_proj = experts.down_proj  # [num_experts, hidden_dim, intermediate]

        num_tokens = flat_input.shape[0]
        intermediate_size = down_proj.shape[2]
        device = flat_input.device
        dtype = flat_input.dtype

        outputs = []

        for expert_idx in range(num_experts):
            # Get expert weights
            gate_up = gate_up_proj[expert_idx]  # [2*I, H]
            down = down_proj[expert_idx]  # [H, I]

            # Forward pass
            gate_up_out = F.linear(flat_input, gate_up)  # [tokens, 2*I]
            gate, up = gate_up_out.chunk(2, dim=-1)  # each [tokens, I]
            hidden = F.silu(gate) * up  # [tokens, I]
            output = F.linear(hidden, down)  # [tokens, H]
            outputs.append(output)

        return torch.stack(outputs, dim=0)  # [num_experts, num_tokens, hidden_dim]

    def _compute_separate_expert_outputs(
        self, module: nn.Module, flat_input: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        """
        Compute outputs for separate experts (individual Linear layers).

        Returns:
            Expert outputs [num_experts, num_tokens, hidden_dim]
        """
        outputs = []

        for expert_idx in range(num_experts):
            expert = module.experts[expert_idx]
            output = expert(flat_input)
            outputs.append(output)

        return torch.stack(outputs, dim=0)  # [num_experts, num_tokens, hidden_dim]

    def get_collected_stats(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Get the collected statistics for all layers.

        Finalizes all layer states and computes final statistics.

        Returns:
            Dictionary mapping layer_idx -> stats dict
        """
        device = torch.device(self.config.device)

        for layer_idx, state in self.layer_states.items():
            if state.expert_frequency is None:
                state.finalize(device, store_on_cpu=self.config.store_on_cpu)

            # Compute saliency scores
            if (
                state.router_logits is not None
                and state.expert_outputs is not None
                and state.saliency_scores is None
            ):
                state.saliency_scores = self._compute_saliency(
                    state.router_logits, state.expert_outputs
                )

        # Convert to output format
        result = {}
        for layer_idx, state in self.layer_states.items():
            result[layer_idx] = {
                "router_logits": state.router_logits,
                "expert_outputs": state.expert_outputs,
                "expert_frequency": state.expert_frequency,
                "saliency_scores": state.saliency_scores,
            }

        return result

    @staticmethod
    def _compute_saliency(
        router_logits: torch.Tensor,  # [num_tokens, num_experts]
        expert_outputs: torch.Tensor,  # [num_experts, num_tokens, hidden_dim]
    ) -> torch.Tensor:
        """
        Compute REAP saliency scores per expert.

        S[i] = mean_{tokens routed to i} ||h_i(x)|| * p_i(x)

        Args:
            router_logits: Router logits for all tokens
            expert_outputs: Expert output hidden states

        Returns:
            Saliency scores [num_experts]
        """
        num_tokens, num_experts = router_logits.shape
        probs = torch.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

        # Get top-k
        top_k = min(num_experts, router_logits.shape[-1])
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)

        saliency = torch.zeros(num_experts, device=router_logits.device)

        for i in range(num_experts):
            # Find tokens where this expert was in top-k
            token_idx, within_topk_idx = torch.where(topk_idx == i)

            if token_idx.numel() == 0:
                continue

            # Get expert outputs for these tokens
            h_i = expert_outputs[i, token_idx]  # [n_i, hidden_dim]
            p_i = topk_vals[token_idx, within_topk_idx]  # [n_i]

            # Compute weighted norm
            norm = h_i.norm(dim=-1)  # [n_i]
            saliency[i] = (norm * p_i).mean()

        return saliency


def _get_num_experts_from_module(module: nn.Module, model_attrs: Dict[str, Any]) -> int:
    """Get number of experts from a MoE module."""
    num_experts_attr = model_attrs.get("num_experts", "num_experts")

    if num_experts_attr.startswith("config."):
        # Try getting from module's config
        if hasattr(module, "config"):
            config_key = num_experts_attr.split(".", 1)[1]
            if hasattr(module.config, config_key):
                return getattr(module.config, config_key)

    # Try direct attribute
    try:
        from functools import reduce
        return reduce(getattr, num_experts_attr.split("."), module)
    except AttributeError:
        pass

    # Count experts
    if hasattr(module, "experts"):
        experts = module.experts
        if isinstance(experts, nn.ModuleList):
            return len(experts)
        elif hasattr(experts, "gate_up_proj"):
            return experts.gate_up_proj.shape[0]

    raise ValueError(f"Cannot determine num_experts for module {module}")


def _get_top_k_from_module(module: nn.Module, model_attrs: Dict[str, Any]) -> int:
    """Get top-k value from a MoE module."""
    top_k_attr = model_attrs.get("num_experts_per_tok", "top_k")

    if top_k_attr.startswith("config."):
        if hasattr(module, "config"):
            config_key = top_k_attr.split(".", 1)[1]
            if hasattr(module.config, config_key):
                return getattr(module.config, config_key)

    try:
        from functools import reduce
        return reduce(getattr, top_k_attr.split("."), module)
    except AttributeError:
        pass

    # Default fallback
    return 1


def observe_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    config: ObserverConfig | None = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Convenience function to observe a model on a single batch.

    Args:
        model: The model to observe
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        config: Observer configuration

    Returns:
        Dictionary of collected statistics per layer
    """
    observer = MoEObserver(model, config or ObserverConfig())
    observer.hook_model()

    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    finally:
        observer.unhook_model()

    return observer.get_collected_stats()

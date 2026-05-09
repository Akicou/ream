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
from functools import reduce
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
    selected_experts: List[torch.Tensor] = field(default_factory=list)
    routing_weights: List[torch.Tensor] = field(default_factory=list)
    expert_frequency: Optional[torch.Tensor] = None
    saliency_scores: Optional[torch.Tensor] = None

    def finalize(self, device: torch.device, store_on_cpu: bool = False, top_k: Optional[int] = None) -> None:
        """
        Convert collected lists to tensors and compute final statistics.

        Args:
            device: Device for forward pass
            store_on_cpu: If True, keep tensors on CPU to save GPU memory
            top_k: Actual number of experts activated per token (from model config).
                   If None, falls back to using all experts (inaccurate for frequency).
        """
        # Determine storage device (CPU if requested to save GPU memory)
        storage_device = torch.device('cpu') if store_on_cpu else device

        if self.router_logits:
            self.router_logits = torch.cat(self.router_logits, dim=0).to(storage_device)

        if self.expert_outputs:
            # Concatenate along token dimension
            # expert_outputs is a list of tensors, each with shape [num_experts, num_tokens, hidden_dim]
            self.expert_outputs = torch.cat(self.expert_outputs, dim=1).to(storage_device)

        if self.selected_experts:
            self.selected_experts = torch.cat(self.selected_experts, dim=0).to(storage_device)

        if self.routing_weights:
            self.routing_weights = torch.cat(self.routing_weights, dim=0).to(storage_device)

        # Compute final statistics (keep on storage device)
        if isinstance(self.router_logits, torch.Tensor):
            num_experts = self.router_logits.shape[-1]

            if isinstance(self.selected_experts, torch.Tensor):
                topk_idx = self.selected_experts
            else:
                probs = torch.softmax(self.router_logits, dim=-1)

                # Use the actual model top_k so frequency counts only routed tokens
                actual_top_k = top_k if top_k is not None else num_experts
                actual_top_k = min(actual_top_k, num_experts)
                _, topk_idx = torch.topk(probs, k=actual_top_k, dim=-1)

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
        self.layer_top_k: Dict[int, int] = {}  # actual top-k per layer from model config

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

            # Store actual top-k for this layer so saliency only counts routed tokens
            try:
                from ream_moe.model_utils import get_top_k
                self.layer_top_k[layer_idx] = get_top_k(self.model, layer_idx)
            except Exception:
                self.layer_top_k[layer_idx] = 1  # safe conservative fallback

            # Create hook function for this layer. Use with_kwargs when
            # available so DeepSeek-V4 hash MoE can read input_ids.
            def make_hook(idx: int):
                def hook(module, args, kwargs, output):
                    return self._forward_hook(idx, module, args, output, kwargs)
                return hook

            try:
                handle = moe_block.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
            except TypeError:
                def make_legacy_hook(idx: int):
                    def hook(module, args, output):
                        return self._forward_hook(idx, module, args, output)
                    return hook

                handle = moe_block.register_forward_hook(make_legacy_hook(layer_idx))
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
        kwargs: Optional[dict[str, Any]] = None,
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

        # Get selected experts and routing weights. DeepSeek-V4 hash MoE layers
        # route via gate.tid2eid[input_ids] rather than top-k router logits, so
        # preserve the actual selected experts when input_ids are available.
        selected_experts, routing_weights = self._extract_selected_experts_and_weights(
            module, args, router_logits, top_k, kwargs
        )

        # Limit tokens to collect
        remaining_tokens = self.config.max_tokens_per_layer - tokens_collected
        if num_tokens > remaining_tokens:
            # Randomly sample tokens
            indices = torch.randperm(num_tokens, device=input_hidden.device)[:remaining_tokens]
            flat_input = flat_input[indices]
            router_logits = router_logits[indices]
            selected_experts = selected_experts[indices]
            routing_weights = routing_weights[indices]
            num_tokens = remaining_tokens

        # Collect router logits and actual routing choices
        state.router_logits.append(router_logits.cpu())
        state.selected_experts.append(selected_experts.cpu())
        state.routing_weights.append(routing_weights.cpu())

        # Compute expert outputs
        expert_outputs_list = []

        experts_obj = getattr(module, self.model_attrs.get("experts", "experts"), None)
        if self.model_attrs.get("fused", False) or hasattr(experts_obj, "gate_up_proj"):
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

    def _extract_selected_experts_and_weights(
        self,
        module: nn.Module,
        args: tuple,
        router_logits: torch.Tensor,
        top_k: int,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract actual routed experts and their routing weights.

        Most MoE models select top-k experts from router probabilities. DeepSeek-V4
        hash layers are different: the first layers select experts from a
        persistent token-id -> expert-id table (`gate.tid2eid`) and only use the
        learned gate scores to weight those selected experts.
        """
        router_attr = self.model_attrs.get("router", "gate")
        router = getattr(module, router_attr, None)

        input_ids = None
        if len(args) > 1 and args[1] is not None:
            input_ids = args[1]
        elif kwargs is not None:
            input_ids = kwargs.get("input_ids")

        if router is not None and hasattr(router, "tid2eid") and input_ids is not None:
            input_ids = input_ids.reshape(-1).to(router.tid2eid.device)
            selected_experts = router.tid2eid[input_ids].to(router_logits.device).long()

            if hasattr(router, "score_fn"):
                scores = router.score_fn(router_logits.float())
            else:
                scores = torch.softmax(router_logits, dim=-1)

            routing_weights = torch.gather(scores, dim=-1, index=selected_experts)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)

            if hasattr(router, "routed_scaling_factor"):
                routing_weights = routing_weights * router.routed_scaling_factor

            return selected_experts, routing_weights.to(router_logits.dtype)

        # DeepSeek-style top-k routers use a non-softmax score function and
        # normalize only the selected expert weights.
        if router is not None and hasattr(router, "score_fn"):
            scores = router.score_fn(router_logits.float())
            selection_scores = scores
            correction_bias = getattr(router, "e_score_correction_bias", None)
            if correction_bias is None:
                correction_bias = getattr(router, "bias", None)
            if correction_bias is not None:
                selection_scores = selection_scores + correction_bias.to(selection_scores.device)

            selected_experts = torch.topk(selection_scores, k=top_k, dim=-1, sorted=False).indices
            topk_vals = torch.gather(scores, dim=-1, index=selected_experts)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-20)

            if hasattr(router, "routed_scaling_factor"):
                topk_vals = topk_vals * router.routed_scaling_factor

            return selected_experts, topk_vals.to(router_logits.dtype)

        probs = torch.softmax(router_logits, dim=-1)
        topk_vals, selected_experts = torch.topk(probs, k=top_k, dim=-1)

        if self.config.renormalize_router_weights:
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return selected_experts, topk_vals

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
            bias = None
            if router_weight_attr and "." in router_weight_attr:
                parts = router_weight_attr.split(".")
                inner = router
                for part in parts[:-1]:
                    inner = getattr(inner, part)
                weight_attr = parts[-1]
                weight = getattr(inner, weight_attr)
                bias_attr = weight_attr.replace("weight", "bias")
                bias = getattr(inner, bias_attr, None)
            elif hasattr(router, "weight"):
                weight = router.weight
                bias = getattr(router, "bias", None)
            elif hasattr(router, "classifier") and hasattr(router.classifier, "weight"):
                weight = router.classifier.weight
                bias = getattr(router.classifier, "bias", None)
            else:
                # Fallback: create zeros
                return torch.zeros(
                    flat_input.shape[0], num_experts,
                    device=flat_input.device, dtype=flat_input.dtype
                )

            # Compute logits. DeepSeek-style routers use `bias` as a correction
            # term after the score function, not as a linear bias.
            linear_bias = None if (hasattr(router, "score_fn") or hasattr(router, "tid2eid")) else bias
            logits = F.linear(flat_input.to(weight.dtype), weight, linear_bias)
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
            top_k = self.layer_top_k.get(layer_idx)
            if state.expert_frequency is None:
                state.finalize(device, store_on_cpu=self.config.store_on_cpu, top_k=top_k)

            # Compute saliency scores
            if (
                isinstance(state.router_logits, torch.Tensor)
                and isinstance(state.expert_outputs, torch.Tensor)
                and state.saliency_scores is None
            ):
                state.saliency_scores = self._compute_saliency(
                    state.router_logits,
                    state.expert_outputs,
                    top_k=top_k,
                    renormalize_topk=self.config.renormalize_router_weights,
                    selected_experts=state.selected_experts if isinstance(state.selected_experts, torch.Tensor) else None,
                    routing_weights=state.routing_weights if isinstance(state.routing_weights, torch.Tensor) else None,
                )

        # Convert to output format
        result = {}
        for layer_idx, state in self.layer_states.items():
            result[layer_idx] = {
                "router_logits": state.router_logits,
                "expert_outputs": state.expert_outputs,
                "expert_frequency": state.expert_frequency,
                "saliency_scores": state.saliency_scores,
                "selected_experts": state.selected_experts if isinstance(state.selected_experts, torch.Tensor) else None,
                "routing_weights": state.routing_weights if isinstance(state.routing_weights, torch.Tensor) else None,
            }

        return result

    @staticmethod
    def _compute_saliency(
        router_logits: torch.Tensor,  # [num_tokens, num_experts]
        expert_outputs: torch.Tensor,  # [num_experts, num_tokens, hidden_dim]
        top_k: Optional[int] = None,
        renormalize_topk: bool = False,
        selected_experts: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute REAP saliency scores per expert.

        S[i] = mean_{tokens routed to i} ||h_i(x)|| * p_i(x)

        Args:
            router_logits: Router logits for all tokens
            expert_outputs: Expert output hidden states
            top_k: Actual number of experts activated per token (from model config).
                   Critical for correctness: only tokens where expert i was in the
                   top-k routing selection should contribute to its saliency score.
            renormalize_topk: Whether to renormalize top-k router probabilities
                   before weighting expert-output norms, matching Mixtral/Qwen-style
                   routing implementations that normalize selected expert weights.
            selected_experts: Optional actual selected experts [num_tokens, top_k].
                   Used by hash-routing models like DeepSeek-V4.
            routing_weights: Optional actual routing weights [num_tokens, top_k].

        Returns:
            Saliency scores [num_experts]
        """
        num_tokens, num_experts = router_logits.shape
        if selected_experts is not None:
            topk_idx = selected_experts.to(router_logits.device)
            if routing_weights is not None:
                topk_vals = routing_weights.to(router_logits.device)
            else:
                probs = torch.softmax(router_logits, dim=-1)
                topk_vals = torch.gather(probs, dim=-1, index=topk_idx)
        else:
            probs = torch.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

            # Use the model's actual top-k so only routed tokens count toward saliency
            actual_top_k = top_k if top_k is not None else num_experts
            actual_top_k = min(actual_top_k, num_experts)
            topk_vals, topk_idx = torch.topk(probs, k=actual_top_k, dim=-1)
            if renormalize_topk:
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

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

    # Fallback: common names on the block and its router.
    for attr_name in ["top_k", "topk", "num_experts_per_tok", "k", "num_selected_experts"]:
        if hasattr(module, attr_name):
            val = getattr(module, attr_name)
            if isinstance(val, int):
                return val

    router_attr = model_attrs.get("router", "gate")
    if hasattr(module, router_attr):
        router = getattr(module, router_attr)
        for attr_name in ["top_k", "topk", "num_experts_per_tok", "k", "num_selected_experts"]:
            if hasattr(router, attr_name):
                val = getattr(router, attr_name)
                if isinstance(val, int):
                    return val

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

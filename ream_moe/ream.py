"""
REAM/REAP-style MoE compression — unified high-level API.

This module provides the :class:`REAMConfig` dataclass and
:class:`REAMCompressor` as a single-entry-point wrapper around the individual
observer / prune / merge pipelines, plus the legacy
:func:`observe_model` / :func:`prune_model` / :func:`merge_model` functions
exported from their own sub-modules.

Typical usage::

    from ream_moe import REAMConfig, REAMCompressor
    from ream_moe.calibration import build_calibration_batches

    compressor = REAMCompressor(model, tokenizer)
    batches = build_calibration_batches(tokenizer, "hardcoded", samples=1000)
    compressor.compress(batches, ratio=0.25)
    compressor.save("./compressed_model")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .calibration import CalibrationBatch
from .model_attr_configs import get_model_attrs, MODEL_ATTRS
from .model_utils import list_moe_layers, get_moe_block, ensure_model_registered
from .observer import MoEObserver, ObserverConfig, LayerObserverState
from .prune import PruningConfig, prune_model, compute_experts_to_prune
from .merge import MergeConfig, merge_model

logger = logging.getLogger(__name__)


@dataclass
class REAMConfig:
    """Unified configuration for REAM/REAP compression.

    Attributes:
        target_ratio:
            Fraction of experts to **keep** per MoE layer (e.g. 0.75 → keep 75 %).
            This is converted to ``MergeConfig.target_ratio`` or the inverse of
            ``PruningConfig.compression_ratio`` depending on ``method``.
        method:
            ``"prune"``, ``"merge"`` or ``"offline"`` (random seed-based pruning).
        group_size:
            Max experts per merge group (merge-only).
        use_gated_similarity:
            Combine router + hidden-state similarity when grouping (merge-only).
        preserve_super_experts:
            If True, experts with unusually high activations are never pruned
            (prune-only).
        super_expert_quantile:
            Quantile threshold for identifying super-experts (prune-only).
        max_tokens_per_layer:
            Maximum calibration tokens collected per layer (observer).
        store_on_cpu:
            Keep observer statistics on CPU to save GPU memory.
    """

    target_ratio: float = 0.75
    method: str = "merge"  # "prune" | "merge" | "offline"
    group_size: int = 16
    use_gated_similarity: bool = True
    preserve_super_experts: bool = False
    super_expert_quantile: float = 99.5
    max_tokens_per_layer: int = 2048 * 512
    store_on_cpu: bool = False


class REAMCompressor:
    """High-level entry point for MoE model compression.

    ``REAMCompressor`` wraps the lower-level observer / prune / merge pipelines
    so you can compress a model with a single call.

    Parameters:
        model:
            A transformers ``PreTrainedModel`` containing MoE layers.
        tokenizer:
            Tokenizer matching *model* (used for calibration).
    """

    def __init__(self, model: nn.Module, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

        model_class = model.__class__.__name__
        if model_class not in MODEL_ATTRS:
            logger.info(
                "Model %s not in MODEL_ATTRS — attempting auto-registration.",
                model_class,
            )
            ensure_model_registered(model)

        self._model_attrs = get_model_attrs(model_class)
        if self._model_attrs is None:
            raise ValueError(
                f"Model {model_class!r} is not registered in MODEL_ATTRS "
                f"and auto-registration failed."
            )

        self._moe_layers = list_moe_layers(model)
        if not self._moe_layers:
            raise ValueError(
                f"No MoE layers detected in model {model_class!r}."
            )

        self._observer: Optional[MoEObserver] = None
        self._observer_data: Optional[Dict[int, Dict[str, torch.Tensor]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        calib_batches: Iterable[CalibrationBatch],
        config: REAMConfig | None = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Run calibration forward passes and return per-layer observer stats.

        This is separated from :meth:`compress` so you can inspect / cache
        observer output before deciding how to compress.
        """
        cfg = config or REAMConfig()
        obs_cfg = ObserverConfig(
            max_tokens_per_layer=cfg.max_tokens_per_layer,
            store_on_cpu=cfg.store_on_cpu,
        )
        observer = MoEObserver(self.model, obs_cfg)
        observer.hook_model()

        try:
            with torch.no_grad():
                for batch in tqdm(calib_batches, desc="Calibrating"):
                    # Handle both dict-like and attribute-based batches.
                    input_ids = getattr(batch, "input_ids", None)
                    if input_ids is None and isinstance(batch, dict):
                        input_ids = batch.get("input_ids")
                    attn_mask = getattr(batch, "attention_mask", None)
                    if attn_mask is None and isinstance(batch, dict):
                        attn_mask = batch.get("attention_mask")
                    self.model(input_ids, attention_mask=attn_mask)
        finally:
            observer.unhook_model()

        self._observer_data = observer.get_collected_stats()
        self._observer = observer
        return self._observer_data

    def compress(
        self,
        calib_batches: Iterable[CalibrationBatch],
        config: REAMConfig | None = None,
    ) -> Dict[int, int]:
        """Observe, then compress all MoE layers in-place.

        Returns a dictionary ``{layer_idx: retained_experts}``.
        """
        cfg = config or REAMConfig()

        # 1. Observe
        observer_data = self.observe(calib_batches, cfg)

        # 2. Compress
        method = cfg.method.lower()
        if method == "prune":
            return self._prune(observer_data, cfg)
        elif method == "merge":
            return self._merge(observer_data, cfg)
        else:
            raise ValueError(
                f"Unknown method {cfg.method!r}. Choose 'prune' or 'merge'."
            )

    def save(self, output_dir: str) -> None:
        """Save the (compressed) model and tokenizer to *output_dir*."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(
        self,
        observer_data: Dict[int, Dict[str, torch.Tensor]],
        cfg: REAMConfig,
    ) -> Dict[int, int]:
        compression_ratio = 1.0 - cfg.target_ratio
        prune_cfg = PruningConfig(
            compression_ratio=compression_ratio,
            preserve_super_experts=cfg.preserve_super_experts,
            super_expert_quantile=cfg.super_expert_quantile,
        )
        return prune_model(self.model, observer_data, prune_cfg)

    def _merge(
        self,
        observer_data: Dict[int, Dict[str, torch.Tensor]],
        cfg: REAMConfig,
    ) -> Dict[int, int]:
        merge_cfg = MergeConfig(
            target_ratio=cfg.target_ratio,
            group_size=cfg.group_size,
            use_gated_similarity=cfg.use_gated_similarity,
        )
        return merge_model(self.model, observer_data, merge_cfg)

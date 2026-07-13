#!/usr/bin/env python3
"""
Sequential MoE compression — one layer at a time (REAM paper approach).

Unlike compress_model.py which observes ALL layers before merging,
this script processes layers sequentially: observe layer i → merge layer i →
re-observe layer i+1 with updated hidden states. This avoids storing
expert outputs for all layers simultaneously, making it suitable for
large models like Tencent Hy3 (192 experts, 80 layers).

Usage:
    python compress_sequential.py \
        --model tencent/Hy3 \
        --output ./hy3-100B \
        --target-ratio 0.333 \
        --samples 200 \
        --max-tokens 4096
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ream_moe.merge import MergeConfig, merge_layer, _compute_saliency_scores, _group_experts_around_centroids, _merge_groups, _update_merged_weights
from ream_moe.calibration import build_calibration_batches
from ream_moe.model_utils import list_moe_layers, get_moe_block, ensure_model_registered
from ream_moe.observer import MoEObserver, ObserverConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sequential REAM compression")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--target-ratio", type=float, default=0.333, help="Fraction of experts to KEEP")
    parser.add_argument("--samples", type=int, default=200, help="Calibration samples per layer")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per layer observation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--cpu-merge", action="store_true", help="Process merge weights on CPU")
    parser.add_argument("--fast-merge", action="store_true", help="Skip Hungarian permutation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model and tokenizer
    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    ensure_model_registered(model)

    moe_layers = list_moe_layers(model)
    logger.info("Found %d MoE layers (indices: %s)", len(moe_layers), moe_layers[:5])

    # Build calibration batches once
    batches = list(build_calibration_batches(
        tokenizer, "hardcoded",
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        samples=args.samples,
    ))
    logger.info("Calibration: %d batches, batch_size=%d, total ~%d tokens",
                len(batches), args.batch_size,
                len(batches) * args.batch_size * args.max_seq_len)

    merge_config = MergeConfig(
        target_ratio=args.target_ratio,
        group_size=args.group_size,
        use_cpu_for_weights=args.cpu_merge,
        skip_permutation=args.fast_merge,
        avg_router=False,  # REAM paper: keep centroid rows only
    )

    retained_counts = {}
    device = next(model.parameters()).device

    # Process layers one at a time
    for layer_idx in tqdm(moe_layers, desc="Layers"):
        logger.info("Layer %d: observing...", layer_idx)

        # Observe ONLY this layer with a small token budget
        obs_config = ObserverConfig(
            max_tokens_per_layer=args.max_tokens,
            device=str(device),
            store_on_cpu=True,
        )
        observer = MoEObserver(model, obs_config)
        observer.hook_model()

        try:
            with torch.no_grad():
                for batch in batches:
                    kwargs = {"input_ids": batch.input_ids.to(device)}
                    if hasattr(batch, "attention_mask") and batch.attention_mask is not None:
                        kwargs["attention_mask"] = batch.attention_mask.to(device)
                    model(**kwargs)
        finally:
            observer.unhook_model()

        all_stats = observer.get_collected_stats()
        layer_stats = all_stats.get(layer_idx)

        if layer_stats is None:
            logger.warning("Layer %d: no stats collected, skipping", layer_idx)
            continue

        router_logits = layer_stats.get("router_logits")
        expert_outputs = layer_stats.get("expert_outputs")
        if router_logits is None or expert_outputs is None:
            logger.warning("Layer %d: missing data, skipping", layer_idx)
            continue

        logger.info("Layer %d: %d tokens, %d experts — merging...",
                    layer_idx, router_logits.shape[0], router_logits.shape[-1])

        # Merge this layer
        retained = merge_layer(model, layer_idx, layer_stats, merge_config)
        retained_counts[layer_idx] = retained

        # Clear GPU cache between layers
        torch.cuda.empty_cache()
        logger.info("Layer %d: %d experts retained", layer_idx, retained)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update config
    if retained_counts:
        unique = set(retained_counts.values())
        final_count = list(retained_counts.values())[0]
        if len(unique) == 1:
            for attr in ["num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"]:
                if hasattr(model.config, attr):
                    setattr(model.config, attr, final_count)
            logger.info("Updated model.config.num_experts = %d", final_count)
        else:
            logger.warning("Different expert counts per layer: %s", unique)

    logger.info("Saving model to %s...", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Done! Model saved to %s", output_dir)

    # Summary
    if retained_counts:
        avg = sum(retained_counts.values()) / len(retained_counts)
        logger.info("Summary: %.1f experts/layer on average across %d layers", avg, len(retained_counts))


if __name__ == "__main__":
    main()

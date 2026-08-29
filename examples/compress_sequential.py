#!/usr/bin/env python3
"""
Sequential MoE compression — one layer at a time, single-layer hooks.

Unlike compress_model.py which observes ALL layers before merging,
this script hooks only the current layer, runs the forward pass to
collect its activations, merges it, then moves to the next layer.
This avoids OOM for large models (192 experts × 79 layers).

Usage:
    python compress_sequential.py \
        --model tencent/Hy3 \
        --output ./hy3-100B \
        --target-ratio 0.333 \
        --max-tokens 2048
"""

import argparse
import logging
import sys
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ream_moe.merge import MergeConfig, merge_layer
from ream_moe.calibration import build_calibration_batches
from ream_moe.model_utils import list_moe_layers, get_moe_block, ensure_model_registered
from ream_moe.model_attr_configs import get_model_attrs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_single_layer(
    model, layer_idx, batches, attrs, device, max_tokens: int = 2048
):
    """Run forward pass, collecting observer stats for ONLY one layer."""
    router_logits_list = []
    expert_outputs_list = []
    tokens_collected = 0

    moe_block = get_moe_block(model, layer_idx)
    router_attr = attrs.get("router", "gate")
    router = getattr(moe_block, router_attr)
    router_weight_attr = attrs.get("router_weight_attr")
    experts_attr = attrs.get("experts", "experts")
    experts = getattr(moe_block, experts_attr)
    num_experts = len(experts) if isinstance(experts, nn.ModuleList) else experts.num_experts
    fused = attrs.get("fused", False) or hasattr(experts, "gate_up_proj")

    def hook_fn(module, args, output):
        nonlocal tokens_collected
        if tokens_collected >= max_tokens:
            return

        flat_input = args[0].reshape(-1, args[0].shape[-1])  # [B*S, H]
        num_tokens = flat_input.shape[0]

        remaining = max_tokens - tokens_collected
        if num_tokens > remaining:
            idx = torch.randperm(num_tokens, device=flat_input.device)[:remaining]
            flat_input = flat_input[idx]
            num_tokens = remaining

        # Extract router logits
        if router_weight_attr and "." in router_weight_attr:
            parts = router_weight_attr.split(".")
            inner = router
            for part in parts[:-1]:
                inner = getattr(inner, part)
            weight = getattr(inner, parts[-1])
            logits = nn.functional.linear(flat_input.to(weight.dtype), weight)
        elif hasattr(router, "weight"):
            logits = nn.functional.linear(flat_input.to(router.weight.dtype), router.weight)
        else:
            logits = torch.zeros(num_tokens, num_experts, device=flat_input.device, dtype=flat_input.dtype)

        router_logits_list.append(logits.cpu())

        # Compute all expert outputs
        if fused:
            gate_up = experts.gate_up_proj
            down = experts.down_proj
            I = gate_up.shape[1] // 2
            outputs = []
            for e in range(num_experts):
                gu = gate_up[e]
                g, u = nn.functional.linear(flat_input, gu).chunk(2, dim=-1)
                h = nn.functional.silu(g) * u
                o = nn.functional.linear(h, down[e])
                outputs.append(o)
            expert_outputs_list.append(torch.stack(outputs, dim=0).cpu())  # [E, T, H]
        else:
            outputs = []
            for e in range(num_experts):
                outputs.append(experts[e](flat_input))
            expert_outputs_list.append(torch.stack(outputs, dim=0).cpu())

        tokens_collected += num_tokens

    handle = moe_block.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch in batches:
                if tokens_collected >= max_tokens:
                    break
                kwargs = {"input_ids": batch.input_ids.to(device)}
                if hasattr(batch, "attention_mask") and batch.attention_mask is not None:
                    kwargs["attention_mask"] = batch.attention_mask.to(device)
                model(**kwargs)
    finally:
        handle.remove()

    if not router_logits_list:
        return None

    router_logits = torch.cat(router_logits_list, dim=0)
    expert_outputs = torch.cat(expert_outputs_list, dim=1)

    return {
        "router_logits": router_logits,
        "expert_outputs": expert_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Sequential REAM compression (single-layer hooks)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-ratio", type=float, default=0.333)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--cpu-merge", action="store_true")
    parser.add_argument("--fast-merge", action="store_true")
    parser.add_argument("--max-memory-per-gpu", type=str, default="60GiB",
                        help="max memory per GPU for device_map (leaves headroom for the observer)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", args.model)
    max_memory = {i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    ensure_model_registered(model)

    attrs = get_model_attrs(model.__class__.__name__)
    if attrs is None:
        raise ValueError(f"Model {model.__class__.__name__} not registered")

    moe_layers = list_moe_layers(model)
    logger.info("Found %d MoE layers: %s", len(moe_layers), moe_layers[:5])

    batches = list(build_calibration_batches(
        tokenizer, "hardcoded",
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        samples=args.samples,
    ))
    logger.info("Calibration: %d batches of %d", len(batches), args.batch_size)

    merge_config = MergeConfig(
        target_ratio=args.target_ratio,
        group_size=args.group_size,
        use_cpu_for_weights=args.cpu_merge,
        skip_permutation=args.fast_merge,
        avg_router=False,
    )

    device = next(model.parameters()).device
    retained_counts = {}

    for layer_idx in tqdm(moe_layers, desc="Layers"):
        logger.info("Layer %d: collecting...", layer_idx)

        layer_stats = collect_single_layer(
            model, layer_idx, batches, attrs, device, max_tokens=args.max_tokens
        )

        if layer_stats is None:
            logger.warning("Layer %d: no data, skipping", layer_idx)
            continue

        logger.info("Layer %d: %d tokens — merging...",
                    layer_idx, layer_stats["router_logits"].shape[0])

        retained = merge_layer(model, layer_idx, layer_stats, merge_config)
        retained_counts[layer_idx] = retained
        torch.cuda.empty_cache()
        logger.info("Layer %d: %d experts retained", layer_idx, retained)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if retained_counts:
        unique = set(retained_counts.values())
        final_count = list(retained_counts.values())[0]
        if len(unique) == 1:
            for attr in ["num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"]:
                if hasattr(model.config, attr):
                    setattr(model.config, attr, final_count)
            logger.info("Updated config.num_experts = %d", final_count)
        else:
            logger.warning("Different expert counts per layer: %s", unique)

    logger.info("Saving model to %s...", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Done!")

    if retained_counts:
        avg = sum(retained_counts.values()) / len(retained_counts)
        logger.info("Average: %.1f experts/layer", avg)


if __name__ == "__main__":
    main()

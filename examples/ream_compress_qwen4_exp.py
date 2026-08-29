#!/usr/bin/env python3
"""
Compress Qwen4-Exp / Qwen3.8-Flash-Next with the REAM (merge) algorithm.

Run this on a multi-GPU worker (e.g. 6x H100 SXM 80GB). It loads the model
across all GPUs, calibrates on the built-in "hardcoded" dataset, then merges
experts so the configured fraction is removed. Defaults target the text MoE
experts of the big multimodal model.

Usage:
    python examples/ream_compress_qwen4_exp.py \
        --model Qwen/Qwen3.8-Flash-Next \
        --output ./compressed \
        --target-ratio 0.60 \
        --dataset hardcoded
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Make the repo importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from ream_moe import (
    MoEObserver,
    ObserverConfig,
    MergeConfig,
    merge_model,
    verify_model_config,
    print_verification_result,
)
from ream_moe.calibration import build_calibration_batches, list_available_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for name in ("huggingface_hub", "datasets", "urllib3", "filelock"):
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger("ream_run")


def parse_args():
    p = argparse.ArgumentParser(description="REAM merge a Qwen4-Exp model")
    p.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next", help="HF model id or path")
    p.add_argument("--output", default="./compressed", help="output dir")
    p.add_argument("--target-ratio", type=float, default=0.60,
                   help="fraction of experts to keep (0.60 merges ~40%% away)")
    p.add_argument("--dataset", default="hardcoded", choices=list_available_datasets())
    p.add_argument("--samples", type=int, default=512, help="calibration samples")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens-per-layer", type=int, default=4096,
                   help="max calibration tokens retained per layer (smaller = less RAM)")
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--upload-name", default=None,
                   help="HF repo id to upload the compressed model to (requires a valid write token)")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    logger.info("Loading model %s (dtype=%s)", args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    res = verify_model_config(args.model, model)
    print_verification_result(res)
    if args.verify_only:
        return
    if not res["valid"]:
        raise SystemExit("Model config invalid, aborting.")

    batches = list(build_calibration_batches(
        tokenizer, args.dataset,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        samples=args.samples,
    ))
    logger.info("Built %d calibration batches", len(batches))

    obs_cfg = ObserverConfig(
        max_tokens_per_layer=args.max_tokens_per_layer,
        store_on_cpu=True,
        device="cuda",
    )

    # Collect per-layer activation statistics across all calibration batches.
    observer = MoEObserver(model, obs_cfg)
    observer.hook_model()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for batch in batches:
                input_ids = getattr(batch, "input_ids", None)
                attn_mask = getattr(batch, "attention_mask", None)
                if input_ids is None and isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    attn_mask = batch.get("attention_mask")
                if input_ids is None:
                    continue
                kwargs = {"input_ids": input_ids.to(device)}
                if attn_mask is not None:
                    kwargs["attention_mask"] = attn_mask.to(device)
                model(**kwargs)
    finally:
        observer.unhook_model()
    observer_data = observer.get_collected_stats()
    logger.info("Observed %d MoE layers", len(observer_data))

    merge_cfg = MergeConfig(
        target_ratio=args.target_ratio,
        group_size=args.group_size,
        use_gated_similarity=True,
        use_cpu_for_weights=True,
    )
    retained = merge_model(model, observer_data, merge_cfg)
    logger.info("Merge summary (layer -> retained experts): %s", retained)

    model.config.save_pretrained(args.output)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    logger.info("Saved compressed model to %s", args.output)

    # Optional upload if a valid HF write token is available.
    if args.upload_name:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import whoami
                whoami(token=hf_token)
                model.push_to_hub(args.upload_name, token=hf_token)
                logger.info("Uploaded compressed model to %s", args.upload_name)
            except Exception as e:
                raise SystemExit(f"HF upload failed (invalid/missing write token): {e}")
        else:
            raise SystemExit("HF_TOKEN not set; cannot upload — set a valid write token.")
    else:
        logger.info("No --upload-name given; skipping HF upload.")


if __name__ == "__main__":
    main()

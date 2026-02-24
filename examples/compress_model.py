#!/usr/bin/env python3
"""
CLI script for compressing MoE models using REAM/REAP.

This script provides a command-line interface for:
1. Loading a MoE model
2. Collecting activation statistics on calibration data
3. Pruning or merging experts
4. Saving the compressed model

Usage:
    python compress_model.py \
        --model Qwen/Qwen3-14B-MoE \
        --output ./compressed_model \
        --compression-ratio 0.25 \
        --method prune \
        --dataset combined
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ream_moe import (
    MoEObserver,
    ObserverConfig,
    PruningConfig,
    MergeConfig,
    prune_model,
    merge_model,
    verify_model_config,
    print_verification_result,
    ensure_model_registered,
    list_supported_models,
)
from ream_moe.calibration import build_calibration_batches, list_available_datasets

# Configure logging - suppress INFO messages from all sources
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Suppress INFO logs from noisy libraries
for logger_name in ["", "ream_moe", "datasets", "transformers", "torch", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress MoE models using REAM/REAP expert pruning/merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prune 25% of experts from a Qwen3 MoE model
  python compress_model.py --model Qwen/Qwen3-14B-MoE --output ./qwen_compressed --compression-ratio 0.25

  # Merge experts to keep 75% (25% compression)
  python compress_model.py --model Qwen/Qwen3-14B-MoE --output ./qwen_merged --target-ratio 0.75 --method merge

  # Use specific calibration dataset
  python compress_model.py --model Qwen/Qwen3-14B-MoE --dataset code --samples 500

  # Verify model configuration only
  python compress_model.py --model Qwen/Qwen3-14B-MoE --verify-only
        """
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (HuggingFace format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for compressed model",
    )

    # Compression arguments
    parser.add_argument(
        "--method",
        type=str,
        choices=["prune", "merge"],
        default="prune",
        help="Compression method: 'prune' removes experts, 'merge' combines experts",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.25,
        help="Fraction of experts to remove (0.25 = remove 25%%, keep 75%%)",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=None,
        help="For merging: fraction of experts to KEEP (0.75 = keep 75%%, compress 25%%)",
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        default=None,
        help="Exact number of experts to prune (overrides compression-ratio)",
    )

    # Calibration arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="hardcoded",
        help=f"Calibration dataset: {', '.join(list_available_datasets())} (default: hardcoded - recommended to avoid OOM)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of samples to use for calibration (default: 2000 for hardcoded dataset)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for calibration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048 * 512,
        help="Maximum tokens to collect per layer",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for model loading and inference",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model loading",
    )

    # Observer options
    parser.add_argument(
        "--renormalize-router",
        action="store_true",
        help="Renormalize router weights after top-k",
    )
    parser.add_argument(
        "--poor",
        action="store_true",
        help="Store observer statistics on CPU/RAM instead of GPU to avoid OOM (slower but uses less GPU memory)",
    )
    parser.add_argument(
        "--cpu-merge",
        action="store_true",
        help="Process merge weight operations on CPU/RAM instead of GPU to avoid OOM during merging (slower but uses less GPU memory)",
    )
    parser.add_argument(
        "--use-sdpa",
        action="store_true",
        help="Use scaled_dot_product_attention for memory-efficient attention computation during calibration",
    )

    # Verification
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify model configuration, don't compress",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip model configuration verification",
    )

    # Other options
    parser.add_argument(
        "--preserve-super-experts",
        action="store_true",
        help="Preserve experts with unusually high activation (super experts)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load the model and tokenizer."""
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Device: {args.device}, dtype: {args.torch_dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if torch_dtype != "auto" else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure model is registered
    ensure_model_registered(model)

    logger.info(f"Loaded model: {model.__class__.__name__}")

    return model, tokenizer


def collect_observer_data(model, tokenizer, args):
    """Collect activation statistics using the observer."""
    logger.info("Collecting activation statistics...")

    if args.poor:
        logger.info("Using CPU/RAM for observer statistics to save GPU memory")

    if args.use_sdpa:
        logger.info("Using SDPA (scaled_dot_product_attention) for memory efficiency")

    observer_config = ObserverConfig(
        max_tokens_per_layer=args.max_tokens,
        renormalize_router_weights=args.renormalize_router,
        device=args.device,
        store_on_cpu=args.poor,
    )

    observer = MoEObserver(model, observer_config)
    observer.hook_model()

    # Enable SDPA if requested
    original_use_sdpa = None
    original_attn_impl = None

    if args.use_sdpa and hasattr(model.config, "use_sdpa"):
        original_use_sdpa = model.config.use_sdpa
        model.config.use_sdpa = True
        logger.info("Enabled SDPA in model config")
    elif args.use_sdpa and hasattr(model.config, "attn_implementation"):
        original_attn_impl = model.config.attn_implementation
        model.config.attn_implementation = "sdpa"
        logger.info(f"Set attn_implementation to 'sdpa' (was: '{original_attn_impl}')")

    try:
        # Build calibration batches
        logger.info(f"Using dataset: {args.dataset}, samples: {args.samples}")
        batches = build_calibration_batches(
            tokenizer,
            args.dataset,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            samples=args.samples,
        )

        # Run forward pass
        model.eval()
        total_batches = 0
        max_batches = args.samples // args.batch_size

        with torch.no_grad():
            for batch in batches:
                if total_batches >= max_batches:
                    break

                # Move to device
                input_ids = batch.input_ids.to(args.device)
                attention_mask = batch.attention_mask.to(args.device)

                # Forward pass with SDPA if enabled
                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                # Add use_cache=False for memory efficiency with SDPA
                if args.use_sdpa:
                    forward_kwargs["use_cache"] = False

                _ = model(**forward_kwargs)

                total_batches += 1

                if total_batches % 10 == 0:
                    logger.info(f"Processed {total_batches} batches...")

    finally:
        observer.unhook_model()

        # Restore original SDPA setting
        if args.use_sdpa and hasattr(model.config, "use_sdpa") and original_use_sdpa is not None:
            model.config.use_sdpa = original_use_sdpa
        elif args.use_sdpa and hasattr(model.config, "attn_implementation") and 'original_attn_impl' in locals():
            model.config.attn_implementation = original_attn_impl

    # Get collected statistics
    observer_data = observer.get_collected_stats()
    logger.info(f"Collected statistics for {len(observer_data)} layers")

    return observer_data


def compress_model(model, observer_data, args):
    """Compress the model based on collected statistics."""
    logger.info(f"Compressing model using method: {args.method}")

    if args.method == "prune":
        config = PruningConfig(
            compression_ratio=args.compression_ratio,
            n_experts_to_prune=args.n_experts,
            preserve_super_experts=args.preserve_super_experts,
        )
        retained_counts = prune_model(model, observer_data, config)

    else:  # merge
        target_ratio = args.target_ratio
        if target_ratio is None:
            target_ratio = 1.0 - args.compression_ratio

        config = MergeConfig(
            target_ratio=target_ratio,
            use_cpu_for_weights=args.cpu_merge,
        )
        retained_counts = merge_model(model, observer_data, config)

    return retained_counts


def save_model(model, tokenizer, output_dir, args, retained_counts=None):
    """Save the compressed model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to: {output_path}")

    # Ensure config is updated before saving
    if retained_counts:
        final_expert_count = list(retained_counts.values())[0] if retained_counts else None
        if final_expert_count:
            for attr_name in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
                if hasattr(model.config, attr_name):
                    old_val = getattr(model.config, attr_name)
                    if old_val != final_expert_count:
                        logger.info(f"Updating model.config.{attr_name}: {old_val} -> {final_expert_count}")
                        setattr(model.config, attr_name, final_expert_count)

    # Save model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save compression info
    info_path = output_path / "compression_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Compression ratio: {args.compression_ratio}\n")
        if args.target_ratio:
            f.write(f"Target ratio: {args.target_ratio}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {args.samples}\n")
        f.write(f"Max seq len: {args.max_seq_len}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Torch dtype: {args.torch_dtype}\n")

    logger.info(f"Model saved successfully to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Verify model configuration
    if not args.skip_verification:
        logger.info("Verifying model configuration...")
        verification = verify_model_config(args.model, model)
        print_verification_result(verification)

        if not verification["valid"]:
            logger.error("Model verification failed!")
            if not args.verify_only:
                logger.error("Use --skip-verification to proceed anyway.")
            sys.exit(1)

    if args.verify_only:
        logger.info("Verification complete. Exiting (--verify-only specified).")
        sys.exit(0)

    # Collect observer data
    observer_data = collect_observer_data(model, tokenizer, args)

    # Compress model
    retained_counts = compress_model(model, observer_data, args)

    # Save compressed model
    save_model(model, tokenizer, args.output, args, retained_counts)

    # Print summary
    logger.info("=" * 70)
    logger.info("Compression complete!")
    logger.info(f"Original model: {args.model}")
    logger.info(f"Compressed model: {args.output}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Compression ratio: {args.compression_ratio:.1%}")

    if retained_counts:
        avg_experts = sum(retained_counts.values()) / len(retained_counts)
        logger.info(f"Average experts per layer: {avg_experts:.1f}")

    logger.info("=" * 70)


def cli_entry():
    """Entry point for CLI script."""
    main()


if __name__ == "__main__":
    main()

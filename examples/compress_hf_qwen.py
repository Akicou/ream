from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ream_moe.ream import REAMCompressor, REAMConfig
from ream_moe.adapters.hf_qwen import HFQwenMoEAdapter
from ream_moe.calibration import build_calibration_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress a HuggingFace Qwen-like MoE model with REAM."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model id or local path (Qwen / gpt-oss / Kimi MoE, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the compressed model.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.75,
        help="Fraction of experts to keep in each MoE layer.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="Max number of experts in each merged group.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for calibration tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for calibration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run compression on (e.g., cuda or cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)

    # TODO: replace this with real calibration data:
    # ideally a mix of c4, math, and code as described in the REAM blog.
    dummy_texts: List[str] = [
        "The quick brown fox jumps over the lazy dog.",
        "Write a Python function to compute the factorial of a number.",
        "Solve the following equation: 2x + 5 = 17.",
        "Explain the difference between breadth-first search and depth-first search.",
    ]

    calib_batches = build_calibration_batches(
        tokenizer,
        dummy_texts,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    adapter = HFQwenMoEAdapter(model, device=args.device)
    cfg = REAMConfig(
        target_ratio=args.target_ratio,
        group_size=args.group_size,
        sequential_merging=True,
    )

    compressor = REAMCompressor(adapter, cfg)
    compressor.compress(calib_batches)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Compressed model saved to {args.output_dir}")


if __name__ == "__main__":
    main()


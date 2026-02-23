#!/usr/bin/env python3
"""
Diagnostic script to identify the exact structure of Qwen3 MoE experts.
Run this on Ubuntu to understand how to access the experts.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

print("=" * 70)
print("Qwen3 MoE Experts Structure Diagnostic")
print("=" * 70)

print(f"\nLoading model: {MODEL_NAME}")

# Load model with GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",  # Use GPU
    trust_remote_code=True,
)

print(f"Model class: {model.__class__.__name__}")
print(f"Config num_experts: {getattr(model.config, 'num_experts', 'N/A')}")
print(f"Config num_experts_per_tok: {getattr(model.config, 'num_experts_per_tok', 'N/A')}")

# Get first MoE layer
if hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
elif hasattr(model, "layers"):
    layers = model.layers
else:
    print("ERROR: Cannot find layers in model")
    sys.exit(1)

print(f"\nTotal layers: {len(layers)}")

# Find first MoE layer
for layer_idx in range(min(5, len(layers))):
    layer = layers[layer_idx]

    print(f"\n{'=' * 70}")
    print(f"Layer {layer_idx} structure:")
    print(f"{'=' * 70}")

    # Check for mlp attribute
    if hasattr(layer, "mlp"):
        moe_block = layer.mlp
        print(f"✓ Found mlp: {moe_block.__class__.__name__}")
    elif hasattr(layer, "block_sparse_moe"):
        moe_block = layer.block_sparse_moe
        print(f"✓ Found block_sparse_moe: {moe_block.__class__.__name__}")
    else:
        print(f"✗ No MoE block found in layer {layer_idx}")
        continue

    # List all attributes of MoE block
    print(f"\nMoE block attributes:")
    for attr in dir(moe_block):
        if not attr.startswith("_"):
            try:
                val = getattr(moe_block, attr)
                if not callable(val):
                    type_name = type(val).__name__
                    if isinstance(val, torch.Tensor):
                        print(f"  {attr}: {type_name} {tuple(val.shape)}")
                    elif hasattr(val, "__len__"):
                        print(f"  {attr}: {type_name} len={len(val)}")
                    else:
                        print(f"  {attr}: {type_name}")
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")

    # Focus on 'experts' attribute
    if hasattr(moe_block, "experts"):
        experts = moe_block.experts
        print(f"\n{'=' * 70}")
        print(f"EXPERTS OBJECT: {experts.__class__.__name__}")
        print(f"{'=' * 70}")

        # List all attributes of experts
        print(f"\nExperts attributes:")
        for attr in dir(experts):
            if not attr.startswith("_"):
                try:
                    val = getattr(experts, attr)
                    if not callable(val):
                        if isinstance(val, torch.Tensor):
                            print(f"  {attr}: Tensor {tuple(val.shape)}")
                        elif hasattr(val, "__len__"):
                            print(f"  {attr}: {type(val).__name__} len={len(val)}")
                        else:
                            print(f"  {attr}: {type(val).__name__}")
                except Exception as e:
                    pass  # Skip errors

        # Check specific tensor attributes
        print(f"\nExpert tensor details:")
        if hasattr(experts, "gate_up_proj"):
            gate_up = experts.gate_up_proj
            print(f"  gate_up_proj: Tensor {tuple(gate_up.shape)}")
            print(f"    - dtype: {gate_up.dtype}")
            print(f"    - device: {gate_up.device}")

        if hasattr(experts, "down_proj"):
            down = experts.down_proj
            print(f"  down_proj: Tensor {tuple(down.shape)}")
            print(f"    - dtype: {down.dtype}")
            print(f"    - device: {down.device}")

        # Check if experts is subscriptable
        print(f"\nIs experts subscriptable?")
        try:
            test = experts[0]
            print(f"  ✓ experts[0] works: {type(test).__name__}")
        except (TypeError, KeyError) as e:
            print(f"  ✗ experts[0] failed: {e}")

        # Check if we can use len()
        try:
            n = len(experts)
            print(f"  ✓ len(experts) = {n}")
        except TypeError as e:
            print(f"  ✗ len(experts) failed: {e}")

    # Check router/gate
    print(f"\n{'=' * 70}")
    print(f"ROUTER/GATE:")
    print(f"{'=' * 70}")

    if hasattr(moe_block, "gate"):
        gate = moe_block.gate
        print(f"  gate: {gate.__class__.__name__}")
        if hasattr(gate, "weight"):
            print(f"    weight shape: {tuple(gate.weight.shape)}")

    if hasattr(moe_block, "router"):
        router = moe_block.router
        print(f"  router: {router.__class__.__name__}")

    # Only check first MoE layer
    break

print(f"\n{'=' * 70}")
print("Diagnostic complete!")
print(f"{'=' * 70}")

#!/usr/bin/env python3
"""
Diagnostic script to identify the exact structure of Kimi-VL model.
Run this to understand how to access the MoE experts in the language model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer

MODEL_NAME = "moonshotai/Kimi-VL-A3B-Thinking-2506"

print("=" * 70)
print("Kimi-VL Model Structure Diagnostic")
print("=" * 70)

print(f"\nLoading model: {MODEL_NAME}")

# Load model with GPU
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",  # Use GPU
    trust_remote_code=True,
)

print(f"Model class: {model.__class__.__name__}")

# Print model config
print(f"\n{'=' * 70}")
print("Model Config:")
print(f"{'=' * 70}")
print(f"model_type: {getattr(model.config, 'model_type', 'N/A')}")
print(f"num_experts: {getattr(model.config.text_config, 'n_routed_experts', 'N/A')}")
print(f"num_experts_per_tok: {getattr(model.config.text_config, 'num_experts_per_tok', 'N/A')}")
print(f"n_shared_experts: {getattr(model.config.text_config, 'n_shared_experts', 'N/A')}")
print(f"num_hidden_layers: {getattr(model.config.text_config, 'num_hidden_layers', 'N/A')}")

# Check top-level attributes
print(f"\n{'=' * 70}")
print("Top-level Model Attributes:")
print(f"{'=' * 70}")
for attr in dir(model):
    if not attr.startswith("_") and not callable(getattr(model, attr)):
        try:
            val = getattr(model, attr)
            type_name = type(val).__name__
            print(f"  {attr}: {type_name}")
        except Exception:
            pass

# Navigate to language model
print(f"\n{'=' * 70}")
print("Language Model Structure:")
print(f"{'=' * 70}")

if hasattr(model, "language_model"):
    lang_model = model.language_model
    print(f"✓ Found language_model: {lang_model.__class__.__name__}")
elif hasattr(model, "model"):
    lang_model = model.model
    print(f"✓ Found model: {lang_model.__class__.__name__}")
else:
    print("ERROR: Cannot find language model")
    sys.exit(1)

# Check language model structure
print(f"\nLanguage model attributes:")
for attr in dir(lang_model):
    if not attr.startswith("_") and not callable(getattr(lang_model, attr)):
        try:
            val = getattr(lang_model, attr)
            type_name = type(val).__name__
            if isinstance(val, torch.nn.Module):
                print(f"  {attr}: {type_name}")
        except Exception:
            pass

# Get layers from language model
if hasattr(lang_model, "model") and hasattr(lang_model.model, "layers"):
    layers = lang_model.model.layers
    print(f"✓ Found layers at language_model.model.layers")
elif hasattr(lang_model, "layers"):
    layers = lang_model.layers
    print(f"✓ Found layers at language_model.layers")
else:
    print("ERROR: Cannot find layers in language model")
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
        if hasattr(experts, "gate_proj"):
            gate = experts.gate_proj
            print(f"  gate_proj: {gate.__class__.__name__")

        if hasattr(experts, "up_proj"):
            up = experts.up_proj
            print(f"  up_proj: {up.__class__.__name__}")

        if hasattr(experts, "down_proj"):
            down = experts.down_proj
            print(f"  down_proj: {down.__class__.__name__}")

        # Check if experts is subscriptable
        print(f"\nIs experts subscriptable?")
        try:
            test = experts[0]
            print(f"  ✓ experts[0] works: {type(test).__name__}")
            # Check individual expert structure
            print(f"\nIndividual expert (experts[0]) attributes:")
            for attr in dir(test):
                if not attr.startswith("_") and not callable(getattr(test, attr)):
                    try:
                        val = getattr(test, attr)
                        if isinstance(val, torch.Tensor):
                            print(f"    {attr}: Tensor {tuple(val.shape)}")
                    except Exception:
                        pass
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

# Check vision model structure
print(f"\n{'=' * 70}")
print("Vision Model Structure:")
print(f"{'=' * 70}")

if hasattr(model, "vision_tower"):
    vision_model = model.vision_tower
    print(f"✓ Found vision_tower: {vision_model.__class__.__name__}")

    # Check if vision model has MoE
    print(f"\nVision model attributes:")
    for attr in ["encoder", "blocks"]:
        if hasattr(vision_model, attr):
            val = getattr(vision_model, attr)
            print(f"  {attr}: {type(val).__name__}")

# Check multimodal projector
print(f"\n{'=' * 70}")
print("Multi-modal Projector Structure:")
print(f"{'=' * 70}")

if hasattr(model, "multi_modal_projector"):
    projector = model.multi_modal_projector
    print(f"✓ Found multi_modal_projector: {projector.__class__.__name__}")

    print(f"\nProjector attributes:")
    for attr in dir(projector):
        if not attr.startswith("_") and not callable(getattr(projector, attr)):
            try:
                val = getattr(projector, attr)
                if isinstance(val, torch.Tensor):
                    print(f"  {attr}: Tensor {tuple(val.shape)}")
                elif isinstance(val, torch.nn.Module):
                    print(f"  {attr}: {type(val).__name__}")
            except Exception:
                pass

print(f"\n{'=' * 70}")
print("Diagnostic complete!")
print(f"{'=' * 70}")

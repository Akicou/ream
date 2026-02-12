# REAM-MoE

REAM-style Mixture-of-Experts (MoE) compression framework with pluggable adapters.

This repo implements a generic version of the **REAM** algorithm described in\
“REAM: Compressing Mixture-of-Experts LLMs” and makes it easy to apply to\
different MoE model families such as Qwen, gpt-oss, Kimi Linear, etc.

## Features

- **Adapter-based design**: a small `MoEAdapter` interface hides model-specific details.
- **REAM core implementation**:
  - REAP-style saliency computation.
  - Gated similarity + pseudo-pruning grouping.
  - Permutation-aware expert merging (Hungarian alignment).
  - Router (gate) weight adjustment.
- **Sequential or non-sequential merging** across MoE layers.
- Example HuggingFace-style adapter and compression script.

## Installation

```bash
pip install -e .
```

or using the `requirements.txt`:

```bash
pip install -r requirements.txt
```

> On Windows, ensure you are in the correct virtual environment before installing.

## Package layout

- `ream_moe/`
  - `__init__.py`
  - `adapters/`
    - `__init__.py`
    - `base.py` – core adapter abstractions.
    - `hf_qwen.py` – example HuggingFace MoE adapter (skeleton).
  - `ream.py` – REAM compressor implementation.
  - `calibration.py` – small helpers for building calibration batches.
- `examples/`
  - `compress_hf_qwen.py` – example CLI for compressing a Qwen-like HF model.

## Quick start (HuggingFace-style model)

This is a sketch for Qwen-like models. You will need to:

- Point `HFQwenMoEAdapter` to the actual MoE layer classes used by your model.
- Implement `forward_collect_calibration` for your specific architecture.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ream_moe.adapters.hf_qwen import HFQwenMoEAdapter
from ream_moe.ream import REAMCompressor, REAMConfig
from ream_moe.calibration import build_calibration_batches

model_name = "Qwen/Qwen2.5-32B-A14B-Instruct"  # example
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="torch.float16").cuda()

adapter = HFQwenMoEAdapter(model, device="cuda")
cfg = REAMConfig(target_ratio=0.75, group_size=16, sequential_merging=True)
ream = REAMCompressor(adapter, cfg)

calib_texts = [...]  # your C4 / math / code mix
calib_batches = build_calibration_batches(tokenizer, calib_texts, max_seq_len=512, batch_size=4)

ream.compress(calib_batches)
model.save_pretrained("compressed-model")
tokenizer.save_pretrained("compressed-model")
```

## Extending to new model families

To support a new MoE architecture (e.g., gpt-oss, Kimi Linear):

1. Create a new adapter in `ream_moe/adapters/` subclassing `MoEAdapter`.
2. Implement:
   - `moe_layers`, `experts_in_layer`, `top_k`.
   - `forward_collect_calibration` (hooks or custom MoE wrapper).
   - `get_expert_weights`, `set_expert_weights`.
   - `get_router_weights`, `set_router_weights`, `router_expert_axis`.
   - `rebuild_caches` (if needed).
3. Use the same `REAMCompressor` without changing the core algorithm.

See `ream_moe/adapters/base.py` and `ream_moe/adapters/hf_qwen.py` for guidance.


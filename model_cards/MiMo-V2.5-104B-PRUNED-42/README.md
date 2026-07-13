---
license: mit
language:
- en
- zh
base_model: XiaomiMiMo/MiMo-V2.5
tags:
- mimo-v2
- mixture-of-experts
- moe
- fp8
- offline-pruned
- seed-42
- text-generation
- multimodal
library_name: transformers
pipeline_tag: text-generation
---

# MiMo-V2.5-104B-PRUNED-42

This is an **offline seed-pruned** version of [`XiaomiMiMo/MiMo-V2.5`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5), produced with [`Akicou/ream`](https://github.com/Akicou/ream).

It is **not an official Xiaomi release**.

## What was changed

The base MiMo-V2.5 checkpoint was pruned directly at the safetensors level without loading the full Transformers model into memory and without GPU calibration.

Pruning details:

| Item | Value |
|---|---:|
| Base model | `XiaomiMiMo/MiMo-V2.5` |
| Method | Offline random seed expert pruning |
| Seed | `42` |
| Routed experts per MoE layer | `256 -> 81` |
| Experts pruned per MoE layer | `175` |
| MoE layers processed | `47` |
| Dense layer | Layer `0` left unchanged |
| Output safetensors payload | ~`107.86 GB` |

Only routed MoE experts and their router tensors were pruned/remapped. Non-MoE weights, tokenizer files, multimodal/audio files, and model code were copied from the base model.

## How it was created

```bash
python examples/compress_model.py \
  --model XiaomiMiMo/MiMo-V2.5 \
  --output ./MiMo-V2.5-104B-pruned \
  --offline-seed-prune \
  --n-experts 175 \
  --seed 42
```

`--n-experts 175` means **175 experts were removed** from each MoE layer, leaving `81 / 256` routed experts.

## Important notes

- This is **random seed pruning**, not calibrated saliency pruning.
- No benchmark evaluation is claimed here.
- Quality may be significantly worse than the original model.
- The model may require a recent `torch`/`transformers` stack because the original MiMo-V2.5 code uses FP8/custom MoE integrations.

## Basic usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Akicou/MiMo-V2.5-104B-PRUNED-42"

tokenizer = AutoTokenizer.from_pretrained(
    "XiaomiMiMo/MiMo-V2.5",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

prompt = "What is a reaper?"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Attribution

All architecture, tokenizer, and original weights are from [`XiaomiMiMo/MiMo-V2.5`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5). This repository only contains an offline-pruned derivative checkpoint.

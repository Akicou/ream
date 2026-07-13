---
language:
- en
- zh
base_model:
- tencent/Hy3
library_name: transformers
tags:
- hy3
- tencent
- ream
- reap
- merge
- merging
- compression
- compressed
- mixture-of-experts
- moe
- ream-moe
pipeline_tag: text-generation
---

# Akicou/Hy3-REAM-100B

This repository contains a **REAM-compressed** version of [Tencent Hy3](https://huggingface.co/tencent/Hy3), produced with [`Akicou/ream`](https://github.com/Akicou/ream) — a REAM/REAP-style Mixture-of-Experts compression framework.

It is **not an official Tencent release**.

## 🧠 Methodology: REAM

**REAM (Router Expert Activation Merging)** — originally proposed by [SamsungSAILMontreal/ream](https://github.com/SamsungSAILMontreal/ream) (Jha et al., 2026) — is a compression technique for MoE models that combines expert merging with pseudo-pruning:

1. **Calibration** — Run a small set of hardcoded prompts through the model to collect router logits and expert output activations.
2. **REAP Saliency** — Compute per-expert importance as `S[i] = mean(||h_i(x)|| × p_i(x))` over tokens routed to expert `i`.
3. **Pseudo-Grouping** — Select the top-K most salient experts as centroids, then assign nearby experts using gated similarity (50% hidden state + 50% router logit distribution).
4. **Merge** — Within each group, align neurons via Hungarian permutation and merge with saliency-weighted averaging.
5. **Prune** — Non-centroid experts are removed. The router is shrunk to match the new expert count, keeping only centroid rows.

This implementation follows the original REAM paper closely, including:
- Centroid-only router update (REAP-style)
- Zero saliency edge-case handling (unused experts get a small non-zero value)
- Single-layer sequential hooks to avoid OOM on large models

## 📊 Compression Details

| Metric | Original (Hy3) | REAM-100B | Change |
|--------|---------------|-----------|--------|
| **Total Parameters** | ~300B | **~100B** | -67% |
| **Routed Experts/Layer** | 192 | **64** | -128 |
| **MoE Layers** | 79 | 79 | — |
| **Calibration** | 100 samples, hardcoded prompts | — | — |
| **Merge Method** | Saliency-weighted avg + Hungarian | — | — |
| **Grouping** | REAM pseudo-group (group_size=16) | — | — |

## 🛠 How It Was Created

```bash
python examples/compress_sequential.py \
    --model tencent/Hy3 \
    --output ./hy3-ream-100B \
    --target-ratio 0.333 \
    --samples 100 \
    --max-seq-len 512 \
    --batch-size 4 \
    --max-tokens 2048 \
    --cpu-merge \
    --fast-merge \
    --seed 42
```

**Hardware:** 6× NVIDIA H200 (141GB each), 192 vCPUs, 3TB RAM

## ⚠️ Important Notes

- This is a **calibrated merge** (100 hardcoded prompts), not a full evaluation-grade calibration.
- No benchmark evaluation is claimed here — this is an experimental release.
- The model uses `trust_remote_code=True` (same as the base Hy3).
- Shared experts and dense layers are left untouched.
- The `e_score_correction_bias` per layer is correctly shrunk alongside router weights.

## 📦 Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Akicou/Hy3-REAM-100B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = "Explain the concept of reinforcement learning."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📜 Citation

```bibtex
@article{jha2026ream,
  title={REAM: Merging Improves Pruning of Experts in LLMs},
  author={Jha, Saurav and Hashemzadeh, Maryam and Pasand, Ali Saheb and Parviz, Ali and Lee, Min-Joong and Knyazev, Boris},
  journal={arXiv preprint arXiv:2604.04356},
  year={2026}
}

@misc{hy3,
  title={Hy3},
  author={Tencent},
  year={2025},
  url={https://huggingface.co/tencent/Hy3}
}
```

## Attribution

All architecture, tokenizer, and original weights are from [tencent/Hy3](https://huggingface.co/tencent/Hy3). This repository contains a REAM-compressed derivative checkpoint.
# REAM-MoE

> REAM/REAP-style Mixture-of-Experts compression framework with production-ready support for multiple model families.

**REAM-MoE** is a Python library for compressing Mixture-of-Experts (MoE) Large Language Models using the REAM (REAM-style expert merging) algorithm. It provides a generic, model-agnostic compression framework with adapter-based architecture for supporting multiple MoE model families.

> **Note:** Model configurations may not be 100% correct for all model families. If you encounter issues with a specific model, please verify the configuration and consider opening an issue or contributing a fix.

## Releases
 - [Akicou/Qwen3-30B-A3B-Instruct-REAMINI](https://huggingface.co/Akicou/Qwen3-30B-A3B-Instruct-REAMINI)
## Features

- **Adapter-based design** - Small `MoEAdapter` interface hides model-specific details
- **REAM/REAP core implementation**:
  - REAP-style saliency computation (router-weighted expert activation)
  - Gated similarity + pseudo-pruning grouping
  - Permutation-aware expert merging (Hungarian alignment)
  - Router (gate) weight adjustment
- **Production-ready model support** for 15+ MoE model families:
  - Qwen (Qwen3Moe, NonUniformQwen3Moe)
  - Llama4 (Llama4ForCausalLM)
  - Mixtral (MixtralForCausalLM)
  - DeepSeek (DeepseekV2ForCausalLM, DeepseekV3ForCausalLM)
  - Kimi (KimiK2ForCausalLM)
  - GLM (Glm4MoeForCausalLM, Glm4MoeLiteForCausalLM, GlmMoeDsaForCausalLM)
  - Ernie (Ernie4_5_MoEForCausalLM, Ernie4_5_MoeForCausalLM)
  - Solar (SolarOpenForCausalLM)
  - Vaetki (VaetkiForCausalLM)
  - MiMo (MiMoV2FlashForCausalLM)
  - LongCat (LongcatCausalLM, LongcatForCausalLM)
  - MiniMax (MiniMaxM2ForCausalLM)
- **Multiple compression methods**:
  - Expert pruning (remove low-saliency experts)
  - Expert merging (combine similar experts)
- **Built-in calibration datasets** (C4, code, math, writing)
- **Auto-registration** for unknown model architectures

## Installation

```bash
pip install -e .
```

Or using the `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Quick Start

For an interactive tutorial, see the [Quickstart Notebook](examples/quickstart.ipynb).

### Using the CLI

The easiest way to compress a model is using the provided CLI script:

```bash
python examples/compress_model.py \
    --model Qwen/Qwen3-14B-MoE \
    --output ./compressed_model \
    --compression-ratio 0.25 \
    --method prune \
    --dataset combined
```

### Using the Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ream_moe import observe_model, prune_model, PruningConfig
from ream_moe.calibration import build_calibration_batches

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B-MoE",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-MoE", trust_remote_code=True)

# Prepare calibration data
# Use built-in datasets: "c4", "code", "math", "writing", "hardcoded", "combined"
batches = list(build_calibration_batches(
    tokenizer,
    "hardcoded",  # Recommended: diverse hardcoded prompts
    max_seq_len=512,
    batch_size=4,
    samples=1000,
))

# Collect activation statistics on calibration data
observer_data = observe_model(
    model,
    batches[0].input_ids,
    batches[0].attention_mask,
)

# Prune 25% of experts
config = PruningConfig(compression_ratio=0.25)
retained_counts = prune_model(model, observer_data, config)

# Save compressed model
model.save_pretrained("./compressed_model")
tokenizer.save_pretrained("./compressed_model")
```

### Using Expert Merging

```python
from ream_moe import merge_model, MergeConfig
# First, load model and collect observer_data as shown above

# Merge experts to keep 75% (25% compression)
config = MergeConfig(target_ratio=0.75)
retained_counts = merge_model(model, observer_data, config)
```

## Supported Models

| Model Family | Model Class | Fused Experts | Notes |
|-------------|-------------|---------------|-------|
| Qwen3 MoE | `Qwen3MoeForCausalLM` | No | Standard Qwen MoE |
| Qwen3 NonUniform | `NonUniformQwen3MoeForCausalLM` | No | Non-uniform expert allocation |
| Llama4 | `Llama4ForCausalLM` | Yes | Fused gate_up_proj |
| Mixtral | `MixtralForCausalLM` | No | Uses w1/w2/w3 naming |
| DeepSeek V2 | `DeepseekV2ForCausalLM` | No | 160 experts, top_k=6 |
| DeepSeek V3 | `DeepseekV3ForCausalLM` | No | 256 experts, MLA attention |
| Kimi K2 | `KimiK2ForCausalLM` | No | DeepSeek V3 based |
| GLM-4 | `Glm4MoeForCausalLM` | No | 64 routed experts |
| GLM-4.7 Flash | `Glm4MoeLiteForCausalLM` | Yes | Layer 0 dense, 1-46 MoE |
| GLM-5 | `GlmMoeDsaForCausalLM` | No | Hybrid routed + shared |
| Ernie 4.5 | `Ernie4_5_MoeForCausalLM` | No | Baidu MoE architecture |
| Solar | `SolarOpenForCausalLM` | No | |
| MiMo V2 | `MiMoV2FlashForCausalLM` | No | 309B parameter model |
| LongCat | `LongcatCausalLM` | No | 512 real + 256 zero experts |
| MiniMax M2.5 | `MiniMaxM2ForCausalLM` | No | Uses w1/w2/w3 naming |

## Calibration Datasets

The following built-in calibration datasets are available:

| Dataset | Description | Use For |
|---------|-------------|---------|
| `c4` | General web text (C4 corpus) | General-purpose compression |
| `code` | Code instruction dataset | Code-focused models |
| `math` | Math instruction dataset | Math/reasoning models |
| `writing` | Creative writing prompts | Creative writing models |
| `combined` | Mix of all categories | Balanced compression |

## CLI Options

```
usage: compress_model.py [-h] --model MODEL --output OUTPUT
                          [--method {prune,merge}]
                          [--compression-ratio COMPRESSION_RATIO]
                          [--target-ratio TARGET_RATIO]
                          [--n-experts N_EXPERTS]
                          [--dataset DATASET] [--samples SAMPLES]
                          [--max-seq-len MAX_SEQ_LEN]
                          [--batch-size BATCH_SIZE]
                          [--max-tokens MAX_TOKENS]
                          [--device DEVICE]
                          [--torch-dtype {auto,float32,float16,bfloat16}]
                          [--renormalize-router] [--verify-only]
                          [--skip-verification]
                          [--preserve-super-experts]
                          [--seed SEED]

options:
  -h, --help            show this help message and exit
  --model MODEL         Model name or path (HuggingFace format)
  --output OUTPUT       Output directory for compressed model
  --method {prune,merge}  Compression method
  --compression-ratio COMPRESSION_RATIO
                        Fraction of experts to remove (default: 0.25)
  --target-ratio TARGET_RATIO
                        For merging: fraction of experts to KEEP
  --n-experts N_EXPERTS  Exact number of experts to prune
  --dataset DATASET     Calibration dataset
  --samples SAMPLES      Number of samples for calibration (default: 1000)
  --max-seq-len MAX_SEQ_LEN
                        Maximum sequence length (default: 512)
  --batch-size BATCH_SIZE
                        Batch size for calibration (default: 4)
  --max-tokens MAX_TOKENS
                        Maximum tokens per layer (default: 1048576)
  --device DEVICE       Device to use (default: cuda if available)
  --torch-dtype {auto,float32,float16,bfloat16}
                        Torch dtype (default: auto)
  --renormalize-router  Renormalize router weights after top-k
  --verify-only         Only verify model config
  --skip-verification   Skip model verification
  --preserve-super-experts
                        Preserve high-activation experts
  --seed SEED           Random seed (default: 42)
```

## Advanced Usage

### Verifying Model Configuration

Before compression, verify that your model is properly supported:

```python
from ream_moe import verify_model_config, print_verification_result

result = verify_model_config("Qwen/Qwen3-14B-MoE")
print_verification_result(result)
```

### Listing Supported Models

```python
from ream_moe import list_supported_models

for model_class in list_supported_models():
    print(model_class)
```

### Custom Calibration Data

```python
from ream_moe.calibration import build_calibration_batches, list_available_datasets

# See available datasets
print(list_available_datasets())  # ['c4', 'code', 'math', 'writing', 'hardcoded', 'combined']

# Use your own texts
my_texts = ["Your text here...", "More text..."]
batches = build_calibration_batches(
    tokenizer,
    my_texts,
    max_seq_len=512,
    batch_size=4,
)

# Or use a built-in dataset (all use hardcoded prompts to avoid OOM)
batches = build_calibration_batches(tokenizer, "hardcoded", samples=1000)

# Individual categories
batches = build_calibration_batches(tokenizer, "code")      # Programming tasks
batches = build_calibration_batches(tokenizer, "math")      # Math problems
batches = build_calibration_batches(tokenizer, "writing")   # Creative prompts
batches = build_calibration_batches(tokenizer, "c4")        # General knowledge

# Combined dataset (mix of all categories)
batches = build_calibration_batches(tokenizer, "combined", samples=2000)
```

**Note:** All built-in datasets use comprehensive hardcoded instruction prompts to avoid OOM issues from HuggingFace dataset downloads. These prompts cover diverse domains:
- **c4**: General knowledge, ML/AI, science, history, business
- **code**: Python, web dev, data science, algorithms, DevOps, security
- **math**: Algebra, calculus, geometry, statistics, probability, number theory
- **writing**: Story prompts, descriptive writing, poetry, dialogue, reflections
- **hardcoded**: Large combined set with all categories (recommended for best calibration)
- **combined**: Smaller mix of all categories

### Preserving Super Experts

To prevent pruning of unusually high-activation experts:

```python
config = PruningConfig(
    compression_ratio=0.25,
    preserve_super_experts=True,
)
```

## Model Configuration

Each model family requires specific configuration stored in `MODEL_ATTRS`:

```python
MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",              # MoE block attribute in decoder layers
        "gate_proj": "gate_proj",        # Gate projection name
        "up_proj": "up_proj",            # Up projection name
        "down_proj": "down_proj",        # Down projection name
        "experts": "experts",            # Experts container
        "fused": False,                  # Whether experts use fused projections
        "router": "gate",                # Router/gate attribute
        "num_experts": "num_experts",    # Config attribute for expert count
        "num_experts_per_tok": "num_experts_per_tok",  # Config for top-k
    },
    # ... more models
}
```

## Auto-Registration

For models not explicitly supported, REAM-MoE can attempt to auto-detect configuration:

```python
from ream_moe import ensure_model_registered

model = AutoModelForCausalLM.from_pretrained("unknown-moe-model")
success = ensure_model_registered(model)

if success:
    print("Model auto-registered successfully!")
else:
    print("Auto-registration failed, please add to MODEL_ATTRS manually")
```

## Algorithm Details

### REAP Saliency

Each expert's importance is computed as:

```
S[i] = mean_{tokens routed to i} ||h_i(x)|| * p_i(x)
```

Where:
- `h_i(x)` = expert output hidden states
- `p_i(x)` = routing probabilities
- `||·||` = L2 norm

### Expert Grouping

1. Select centroid experts (highest saliency)
2. For each centroid, group nearby experts using gated similarity
3. Most low-saliency experts remain singletons (pseudo-pruning)

### Merging

1. For each group, align expert weights using Hungarian algorithm
2. Merge with saliency-weighted averaging
3. Update router to only output centroids

## Project Structure

```
ream-moe/
├── ream_moe/
│   ├── __init__.py              # Public API exports
│   ├── ream.py                  # Core REAM compressor
│   ├── calibration.py           # Dataset registry and calibration
│   ├── observer.py              # Activation collection
│   ├── prune.py                 # Expert pruning
│   ├── merge.py                 # Expert merging
│   ├── model_attr_configs.py    # MODEL_ATTRS registry
│   ├── observer_configs.py      # Observer config registry
│   └── model_utils.py           # Helper functions
├── examples/
│   └── compress_model.py        # CLI script
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
└── README.md
```

## Contributing

To add support for a new model family:

1. Add model configuration to `model_attr_configs.py`:

```python
MODEL_ATTRS["YourMoEModelForCausalLM"] = {
    "moe_block": "mlp",           # or "block_sparse_moe", etc.
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
    "experts": "experts",
    "fused": False,
    "router": "gate",
    "num_experts": "num_experts",
    "num_experts_per_tok": "num_experts_per_tok",
}
```

2. Add observer config to `observer_configs.py`:

```python
OBSERVER_CONFIG_REGISTRY["YourMoEModelForCausalLM"] = type(
    "YourMoEObserverConfig",
    (ObserverHookConfig,),
    {"module_class_name_to_hook_regex": "YourMoEBlock"},
)
```

3. Test with `--verify-only` first!

## License

MIT

## References

- Blog post: [Understanding MoE Compression](https://bknyaz.github.io/blog/2026/moe/) - Detailed explanation of the REAM/REAP algorithm and theory
- Based on insights from [Cerebras Research/reap](https://github.com/CerebrasResearch/reap)

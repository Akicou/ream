

# REAM-MoE

> Marco de trabajo de compresión Mixture-of-Experts estilo REAM/REAP con soporte listo para producción para múltiples familias de modelos.

**REAM-MoE** es una biblioteca de Python para comprimir Modelos de Lenguaje Grande Mixture-of-Experts (MoE) utilizando el algoritmo REAM (fusión de expertos estilo REAM). Proporciona un marco de compresión genérico y agnóstico al modelo con una arquitectura basada en adaptadores para soportar múltiples familias de modelos MoE.

> **Nota:** Las configuraciones de los modelos pueden no ser 100 % correctas para todas las familias de modelos. Si encuentra problemas con un modelo específico, verifique la configuración y considere abrir un problema o contribuir con una corrección.

## Lanzamientos
 - [Akicou/Qwen3-30B-A3B-Instruct-REAMINI](https://huggingface.co/Akicou/Qwen3-30B-A3B-Instruct-REAMINI)
 - [Akicou/Hy3-REAM-100B](https://huggingface.co/Akicou/Hy3-REAM-100B) — Tencent Hy3 300B→100B
## Características

- **Diseño basado en adaptadores** - La pequeña interfaz `MoEAdapter` oculta los detalles específicos del modelo
- **Implementación central de REAM/REAP**:
  - Cálculo de saliencia estilo REAP (activación de expertos ponderada por el enrutador)
  - Agrupación por similitud con compuerta + pseudopoda
  - Fusión de expertos consciente de permutación (alineación húngara)
  - Ajuste de pesos del enrutador (compuerta)
- **Soporte de modelos listo para producción** para más de 17 familias de modelos MoE:
  - Qwen (Qwen3Moe, NonUniformQwen3Moe, Qwen3.5Moe)
  - Llama4 (Llama4ForCausalLM)
  - Mixtral (MixtralForCausalLM)
  - DeepSeek (DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV4ForCausalLM)
  - Kimi (KimiK2ForCausalLM)
  - GLM (Glm4MoeForCausalLM, Glm4MoeLiteForCausalLM, GlmMoeDsaForCausalLM)
  - Ernie (Ernie4_5_MoEForCausalLM, Ernie4_5_MoeForCausalLM)
  - Solar (SolarOpenForCausalLM)
  - Vaetki (VaetkiForCausalLM)
  - MiMo (MiMoV2ForCausalLM, MiMoV2FlashForCausalLM)
  - LongCat (LongcatCausalLM, LongcatForCausalLM)
  - MiniMax (MiniMaxM2ForCausalLM, MiniMaxM3Sparse)
  - DiffusionGemma (DiffusionGemmaForBlockDiffusion)
  - Tencent Hy3 (HYV3ForCausalLM)
- **Múltiples métodos de compresión**:
  - Poda de expertos (eliminar expertos de baja saliencia)
  - Fusión de expertos (combinar expertos similares)
- **Conjuntos de datos de calibración integrados** (C4, código, matemáticas, escritura)
- **Registro automático** para arquitecturas de modelos desconocidas

## Instalación

```bash
pip install -e .
```

O utilizando el `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Inicio Rápido

Para un tutorial interactivo, consulte el [Cuaderno de Inicio Rápido](examples/quickstart.ipynb).

### Uso de la CLI

La forma más fácil de comprimir un modelo es utilizar el script de CLI proporcionado:

```bash
python examples/compress_model.py \
    --model Qwen/Qwen3-14B-MoE \
    --output ./compressed_model \
    --compression-ratio 0.25 \
    --method prune \
    --dataset combined
```

### Uso de la API de Python

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

### Uso de la Fusión de Expertos

```python
from ream_moe import merge_model, MergeConfig
# First, load model and collect observer_data as shown above

# Merge experts to keep 75% (25% compression)
config = MergeConfig(target_ratio=0.75)
retained_counts = merge_model(model, observer_data, config)
```

## Modelos Soportados

| Familia de Modelo | Clase de Modelo | Expertos Fusionados | Notas |
|-------------|-------------|---------------|-------|
| Qwen3 MoE | `Qwen3MoeForCausalLM` | No | Qwen MoE estándar, proyecciones separadas |
| Qwen3 NonUniform | `NonUniformQwen3MoeForCausalLM` | No | Asignación de expertos no uniforme |
| Qwen3.5 MoE | `Qwen3_5MoeForConditionalGeneration` | Sí | 512 expertos, top_k=10, multimodal, expertos compartidos |
| Llama4 | `Llama4ForCausalLM` | Sí | gate_up_proj fusionado |
| Mixtral | `MixtralForCausalLM` | No | Utiliza nomenclatura w1/w2/w3 |
| DeepSeek V2 | `DeepseekV2ForCausalLM` | No | 160 expertos, top_k=6 |
| DeepSeek V3 | `DeepseekV3ForCausalLM` | No | 256 expertos, atención MLA |
| DeepSeek V4 Flash/Base | `DeepseekV4ForCausalLM` | Automático | 256 expertos, top_k=6, primeras capas enrutadas por hash |
| Kimi K2 | `KimiK2ForCausalLM` | No | Basado en DeepSeek V3 |
| GLM-4 | `Glm4MoeForCausalLM` | No | 64 expertos enrutados |
| GLM-4.7 Flash | `Glm4MoeLiteForCausalLM` | Sí | Capa 0 densa, 1-46 MoE |
| GLM-5 | `GlmMoeDsaForCausalLM` | No | Enrutado híbrido + compartido |
| Ernie 4.5 | `Ernie4_5_MoeForCausalLM` | No | Arquitectura MoE de Baidu |
| Solar | `SolarOpenForCausalLM` | No | |
| MiMo V2.5 | `MiMoV2ForCausalLM` | No | 256 expertos, top_k=8, multimodal/audio |
| MiMo V2 | `MiMoV2FlashForCausalLM` | No | Modelo de 309B parámetros |
| LongCat | `LongcatCausalLM` | No | 512 expertos reales + 256 expertos cero |
| MiniMax M2.5 | `MiniMaxM2ForCausalLM` | No | Solo texto, nomenclatura w1/w2/w3 |
| MiniMax M3 | `MiniMaxM3SparseForConditionalGeneration` | No | 128 expertos, top_k=4, multimodal, expertos compartidos, e_score_correction_bias |
| DiffusionGemma 26B-A4B | `DiffusionGemmaForBlockDiffusion` | Sí | Los expertos residen directamente en `model.decoder.layers.*` |
| Tencent Hy3 | `HYV3ForCausalLM` | Sí | 192 expertos, top_k=8, gate_up_proj fusionado, expertos compartidos, enrutador sigmoide, capa MTP |

## Conjuntos de Datos de Calibración

Los siguientes conjuntos de datos de calibración integrados están disponibles:

| Conjunto de Datos | Descripción | Uso Para |
|---------|-------------|---------|
| `c4` | Texto web general (corpus C4) | Compresión de propósito general |
| `code` | Conjunto de datos de instrucciones de código | Modelos enfocados en código |
| `math` | Conjunto de datos de instrucciones matemáticas | Modelos de matemáticas/razonamiento |
| `writing` | Ideas para escritura creativa | Modelos de escritura creativa |
| `combined` | Mezcla de todas las categorías | Compresión equilibrada |

## Opciones de la CLI

```
usage: compress_model.py [-h] --model MODEL --output OUTPUT
                          [--method {prune,merge}]
                          [--compression-ratio COMPRESSION_RATIO]
                          [--target-ratio TARGET_RATIO]
                          [--n-experts N_EXPERTS]
                          [--offline-seed-prune]
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
  -h, --help            mostrar este mensaje de ayuda y salir
  --model MODEL         Nombre o ruta del modelo (formato HuggingFace)
  --output OUTPUT       Directorio de salida para el modelo comprimido
  --method {prune,merge}  Método de compresión
  --compression-ratio COMPRESSION_RATIO
                        Fracción de expertos a eliminar (predeterminado: 0.25)
  --target-ratio TARGET_RATIO
                        Para fusión: fracción de expertos a MANTENER
  --n-experts N_EXPERTS  Número exacto de expertos a podar
  --offline-seed-prune  Poda aleatoria directamente en safetensors sin cargar el modelo
  --dataset DATASET     Conjunto de datos de calibración
  --samples SAMPLES      Número de muestras para calibración (predeterminado: 1000)
  --max-seq-len MAX_SEQ_LEN
                        Longitud máxima de secuencia (predeterminado: 512)
  --batch-size BATCH_SIZE
                        Tamaño del lote para calibración (predeterminado: 4)
  --max-tokens MAX_TOKENS
                        Tokens máximos por capa (predeterminado: 1048576)
  --device DEVICE       Dispositivo a usar (predeterminado: cuda si está disponible)
  --torch-dtype {auto,float32,float16,bfloat16}
                        Tipo de datos de Torch (predeterminado: auto)
  --renormalize-router  Renormalizar pesos del enrutador después de top-k
  --verify-only         Solo verificar la configuración del modelo
  --skip-verification   Omitir verificación del modelo
  --preserve-super-experts
                        Preservar expertos de alta activación
  --seed SEED           Semilla aleatoria (predeterminado: 42)
```

## Uso Avanzado

### Verificación de la Configuración del Modelo

Antes de la compresión, verifique que su modelo tenga soporte adecuado:

```python
from ream_moe import verify_model_config, print_verification_result

result = verify_model_config("Qwen/Qwen3-14B-MoE")
print_verification_result(result)
```

### Listado de Modelos Soportados

```python
from ream_moe import list_supported_models

for model_class in list_supported_models():
    print(model_class)
```

### Datos de Calibración Personalizados

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

**Nota:** Todos los conjuntos de datos integrados utilizan instrucciones codificadas de forma integral para evitar problemas de memoria agotada (OOM) por las descargas de conjuntos de datos de HuggingFace. Estas instrucciones cubren dominios diversos:
- **c4**: Conocimiento general, ML/IA, ciencia, historia, negocios
- **code**: Python, desarrollo web, ciencia de datos, algoritmos, DevOps, seguridad
- **math**: Álgebra, cálculo, geometría, estadística, probabilidad, teoría de números
- **writing**: Ideas para historias, escritura descriptiva, poesía, diálogos, reflexiones
- **hardcoded**: Conjunto combinado grande con todas las categorías (recomendado para la mejor calibración)
- **combined**: Mezcla más pequeña de todas las categorías

### Preservar Super Expertos

Para evitar la poda de expertos con activación inusualmente alta:

```python
config = PruningConfig(
    compression_ratio=0.25,
    preserve_super_experts=True,
)
```

## Configuración del Modelo

Cada familia de modelos requiere una configuración específica almacenada en `MODEL_ATTRS`:

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

## Registro Automático

Para modelos no soportados explícitamente, REAM-MoE puede intentar detectar automáticamente la configuración:

```python
from ream_moe import ensure_model_registered

model = AutoModelForCausalLM.from_pretrained("unknown-moe-model")
success = ensure_model_registered(model)

if success:
    print("Model auto-registered successfully!")
else:
    print("Auto-registration failed, please add to MODEL_ATTRS manually")
```

## Detalles del Algoritmo

### Saliencia REAP

La importancia de cada experto se calcula como:

```
S[i] = mean_{tokens routed to i} ||h_i(x)|| * p_i(x)
```

Donde:
- `h_i(x)` = estados ocultos de salida del experto
- `p_i(x)` = probabilidades de enrutamiento
- `||·||` = norma L2

### Agrupación de Expertos

1. Seleccionar expertos centroides (mayor saliencia)
2. Para cada centroide, agrupar expertos cercanos usando similitud con compuerta
3. La mayoría de los expertos de baja saliencia permanecen como singletes (pseudopoda)

### Fusión

1. Para cada grupo, alinear los pesos de los expertos usando el algoritmo húngaro
2. Fusionar con promedio ponderado por saliencia
3. Actualizar el enrutador para que solo emita centroides

## Estructura del Proyecto

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

## Contribuir

Para agregar soporte para una nueva familia de modelos:

1. Agregue la configuración del modelo a `model_attr_configs.py`:

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

2. Agregue la configuración del observador a `observer_configs.py`:

```python
OBSERVER_CONFIG_REGISTRY["YourMoEModelForCausalLM"] = type(
    "YourMoEObserverConfig",
    (ObserverHookConfig,),
    {"module_class_name_to_hook_regex": "YourMoEBlock"},
)
```

3. ¡Pruebe primero con `--verify-only`!

## Licencia

MIT

## Referencias

- Artículo de blog: [Understanding MoE Compression](https://bknyaz.github.io/blog/2026/moe/) - Explicación detallada del algoritmo y la teoría de REAM/REAP
- Basado en observaciones de [Cerebras Research/reap](https://github.com/CerebrasResearch/reap)

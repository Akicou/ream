"""
Offline safetensors pruning utilities.

This module prunes MoE experts directly in safetensors checkpoints without
instantiating a Transformers model and without using CUDA/GPU. It is intended
for very large checkpoints where calibration/model loading is impractical.

Notes:
- This is random/seeded pruning, not saliency pruning.
- It processes one safetensors shard at a time on CPU.
- The safetensors writer requires the output tensors for a shard to be present
  in CPU memory while that shard is written; it never loads the full model.
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ream_moe.model_attr_configs import get_model_attrs

logger = logging.getLogger(__name__)


@dataclass
class OfflinePruneResult:
    """Summary of an offline safetensors pruning run."""

    source_dir: Path
    output_dir: Path
    model_class: str
    original_experts: int
    retained_experts: int
    pruned_experts: int
    num_layers: int
    original_total_size_bytes: Optional[int]
    output_total_size_bytes: int


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def _resolve_model_dir(model: str) -> Path:
    """Return a local model directory, downloading from HF if needed."""
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "--offline-seed-prune with a remote model id requires huggingface_hub. "
            "Install transformers/huggingface_hub or pass a local model directory."
        ) from exc

    logger.warning(
        "Model path does not exist locally; downloading safetensors/tokenizer/config "
        "snapshot for offline pruning: %s",
        model,
    )
    local_dir = snapshot_download(
        repo_id=model,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.model",
            "*.txt",
            "*.py",
            "*.md",
            "LICENSE*",
            ".gitattributes",
        ],
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5"],
    )
    return Path(local_dir).resolve()


def _load_config(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _model_class_from_config(config: Dict[str, Any]) -> str:
    architectures = config.get("architectures") or []
    if architectures:
        return str(architectures[0])
    model_type = str(config.get("model_type", ""))
    if model_type == "deepseek_v4":
        return "DeepseekV4ForCausalLM"
    raise ValueError("Could not infer model class from config.json architectures/model_type")


def _get_config_int(config: Dict[str, Any], attr: str) -> Optional[int]:
    if attr.startswith("config."):
        attr = attr.split(".", 1)[1]
    if "." in attr:
        return None
    value = config.get(attr)
    return int(value) if isinstance(value, int) else None


def _load_weight_map(model_dir: Path) -> Tuple[Dict[str, str], Optional[Dict[str, Any]]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return dict(index.get("weight_map", {})), index

    weight_map: Dict[str, str] = {}
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for file_path in safetensor_files:
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = file_path.name
    return weight_map, None


def _infer_layers_and_experts(weight_map: Dict[str, str], moe_paths: List[str]) -> Tuple[List[int], int]:
    layers: set[int] = set()
    experts: set[int] = set()

    for key in weight_map:
        match = _match_numbered_expert_key(key, moe_paths)
        if match is None:
            continue
        _prefix, layer_idx, _moe_path, expert_idx, _rest = match
        layers.add(layer_idx)
        experts.add(expert_idx)

    if not layers or not experts:
        raise ValueError(
            "Could not find numbered expert tensors in safetensors index. "
            f"Checked MoE paths: {moe_paths}"
        )

    return sorted(layers), max(experts) + 1


def _build_retained_indices(
    layers: List[int],
    num_experts: int,
    retained_experts: int,
    seed: int,
) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    retained_by_layer: Dict[int, List[int]] = {}
    for layer_idx in layers:
        retained = sorted(rng.sample(range(num_experts), retained_experts))
        retained_by_layer[layer_idx] = retained
    return retained_by_layer


def _build_old_to_new(retained: List[int], num_experts: int) -> torch.Tensor:
    """Map old expert ids to compact new ids; pruned ids map to expert 0."""
    mapping = torch.zeros(num_experts, dtype=torch.long)
    for new_idx, old_idx in enumerate(retained):
        mapping[old_idx] = new_idx
    return mapping


def _match_numbered_expert_key(
    key: str,
    moe_paths: List[str],
) -> Optional[Tuple[str, int, str, int, str]]:
    for moe_path in moe_paths:
        pattern = rf"^((?:model\.)?)layers\.(\d+)\.{re.escape(moe_path)}\.experts\.(\d+)\.(.+)$"
        match = re.match(pattern, key)
        if match:
            return match.group(1), int(match.group(2)), moe_path, int(match.group(3)), match.group(4)
    return None


def _match_fused_experts_key(
    key: str,
    moe_paths: List[str],
) -> Optional[Tuple[str, int, str, str]]:
    for moe_path in moe_paths:
        pattern = rf"^((?:model\.)?)layers\.(\d+)\.{re.escape(moe_path)}\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$"
        match = re.match(pattern, key)
        if match:
            return match.group(1), int(match.group(2)), moe_path, match.group(3)
    return None


def _match_router_key(
    key: str,
    moe_paths: List[str],
    router_attr: str,
) -> Optional[Tuple[str, int, str, str]]:
    for moe_path in moe_paths:
        pattern = rf"^((?:model\.)?)layers\.(\d+)\.{re.escape(moe_path)}\.{re.escape(router_attr)}\.(.+)$"
        match = re.match(pattern, key)
        if match:
            return match.group(1), int(match.group(2)), moe_path, match.group(3)
    return None


def _copy_non_weight_files(source_dir: Path, output_dir: Path, skip_safetensors: set[Path]) -> None:
    for src in source_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source_dir)
        if ".git" in rel.parts or ".cache" in rel.parts:
            continue
        if src.name == "model.safetensors.index.json":
            continue
        if src.suffix == ".safetensors" and rel in skip_safetensors:
            continue
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _update_config(
    config: Dict[str, Any],
    retained_experts: int,
) -> Dict[str, Any]:
    updated = dict(config)
    for key in ["n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"]:
        if key in updated:
            updated[key] = retained_experts

    for topk_key in ["num_experts_per_tok", "moe_k", "top_k", "num_selected_experts"]:
        if isinstance(updated.get(topk_key), int) and updated[topk_key] > retained_experts:
            updated[topk_key] = retained_experts

    return updated


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _transform_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    moe_paths: List[str],
    router_attr: str,
    num_experts: int,
    retained_by_layer: Dict[int, List[int]],
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    numbered = _match_numbered_expert_key(key, moe_paths)
    if numbered is not None:
        prefix, layer_idx, moe_path, old_expert_idx, rest = numbered
        retained = retained_by_layer[layer_idx]
        if old_expert_idx not in retained:
            return None, None
        new_expert_idx = retained.index(old_expert_idx)
        new_key = f"{prefix}layers.{layer_idx}.{moe_path}.experts.{new_expert_idx}.{rest}"
        return new_key, tensor

    fused = _match_fused_experts_key(key, moe_paths)
    if fused is not None:
        _prefix, layer_idx, _moe_path, _name = fused
        retained = retained_by_layer[layer_idx]
        if tensor.ndim > 0 and tensor.shape[0] == num_experts:
            idx = torch.as_tensor(retained, dtype=torch.long)
            return key, tensor.index_select(0, idx)
        return key, tensor

    router = _match_router_key(key, moe_paths, router_attr)
    if router is not None:
        _prefix, layer_idx, _moe_path, router_leaf = router
        retained = retained_by_layer[layer_idx]
        idx = torch.as_tensor(retained, dtype=torch.long)

        if router_leaf == "tid2eid":
            mapping = _build_old_to_new(retained, num_experts).to(tensor.device)
            clamped = tensor.long().clamp(min=0, max=num_experts - 1)
            return key, mapping[clamped].to(dtype=tensor.dtype)

        # Router rows / correction biases are indexed by expert id.
        if tensor.ndim > 0 and tensor.shape[0] == num_experts:
            return key, tensor.index_select(0, idx)
        if tensor.ndim > 0 and tensor.shape[-1] == num_experts:
            return key, tensor.index_select(tensor.ndim - 1, idx)

    return key, tensor


def offline_seed_prune_safetensors(
    model: str,
    output: str,
    *,
    n_experts_to_prune: Optional[int] = None,
    compression_ratio: float = 0.25,
    seed: int = 42,
) -> OfflinePruneResult:
    """
    Randomly prune MoE experts directly in safetensors files.

    Args:
        model: Local model directory or Hugging Face repo id.
        output: Output directory for the pruned checkpoint.
        n_experts_to_prune: Exact number of routed experts to remove per layer.
        compression_ratio: Fraction of routed experts to remove if n_experts_to_prune is None.
        seed: Seed controlling the deterministic random retained expert set.

    Returns:
        OfflinePruneResult summary.
    """
    source_dir = _resolve_model_dir(model)
    output_dir = Path(output).expanduser().resolve()

    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be the same as or inside the source model directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(source_dir)
    model_class = _model_class_from_config(config)
    attrs = get_model_attrs(model_class)
    if attrs is None:
        raise ValueError(f"Model class {model_class!r} is not registered in MODEL_ATTRS")

    moe_paths = _as_list(attrs.get("moe_block", "mlp"))
    router_attr = str(attrs.get("router", "gate"))

    weight_map, original_index = _load_weight_map(source_dir)
    layers, inferred_num_experts = _infer_layers_and_experts(weight_map, moe_paths)

    config_num_experts = _get_config_int(config, str(attrs.get("num_experts", "")))
    num_experts = config_num_experts or inferred_num_experts

    if num_experts != inferred_num_experts:
        logger.warning(
            "Config expert count (%s) differs from checkpoint inferred count (%s); using checkpoint count",
            num_experts,
            inferred_num_experts,
        )
        num_experts = inferred_num_experts

    if n_experts_to_prune is None:
        n_experts_to_prune = int(num_experts * compression_ratio)

    n_experts_to_prune = max(0, min(int(n_experts_to_prune), num_experts - 1))
    retained_experts = num_experts - n_experts_to_prune

    retained_by_layer = _build_retained_indices(layers, num_experts, retained_experts, seed)

    logger.warning(
        "Offline seed pruning %s: %s -> %s experts/layer across %s layers (seed=%s). "
        "No Transformers model is loaded; safetensors shards are processed on CPU.",
        model_class,
        num_experts,
        retained_experts,
        len(layers),
        seed,
    )

    source_files = sorted(set(weight_map.values()))
    skip_safetensors = {Path(filename) for filename in source_files}

    _copy_non_weight_files(source_dir, output_dir, skip_safetensors)
    (output_dir / "config.json").write_text(
        json.dumps(_update_config(config, retained_experts), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    new_weight_map: Dict[str, str] = {}
    output_total_size = 0

    for filename in source_files:
        src_file = source_dir / filename
        dst_file = output_dir / filename
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        out_tensors: Dict[str, torch.Tensor] = {}
        metadata = None

        logger.warning("Processing safetensors shard: %s", filename)
        with safe_open(str(src_file), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                new_key, new_tensor = _transform_tensor(
                    key,
                    tensor,
                    moe_paths=moe_paths,
                    router_attr=router_attr,
                    num_experts=num_experts,
                    retained_by_layer=retained_by_layer,
                )
                if new_key is None or new_tensor is None:
                    continue
                out_tensors[new_key] = new_tensor.contiguous()
                new_weight_map[new_key] = filename
                output_total_size += _tensor_nbytes(new_tensor)

        save_file(out_tensors, str(dst_file), metadata=metadata)
        del out_tensors

    original_total_size = None
    if original_index is not None:
        metadata = original_index.get("metadata") or {}
        if isinstance(metadata.get("total_size"), int):
            original_total_size = metadata["total_size"]

        new_index = dict(original_index)
        new_metadata = dict(metadata)
        new_metadata["total_size"] = output_total_size
        new_index["metadata"] = new_metadata
        new_index["weight_map"] = new_weight_map
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif len(source_files) > 1:
        new_index = {
            "metadata": {"total_size": output_total_size},
            "weight_map": new_weight_map,
        }
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    info = {
        "method": "offline_seed_prune_safetensors",
        "model_class": model_class,
        "source_dir": str(source_dir),
        "original_experts": num_experts,
        "retained_experts": retained_experts,
        "pruned_experts": n_experts_to_prune,
        "seed": seed,
        "num_layers": len(layers),
        "layers": layers,
        "retained_indices_by_layer": {str(k): v for k, v in retained_by_layer.items()},
        "output_total_size_bytes": output_total_size,
    }
    (output_dir / "offline_prune_info.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )

    return OfflinePruneResult(
        source_dir=source_dir,
        output_dir=output_dir,
        model_class=model_class,
        original_experts=num_experts,
        retained_experts=retained_experts,
        pruned_experts=n_experts_to_prune,
        num_layers=len(layers),
        original_total_size_bytes=original_total_size,
        output_total_size_bytes=output_total_size,
    )

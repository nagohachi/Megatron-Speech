"""Reading and writing Hugging Face checkpoints (safetensors)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
SKIP_FILES = {"model.safetensors.index.json", ".gitattributes", "README.md"}


def resolve_hf_path(path_or_repo: str) -> Path:
    """Local directory of an HF checkpoint given a path or a hub repo id (uses the HF cache)."""
    if os.path.isdir(path_or_repo):
        return Path(path_or_repo)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(path_or_repo, local_files_only=True))
    except Exception:
        return Path(snapshot_download(path_or_repo))


def load_hf_config(hf_dir: Path) -> dict:
    with open(hf_dir / "config.json") as f:
        return json.load(f)


class SafetensorsReader:
    """Lazy tensor access over a single-file or sharded safetensors checkpoint."""

    def __init__(self, hf_dir: Path):
        hf_dir = Path(hf_dir)
        index = hf_dir / "model.safetensors.index.json"
        if index.exists():
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
            self.key_to_file = {k: hf_dir / v for k, v in weight_map.items()}
        else:
            files = sorted(hf_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors files in {hf_dir}")
            self.key_to_file = {}
            for file in files:
                with safe_open(file, framework="pt") as f:
                    for k in f.keys():
                        self.key_to_file[k] = file
        self._handles = {}

    def keys(self) -> set[str]:
        return set(self.key_to_file)

    def get(self, key: str) -> torch.Tensor:
        file = self.key_to_file[key]
        if file not in self._handles:
            self._handles[file] = safe_open(file, framework="pt")
        return self._handles[file].get_tensor(key)


def save_hf_state_dict(state_dict: dict[str, torch.Tensor], out_dir: Path, max_shard_bytes: int = 5 * 1024**3) -> None:
    """Write ``state_dict`` as (sharded) safetensors with an index when sharded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().contiguous().cpu()
        nbytes = tensor.numel() * tensor.element_size()
        if shards[-1] and size + nbytes > max_shard_bytes:
            shards.append({})
            size = 0
        shards[-1][key] = tensor
        size += nbytes
    if len(shards) == 1:
        save_file(shards[0], out_dir / "model.safetensors", metadata={"format": "pt"})
        return
    weight_map = {}
    total = 0
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, out_dir / name, metadata={"format": "pt"})
        for key, tensor in shard.items():
            weight_map[key] = name
            total += tensor.numel() * tensor.element_size()
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def copy_non_weight_files(src_dir: Path, out_dir: Path) -> None:
    """Copy config, tokenizer, processor and template files of an HF checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(src_dir).iterdir():
        if path.is_dir() or path.name in SKIP_FILES or path.name.endswith(WEIGHT_SUFFIXES):
            continue
        shutil.copy(path, out_dir / path.name)  # follows HF-cache symlinks

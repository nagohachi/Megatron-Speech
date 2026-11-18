"""Generic HF -> Megatron weight loading and Megatron -> HF export, driven by a MappingTable."""

from __future__ import annotations

import torch

from megatron_speech import compat
from megatron_speech.bridges.hf_io import SafetensorsReader
from megatron_speech.bridges.mapping import MappingTable

# Wrapper segments that PEFT inserts into parameter names (megatron_speech.peft.lora).
_WRAPPER_SEGMENTS = (".to_wrap.",)
_ADAPTER_SEGMENT = ".adapter."


def normalize_param_name(name: str) -> str:
    """Strip DDP/Float16Module prefixes and PEFT wrapper segments from a parameter name."""
    for prefix in ("module.",):
        while name.startswith(prefix):
            name = name[len(prefix):]
    for seg in _WRAPPER_SEGMENTS:
        name = name.replace(seg, ".")
    return name


def iter_base_parameters(model: torch.nn.Module):
    """Yield ``(normalized_name, param)`` of base (non-adapter) parameters."""
    for name, param in model.named_parameters():
        if _ADAPTER_SEGMENT in name:
            continue
        yield normalize_param_name(name), param


@torch.no_grad()
def load_hf_weights(model: torch.nn.Module, table: MappingTable, reader: SafetensorsReader, allow_unused_hf: set[str] = frozenset()) -> None:
    """Copy HF tensors into the (possibly tensor-parallel) Megatron model in place."""
    tp_rank = compat.get_tensor_model_parallel_rank()
    tp_size = compat.get_tensor_model_parallel_world_size()
    missing, used = [], set()
    for name, param in iter_base_parameters(model):
        mapping = table.get(name)
        if mapping is None:
            missing.append(name)
            continue
        hf = {n: reader.get(n) for n in mapping.hf_names}
        used.update(mapping.hf_names)
        full = mapping.to_megatron(hf)
        local = mapping.shard(full, param.shape, tp_rank, tp_size)
        if tuple(local.shape) != tuple(param.shape):
            raise ValueError(f"{name}: HF tensor gives shape {tuple(local.shape)}, model has {tuple(param.shape)}")
        param.copy_(local.to(device=param.device, dtype=param.dtype))
    if missing:
        raise KeyError(f"no HF mapping for Megatron parameters: {missing[:10]} ({len(missing)} total)")
    unused = reader.keys() - used - set(allow_unused_hf)
    if unused:
        raise KeyError(f"HF tensors not loaded into the model: {sorted(unused)[:10]} ({len(unused)} total)")


def _gather_tp(param: torch.Tensor) -> list[torch.Tensor]:
    tp_size = compat.get_tensor_model_parallel_world_size()
    if tp_size == 1 or not getattr(param, "tensor_model_parallel", False):
        return [param.detach()]
    shards = [torch.empty_like(param) for _ in range(tp_size)]
    torch.distributed.all_gather(shards, param.detach().contiguous(), group=compat.get_tensor_model_parallel_group())
    return shards


@torch.no_grad()
def export_hf_state_dict(model: torch.nn.Module, table: MappingTable, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Full HF state dict of the model's base weights (collective over the TP group).

    Every TP rank returns the same dict; the caller writes it from one rank.
    """
    out: dict[str, torch.Tensor] = {}
    for name, param in iter_base_parameters(model):
        mapping = table.get(name)
        if mapping is None:
            raise KeyError(f"no HF mapping for Megatron parameter {name}")
        shards = _gather_tp(param)
        partition_dim = getattr(param, "partition_dim", None) if len(shards) > 1 else None
        full = mapping.unshard(shards, partition_dim)
        for hf_name, tensor in mapping.to_hf(full).items():
            out[hf_name] = tensor.to(dtype=dtype or tensor.dtype).cpu()
    return out

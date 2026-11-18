"""Export LoRA-trained models to Hugging Face: merged weights or a separate PEFT adapter.

Adapter export reuses the base-weight mapping of each wrapped layer: ``lora_B`` has the same
row layout as the base weight, so the mapping that splits a fused ``linear_qkv`` weight into
q/k/v (or ``linear_fc1`` into gate/up) splits ``lora_B`` the same way, and the shared
``lora_A`` is repeated for each HF projection. The result is exact: the PEFT model computes
``W x + s * B_q A x`` for q, and so on.

Components trained in full alongside LoRA (``mode: full``, e.g. the projector) are saved in
the adapter through PEFT's ``modules_to_save``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from megatron_speech import compat
from megatron_speech.bridges.convert import iter_base_parameters, normalize_param_name
from megatron_speech.bridges.mapping import MappingTable
from megatron_speech.peft.apply import COMPONENT_PREFIXES
from megatron_speech.peft.lora import LoRALinear

PEFT_PREFIX = "base_model.model."


def iter_lora_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield normalize_param_name(name + "."), module


def _gather(t: torch.Tensor, param: torch.Tensor) -> list[torch.Tensor]:
    tp_size = compat.get_tensor_model_parallel_world_size()
    if tp_size == 1 or not getattr(param, "tensor_model_parallel", False):
        return [t]
    shards = [torch.empty_like(t) for _ in range(tp_size)]
    torch.distributed.all_gather(shards, t.contiguous(), group=compat.get_tensor_model_parallel_group())
    return shards


def _gather_param(value: torch.Tensor, param: torch.Tensor) -> tuple[list[torch.Tensor], int | None]:
    shards = _gather(value.detach(), param)
    return shards, (getattr(param, "partition_dim", None) if len(shards) > 1 else None)


def export_lora_tensors(model: torch.nn.Module, table: MappingTable, use_grad: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    """``{hf_module: {"lora_A": A, "lora_B": B}}`` for every LoRA layer (collective over TP).

    With ``use_grad=True`` the gradients are exported instead (for tests). For the ``fused``
    variant the gradient of the shared ``lora_A`` equals the sum of the HF per-projection
    ``lora_A`` gradients.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in iter_lora_modules(model):
        prefix = name.rstrip(".")
        mapping = table.get(f"{prefix}.weight")
        if mapping is None:
            raise KeyError(f"no base-weight mapping for LoRA layer {prefix}")
        adapter = module.adapter
        a_param = adapter.lora_A.weight
        shards, dim = _gather_param(a_param.grad if use_grad else a_param, a_param)
        full_a = shards[0] if dim is None else torch.cat(shards, dim=dim)
        b_full = []
        for b in adapter.lora_B:
            shards, dim = _gather_param(b.weight.grad if use_grad else b.weight, b.weight)
            b_full.append((shards, dim))
        if len(adapter.lora_B) == 1:
            # One pair for the whole (possibly fused) layer: B follows the base weight layout.
            shards, dim = b_full[0]
            for hf_weight, b in mapping.to_hf(mapping.unshard(shards, dim)).items():
                out[hf_weight.removesuffix(".weight")] = {"lora_A": full_a, "lora_B": b}
        else:
            # Canonical: pair i belongs to HF projection i (q/k/v or gate/up), rows in HF order.
            if len(mapping.hf_names) != len(adapter.lora_B):
                raise ValueError(f"{prefix}: {len(adapter.lora_B)} LoRA pairs but {len(mapping.hf_names)} HF tensors")
            r = adapter.rank
            for i, (hf_weight, (shards, dim)) in enumerate(zip(mapping.hf_names, b_full)):
                b = shards[0] if dim is None else torch.cat(shards, dim=dim)
                out[hf_weight.removesuffix(".weight")] = {"lora_A": full_a[i * r : (i + 1) * r], "lora_B": b}
    return out


def _component_hf_prefix(table: MappingTable, megatron_prefix: str) -> str:
    names = [n for m in table if m.megatron_name.startswith(megatron_prefix) for n in m.hf_names]
    common = os.path.commonprefix(names)
    return common[: common.rfind(".")]


def export_peft_adapter(
    model: torch.nn.Module,
    table: MappingTable,
    peft_cfg: dict[str, Any],
    base_model_name_or_path: str,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """PEFT ``adapter_model.safetensors`` state dict and ``adapter_config.json`` contents."""
    lora = export_lora_tensors(model, table)
    if not lora:
        raise ValueError("the model has no LoRA layers")
    state_dict: dict[str, torch.Tensor] = {}
    for hf_module, t in lora.items():
        for key in ("lora_A", "lora_B"):
            state_dict[f"{PEFT_PREFIX}{hf_module}.{key}.weight"] = t[key].to(dtype or t[key].dtype).cpu()

    lora_components = {c: cfg for c, cfg in peft_cfg.items() if (cfg or {}).get("mode") == "lora"}
    ranks = {(int(c.get("rank", 16)), float(c.get("alpha", 32))) for c in lora_components.values()}
    if len(ranks) != 1:
        raise NotImplementedError("PEFT export with different rank/alpha per component is not implemented")
    (rank, alpha), = ranks
    dropout = max(float(c.get("dropout", 0.0)) for c in lora_components.values())

    # Fully trained components go into modules_to_save; their weights are stored under the
    # module path without PEFT's "modules_to_save.default." infix.
    modules_to_save = []
    for component, prefix in COMPONENT_PREFIXES.items():
        if (peft_cfg.get(component) or {}).get("mode", "full") != "full":
            continue
        if not any(n.startswith(prefix) for n, _ in iter_base_parameters(model)):
            continue
        modules_to_save.append(_component_hf_prefix(table, prefix))
    if modules_to_save:
        from megatron_speech.bridges.convert import export_hf_state_dict

        for hf_name, tensor in export_hf_state_dict(model, table, dtype=dtype).items():
            if any(hf_name.startswith(m + ".") for m in modules_to_save):
                state_dict[PEFT_PREFIX + hf_name] = tensor

    target_regex = "|".join(re.escape(m) for m in sorted(lora))
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_name_or_path,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_regex,
        "modules_to_save": modules_to_save or None,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
        "task_type": None,
    }
    return state_dict, config


def save_peft_adapter(state_dict: dict[str, torch.Tensor], config: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # clone: canonical lora_A blocks are views of one tensor, which safetensors refuses to save.
    save_file({k: v.detach().cpu().clone() for k, v in state_dict.items()}, out_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


@torch.no_grad()
def merge_lora_weights(model: torch.nn.Module) -> int:
    """Fold every adapter into its base weight and zero ``lora_B``; returns the layer count."""
    n = 0
    for _, module in iter_lora_modules(model):
        module.merge_into_base()
        for b in module.adapter.lora_B:
            b.weight.zero_()
        n += 1
    return n

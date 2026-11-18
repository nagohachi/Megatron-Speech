"""Apply per-component training modes (full / frozen / lora) to a speech LLM."""

from __future__ import annotations

from typing import Any, Callable

import torch

from megatron_speech.peft.lora import LoRALinear

# Module-name prefixes of the components inside MimoModel.
COMPONENT_PREFIXES = {
    "llm": "language_model.",
    "encoder": "modality_submodules.audio.encoders.",
    "projector": "modality_submodules.audio.input_projections.",
}
DEFAULT_LORA_TARGETS = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
MODES = ("full", "frozen", "lora")


def _set_submodule(root: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


def apply_lora(root: torch.nn.Module, prefix: str, rank: int, alpha: float, dropout: float, target_modules: list[str],
               variant: str = "canonical") -> int:
    """Wrap every linear under ``prefix`` whose last name is in ``target_modules``."""
    targets = [
        name
        for name, module in root.named_modules()
        if name.startswith(prefix)
        and name.rsplit(".", 1)[-1] in target_modules
        and not isinstance(module, LoRALinear)
        and hasattr(module, "weight")
    ]
    for name in targets:
        module = LoRALinear(
            root.get_submodule(name), name.rsplit(".", 1)[-1], rank=rank, alpha=alpha, dropout=dropout, variant=variant
        )
        _set_submodule(root, name, module)
    return len(targets)


def apply_peft(model: torch.nn.Module, peft_cfg: dict[str, Any]) -> dict[str, int]:
    """Freeze or add LoRA per component. Returns the number of wrapped layers per component."""
    unknown = set(peft_cfg) - set(COMPONENT_PREFIXES)
    if unknown:
        raise ValueError(f"model.peft has unknown components {sorted(unknown)}; use {sorted(COMPONENT_PREFIXES)}")
    wrapped = {}
    for component, prefix in COMPONENT_PREFIXES.items():
        cfg = dict(peft_cfg.get(component) or {})
        mode = cfg.pop("mode", "full")
        if mode not in MODES:
            raise ValueError(f"model.peft.{component}.mode must be one of {MODES}, got {mode!r}")
        if mode == "full":
            continue
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad_(False)
        if mode == "lora":
            wrapped[component] = apply_lora(
                model,
                prefix,
                rank=int(cfg.get("rank", 16)),
                alpha=float(cfg.get("alpha", 32)),
                dropout=float(cfg.get("dropout", 0.0)),
                target_modules=list(cfg.get("target_modules", DEFAULT_LORA_TARGETS)),
                variant=cfg.get("variant", "canonical"),
            )
            if wrapped[component] == 0:
                raise ValueError(f"model.peft.{component}: no layer matched {cfg.get('target_modules', DEFAULT_LORA_TARGETS)}")
    return wrapped


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def apply_peft_hook(peft_cfg: dict[str, Any]) -> Callable:
    """``pre_wrap_hook`` applying ``apply_peft`` to every model chunk."""

    def hook(model_chunks):
        from megatron_speech import compat

        for chunk in model_chunks:
            wrapped = apply_peft(chunk, peft_cfg)
            trainable, total = count_parameters(chunk)
            compat.print_rank_0(
                f"> PEFT: LoRA layers {wrapped or 'none'}; trainable parameters on this rank "
                f"{trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)"
            )
        return model_chunks

    return hook

"""YAML run configuration with dotted command-line overrides.

A run config has three top-level sections::

    model:     # what to build (recipe name, HF checkpoint, components, PEFT)
    data:      # recipe-specific data settings
    megatron:  # Megatron-LM command-line arguments, e.g. {micro-batch-size: 1, bf16: true}

Overrides are ``key.subkey=value`` tokens; values are parsed as YAML scalars
(``megatron.lr=1e-5``, ``model.peft.llm.mode=lora``). A config may name another file in
``base:``; it is loaded first and the current file is merged on top.
"""

import copy
from pathlib import Path

import yaml

type ConfigValue = str | int | float | bool | None | list[ConfigValue] | dict[str, ConfigValue]
type ConfigDict = dict[str, ConfigValue]


def merge_dicts(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config_file(path: str | Path) -> ConfigDict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        base_path = (path.parent / base) if not Path(base).is_absolute() else Path(base)
        cfg = merge_dicts(load_config_file(base_path), cfg)
    return cfg


def apply_overrides(cfg: ConfigDict, overrides: list[str]) -> ConfigDict:
    cfg = copy.deepcopy(cfg)
    for token in overrides:
        if "=" not in token:
            raise ValueError(f"override must look like key.subkey=value, got {token!r}")
        dotted, raw = token.split("=", 1)
        value = yaml.safe_load(raw) if raw != "" else None
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {dotted}: {key} is not a mapping")
        node[keys[-1]] = value
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> ConfigDict:
    cfg = load_config_file(path)
    cfg = apply_overrides(cfg, overrides or [])
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("megatron", {})
    return cfg

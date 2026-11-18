"""Registry of speech components, keyed by the ``type`` field of their config.

A component config is a plain dict (from YAML or from a bridge), e.g.::

    encoder:   {type: qwen3_asr, d_model: 1024, num_layers: 24, ...}
    projector: {type: mlp, input_size: 1024, hidden_size: 1024, output_size: 2048}
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASREncoderConfig
from megatron_speech.components.projectors.mlp import MLPProjector, MLPProjectorConfig

ENCODERS: dict[str, tuple[type, type]] = {
    "qwen3_asr": (Qwen3ASREncoderConfig, Qwen3ASRAudioEncoder),
}

PROJECTORS: dict[str, tuple[type, type]] = {
    "mlp": (MLPProjectorConfig, MLPProjector),
}


def _parse(component_cfg: dict[str, Any], registry: dict[str, tuple[type, type]], kind: str):
    cfg = dict(component_cfg)
    type_name = cfg.pop("type")
    if type_name not in registry:
        raise ValueError(f"unknown {kind} type {type_name!r}; available: {sorted(registry)}")
    config_cls, module_cls = registry[type_name]
    known = {f.name for f in fields(config_cls)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"{kind} {type_name!r} does not take {sorted(unknown)}")
    return config_cls(**cfg), module_cls


def parse_encoder_config(component_cfg: dict[str, Any]):
    return _parse(component_cfg, ENCODERS, "encoder")[0]


def build_encoder_spec(component_cfg: dict[str, Any], llm_config, pg_collection) -> ModuleSpec:
    config, module_cls = _parse(component_cfg, ENCODERS, "encoder")
    return ModuleSpec(
        module=module_cls,
        params={"encoder_config": config, "llm_config": llm_config, "pg_collection": pg_collection},
    )


def build_projector_spec(component_cfg: dict[str, Any], llm_config, pg_collection) -> ModuleSpec:
    config, module_cls = _parse(component_cfg, PROJECTORS, "projector")
    return ModuleSpec(
        module=module_cls,
        params={"projector_config": config, "llm_config": llm_config, "pg_collection": pg_collection},
    )

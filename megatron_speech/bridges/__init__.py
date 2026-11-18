"""Hugging Face <-> Megatron weight bridges."""

from __future__ import annotations

from typing import Any, Callable

from megatron_speech.bridges.base import SpeechLLMBridge


def get_bridge(name: str, hf_path: str) -> SpeechLLMBridge:
    if name == "qwen3_asr":
        from megatron_speech.bridges.qwen3_asr import Qwen3ASRBridge

        return Qwen3ASRBridge(hf_path)
    raise ValueError(f"unknown bridge {name!r} (available: qwen3_asr)")


def get_bridge_for_model_section(model_section: dict[str, Any]) -> SpeechLLMBridge:
    if not model_section.get("bridge") or not model_section.get("hf_init"):
        raise ValueError("model.bridge and model.hf_init are required to load or export HF weights")
    return get_bridge(model_section["bridge"], model_section["hf_init"])


def load_hf_weights_hook(model_section: dict[str, Any]) -> Callable:
    """``pre_wrap_hook`` that loads the HF checkpoint ``model.hf_init`` into every model chunk."""

    def hook(model_chunks):
        from megatron_speech import compat

        bridge = get_bridge_for_model_section(model_section)
        compat.print_rank_0(f"> loading HF weights from {bridge.hf_dir} ({bridge.name} bridge)")
        for chunk in model_chunks:
            bridge.load_into(chunk, model_section)
        return model_chunks

    return hook

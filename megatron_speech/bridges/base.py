"""Base class of model bridges (HF checkpoint <-> Megatron speech LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from megatron_speech.bridges.convert import export_hf_state_dict, load_hf_weights
from megatron_speech.bridges.hf_io import (
    SafetensorsReader,
    copy_non_weight_files,
    load_hf_config,
    resolve_hf_path,
    save_hf_state_dict,
)
from megatron_speech.bridges.mapping import MappingTable


class SpeechLLMBridge:
    """A bridge knows, for one HF architecture:

    * the Megatron CLI arguments of the LLM (``megatron_args``),
    * the component configs (``speech_defaults``: encoder, projector, audio token id),
    * the weight mapping (``mapping_table``).

    Loading and export are generic over the mapping table.
    """

    name: str = ""

    def __init__(self, hf_path: str | Path):
        self.hf_dir = resolve_hf_path(str(hf_path))
        self.hf_config = load_hf_config(self.hf_dir)

    def megatron_args(self) -> dict[str, Any]:
        raise NotImplementedError

    def speech_defaults(self) -> dict[str, Any]:
        raise NotImplementedError

    def mapping_table(self, speech: dict[str, Any]) -> MappingTable:
        raise NotImplementedError

    def hf_torch_dtype(self) -> torch.dtype:
        name = self.hf_config.get("dtype") or self.hf_config.get("torch_dtype") or "bfloat16"
        return getattr(torch, name)

    # -- generic -----------------------------------------------------------------------------
    def load_into(self, model: torch.nn.Module, speech: dict[str, Any]) -> None:
        load_hf_weights(model, self.mapping_table(speech), SafetensorsReader(self.hf_dir))

    def export_state_dict(self, model: torch.nn.Module, speech: dict[str, Any]) -> dict[str, torch.Tensor]:
        return export_hf_state_dict(model, self.mapping_table(speech), dtype=self.hf_torch_dtype())

    def save_hf(self, state_dict: dict[str, torch.Tensor], out_dir: str | Path) -> None:
        """Write weights plus config/tokenizer/processor files copied from the source checkpoint."""
        copy_non_weight_files(self.hf_dir, Path(out_dir))
        save_hf_state_dict(state_dict, Path(out_dir))

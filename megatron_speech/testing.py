"""Building a speech LLM outside of ``pretrain()`` (tests, conversion and export scripts)."""

from __future__ import annotations

import copy
import os
from typing import Any

import torch

from megatron_speech import compat

QWEN3_ASR_HF = os.environ.get("MS_QWEN3_ASR_HF", "Qwen/Qwen3-ASR-1.7B-hf")

# Arguments Megatron validation requires even when no training loop runs.
_MINIMAL_TRAINING_ARGS = {
    "bf16": True,
    "global-batch-size": 1,
    "train-iters": 1,
    "lr": 1.0e-5,
    "seq-length": 4096,
}


def make_run_config(model_section: dict[str, Any], tensor_parallel: int = 1, megatron: dict | None = None) -> dict:
    return {
        "model": {"type": "speech_llm", **copy.deepcopy(model_section)},
        "data": {},
        "megatron": {**_MINIMAL_TRAINING_ARGS, "tensor-model-parallel-size": tensor_parallel, **(megatron or {})},
    }


def reset_megatron_globals() -> None:
    from megatron.training.global_vars import unset_global_variables

    unset_global_variables()


def setup_speech_llm(run_cfg: dict[str, Any]):
    """Initialize distributed state and parse Megatron args; return ``(args, model_cfg)``."""
    from megatron_speech.training.speech_llm_recipe import SpeechLLMRecipe

    tp = int(run_cfg["megatron"].get("tensor-model-parallel-size", 1))
    compat.initialize_model_parallel_for_tests(tensor_model_parallel_size=tp)
    reset_megatron_globals()
    recipe = SpeechLLMRecipe(run_cfg)
    args = compat.parse_megatron_args(recipe.megatron_arg_dict())
    model_cfg = recipe.build_model_config(args)
    return args, model_cfg


def build_speech_llm_for_eval(model_cfg, load_hf: bool = True, apply_peft: bool = True, dtype=torch.bfloat16):
    """Build the MIMO speech LLM on the current GPU with HF weights, without DDP wrapping."""
    from megatron_speech.bridges import get_bridge_for_model_section
    from megatron_speech.models.speech_llm import build_speech_llm
    from megatron_speech.peft import apply_peft as _apply_peft

    model = build_speech_llm(model_cfg)
    if load_hf:
        get_bridge_for_model_section(model_cfg.speech).load_into(model, model_cfg.speech)
    if apply_peft:
        _apply_peft(model, model_cfg.speech.get("peft") or {})
    model = model.cuda().to(dtype)
    model.eval()
    return model


def gather_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """Concatenate vocab-parallel logits ``[..., V/tp]`` from all TP ranks into ``[..., V]``."""
    if compat.get_tensor_model_parallel_world_size() == 1:
        return logits
    from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region

    return gather_from_tensor_model_parallel_region(logits)

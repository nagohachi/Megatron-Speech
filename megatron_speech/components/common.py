"""Helpers shared by component implementations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig

# Parallelism and precision settings that every component inherits from the LLM config.
_INHERITED_FIELDS = (
    "tensor_model_parallel_size",
    "params_dtype",
    "bf16",
    "fp16",
    "use_cpu_initialization",
    "perform_initialization",
    "gradient_accumulation_fusion",
    "deterministic_mode",
    "attention_backend",
    "init_method_std",
    "masked_softmax_fusion",
    "bias_dropout_fusion",
    "use_te_rng_tracker",
)

ACTIVATIONS = {
    "gelu": F.gelu,  # exact (erf) GELU, as torch.nn.GELU() and HF ACT2FN["gelu"]
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "relu": F.relu,
}


def make_component_transformer_config(llm_config: TransformerConfig, **arch: Any) -> TransformerConfig:
    """Build the ``TransformerConfig`` of a component (encoder, projector, ...).

    The component inherits tensor parallelism and precision from the LLM, and runs without
    sequence, context or pipeline parallelism (the component sits on the first pipeline stage
    and sees whole sequences). ``arch`` sets the architecture fields.
    """
    kwargs = {name: getattr(llm_config, name) for name in _INHERITED_FIELDS if hasattr(llm_config, name)}
    kwargs.update(
        sequence_parallel=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        tp_comm_overlap=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs.update(arch)
    activation = kwargs.get("activation_func")
    if isinstance(activation, str):
        kwargs["activation_func"] = ACTIVATIONS[activation]
    return TransformerConfig(**kwargs)


def flatten_valid_positions(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Select rows ``index`` of ``x`` viewed as ``[-1, hidden]``."""
    return torch.index_select(x.reshape(-1, x.shape[-1]), 0, index)

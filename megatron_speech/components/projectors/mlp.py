"""Two-layer MLP projector from encoder outputs to the LLM embedding space.

Uses Megatron Core's ``MultimodalProjector`` (tensor-parallel fc1/fc2 with Transformer Engine).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.mlp import MLPSubmodules

from megatron_speech.components.common import make_component_transformer_config


@dataclass
class MLPProjectorConfig:
    input_size: int
    hidden_size: int
    output_size: int
    activation: str = "gelu"
    bias: bool = True


class MLPProjector(MegatronModule):
    """``output = fc2(act(fc1(x)))`` applied to ``[num_tokens, input_size]``."""

    def __init__(self, projector_config: MLPProjectorConfig, llm_config: TransformerConfig, pg_collection=None):
        config = make_component_transformer_config(
            llm_config,
            num_layers=1,
            hidden_size=projector_config.output_size,
            ffn_hidden_size=projector_config.hidden_size,
            # Unused by the MLP; TransformerConfig only requires it to be a multiple of TP.
            num_attention_heads=llm_config.tensor_model_parallel_size,
            add_bias_linear=projector_config.bias,
            gated_linear_unit=False,
            activation_func=projector_config.activation,
            bias_activation_fusion=False,
        )
        super().__init__(config=config)
        self.projector = MultimodalProjector(
            config=config,
            submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
            projector_type="mlp",
            input_size=projector_config.input_size,
            pg_collection=pg_collection,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_states.unsqueeze(1)).squeeze(1)

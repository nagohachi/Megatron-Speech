"""Two-layer MLP projector from encoder outputs to the LLM embedding space."""

from dataclasses import dataclass

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.mlp import MLPSubmodules

from megatron_speech.components.transformer_config import build_component_transformer_config


@dataclass
class MLPProjectorConfig:
    """Sizes of ``MLPProjector``."""

    input_size: int
    hidden_size: int
    output_size: int


class MLPProjector(MegatronModule):
    """``fc2(gelu(fc1(x)))`` on ``[num_tokens, input_size]``, with biases and exact GELU.

    fc1 is split over its output features and fc2 over its input features across the tensor
    parallel ranks (Megatron Core's ``MultimodalProjector`` with Transformer Engine linears).
    """

    def __init__(
        self,
        projector_config: MLPProjectorConfig,
        llm_config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Build fc1 and fc2 with the tensor parallelism and precision of ``llm_config``."""
        config = build_component_transformer_config(
            llm_config,
            activation_func=torch.nn.functional.gelu,
            num_layers=1,
            hidden_size=projector_config.output_size,
            ffn_hidden_size=projector_config.hidden_size,
            # Unused by the MLP; TransformerConfig only requires a multiple of the TP size.
            num_attention_heads=llm_config.tensor_model_parallel_size,
            add_bias_linear=True,
            gated_linear_unit=False,
            # The fused bias-GELU kernel uses the tanh approximation of GELU.
            bias_activation_fusion=False,
        )
        super().__init__(config=config)
        self.projector = MultimodalProjector(
            config=config,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            projector_type="mlp",
            input_size=projector_config.input_size,
            pg_collection=pg_collection,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project ``[num_tokens, input_size]`` to ``[num_tokens, output_size]``."""
        # Megatron's MLP takes [sequence, batch, hidden].
        return self.projector(hidden_states.unsqueeze(1)).squeeze(1)

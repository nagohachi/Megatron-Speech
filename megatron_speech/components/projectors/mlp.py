from dataclasses import dataclass
from itertools import pairwise

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer import MegatronModule, TransformerConfig

from megatron_speech.components.activations import ACTIVATIONS
from megatron_speech.components.transformer_config import build_component_transformer_config


@dataclass
class MLPProjectorConfig:
    """Shape and activation of ``MLPProjector``.

    ``num_layers`` linear layers map ``input_size`` features to ``output_size`` features. The
    features between two layers number ``hidden_size``, which a single layer does not use.
    ``activation`` (a key of ``ACTIVATIONS``) follows every layer but the last.
    """

    input_size: int
    output_size: int
    num_layers: int = 2
    hidden_size: int | None = None
    activation: str = "gelu"


class MLPProjector(MegatronModule):
    """Linear layers with biases and an activation between them, on ``[num_tokens, input_size]``.

    Layers alternate between splitting their output features (column parallel) and their input
    features (row parallel) across the tensor parallel ranks, so the activation after a
    column-parallel layer runs on split features and only row-parallel layers communicate.
    A column-parallel last layer gathers its output.
    """

    def __init__(
        self,
        projector_config: MLPProjectorConfig,
        llm_config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Build the layers with the tensor parallelism and precision of ``llm_config``."""
        num_layers = projector_config.num_layers
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if num_layers > 1 and projector_config.hidden_size is None:
            raise ValueError("hidden_size is required when num_layers > 1")
        sizes = [
            projector_config.input_size,
            *[projector_config.hidden_size] * (num_layers - 1),
            projector_config.output_size,
        ]
        tp_size = llm_config.tensor_model_parallel_size
        if pg_collection is not None and pg_collection.tp.size() != tp_size:
            raise ValueError(
                f"pg_collection.tp has {pg_collection.tp.size()} ranks, but "
                f"llm_config.tensor_model_parallel_size is {tp_size}"
            )
        # Column-parallel layers (index 0, 2, ...) split their output features over the ranks.
        for size in sizes[1::2]:
            if size % tp_size:
                raise ValueError(
                    f"{size} features cannot be split over {tp_size} tensor parallel ranks; "
                    "hidden_size, and output_size when num_layers is odd, must be divisible by "
                    "the TP size"
                )
        config = build_component_transformer_config(
            llm_config,
            # The layers below are built directly; TransformerConfig only requires these
            # fields, with num_attention_heads a multiple of the TP size.
            num_layers=1,
            hidden_size=projector_config.output_size,
            num_attention_heads=llm_config.tensor_model_parallel_size,
        )
        super().__init__(config=config)
        self.activation_func = ACTIVATIONS[projector_config.activation]
        tp_group = pg_collection.tp if pg_collection is not None else None
        self.layers = torch.nn.ModuleList()
        for index, (in_features, out_features) in enumerate(pairwise(sizes)):
            if index % 2 == 0:
                layer = TEColumnParallelLinear(
                    in_features,
                    out_features,
                    config=config,
                    init_method=config.init_method,
                    gather_output=False,
                    bias=True,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_group=tp_group,
                )
            else:
                layer = TERowParallelLinear(
                    in_features,
                    out_features,
                    config=config,
                    init_method=config.init_method,
                    bias=True,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_group=tp_group,
                )
            self.layers.append(layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project ``[num_tokens, input_size]`` to ``[num_tokens, output_size]``."""
        for index, layer in enumerate(self.layers):
            hidden_states, _ = layer(hidden_states)
            if index < len(self.layers) - 1:
                hidden_states = self.activation_func(hidden_states)
        if len(self.layers) % 2 == 1:
            # The last layer is column parallel; its output features are split.
            hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
        return hidden_states

from dataclasses import dataclass

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer import MegatronModule, TransformerConfig

from megatron_speech.components.activations import ACTIVATIONS
from megatron_speech.components.transformer_config import build_component_transformer_config


@dataclass
class QFormerProjectorConfig:
    """Shape of ``QFormerProjector``.

    Fields that Hugging Face's ``Blip2QFormerConfig`` also has, with its names:

    - ``input_size``: ``encoder_hidden_size`` (the encoder's output size)
    - ``hidden_size``: ``hidden_size``
    - ``num_layers``: ``num_hidden_layers``
    - ``num_attention_heads``: ``num_attention_heads``
    - ``ffn_hidden_size``: ``intermediate_size``
    - ``cross_attention_frequency``: ``cross_attention_frequency``
    - ``activation``: ``hidden_act``
    - ``layernorm_epsilon``: ``layer_norm_eps``

    ``output_size`` is the LLM's hidden size.

    Each utterance is cut into windows of ``window_size`` encoder frames, and ``num_queries``
    learned queries read each window. ``window_size=None`` makes the whole utterance one window,
    as BLIP-2 does for an image.
    """

    input_size: int
    output_size: int
    hidden_size: int = 768
    num_layers: int = 2
    num_attention_heads: int = 12
    ffn_hidden_size: int = 3072
    cross_attention_frequency: int = 1
    num_queries: int = 1
    window_size: int | None = 17
    activation: str = "gelu"
    layernorm_epsilon: float = 1e-12


def build_column_parallel_linear(
    input_size: int,
    output_size: int,
    config: TransformerConfig,
    tp_group: torch.distributed.ProcessGroup | None,
) -> TEColumnParallelLinear:
    """Linear layer with a bias whose output features are split over the TP ranks."""
    return TEColumnParallelLinear(
        input_size,
        output_size,
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )


def build_row_parallel_linear(
    input_size: int,
    output_size: int,
    config: TransformerConfig,
    tp_group: torch.distributed.ProcessGroup | None,
) -> TERowParallelLinear:
    """Linear layer with a bias whose input features are split over the TP ranks."""
    return TERowParallelLinear(
        input_size,
        output_size,
        config=config,
        init_method=config.init_method,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )


class QFormerAttention(MegatronModule):
    """Multi-head attention followed by a residual connection and LayerNorm (BERT order).

    The heads are split over the TP ranks. Self-attention reads keys and values from the
    queries themselves; cross-attention reads them from the encoder frames.
    """

    def __init__(
        self,
        config: TransformerConfig,
        key_value_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        """Build the projections; keys and values come from inputs of ``key_value_size``."""
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.num_local_heads = config.num_attention_heads // config.tensor_model_parallel_size
        self.head_size = config.kv_channels
        self.query = build_column_parallel_linear(hidden_size, hidden_size, config, tp_group)
        self.key = build_column_parallel_linear(key_value_size, hidden_size, config, tp_group)
        self.value = build_column_parallel_linear(key_value_size, hidden_size, config, tp_group)
        self.dense = build_row_parallel_linear(hidden_size, hidden_size, config, tp_group)
        self.layernorm = torch.nn.LayerNorm(
            hidden_size, eps=config.layernorm_epsilon, dtype=config.params_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend from ``[batch, queries, hidden]`` to ``[batch, keys, key_value_size]``.

        ``key_mask`` is ``[batch, keys]`` and is True for the keys that take part.
        """
        batch, num_queries, _ = hidden_states.shape
        num_keys = key_value_states.shape[1]
        query, _ = self.query(hidden_states)
        key, _ = self.key(key_value_states)
        value, _ = self.value(key_value_states)
        # [batch, length, local hidden] -> [batch, local heads, length, head size]
        heads = (self.num_local_heads, self.head_size)
        query = query.view(batch, num_queries, *heads).transpose(1, 2)
        key = key.view(batch, num_keys, *heads).transpose(1, 2)
        value = value.view(batch, num_keys, *heads).transpose(1, 2)
        attn_mask = None if key_mask is None else key_mask[:, None, None, :]
        context = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )
        context = context.transpose(1, 2).reshape(
            batch, num_queries, self.num_local_heads * self.head_size
        )
        output, _ = self.dense(context)
        return self.layernorm(output + hidden_states)


class QFormerLayer(MegatronModule):
    """Self-attention over the queries, optional cross-attention to the frames, and an MLP."""

    def __init__(
        self,
        config: TransformerConfig,
        projector_config: QFormerProjectorConfig,
        has_cross_attention: bool,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        """Build the attention blocks and the MLP of one layer."""
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.self_attention = QFormerAttention(config, hidden_size, tp_group)
        self.cross_attention = (
            QFormerAttention(config, projector_config.input_size, tp_group)
            if has_cross_attention
            else None
        )
        self.activation_func = ACTIVATIONS[projector_config.activation]
        self.fc1 = build_column_parallel_linear(
            hidden_size, config.ffn_hidden_size, config, tp_group
        )
        self.fc2 = build_row_parallel_linear(config.ffn_hidden_size, hidden_size, config, tp_group)
        self.mlp_layernorm = torch.nn.LayerNorm(
            hidden_size, eps=config.layernorm_epsilon, dtype=config.params_dtype
        )

    def forward(
        self, queries: torch.Tensor, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """Update ``[windows, queries, hidden]`` from ``[windows, frames, input_size]``."""
        queries = self.self_attention(queries, queries, None)
        if self.cross_attention is not None:
            queries = self.cross_attention(queries, frames, frame_mask)
        hidden, _ = self.fc1(queries)
        output, _ = self.fc2(self.activation_func(hidden))
        return self.mlp_layernorm(output + queries)


class QFormerProjector(MegatronModule):
    """Q-Former that reads windows of encoder frames with learned queries (BLIP-2, SALMONN).

    The layers match Hugging Face's ``Blip2QFormerModel`` without its text input: a LayerNorm
    on the queries, then layers of self-attention, cross-attention (in layers whose index is a
    multiple of ``cross_attention_frequency``) and an MLP, each followed by a residual
    connection and LayerNorm. A linear layer maps the query outputs to ``output_size``.
    """

    def __init__(
        self,
        projector_config: QFormerProjectorConfig,
        llm_config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Build the queries and layers with the tensor parallelism and precision of the LLM."""
        tp_size = llm_config.tensor_model_parallel_size
        if pg_collection is not None and pg_collection.tp.size() != tp_size:
            raise ValueError(
                f"pg_collection.tp has {pg_collection.tp.size()} ranks, but "
                f"llm_config.tensor_model_parallel_size is {tp_size}"
            )
        split_sizes = {
            "num_attention_heads": projector_config.num_attention_heads,
            "ffn_hidden_size": projector_config.ffn_hidden_size,
            "output_size": projector_config.output_size,
        }
        for name, size in split_sizes.items():
            if size % tp_size:
                raise ValueError(
                    f"{name}={size} cannot be split over {tp_size} tensor parallel ranks"
                )
        if projector_config.hidden_size % projector_config.num_attention_heads:
            raise ValueError(
                f"hidden_size={projector_config.hidden_size} is not a multiple of "
                f"num_attention_heads={projector_config.num_attention_heads}"
            )
        counts = {
            "num_layers": projector_config.num_layers,
            "num_queries": projector_config.num_queries,
            "cross_attention_frequency": projector_config.cross_attention_frequency,
            # None makes the whole utterance one window.
            "window_size": 1
            if projector_config.window_size is None
            else projector_config.window_size,
        }
        for name, count in counts.items():
            if count < 1:
                raise ValueError(f"{name} must be at least 1, got {count}")
        config = build_component_transformer_config(
            llm_config,
            num_layers=projector_config.num_layers,
            hidden_size=projector_config.hidden_size,
            num_attention_heads=projector_config.num_attention_heads,
            ffn_hidden_size=projector_config.ffn_hidden_size,
            layernorm_epsilon=projector_config.layernorm_epsilon,
        )
        super().__init__(config=config)
        self.window_size = projector_config.window_size
        tp_group = pg_collection.tp if pg_collection is not None else None
        self.query_tokens = torch.nn.Parameter(
            torch.empty(
                projector_config.num_queries,
                projector_config.hidden_size,
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            # Hugging Face starts from zeros, but equal queries get equal gradients and stay
            # equal; LAVIS's BLIP-2 draws them from a normal distribution, as done here.
            config.init_method(self.query_tokens)
        self.layernorm = torch.nn.LayerNorm(
            projector_config.hidden_size,
            eps=projector_config.layernorm_epsilon,
            dtype=config.params_dtype,
        )
        self.layers = torch.nn.ModuleList(
            QFormerLayer(
                config,
                projector_config,
                has_cross_attention=index % projector_config.cross_attention_frequency == 0,
                tp_group=tp_group,
            )
            for index in range(projector_config.num_layers)
        )
        self.projection = build_column_parallel_linear(
            projector_config.hidden_size, projector_config.output_size, config, tp_group
        )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Project the encoder frames of the utterances that ``cu_seqlens`` delimits.

        ``hidden_states`` is ``[num_frames, input_size]``, utterance after utterance, and
        ``cu_seqlens`` the ``[num_utterances + 1]`` cumulative frame counts. Each utterance is
        cut into windows of ``window_size`` frames (the last one may be shorter). Returns
        ``[num_windows * num_queries, output_size]``, utterance by utterance and window by
        window. Counting the windows reads the utterance lengths on the host.
        """
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if self.window_size is not None:
            window_size = self.window_size
        else:
            window_size = max(int(lengths.max()), 1) if lengths.numel() else 1
        num_windows = (lengths + window_size - 1) // window_size
        window_utterance = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=lengths.device), num_windows
        )
        first_window = torch.cumsum(num_windows, dim=0) - num_windows
        window_offset = (
            torch.arange(window_utterance.numel(), device=lengths.device)
            - first_window[window_utterance]
        ) * window_size
        # Frame offsets within the utterance, [windows, window_size].
        offsets = window_offset[:, None] + torch.arange(window_size, device=lengths.device)
        frame_mask = offsets < lengths[window_utterance][:, None]
        frame_index = (cu_seqlens[window_utterance][:, None] + offsets).clamp(
            max=max(hidden_states.shape[0] - 1, 0)
        )
        frames = hidden_states[frame_index]

        queries = self.layernorm(self.query_tokens)
        queries = queries.expand(frames.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, frames, frame_mask)
        output, _ = self.projection(queries)
        output = gather_from_tensor_model_parallel_region(output)
        return output.reshape(-1, output.shape[-1])

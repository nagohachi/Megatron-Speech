import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron_speech.components.activations import ACTIVATIONS
from megatron_speech.components.transformer_config import build_component_transformer_config

MLP_SUBMODULES = MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear)


@dataclass
class AdaLNMLPHeadConfig:
    """Shape and conditions of ``AdaLNMLPHead``.

    ``num_layers`` AdaLN layers of width ``hidden_size`` map a ``latent_size`` latent to a
    ``latent_size`` output. The MLP of each layer has ``ffn_hidden_size`` features, applies
    ``activation`` (a key of ``ACTIVATIONS``), and multiplies it by a second projection when
    ``gated_linear_unit`` is set (SwiGLU for ``"silu"``). ``scalar_condition_names`` name the
    scalars, such as the diffusion time, that condition the head besides the LLM hidden state;
    each is embedded by ``frequency_embedding_size`` sinusoidal features, whose frequencies fall
    from 1 to about ``1 / max_period``, and a two-layer MLP.
    """

    latent_size: int
    hidden_size: int
    ffn_hidden_size: int
    num_layers: int
    scalar_condition_names: tuple[str, ...] = ("time",)
    activation: str = "silu"
    gated_linear_unit: bool = True
    frequency_embedding_size: int = 256
    max_period: float = 10000.0
    norm_eps: float = 1e-5


def _build_column_parallel_linear(
    config: TransformerConfig,
    in_features: int,
    out_features: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    tp_group: torch.distributed.ProcessGroup | None,
) -> TEColumnParallelLinear:
    """Build a linear layer without bias whose output features the TP ranks split."""
    return TEColumnParallelLinear(
        in_features,
        out_features,
        config=config,
        init_method=init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )


def _normalize_by_rms(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last dim by its root mean square in fp32, as Haru-4B's head does.

    Transformer Engine's RMSNorm fails on an empty input, which a batch without audio gives.
    """
    normalized = torch.nn.functional.rms_norm(
        hidden_states.float(), (hidden_states.shape[-1],), eps=eps
    )
    return normalized.to(hidden_states.dtype)


class AdaLNLayer(MegatronModule):
    """RMSNorm, AdaLN shift and scale, an MLP, and a gated residual connection."""

    def __init__(
        self, config: TransformerConfig, tp_group: torch.distributed.ProcessGroup | None
    ) -> None:
        """Build the layer; the modulation starts at zero, which makes the layer the identity."""
        super().__init__(config=config)
        self.norm_weight = torch.nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.params_dtype)
        )
        self.modulation = _build_column_parallel_linear(
            config, config.hidden_size, 3 * config.hidden_size, torch.nn.init.zeros_, tp_group
        )
        self.mlp = MLP(
            config, MLP_SUBMODULES, ffn_hidden_size=config.ffn_hidden_size, tp_group=tp_group
        )

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Update ``[..., hidden_size]`` hidden states with the ``condition`` after SiLU."""
        modulation = gather_from_tensor_model_parallel_region(self.modulation(condition)[0])
        shift, scale, gate = modulation.chunk(3, dim=-1)
        normalized = (
            _normalize_by_rms(hidden_states, self.config.layernorm_epsilon) * self.norm_weight
        )
        mlp_output, _ = self.mlp(normalized * (1 + scale) + shift)
        return hidden_states + gate * mlp_output


class AdaLNMLPHead(MegatronModule):
    """MLP head that predicts one latent from a noisy latent and the LLM hidden state.

    The structure is that of the MeanFlow head of Haru-4B and of the diffusion head of LatentLM
    (arXiv 2412.08635, AdaLN-Zero after DiT). The LLM hidden state is projected and added to
    the embeddings of the scalar conditions; after SiLU, this condition sets the shift, scale
    and gate of every AdaLN layer and the shift and scale before the output projection. The
    linear layers have no bias, as in Haru-4B.

    The residual stream is whole on every tensor parallel rank. The MLPs split their features
    over the ranks; the other linear layers split their output features and gather them.
    """

    def __init__(
        self,
        head_config: AdaLNMLPHeadConfig,
        llm_config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Build the layers with the tensor parallelism and precision of ``llm_config``.

        The output projection and the modulations start at zero (AdaLN-Zero), so a new head
        outputs zeros.
        """
        tp_size = llm_config.tensor_model_parallel_size

        def check_config() -> None:
            """Raise ValueError for a config that cannot be built on ``tp_size`` ranks."""
            if pg_collection is not None and pg_collection.tp.size() != tp_size:
                raise ValueError(
                    f"pg_collection.tp has {pg_collection.tp.size()} ranks, but "
                    f"llm_config.tensor_model_parallel_size is {tp_size}"
                )
            for name in ("latent_size", "hidden_size", "ffn_hidden_size"):
                if getattr(head_config, name) % tp_size:
                    raise ValueError(
                        f"{name}={getattr(head_config, name)} cannot be split over "
                        f"{tp_size} tensor parallel ranks"
                    )
            if head_config.frequency_embedding_size % 2:
                raise ValueError(
                    "frequency_embedding_size must be even (cosine and sine features), got "
                    f"{head_config.frequency_embedding_size}"
                )
            names = head_config.scalar_condition_names
            if len(set(names)) != len(names):
                raise ValueError(f"scalar_condition_names has duplicates: {names}")

        check_config()
        config = build_component_transformer_config(
            llm_config,
            num_layers=head_config.num_layers,
            hidden_size=head_config.hidden_size,
            ffn_hidden_size=head_config.ffn_hidden_size,
            # The head has no attention; TransformerConfig only requires num_attention_heads to
            # be a multiple of the TP size.
            num_attention_heads=tp_size,
            layernorm_epsilon=head_config.norm_eps,
            add_bias_linear=False,
            gated_linear_unit=head_config.gated_linear_unit,
            activation_func=ACTIVATIONS[head_config.activation],
        )
        super().__init__(config=config)
        self.head_config = head_config
        tp_group = pg_collection.tp if pg_collection is not None else None
        hidden_size = head_config.hidden_size
        self.latent_projection = _build_column_parallel_linear(
            config, head_config.latent_size, hidden_size, config.init_method, tp_group
        )
        self.condition_projection = _build_column_parallel_linear(
            config, llm_config.hidden_size, hidden_size, config.init_method, tp_group
        )
        scalar_embedding_config = replace(
            config,
            ffn_hidden_size=hidden_size,
            gated_linear_unit=False,
            activation_func=torch.nn.functional.silu,
        )
        self.scalar_embeddings = torch.nn.ModuleDict(
            {
                name: MLP(
                    scalar_embedding_config,
                    MLP_SUBMODULES,
                    input_size=head_config.frequency_embedding_size,
                    ffn_hidden_size=hidden_size,
                    tp_group=tp_group,
                )
                for name in head_config.scalar_condition_names
            }
        )
        num_frequencies = head_config.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(head_config.max_period) * torch.arange(num_frequencies) / num_frequencies
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.layers = torch.nn.ModuleList(
            AdaLNLayer(config, tp_group) for _ in range(head_config.num_layers)
        )
        self.final_modulation = _build_column_parallel_linear(
            config, hidden_size, 2 * hidden_size, torch.nn.init.zeros_, tp_group
        )
        self.output_projection = _build_column_parallel_linear(
            config, hidden_size, head_config.latent_size, torch.nn.init.zeros_, tp_group
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        hidden_states: torch.Tensor,
        scalar_conditions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict ``[num_positions, num_samples, latent_size]`` from as many noisy latents.

        ``hidden_states`` is ``[num_positions, llm_hidden_size]``; every sample of a position
        shares its projection. ``scalar_conditions`` maps each name of
        ``scalar_condition_names`` to ``[num_positions, num_samples]`` values.
        """
        expected_names = set(self.head_config.scalar_condition_names)
        if set(scalar_conditions) != expected_names:
            raise ValueError(
                f"scalar_conditions has {sorted(scalar_conditions)}, but the head is "
                f"conditioned on {sorted(expected_names)}"
            )
        dtype = self.config.params_dtype
        condition = gather_from_tensor_model_parallel_region(
            self.condition_projection(hidden_states.to(dtype))[0]
        ).unsqueeze(1)
        for name, embedding in self.scalar_embeddings.items():
            phases = scalar_conditions[name].float().unsqueeze(-1) * self.frequencies
            features = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
            condition = condition + embedding(features.to(dtype))[0]
        condition = torch.nn.functional.silu(condition)
        hidden = gather_from_tensor_model_parallel_region(
            self.latent_projection(noisy_latents.to(dtype))[0]
        )
        for layer in self.layers:
            hidden = layer(hidden, condition)
        modulation = gather_from_tensor_model_parallel_region(self.final_modulation(condition)[0])
        shift, scale = modulation.chunk(2, dim=-1)
        normalized = _normalize_by_rms(hidden, self.head_config.norm_eps)
        output = self.output_projection(normalized * (1 + scale) + shift)[0]
        return gather_from_tensor_model_parallel_region(output)

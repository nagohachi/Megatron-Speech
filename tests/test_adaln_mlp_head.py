import logging
import math
from dataclasses import dataclass, replace
from typing import Self

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.activations import ACTIVATIONS
from megatron_speech.components.heads.adaln_mlp import AdaLNMLPHead, AdaLNMLPHeadConfig
from tests.numerical_error import TOLERANCES, WITH_DTYPES, calc_relative_error
from tests.tensor_parallel_shard import take_shard

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

# The head's bf16 gradient errors reach 3.6e-2 over 20 random seeds at TP=1 and TP=2, above
# the shared 2e-2.
GRADIENT_TOLERANCES = TOLERANCES | {torch.bfloat16: 5e-2}
LLM_HIDDEN_SIZE = 48
NUM_POSITIONS = 5
NUM_SAMPLES = 3
BASE_CONFIG = AdaLNMLPHeadConfig(
    latent_size=16,
    hidden_size=64,
    ffn_hidden_size=96,
    num_layers=2,
    frequency_embedding_size=32,
)
HEAD_CONFIGS = [
    # Haru-4B's MeanFlow head: SwiGLU and four scalar conditions.
    replace(
        BASE_CONFIG,
        scalar_condition_names=("time", "interval_embed", "interval", "guidance"),
        activation="silu",
        gated_linear_unit=True,
    ),
    # A DDPM head conditioned on the time only, with a GELU MLP.
    replace(BASE_CONFIG, activation="gelu", gated_linear_unit=False),
]
WITH_HEAD_CONFIGS = pytest.mark.parametrize(
    "head_config",
    HEAD_CONFIGS,
    ids=lambda config: f"{config.activation}-gated{config.gated_linear_unit}",
)


class ReferenceMLP(torch.nn.Module):
    """Two linear layers without bias, with Megatron's ``MLP`` parameter names.

    When gated, ``linear_fc1`` outputs the gate and then the value, as Megatron's ``MLP`` does.
    """

    def __init__(
        self,
        input_size: int,
        ffn_hidden_size: int,
        output_size: int,
        activation: str,
        gated_linear_unit: bool,
    ) -> None:
        """Build the layers with PyTorch's default initialization."""
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(
            input_size, ffn_hidden_size * (2 if gated_linear_unit else 1), bias=False
        )
        self.linear_fc2 = torch.nn.Linear(ffn_hidden_size, output_size, bias=False)
        self.activation_func = ACTIVATIONS[activation]
        self.gated_linear_unit = gated_linear_unit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the layers and the activation between them."""
        intermediate = self.linear_fc1(hidden_states)
        if self.gated_linear_unit:
            gate, value = intermediate.chunk(2, dim=-1)
            intermediate = self.activation_func(gate) * value
        else:
            intermediate = self.activation_func(intermediate)
        return self.linear_fc2(intermediate)


class ReferenceLayer(torch.nn.Module):
    """AdaLN layer in plain PyTorch."""

    def __init__(self, head_config: AdaLNMLPHeadConfig) -> None:
        """Build the norm, the modulation and the MLP."""
        super().__init__()
        hidden_size = head_config.hidden_size
        self.norm_weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.norm_eps = head_config.norm_eps
        self.modulation = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.mlp = ReferenceMLP(
            hidden_size,
            head_config.ffn_hidden_size,
            hidden_size,
            head_config.activation,
            head_config.gated_linear_unit,
        )

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply the layer as ``AdaLNLayer`` does."""
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized_states = torch.nn.functional.rms_norm(
            hidden_states, (hidden_states.shape[-1],), self.norm_weight, self.norm_eps
        )
        return hidden_states + gate * self.mlp(normalized_states * (1 + scale) + shift)


class ReferenceHead(torch.nn.Module):
    """AdaLN MLP head in plain PyTorch, written after Haru-4B's ``MeanFlowHead``."""

    def __init__(self, head_config: AdaLNMLPHeadConfig) -> None:
        """Build the layers with random weights, the modulations included."""
        super().__init__()
        hidden_size = head_config.hidden_size
        self.head_config = head_config
        self.latent_projection = torch.nn.Linear(head_config.latent_size, hidden_size, bias=False)
        self.condition_projection = torch.nn.Linear(LLM_HIDDEN_SIZE, hidden_size, bias=False)
        self.scalar_embeddings = torch.nn.ModuleDict(
            {
                name: ReferenceMLP(
                    head_config.frequency_embedding_size, hidden_size, hidden_size, "silu", False
                )
                for name in head_config.scalar_condition_names
            }
        )
        self.layers = torch.nn.ModuleList(
            ReferenceLayer(head_config) for _ in range(head_config.num_layers)
        )
        self.final_modulation = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.output_projection = torch.nn.Linear(hidden_size, head_config.latent_size, bias=False)
        for layer in self.layers:
            torch.nn.init.normal_(layer.norm_weight, mean=1.0, std=0.1)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        hidden_states: torch.Tensor,
        scalar_conditions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the head as ``AdaLNMLPHead`` does."""
        num_frequencies = self.head_config.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(self.head_config.max_period)
            * torch.arange(num_frequencies, device="cuda")
            / num_frequencies
        )
        condition = self.condition_projection(hidden_states).unsqueeze(1)
        for name, embedding in self.scalar_embeddings.items():
            phases = scalar_conditions[name].unsqueeze(-1) * frequencies
            condition = condition + embedding(torch.cat([phases.cos(), phases.sin()], dim=-1))
        condition = torch.nn.functional.silu(condition)
        hidden = self.latent_projection(noisy_latents)
        for layer in self.layers:
            hidden = layer(hidden, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        normalized_hidden = torch.nn.functional.rms_norm(
            hidden, (hidden.shape[-1],), eps=self.head_config.norm_eps
        )
        return self.output_projection(normalized_hidden * (1 + scale) + shift)


def take_reference_shards(
    reference_tensors: dict[str, torch.Tensor], head_config: AdaLNMLPHeadConfig
) -> dict[str, torch.Tensor]:
    """Cut out this rank's part of each reference parameter (or its gradient), by name.

    The MLPs split ``linear_fc1`` by its output and ``linear_fc2`` by its input features; a
    gated ``linear_fc1`` holds this rank's part of the gate and then of the value. The norms
    are whole on every rank, and the other linear layers split their output features.
    """
    shards = {}
    for name, tensor in reference_tensors.items():
        if name.startswith("layers.") and name.endswith("linear_fc1.weight"):
            if head_config.gated_linear_unit:
                gate, value = tensor.chunk(2, dim=0)
                shards[name] = torch.cat([take_shard(gate, 0), take_shard(value, 0)])
            else:
                shards[name] = take_shard(tensor, 0)
        elif name.endswith("linear_fc2.weight"):
            shards[name] = take_shard(tensor, 1)
        elif name.endswith("norm_weight"):
            shards[name] = tensor
        else:
            shards[name] = take_shard(tensor, 0)
    return shards


@dataclass
class HeadInputs:
    """Inputs of the head: noisy latents, LLM hidden states and scalar conditions."""

    noisy_latents: torch.Tensor
    hidden_states: torch.Tensor
    scalar_conditions: dict[str, torch.Tensor]

    def to(self, dtype: torch.dtype) -> Self:
        """Return the inputs with the latents and hidden states in ``dtype``, as new leaves."""
        return HeadInputs(
            self.noisy_latents.detach().to(dtype).requires_grad_(),
            self.hidden_states.detach().to(dtype).requires_grad_(),
            self.scalar_conditions,
        )


def build_llm_config(tensor_parallel_size: int, dtype: torch.dtype) -> TransformerConfig:
    """Build an LLM config with the hidden size of the test and the given TP size and dtype."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=LLM_HIDDEN_SIZE,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )


@pytest.fixture(name="reference")
def build_reference(head_config: AdaLNMLPHeadConfig) -> ReferenceHead:
    """Build an fp32 reference head with random weights, the same on every rank."""
    torch.manual_seed(0)
    return ReferenceHead(head_config).cuda()


@pytest.fixture(name="head")
def build_head(
    head_config: AdaLNMLPHeadConfig,
    reference: ReferenceHead,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> AdaLNMLPHead:
    """Build an AdaLNMLPHead that holds this rank's shards of the reference weights."""
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    head = AdaLNMLPHead(
        head_config, build_llm_config(tensor_parallel_size, dtype), pg_collection
    ).cuda()
    shards = take_reference_shards(dict(reference.named_parameters()), head_config)
    parameters = dict(head.named_parameters())
    assert parameters.keys() == shards.keys()
    with torch.no_grad():
        for name, parameter in parameters.items():
            assert parameter.shape == shards[name].shape, name
            parameter.copy_(shards[name])
    return head


@pytest.fixture(name="inputs")
def build_inputs(head_config: AdaLNMLPHeadConfig) -> HeadInputs:
    """Build fp32 inputs for 5 positions of 3 samples each, the same on every rank."""
    torch.manual_seed(1)
    return HeadInputs(
        torch.randn(NUM_POSITIONS, NUM_SAMPLES, head_config.latent_size, device="cuda"),
        torch.randn(NUM_POSITIONS, LLM_HIDDEN_SIZE, device="cuda"),
        {
            name: torch.rand(NUM_POSITIONS, NUM_SAMPLES, device="cuda") * 1000
            for name in head_config.scalar_condition_names
        },
    )


@pytest.fixture(name="reference_output")
def calc_reference_output(reference: ReferenceHead, inputs: HeadInputs) -> torch.Tensor:
    """Calculate the reference head's output for the inputs without gradients."""
    with torch.no_grad():
        return reference(inputs.noisy_latents, inputs.hidden_states, inputs.scalar_conditions)


@pytest.fixture(name="head_output")
def calc_head_output(head: AdaLNMLPHead, inputs: HeadInputs, dtype: torch.dtype) -> torch.Tensor:
    """Calculate the output of the head under test for the inputs in its dtype."""
    head_inputs = inputs.to(dtype)
    with torch.no_grad():
        return head(
            head_inputs.noisy_latents, head_inputs.hidden_states, head_inputs.scalar_conditions
        )


@WITH_DTYPES
@WITH_HEAD_CONFIGS
def test_output_shape_matches_torch(
    head_output: torch.Tensor, reference_output: torch.Tensor
) -> None:
    """Check that the head output has the shape of the reference head's output."""
    assert head_output.shape == reference_output.shape


@WITH_DTYPES
@WITH_HEAD_CONFIGS
def test_output_has_parameter_dtype(head_output: torch.Tensor, dtype: torch.dtype) -> None:
    """Check that the head output has the dtype of the head's parameters."""
    assert head_output.dtype == dtype


@WITH_DTYPES
@WITH_HEAD_CONFIGS
def test_output_matches_torch(
    head_output: torch.Tensor,
    reference_output: torch.Tensor,
    head_config: AdaLNMLPHeadConfig,
    dtype: torch.dtype,
) -> None:
    """Check that the head output equals the reference head's output."""
    error = calc_relative_error(head_output, reference_output)
    logger.info("%s %s: output max relative error %.2e", head_config.activation, dtype, error)
    assert error < TOLERANCES[dtype]


@WITH_DTYPES
@WITH_HEAD_CONFIGS
def test_gradients_match_torch(
    head: AdaLNMLPHead,
    reference: ReferenceHead,
    head_config: AdaLNMLPHeadConfig,
    inputs: HeadInputs,
    dtype: torch.dtype,
) -> None:
    """Check that gradients of the inputs and of every parameter equal the reference ones."""
    reference_inputs = inputs.to(torch.float32)
    head_inputs = inputs.to(dtype)
    reference_output = reference(
        reference_inputs.noisy_latents,
        reference_inputs.hidden_states,
        reference_inputs.scalar_conditions,
    )
    head_output = head(
        head_inputs.noisy_latents, head_inputs.hidden_states, head_inputs.scalar_conditions
    )
    grad_output = torch.randn_like(reference_output)
    reference_output.backward(grad_output)
    head_output.backward(grad_output.to(dtype))

    errors = {
        "noisy latents": calc_relative_error(
            head_inputs.noisy_latents.grad, reference_inputs.noisy_latents.grad
        ),
        "hidden states": calc_relative_error(
            head_inputs.hidden_states.grad, reference_inputs.hidden_states.grad
        ),
    }
    reference_gradients = take_reference_shards(
        {name: parameter.grad for name, parameter in reference.named_parameters()}, head_config
    )
    for name, parameter in head.named_parameters():
        errors[name] = calc_relative_error(parameter.grad, reference_gradients[name])
    name, error = max(errors.items(), key=lambda item: item[1])
    logger.info(
        "%s %s: gradient max relative error %.2e (%s)", head_config.activation, dtype, error, name
    )
    assert error < GRADIENT_TOLERANCES[dtype], errors


@pytest.fixture(name="new_head")
def build_new_head(tensor_parallel_size: int) -> AdaLNMLPHead:
    """Build an fp32 head of ``BASE_CONFIG`` with its initial weights."""
    return AdaLNMLPHead(BASE_CONFIG, build_llm_config(tensor_parallel_size, torch.float32)).cuda()


def test_new_head_outputs_zeros(new_head: AdaLNMLPHead) -> None:
    """Check that a new head outputs zeros, as the zero-initialized output projection gives."""
    output = new_head(
        torch.randn(NUM_POSITIONS, NUM_SAMPLES, BASE_CONFIG.latent_size, device="cuda"),
        torch.randn(NUM_POSITIONS, LLM_HIDDEN_SIZE, device="cuda"),
        {"time": torch.rand(NUM_POSITIONS, NUM_SAMPLES, device="cuda")},
    )
    assert torch.equal(output, torch.zeros_like(output))


def test_accepts_zero_positions(new_head: AdaLNMLPHead) -> None:
    """Check that the head accepts no positions, which a batch without audio gives."""
    output = new_head(
        torch.zeros(0, NUM_SAMPLES, BASE_CONFIG.latent_size, device="cuda"),
        torch.zeros(0, LLM_HIDDEN_SIZE, device="cuda"),
        {"time": torch.zeros(0, NUM_SAMPLES, device="cuda")},
    )
    assert output.shape == (0, NUM_SAMPLES, BASE_CONFIG.latent_size)


def test_rejects_scalar_conditions_of_other_names(new_head: AdaLNMLPHead) -> None:
    """Check that scalar conditions other than the configured ones raise ValueError."""
    with pytest.raises(ValueError, match="scalar_conditions has"):
        new_head(
            torch.zeros(NUM_POSITIONS, NUM_SAMPLES, BASE_CONFIG.latent_size, device="cuda"),
            torch.zeros(NUM_POSITIONS, LLM_HIDDEN_SIZE, device="cuda"),
            {"guidance": torch.zeros(NUM_POSITIONS, NUM_SAMPLES, device="cuda")},
        )


@pytest.mark.parametrize("field", ["latent_size", "hidden_size", "ffn_hidden_size"])
def test_rejects_sizes_not_divisible_by_tp(tensor_parallel_size: int, field: str) -> None:
    """Check that a size that TP cannot split raises ValueError naming the field."""
    if tensor_parallel_size == 1:
        pytest.skip("every size splits over one rank")
    head_config = replace(BASE_CONFIG, **{field: tensor_parallel_size * 8 + 1})
    with pytest.raises(ValueError, match=f"{field}=.* cannot be split"):
        AdaLNMLPHead(head_config, build_llm_config(tensor_parallel_size, torch.float32))


# A value of each setting that the head cannot use.
INVALID_SETTINGS = {"num_layers": 0, "activation": "swish", "max_period": 0.0}


@pytest.mark.parametrize("field", list(INVALID_SETTINGS))
def test_rejects_invalid_settings(tensor_parallel_size: int, field: str) -> None:
    """Check that a setting the head cannot use raises ValueError naming the field."""
    head_config = replace(BASE_CONFIG, **{field: INVALID_SETTINGS[field]})
    with pytest.raises(ValueError, match=field):
        AdaLNMLPHead(head_config, build_llm_config(tensor_parallel_size, torch.float32))


def test_rejects_odd_frequency_embedding_size(tensor_parallel_size: int) -> None:
    """Check that an odd number of sinusoidal features raises ValueError."""
    head_config = replace(BASE_CONFIG, frequency_embedding_size=31)
    with pytest.raises(ValueError, match="frequency_embedding_size must be even"):
        AdaLNMLPHead(head_config, build_llm_config(tensor_parallel_size, torch.float32))


def test_rejects_duplicate_scalar_condition_names(tensor_parallel_size: int) -> None:
    """Check that a scalar condition named twice raises ValueError."""
    head_config = replace(BASE_CONFIG, scalar_condition_names=("time", "time"))
    with pytest.raises(ValueError, match="duplicates"):
        AdaLNMLPHead(head_config, build_llm_config(tensor_parallel_size, torch.float32))


def test_rejects_pg_collection_of_another_tp_size(tensor_parallel_size: int) -> None:
    """Check that a pg_collection TP group of another size than llm_config raises ValueError."""
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    with pytest.raises(ValueError, match=r"pg_collection\.tp has"):
        AdaLNMLPHead(
            BASE_CONFIG,
            build_llm_config(tensor_parallel_size + 1, torch.float32),
            pg_collection,
        )

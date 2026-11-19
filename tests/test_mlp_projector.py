import logging
from dataclasses import replace
from itertools import pairwise

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.activations import ACTIVATIONS
from megatron_speech.components.projectors.mlp import MLPProjector, MLPProjectorConfig
from tests.reference_comparison import TOLERANCES, calc_relative_error, take_shard

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

BASE_CONFIG = MLPProjectorConfig(input_size=48, output_size=64, hidden_size=96)
PROJECTOR_CONFIGS = [
    replace(BASE_CONFIG, num_layers=1, activation="gelu"),
    replace(BASE_CONFIG, num_layers=2, activation="gelu"),
    replace(BASE_CONFIG, num_layers=3, activation="silu"),
]


@pytest.fixture(
    name="projector_config",
    params=PROJECTOR_CONFIGS,
    ids=lambda config: f"{config.num_layers}layers-{config.activation}",
)
def get_projector_config(request: pytest.FixtureRequest) -> MLPProjectorConfig:
    """Return a projector config with one to three layers."""
    return request.param


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the projector under test."""
    return request.param


@pytest.fixture(name="reference")
def build_reference(projector_config: MLPProjectorConfig) -> list[torch.nn.Linear]:
    """Build fp32 ``torch.nn.Linear`` layers with random weights, the same on every rank."""
    torch.manual_seed(0)
    sizes = [
        projector_config.input_size,
        *[projector_config.hidden_size] * (projector_config.num_layers - 1),
        projector_config.output_size,
    ]
    return [torch.nn.Linear(n_in, n_out).cuda() for n_in, n_out in pairwise(sizes)]


def get_split_dim(index: int) -> int:
    """Return the weight dim that layer ``index`` splits: output (0) or input features (1)."""
    return 0 if index % 2 == 0 else 1


@pytest.fixture(name="projector")
def build_projector(
    projector_config: MLPProjectorConfig,
    reference: list[torch.nn.Linear],
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> MLPProjector:
    """Build an MLPProjector that holds this rank's shards of the reference weights."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    projector = MLPProjector(projector_config, llm_config, pg_collection).cuda()
    with torch.no_grad():
        for index, (layer, ref) in enumerate(zip(projector.layers, reference, strict=True)):
            dim = get_split_dim(index)
            weight = take_shard(ref.weight, dim)
            assert layer.weight.shape == weight.shape
            layer.weight.copy_(weight)
            layer.bias.copy_(take_shard(ref.bias, 0 if dim == 0 else None))
    return projector


@pytest.fixture(name="inputs")
def build_inputs() -> torch.Tensor:
    """Build an fp32 ``[10, 48]`` input, the same on every rank."""
    torch.manual_seed(1)
    return torch.randn(10, 48, device="cuda", requires_grad=True)


def run_reference_layers(
    reference: list[torch.nn.Linear], x: torch.Tensor, activation: str
) -> torch.Tensor:
    """Apply the reference layers with ``activation`` between them."""
    for index, layer in enumerate(reference):
        x = layer(x)
        if index < len(reference) - 1:
            x = ACTIVATIONS[activation](x)
    return x


@pytest.fixture(name="reference_output")
def calc_reference_output(
    reference: list[torch.nn.Linear], projector_config: MLPProjectorConfig, inputs: torch.Tensor
) -> torch.Tensor:
    """Calculate the reference layers' output for the input without gradients."""
    with torch.no_grad():
        return run_reference_layers(reference, inputs, projector_config.activation)


@pytest.fixture(name="projector_output")
def calc_projector_output(
    projector: MLPProjector, inputs: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Calculate the output of the projector under test for the input in its dtype."""
    with torch.no_grad():
        return projector(inputs.to(dtype))


def test_output_shape_matches_torch(
    projector_output: torch.Tensor, reference_output: torch.Tensor
) -> None:
    """Check that the projector output has the shape of the reference layers' output."""
    assert projector_output.shape == reference_output.shape


def test_output_has_parameter_dtype(projector_output: torch.Tensor, dtype: torch.dtype) -> None:
    """Check that the projector output has the dtype of the projector's parameters."""
    assert projector_output.dtype == dtype


def test_output_matches_torch(
    projector_output: torch.Tensor,
    reference_output: torch.Tensor,
    projector_config: MLPProjectorConfig,
    dtype: torch.dtype,
) -> None:
    """Check that the projector output equals the reference layers' output."""
    error = calc_relative_error(projector_output, reference_output)
    logger.info(
        "layers=%d %s: output max relative error %.2e", projector_config.num_layers, dtype, error
    )
    assert error < TOLERANCES[dtype]


def test_gradients_match_torch(
    projector: MLPProjector,
    reference: list[torch.nn.Linear],
    projector_config: MLPProjectorConfig,
    inputs: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Check that gradients of the input and of every weight equal the reference ones."""
    projector_inputs = inputs.detach().to(dtype).requires_grad_()
    expected = run_reference_layers(reference, inputs, projector_config.activation)
    actual = projector(projector_inputs)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    errors = {"input": calc_relative_error(projector_inputs.grad, inputs.grad)}
    for index, (layer, ref) in enumerate(zip(projector.layers, reference, strict=True)):
        target = take_shard(ref.weight.grad, get_split_dim(index))
        errors[f"layer {index} weight"] = calc_relative_error(layer.weight.grad, target)
    logger.info(
        "layers=%d %s: gradient max relative error %s",
        projector_config.num_layers,
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


def test_accepts_zero_frames(tensor_parallel_size: int) -> None:
    """Check that the projector accepts no encoder frames, which a batch without audio gives."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector = MLPProjector(BASE_CONFIG, llm_config).cuda()
    assert projector(torch.zeros(0, 48, device="cuda")).shape == (0, 64)


def test_rejects_features_not_divisible_by_tp(tensor_parallel_size: int) -> None:
    """Check that a column-parallel layer whose outputs TP cannot split raises ValueError."""
    if tensor_parallel_size == 1:
        pytest.skip("every size splits over one rank")
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = replace(BASE_CONFIG, hidden_size=tensor_parallel_size * 48 + 1)
    with pytest.raises(ValueError, match="cannot be split"):
        MLPProjector(projector_config, llm_config)


def test_rejects_pg_collection_of_another_tp_size(tensor_parallel_size: int) -> None:
    """Check that a pg_collection TP group of another size than llm_config raises ValueError."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size + 1,
        tensor_model_parallel_size=tensor_parallel_size + 1,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    with pytest.raises(ValueError, match=r"pg_collection\.tp has"):
        MLPProjector(BASE_CONFIG, llm_config, pg_collection)

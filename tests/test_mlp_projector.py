import logging
from itertools import pairwise

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.projectors.mlp import ACTIVATIONS, MLPProjector, MLPProjectorConfig

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

# Largest relative error allowed against the fp32 reference, by parameter dtype.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 2e-2}
# (num_layers, activation)
SHAPES = [(1, "gelu"), (2, "gelu"), (3, "silu")]


@pytest.fixture(name="projector_config", params=SHAPES, ids=str)
def build_projector_config(request: pytest.FixtureRequest) -> MLPProjectorConfig:
    """Projectors of one to three layers."""
    num_layers, activation = request.param
    return MLPProjectorConfig(
        input_size=48, output_size=64, num_layers=num_layers, hidden_size=96, activation=activation
    )


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Parameter dtype of the projector under test."""
    return request.param


@pytest.fixture(name="reference")
def build_reference(projector_config: MLPProjectorConfig) -> list[torch.nn.Linear]:
    """fp32 ``torch.nn.Linear`` layers with random weights, the same on every rank."""
    torch.manual_seed(0)
    sizes = [
        projector_config.input_size,
        *[projector_config.hidden_size] * (projector_config.num_layers - 1),
        projector_config.output_size,
    ]
    return [torch.nn.Linear(n_in, n_out).cuda() for n_in, n_out in pairwise(sizes)]


def take_shard(tensor: torch.Tensor, dim: int | None) -> torch.Tensor:
    """This rank's part of ``tensor`` split along ``dim``, or all of it for None."""
    if dim is None:
        return tensor
    size = parallel_state.get_tensor_model_parallel_world_size()
    return tensor.chunk(size, dim)[parallel_state.get_tensor_model_parallel_rank()]


def get_split_dim(index: int) -> int:
    """Weight dim that layer ``index`` splits: output features (0) or input features (1)."""
    return 0 if index % 2 == 0 else 1


@pytest.fixture(name="projector")
def build_projector(
    projector_config: MLPProjectorConfig,
    reference: list[torch.nn.Linear],
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> MLPProjector:
    """MLPProjector that holds this rank's shards of the reference weights."""
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
    """fp32 ``[10, 48]`` input, the same on every rank."""
    torch.manual_seed(1)
    return torch.randn(10, 48, device="cuda", requires_grad=True)


def run_reference(
    reference: list[torch.nn.Linear], x: torch.Tensor, activation: str
) -> torch.Tensor:
    """Apply the reference layers with ``activation`` between them."""
    for index, layer in enumerate(reference):
        x = layer(x)
        if index < len(reference) - 1:
            x = ACTIVATIONS[activation](x)
    return x


def calc_relative_error(value: torch.Tensor, target: torch.Tensor) -> float:
    """Largest absolute difference divided by the largest absolute target value."""
    return ((value.float() - target).abs().max() / target.abs().max()).item()


def test_output_matches_torch(
    projector: MLPProjector,
    reference: list[torch.nn.Linear],
    projector_config: MLPProjectorConfig,
    inputs: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """The projector output equals the reference layers' output."""
    with torch.no_grad():
        expected = run_reference(reference, inputs, projector_config.activation)
        actual = projector(inputs.to(dtype))
    error = calc_relative_error(actual, expected)
    logger.info(
        "layers=%d %s: output max relative error %.2e", projector_config.num_layers, dtype, error
    )
    assert actual.shape == (10, 64)
    assert actual.dtype == dtype
    assert error < TOLERANCES[dtype]


def test_gradients_match_torch(
    projector: MLPProjector,
    reference: list[torch.nn.Linear],
    projector_config: MLPProjectorConfig,
    inputs: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Gradients of the input and of every weight equal the reference ones."""
    our_inputs = inputs.detach().to(dtype).requires_grad_()
    expected = run_reference(reference, inputs, projector_config.activation)
    actual = projector(our_inputs)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    errors = {"input": calc_relative_error(our_inputs.grad, inputs.grad)}
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


def test_accepts_zero_tokens(tensor_parallel_size: int) -> None:
    """A batch without audio gives the projector no tokens."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = MLPProjectorConfig(input_size=48, output_size=64, hidden_size=96)
    projector = MLPProjector(projector_config, llm_config).cuda()
    assert projector(torch.zeros(0, 48, device="cuda")).shape == (0, 64)


def test_rejects_features_not_divisible_by_tp(tensor_parallel_size: int) -> None:
    """A column-parallel layer whose output features do not split evenly raises ValueError."""
    if tensor_parallel_size == 1:
        pytest.skip("every size splits over one rank")
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = MLPProjectorConfig(
        input_size=48, output_size=64, hidden_size=tensor_parallel_size * 48 + 1
    )
    with pytest.raises(ValueError, match="cannot be split"):
        MLPProjector(projector_config, llm_config)


def test_rejects_pg_collection_of_another_tp_size(tensor_parallel_size: int) -> None:
    """The TP group in pg_collection must have as many ranks as llm_config says."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size + 1,
        tensor_model_parallel_size=tensor_parallel_size + 1,
    )
    projector_config = MLPProjectorConfig(input_size=48, output_size=64, hidden_size=96)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    with pytest.raises(ValueError, match=r"pg_collection\.tp has"):
        MLPProjector(projector_config, llm_config, pg_collection)

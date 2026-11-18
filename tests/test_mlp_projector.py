import logging
from itertools import pairwise

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.projectors.mlp import ACTIVATIONS, MLPProjector, MLPProjectorConfig

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)


# Largest relative error allowed against the fp32 reference, by parameter dtype.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 2e-2}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("num_layers", "activation"), [(1, "gelu"), (2, "gelu"), (3, "silu")])
def test_matches_torch_linear_layers(
    tensor_parallel_size: int, num_layers: int, activation: str, dtype: torch.dtype
) -> None:
    """Copy fp32 reference weights into the TP shards; compare outputs and gradients."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    projector_config = MLPProjectorConfig(
        input_size=48,
        output_size=64,
        num_layers=num_layers,
        hidden_size=96,
        activation=activation,
    )
    projector = MLPProjector(projector_config, llm_config).cuda()

    torch.manual_seed(0)  # the same reference weights and inputs on every rank
    sizes = [48, *[96] * (num_layers - 1), 64]
    reference = [torch.nn.Linear(n_in, n_out).cuda() for n_in, n_out in pairwise(sizes)]
    x_ref = torch.randn(10, 48, device="cuda", requires_grad=True)
    x = x_ref.detach().to(dtype).requires_grad_()
    grad_output = torch.randn(10, 64, device="cuda")

    # Column-parallel layers (even index) split output features, row-parallel ones input features.
    split_dims = [0 if index % 2 == 0 else 1 for index in range(num_layers)]
    rank = parallel_state.get_tensor_model_parallel_rank()
    with torch.no_grad():
        for layer, ref, dim in zip(projector.layers, reference, split_dims, strict=True):
            shard = ref.weight.chunk(tensor_parallel_size, dim=dim)[rank]
            assert layer.weight.shape == shard.shape
            layer.weight.copy_(shard)
            layer.bias.copy_(ref.bias.chunk(tensor_parallel_size)[rank] if dim == 0 else ref.bias)

    expected = x_ref
    for index, ref in enumerate(reference):
        expected = ref(expected)
        if index < num_layers - 1:
            expected = ACTIVATIONS[activation](expected)
    actual = projector(x)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    compared = {
        "output": (actual, expected),
        "input grad": (x.grad, x_ref.grad),
        **{
            f"layer {index} weight grad": (
                layer.weight.grad,
                ref.weight.grad.chunk(tensor_parallel_size, dim=dim)[rank],
            )
            for index, (layer, ref, dim) in enumerate(
                zip(projector.layers, reference, split_dims, strict=True)
            )
        },
    }
    errors = {
        name: ((value.float() - target).abs().max() / target.abs().max()).item()
        for name, (value, target) in compared.items()
    }
    logger.info(
        "TP=%d layers=%d %s: max relative error %s",
        tensor_parallel_size,
        num_layers,
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert actual.shape == (10, 64)
    assert actual.dtype == dtype
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

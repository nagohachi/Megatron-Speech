from itertools import pairwise

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.projectors.mlp import ACTIVATIONS, MLPProjector, MLPProjectorConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(("num_layers", "activation"), [(1, "gelu"), (2, "gelu"), (3, "silu")])
def test_matches_torch_linear_layers(
    tensor_parallel_size: int, num_layers: int, activation: str
) -> None:
    """Copy reference weights into the TP shards and compare the outputs in fp32."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = MLPProjectorConfig(
        input_size=48,
        output_size=64,
        num_layers=num_layers,
        hidden_size=96,
        activation=activation,
    )
    projector = MLPProjector(projector_config, llm_config).cuda()

    torch.manual_seed(0)  # the same reference weights and input on every rank
    sizes = [48, *[96] * (num_layers - 1), 64]
    reference = [torch.nn.Linear(n_in, n_out).cuda() for n_in, n_out in pairwise(sizes)]
    x = torch.randn(10, 48, device="cuda")

    rank = parallel_state.get_tensor_model_parallel_rank()
    with torch.no_grad():
        for index, (layer, ref) in enumerate(zip(projector.layers, reference, strict=True)):
            if index % 2 == 0:  # column parallel: output features are split
                layer.weight.copy_(ref.weight.chunk(tensor_parallel_size, dim=0)[rank])
                layer.bias.copy_(ref.bias.chunk(tensor_parallel_size, dim=0)[rank])
            else:  # row parallel: input features are split
                layer.weight.copy_(ref.weight.chunk(tensor_parallel_size, dim=1)[rank])
                layer.bias.copy_(ref.bias)
        expected = x
        for index, ref in enumerate(reference):
            expected = ref(expected)
            if index < num_layers - 1:
                expected = ACTIVATIONS[activation](expected)
        actual = projector(x)

    rel_error = ((actual - expected).abs().max() / expected.abs().max()).item()
    print(f"TP={tensor_parallel_size} layers={num_layers}: max relative error {rel_error:.2e}")
    assert actual.shape == (10, 64)
    assert rel_error < 1e-5

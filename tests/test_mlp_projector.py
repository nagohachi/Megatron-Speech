"""``MLPProjector`` matches ``torch.nn.Linear`` layers that hold the same weights.

    apptainer/run.sh python -m pytest tests/test_mlp_projector.py
    CUDA_VISIBLE_DEVICES=0,1 apptainer/run.sh python -m torch.distributed.run --nproc-per-node 2 \
        -m pytest tests/test_mlp_projector.py
"""

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.projectors.mlp import MLPProjector, MLPProjectorConfig

pytestmark = pytest.mark.gpu


def test_matches_torch_linear_layers(tensor_parallel_size: int) -> None:
    """Copy reference weights into the TP shards and compare ``fc2(gelu(fc1(x)))`` in fp32."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = MLPProjectorConfig(input_size=48, hidden_size=96, output_size=64)
    projector = MLPProjector(projector_config, llm_config).cuda()

    torch.manual_seed(0)  # the same reference weights and input on every rank
    fc1 = torch.nn.Linear(48, 96).cuda()
    fc2 = torch.nn.Linear(96, 64).cuda()
    x = torch.randn(10, 48, device="cuda")

    rank = parallel_state.get_tensor_model_parallel_rank()
    mlp = projector.projector.encoder
    with torch.no_grad():
        mlp.linear_fc1.weight.copy_(fc1.weight.chunk(tensor_parallel_size, dim=0)[rank])
        mlp.linear_fc1.bias.copy_(fc1.bias.chunk(tensor_parallel_size, dim=0)[rank])
        mlp.linear_fc2.weight.copy_(fc2.weight.chunk(tensor_parallel_size, dim=1)[rank])
        mlp.linear_fc2.bias.copy_(fc2.bias)
        expected = fc2(torch.nn.functional.gelu(fc1(x)))
        actual = projector(x)

    rel_error = ((actual - expected).abs().max() / expected.abs().max()).item()
    print(f"TP={tensor_parallel_size}: max relative error {rel_error:.2e}")
    assert actual.shape == (10, 64)
    assert rel_error < 1e-5

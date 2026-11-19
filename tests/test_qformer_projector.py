import logging
from dataclasses import dataclass, replace

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from transformers import Blip2QFormerConfig, Blip2QFormerModel

from megatron_speech.components.projectors.qformer import QFormerProjector, QFormerProjectorConfig
from tests.reference_comparison import TOLERANCES, calc_relative_error, take_shard

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

# The Q-Former's bf16 gradient errors reach 3.15e-2 at TP=2, above the shared 2e-2, so this
# test keeps the 5e-2 it had before the tolerances were shared.
QFORMER_TOLERANCES = TOLERANCES | {torch.bfloat16: 5e-2}

UTTERANCE_LENGTHS = [7, 4, 1]
CU_SEQLENS = [0, 7, 11, 12]
BASE_CONFIG = QFormerProjectorConfig(
    input_size=48, output_size=80, hidden_size=64, num_attention_heads=4, ffn_hidden_size=128
)
PROJECTOR_CONFIGS = [
    replace(BASE_CONFIG, num_layers=2, cross_attention_frequency=1, num_queries=2, window_size=4),
    replace(
        BASE_CONFIG, num_layers=3, cross_attention_frequency=2, num_queries=3, window_size=None
    ),
]


@dataclass
class Reference:
    """Hugging Face Q-Former, output projection and query tokens in fp32."""

    qformer: Blip2QFormerModel
    projection: torch.nn.Linear
    query_tokens: torch.nn.Parameter


def describe_config(config: QFormerProjectorConfig) -> str:
    """Name a config for test ids and logs, such as ``2layers-2queries-window4``."""
    window = "whole-utterance" if config.window_size is None else f"window{config.window_size}"
    return f"{config.num_layers}layers-{config.num_queries}queries-{window}"


@pytest.fixture(name="projector_config", params=PROJECTOR_CONFIGS, ids=describe_config)
def get_projector_config(request: pytest.FixtureRequest) -> QFormerProjectorConfig:
    """Return a config that cuts each utterance into windows, or one that reads it whole."""
    return request.param


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the projector under test."""
    return request.param


@pytest.fixture(name="reference")
def build_reference(projector_config: QFormerProjectorConfig) -> Reference:
    """Build random reference weights, the same on every rank."""
    torch.manual_seed(0)
    hf_config = Blip2QFormerConfig(
        hidden_size=projector_config.hidden_size,
        num_hidden_layers=projector_config.num_layers,
        num_attention_heads=projector_config.num_attention_heads,
        intermediate_size=projector_config.ffn_hidden_size,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        cross_attention_frequency=projector_config.cross_attention_frequency,
        encoder_hidden_size=projector_config.input_size,
    )
    hf_config._attn_implementation = "eager"
    qformer = Blip2QFormerModel(hf_config).cuda()
    with torch.no_grad():  # LayerNorms away from (1, 0), so that a missed copy shows
        for name, param in qformer.named_parameters():
            if "layernorm" in name.lower():
                param.normal_(mean=1.0 if name.endswith("weight") else 0.0, std=0.1)
    return Reference(
        qformer=qformer,
        projection=torch.nn.Linear(
            projector_config.hidden_size, projector_config.output_size
        ).cuda(),
        query_tokens=torch.nn.Parameter(
            torch.randn(projector_config.num_queries, projector_config.hidden_size, device="cuda")
        ),
    )


def list_reference_parameters(reference: Reference) -> dict[str, torch.nn.Parameter]:
    """List the Q-Former's parameters under Hugging Face's names, the projection and the queries."""
    return {
        **dict(reference.qformer.named_parameters()),
        "projection.weight": reference.projection.weight,
        "projection.bias": reference.projection.bias,
        "query_tokens": reference.query_tokens,
    }


def map_hf_tensors(
    hf_tensors: dict[str, torch.Tensor], config: QFormerProjectorConfig
) -> dict[str, torch.Tensor]:
    """Map reference weights or gradients to the projector's parameter names, before splitting."""
    mapped = {
        name: hf_tensors[name] for name in ("query_tokens", "projection.weight", "projection.bias")
    }
    for suffix in ("weight", "bias"):
        mapped[f"layernorm.{suffix}"] = hf_tensors[f"layernorm.{suffix}"]
    for index in range(config.num_layers):
        layer, hf_layer = f"layers.{index}.", f"encoder.layer.{index}."
        attentions = {"self_attention": "attention"}
        if index % config.cross_attention_frequency == 0:
            attentions["cross_attention"] = "crossattention"
        for suffix in ("weight", "bias"):
            for attention, hf_attention in attentions.items():
                for projection in ("query", "key", "value"):
                    mapped[f"{layer}{attention}.{projection}.{suffix}"] = hf_tensors[
                        f"{hf_layer}{hf_attention}.attention.{projection}.{suffix}"
                    ]
                mapped[f"{layer}{attention}.dense.{suffix}"] = hf_tensors[
                    f"{hf_layer}{hf_attention}.output.dense.{suffix}"
                ]
                mapped[f"{layer}{attention}.layernorm.{suffix}"] = hf_tensors[
                    f"{hf_layer}{hf_attention}.output.LayerNorm.{suffix}"
                ]
            mapped[f"{layer}fc1.{suffix}"] = hf_tensors[
                f"{hf_layer}intermediate_query.dense.{suffix}"
            ]
            mapped[f"{layer}fc2.{suffix}"] = hf_tensors[f"{hf_layer}output_query.dense.{suffix}"]
            mapped[f"{layer}mlp_layernorm.{suffix}"] = hf_tensors[
                f"{hf_layer}output_query.LayerNorm.{suffix}"
            ]
    return mapped


def get_split_dim(name: str) -> int | None:
    """Return the dim of the projector parameter ``name`` that TP splits, if any."""
    column_parallel = ("query", "key", "value", "fc1", "projection")
    if name.endswith(
        tuple(f"{layer}.{s}" for layer in column_parallel for s in ("weight", "bias"))
    ):
        return 0  # output features
    if name.endswith(("dense.weight", "fc2.weight")):
        return 1  # input features
    return None


@pytest.fixture(name="projector")
def build_projector(
    projector_config: QFormerProjectorConfig,
    reference: Reference,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> QFormerProjector:
    """Build a QFormerProjector that holds this rank's shards of the reference weights."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    projector = QFormerProjector(projector_config, llm_config, pg_collection).cuda()
    projector_parameters = dict(projector.named_parameters())
    reference_weights = {n: p.detach() for n, p in list_reference_parameters(reference).items()}
    hf_weights = map_hf_tensors(reference_weights, projector_config)
    assert hf_weights.keys() == projector_parameters.keys()
    with torch.no_grad():
        for name, weight in hf_weights.items():
            shard = take_shard(weight, get_split_dim(name))
            assert projector_parameters[name].shape == shard.shape
            projector_parameters[name].copy_(shard)
    return projector


@pytest.fixture(name="frames")
def build_frames() -> torch.Tensor:
    """Build fp32 encoder frames of UTTERANCE_LENGTHS, the same on every rank."""
    torch.manual_seed(1)
    return torch.randn(sum(UTTERANCE_LENGTHS), 48, device="cuda", requires_grad=True)


def run_hf_qformer(
    reference: Reference, frames: torch.Tensor, window_size: int | None
) -> torch.Tensor:
    """Cut the utterances into zero-padded windows and run the Hugging Face model on them.

    Written independently of ``QFormerProjector.forward``.
    """
    size = window_size or max(UTTERANCE_LENGTHS)
    windows, masks = [], []
    start = 0
    for length in UTTERANCE_LENGTHS:
        for offset in range(0, length, size):
            count = min(size, length - offset)
            window = frames[start + offset : start + offset + count]
            windows.append(torch.nn.functional.pad(window, (0, 0, 0, size - count)))
            masks.append(torch.arange(size, device="cuda") < count)
        start += length
    hidden = reference.qformer(
        query_embeds=reference.query_tokens.expand(len(windows), -1, -1),
        encoder_hidden_states=torch.stack(windows),
        encoder_attention_mask=torch.stack(masks).long(),
    ).last_hidden_state
    output = reference.projection(hidden)
    return output.reshape(-1, output.shape[-1])


@pytest.fixture(name="reference_output")
def calc_reference_output(
    reference: Reference, projector_config: QFormerProjectorConfig, frames: torch.Tensor
) -> torch.Tensor:
    """Calculate the Hugging Face Q-Former's projected output for the frames without gradients."""
    with torch.no_grad():
        return run_hf_qformer(reference, frames, projector_config.window_size)


@pytest.fixture(name="projector_output")
def calc_projector_output(
    projector: QFormerProjector, frames: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Calculate the output of the projector under test for the frames in its dtype."""
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")
    with torch.no_grad():
        return projector(frames.to(dtype), cu_seqlens)


def test_output_shape_matches_hf(
    projector_output: torch.Tensor, reference_output: torch.Tensor
) -> None:
    """Check that the projector output has the shape of the reference output."""
    assert projector_output.shape == reference_output.shape


def test_output_matches_hf(
    projector_output: torch.Tensor,
    reference_output: torch.Tensor,
    projector_config: QFormerProjectorConfig,
    dtype: torch.dtype,
) -> None:
    """Check that the output equals the Hugging Face Q-Former followed by the projection."""
    error = calc_relative_error(projector_output, reference_output)
    logger.info(
        "%s %s: output max relative error %.2e", describe_config(projector_config), dtype, error
    )
    assert error < QFORMER_TOLERANCES[dtype]


def test_gradients_match_hf(
    projector: QFormerProjector,
    reference: Reference,
    projector_config: QFormerProjectorConfig,
    frames: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Check that gradients of the input and of every parameter equal the reference ones."""
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")
    projector_frames = frames.detach().to(dtype).requires_grad_()
    expected = run_hf_qformer(reference, frames, projector_config.window_size)
    actual = projector(projector_frames, cu_seqlens)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    reference_gradients = {n: p.grad for n, p in list_reference_parameters(reference).items()}
    hf_gradients = map_hf_tensors(reference_gradients, projector_config)
    errors = {
        "input": calc_relative_error(projector_frames.grad, frames.grad),
        # The key bias adds the same value to every score of a query, which softmax ignores,
        # so its true gradient is zero and both sides hold only rounding errors.
        "parameters": max(
            calc_relative_error(param.grad, take_shard(hf_gradients[name], get_split_dim(name)))
            for name, param in projector.named_parameters()
            if not name.endswith("key.bias")
        ),
    }
    logger.info(
        "%s %s: gradient max relative error %s",
        describe_config(projector_config),
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < QFORMER_TOLERANCES[dtype], errors


@pytest.mark.parametrize("cu_seqlens", [[0], [0, 0]], ids=["no-utterances", "empty-utterance"])
def test_accepts_no_frames(tensor_parallel_size: int, cu_seqlens: list[int]) -> None:
    """Check that no utterances, or one utterance without frames, give no output vectors."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector = QFormerProjector(BASE_CONFIG, llm_config).cuda()
    output = projector(
        torch.zeros(0, 48, device="cuda"),
        torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
    )
    assert output.shape == (0, 80)


def test_rejects_heads_not_divisible_by_tp(tensor_parallel_size: int) -> None:
    """Check that attention heads that TP cannot split evenly raise ValueError."""
    if tensor_parallel_size == 1:
        pytest.skip("every head count splits over one rank")
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = replace(
        BASE_CONFIG,
        hidden_size=tensor_parallel_size * 64 + 64,
        num_attention_heads=tensor_parallel_size + 1,
    )
    with pytest.raises(ValueError, match="num_attention_heads"):
        QFormerProjector(projector_config, llm_config)


@pytest.mark.parametrize(
    "field", ["num_layers", "num_queries", "cross_attention_frequency", "window_size"]
)
def test_rejects_counts_below_one(field: str) -> None:
    """Check that layer, query, cross-attention and window counts below one raise ValueError."""
    llm_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=1)
    projector_config = replace(BASE_CONFIG, **{field: 0})
    with pytest.raises(ValueError, match=field):
        QFormerProjector(projector_config, llm_config)

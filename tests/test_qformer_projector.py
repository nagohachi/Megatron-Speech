import logging
from dataclasses import dataclass, replace

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from transformers import Blip2QFormerConfig, Blip2QFormerModel

from megatron_speech.components.projectors.qformer import QFormerProjector, QFormerProjectorConfig
from tests.reference_comparison import calc_relative_error, take_shard

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

# Largest relative error allowed against the fp32 reference, by parameter dtype.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 5e-2}
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
    """Short name of a config for test ids and logs, such as ``2layers-2queries-window4``."""
    window = "whole-utterance" if config.window_size is None else f"window{config.window_size}"
    return f"{config.num_layers}layers-{config.num_queries}queries-{window}"


@pytest.fixture(name="projector_config", params=PROJECTOR_CONFIGS, ids=describe_config)
def get_projector_config(request: pytest.FixtureRequest) -> QFormerProjectorConfig:
    """A config that cuts each utterance into several windows, or one that reads it whole."""
    return request.param


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Parameter dtype of the projector under test."""
    return request.param


@pytest.fixture(name="reference")
def build_reference(projector_config: QFormerProjectorConfig) -> Reference:
    """Random reference weights, the same on every rank."""
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


def list_weight_pairs(
    projector: QFormerProjector, reference: Reference
) -> list[tuple[torch.nn.Module, torch.nn.Module, int | None]]:
    """Pair each module of ``projector`` with its reference module and its split dim.

    Dim 0 splits output features (column parallel), dim 1 input features (row parallel), and
    None means the module is not split.
    """
    pairs = [(projector.layernorm, reference.qformer.layernorm, None)]
    for ours, theirs in zip(projector.layers, reference.qformer.encoder.layer, strict=True):
        assert (ours.cross_attention is not None) == theirs.has_cross_attention
        attentions = [(ours.self_attention, theirs.attention)]
        if ours.cross_attention is not None:
            attentions.append((ours.cross_attention, theirs.crossattention))
        for our_attention, their_attention in attentions:
            pairs += [
                (our_attention.query, their_attention.attention.query, 0),
                (our_attention.key, their_attention.attention.key, 0),
                (our_attention.value, their_attention.attention.value, 0),
                (our_attention.dense, their_attention.output.dense, 1),
                (our_attention.layernorm, their_attention.output.LayerNorm, None),
            ]
        pairs += [
            (ours.fc1, theirs.intermediate_query.dense, 0),
            (ours.fc2, theirs.output_query.dense, 1),
            (ours.mlp_layernorm, theirs.output_query.LayerNorm, None),
        ]
    pairs.append((projector.projection, reference.projection, 0))
    return pairs


@pytest.fixture(name="projector")
def build_projector(
    projector_config: QFormerProjectorConfig,
    reference: Reference,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> QFormerProjector:
    """QFormerProjector that holds this rank's shards of the reference weights."""
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
    with torch.no_grad():
        projector.query_tokens.copy_(reference.query_tokens)
        for ours, theirs, dim in list_weight_pairs(projector, reference):
            weight = take_shard(theirs.weight, dim)
            assert ours.weight.shape == weight.shape
            ours.weight.copy_(weight)
            ours.bias.copy_(take_shard(theirs.bias, 0 if dim == 0 else None))
    return projector


@pytest.fixture(name="frames")
def build_frames() -> torch.Tensor:
    """fp32 encoder frames of UTTERANCE_LENGTHS, the same on every rank."""
    torch.manual_seed(1)
    return torch.randn(sum(UTTERANCE_LENGTHS), 48, device="cuda", requires_grad=True)


def run_reference(
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


def test_output_matches_hf(
    projector: QFormerProjector,
    reference: Reference,
    projector_config: QFormerProjectorConfig,
    frames: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """The projector output equals the Hugging Face Q-Former followed by the projection."""
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")
    with torch.no_grad():
        expected = run_reference(reference, frames, projector_config.window_size)
        actual = projector(frames.to(dtype), cu_seqlens)
    error = calc_relative_error(actual, expected)
    logger.info(
        "%s %s: output max relative error %.2e", describe_config(projector_config), dtype, error
    )
    assert actual.shape == expected.shape
    assert error < TOLERANCES[dtype]


def test_gradients_match_hf(
    projector: QFormerProjector,
    reference: Reference,
    projector_config: QFormerProjectorConfig,
    frames: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Gradients of the input, the queries and every weight equal the reference ones."""
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")
    our_frames = frames.detach().to(dtype).requires_grad_()
    expected = run_reference(reference, frames, projector_config.window_size)
    actual = projector(our_frames, cu_seqlens)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    errors = {
        "input": calc_relative_error(our_frames.grad, frames.grad),
        "queries": calc_relative_error(projector.query_tokens.grad, reference.query_tokens.grad),
        "weights": max(
            calc_relative_error(ours.weight.grad, take_shard(theirs.weight.grad, dim))
            for ours, theirs, dim in list_weight_pairs(projector, reference)
        ),
    }
    logger.info(
        "%s %s: gradient max relative error %s",
        describe_config(projector_config),
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


@pytest.mark.parametrize("cu_seqlens", [[0], [0, 0]], ids=["no-utterances", "empty-utterance"])
def test_accepts_no_frames(tensor_parallel_size: int, cu_seqlens: list[int]) -> None:
    """No utterances, or one utterance without frames, give no output vectors."""
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
    """Attention heads that do not split evenly over the TP ranks raise ValueError."""
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
    """Layer, query, cross-attention and window counts below one raise ValueError."""
    llm_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=1)
    projector_config = replace(BASE_CONFIG, **{field: 0})
    with pytest.raises(ValueError, match=field):
        QFormerProjector(projector_config, llm_config)

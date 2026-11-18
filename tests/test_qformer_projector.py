import logging

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from transformers import Blip2QFormerConfig, Blip2QFormerModel

from megatron_speech.components.projectors.qformer import QFormerProjector, QFormerProjectorConfig

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

# Largest relative error allowed against the fp32 reference, by parameter dtype.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 5e-2}
UTTERANCE_LENGTHS = [7, 4, 1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("num_layers", "cross_attention_frequency", "num_queries", "window_size"),
    [(2, 1, 2, 4), (3, 2, 3, None)],
)
def test_matches_hf_blip2_qformer(
    tensor_parallel_size: int,
    num_layers: int,
    cross_attention_frequency: int,
    num_queries: int,
    window_size: int | None,
    dtype: torch.dtype,
) -> None:
    """Copy HF Blip2QFormerModel weights into the TP shards; compare outputs and gradients."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    projector_config = QFormerProjectorConfig(
        input_size=48,
        output_size=80,
        hidden_size=64,
        num_layers=num_layers,
        num_attention_heads=4,
        ffn_hidden_size=128,
        cross_attention_frequency=cross_attention_frequency,
        num_queries=num_queries,
        window_size=window_size,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    projector = QFormerProjector(projector_config, llm_config, pg_collection).cuda()

    torch.manual_seed(0)  # the same reference weights and inputs on every rank
    hf_config = Blip2QFormerConfig(
        hidden_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        cross_attention_frequency=cross_attention_frequency,
        encoder_hidden_size=48,
    )
    hf_config._attn_implementation = "eager"
    reference = Blip2QFormerModel(hf_config).cuda()
    projection = torch.nn.Linear(64, 80).cuda()
    query_tokens = torch.nn.Parameter(torch.randn(num_queries, 64, device="cuda"))
    with torch.no_grad():  # LayerNorms away from (1, 0), so that a missed copy shows
        for name, param in reference.named_parameters():
            if "layernorm" in name.lower():
                param.normal_(mean=1.0 if name.endswith("weight") else 0.0, std=0.1)

    # (our module, reference module, split dim): 0 splits output features (column parallel),
    # 1 splits input features (row parallel), None copies the whole module.
    pairs = [(projector.layernorm, reference.layernorm, None)]
    for ours, theirs in zip(projector.layers, reference.encoder.layer, strict=True):
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
    pairs.append((projector.projection, projection, 0))

    rank = parallel_state.get_tensor_model_parallel_rank()
    with torch.no_grad():
        projector.query_tokens.copy_(query_tokens)
        for ours, theirs, dim in pairs:
            weight = (
                theirs.weight
                if dim is None
                else theirs.weight.chunk(tensor_parallel_size, dim)[rank]
            )
            bias = theirs.bias.chunk(tensor_parallel_size)[rank] if dim == 0 else theirs.bias
            assert ours.weight.shape == weight.shape
            ours.weight.copy_(weight)
            ours.bias.copy_(bias)

    cu_seqlens = torch.tensor([0, 7, 11, 12], dtype=torch.int32, device="cuda")
    frames_ref = torch.randn(sum(UTTERANCE_LENGTHS), 48, device="cuda", requires_grad=True)
    frames = frames_ref.detach().to(dtype).requires_grad_()

    # Cut the utterances into zero-padded windows, written independently of QFormerProjector.
    size = window_size or max(UTTERANCE_LENGTHS)
    windows, masks = [], []
    for start, length in zip(cu_seqlens[:-1].tolist(), UTTERANCE_LENGTHS, strict=True):
        for offset in range(0, length, size):
            count = min(size, length - offset)
            window = frames_ref[start + offset : start + offset + count]
            windows.append(torch.nn.functional.pad(window, (0, 0, 0, size - count)))
            masks.append(torch.arange(size, device="cuda") < count)
    hf_output = reference(
        query_embeds=query_tokens.expand(len(windows), -1, -1),
        encoder_hidden_states=torch.stack(windows),
        encoder_attention_mask=torch.stack(masks).long(),
    ).last_hidden_state
    expected = projection(hf_output).reshape(-1, 80)
    actual = projector(frames, cu_seqlens)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    compared = {
        "output": (actual, expected),
        "input grad": (frames.grad, frames_ref.grad),
        "query grad": (projector.query_tokens.grad, query_tokens.grad),
    }
    for index, (ours, theirs, dim) in enumerate(pairs):
        target = theirs.weight.grad
        if dim is not None:
            target = target.chunk(tensor_parallel_size, dim)[rank]
        compared[f"weight grad {index}"] = (ours.weight.grad, target)
    errors = {
        name: ((value.float() - target).abs().max() / target.abs().max()).item()
        for name, (value, target) in compared.items()
    }
    weight_errors = [error for name, error in errors.items() if name.startswith("weight grad")]
    logger.info(
        "TP=%d layers=%d queries=%d window=%s %s: max relative error %s",
        tensor_parallel_size,
        num_layers,
        num_queries,
        window_size,
        dtype,
        f"output {errors['output']:.2e}, input grad {errors['input grad']:.2e}, "
        f"query grad {errors['query grad']:.2e}, weight grads up to {max(weight_errors):.2e}",
    )
    assert actual.shape == (len(windows) * num_queries, 80)
    assert max(errors.values()) < TOLERANCES[dtype], errors


@pytest.mark.parametrize("cu_seqlens", [[0], [0, 0]])
def test_accepts_no_frames(tensor_parallel_size: int, cu_seqlens: list[int]) -> None:
    """No utterances, or one utterance without frames, give no output tokens."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
    )
    projector_config = QFormerProjectorConfig(
        input_size=48, output_size=80, hidden_size=64, num_attention_heads=4, ffn_hidden_size=128
    )
    projector = QFormerProjector(projector_config, llm_config).cuda()
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
    projector_config = QFormerProjectorConfig(
        input_size=48,
        output_size=80,
        hidden_size=tensor_parallel_size * 64 + 64,
        num_attention_heads=tensor_parallel_size + 1,
        ffn_hidden_size=128,
    )
    with pytest.raises(ValueError, match="num_attention_heads"):
        QFormerProjector(projector_config, llm_config)


@pytest.mark.parametrize(
    "field", ["num_layers", "num_queries", "cross_attention_frequency", "window_size"]
)
def test_rejects_counts_below_one(field: str) -> None:
    """Layer, query, cross-attention and window counts below one raise ValueError."""
    llm_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=1)
    projector_config = QFormerProjectorConfig(
        input_size=48, output_size=80, hidden_size=64, num_attention_heads=4, ffn_hidden_size=128
    )
    setattr(projector_config, field, 0)
    with pytest.raises(ValueError, match=field):
        QFormerProjector(projector_config, llm_config)

import logging
from dataclasses import dataclass
from itertools import accumulate

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

from megatron_speech.models.language_model import MultiStreamLanguageModel, SpeechLanguageModel
from tests.numerical_error import TOLERANCES, WITH_DTYPES, calc_relative_error

pytestmark = [
    pytest.mark.gpu,
    WITH_DTYPES,
    # How the batch holds its sequences: one per row, or packed into one row.
    pytest.mark.parametrize("qkv_format", ["sbhd", "thd"]),
]
logger = logging.getLogger(__name__)

VOCAB_SIZE = 128
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16
# Sequences that the thd batch packs into one row of SEQUENCE_LENGTH tokens.
PACKED_SEQUENCE_LENGTHS = [7, 5, 4]
# Which positions of the batch are targets: some of them, or all.
WITH_TARGETS = pytest.mark.parametrize("target_pattern", ["some", "all"])


@dataclass
class Batch:
    """Token ids, positions, labels and loss mask of a batch, each ``[batch, sequence]``.

    ``packed_seq_params`` delimits the sequences packed into a row, or is None for one
    sequence per row.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    packed_seq_params: PackedSeqParams | None


@pytest.fixture(name="sequence_parallel", params=[False, True], ids=["no-sp", "sp"])
def get_sequence_parallel(request: pytest.FixtureRequest, tensor_parallel_size: int) -> bool:
    """Return whether the models run with sequence parallelism, which needs two TP ranks."""
    if request.param and tensor_parallel_size == 1:
        pytest.skip("sequence parallelism needs more than one TP rank")
    return request.param


def build_language_model(
    model_class: type[GPTModel],
    dtype: torch.dtype,
    sequence_parallel: bool,
    tensor_parallel_size: int,
) -> GPTModel:
    """Build a two-layer ``model_class`` without dropout, with tied input and output embeddings."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_parallel_size,
        sequence_parallel=sequence_parallel,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return model_class(
        config,
        get_gpt_layer_with_transformer_engine_spec(),
        VOCAB_SIZE,
        SEQUENCE_LENGTH,
        share_embeddings_and_output_weights=True,
    ).cuda()


@pytest.fixture(name="reference")
def build_reference(sequence_parallel: bool, tensor_parallel_size: int) -> GPTModel:
    """Build an fp32 ``GPTModel`` with random weights."""
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    return build_language_model(GPTModel, torch.float32, sequence_parallel, tensor_parallel_size)


@pytest.fixture(name="model")
def build_model(
    reference: GPTModel, dtype: torch.dtype, sequence_parallel: bool, tensor_parallel_size: int
) -> SpeechLanguageModel:
    """Build a ``SpeechLanguageModel`` in ``dtype`` that holds the reference weights."""
    model = build_language_model(
        SpeechLanguageModel, dtype, sequence_parallel, tensor_parallel_size
    )
    model.load_state_dict(reference.state_dict())
    return model


@pytest.fixture(name="multi_stream_model")
def build_multi_stream_model(
    reference: GPTModel, dtype: torch.dtype, sequence_parallel: bool, tensor_parallel_size: int
) -> MultiStreamLanguageModel:
    """Build a ``MultiStreamLanguageModel`` in ``dtype`` that holds the reference weights."""
    model = build_language_model(
        MultiStreamLanguageModel, dtype, sequence_parallel, tensor_parallel_size
    )
    model.load_state_dict(reference.state_dict())
    return model


@pytest.fixture(name="batch")
def build_batch(target_pattern: str, qkv_format: str) -> Batch:
    """Build a batch in the layout ``qkv_format`` with the targets ``target_pattern`` names.

    ``sbhd`` holds two sequences of 16 tokens, one per row; ``some`` puts the targets of the
    first at 10 to 15 and of the second at 3 to 5. ``thd`` packs sequences of 7, 5 and 4 tokens
    into one row, each with positions from 0; ``some`` puts the targets at the last two tokens
    of each sequence. ``all`` makes every position a target, and ``none`` no position.
    """
    torch.manual_seed(1)
    packed_seq_params = None
    if qkv_format == "thd":
        cu_seqlens = [0, *accumulate(PACKED_SEQUENCE_LENGTHS)]
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
            max_seqlen_q=max(PACKED_SEQUENCE_LENGTHS),
            max_seqlen_kv=max(PACKED_SEQUENCE_LENGTHS),
        )
        position_ids = torch.cat([torch.arange(n) for n in PACKED_SEQUENCE_LENGTHS])[None].cuda()
    else:
        position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda").expand(BATCH_SIZE, -1)
    loss_mask = torch.zeros(position_ids.shape, device="cuda")
    if target_pattern == "all":
        loss_mask[:] = 1
    elif target_pattern == "some" and qkv_format == "thd":
        for end in cu_seqlens[1:]:
            loss_mask[0, end - 2 : end] = 1
    elif target_pattern == "some":
        loss_mask[0, 10:] = 1
        loss_mask[1, 3:6] = 1
    return Batch(
        input_ids=torch.randint(VOCAB_SIZE, position_ids.shape, device="cuda"),
        position_ids=position_ids,
        labels=torch.randint(VOCAB_SIZE, position_ids.shape, device="cuda"),
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )


def run_reference(reference: GPTModel, batch: Batch) -> torch.Tensor:
    """Run ``GPTModel``, which computes the logits and the loss at every position."""
    return reference(
        batch.input_ids,
        batch.position_ids,
        None,
        labels=batch.labels,
        packed_seq_params=batch.packed_seq_params,
    )


def calc_loss_at_targets(model: SpeechLanguageModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss of the model under test with the labels and loss mask of ``batch``."""
    return model(
        batch.input_ids,
        batch.position_ids,
        None,
        labels=batch.labels,
        loss_mask=batch.loss_mask,
        packed_seq_params=batch.packed_seq_params,
    )


@pytest.fixture(name="reference_loss")
def calc_reference_loss(reference: GPTModel, batch: Batch) -> torch.Tensor:
    """Calculate the reference loss ``[batch, sequence]`` without gradients."""
    with torch.no_grad():
        return run_reference(reference, batch)


@pytest.fixture(name="model_loss")
def calc_model_loss(model: SpeechLanguageModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss ``[batch, sequence]`` of the model under test without gradients."""
    with torch.no_grad():
        return calc_loss_at_targets(model, batch)


@pytest.fixture(name="reference_gradients")
def calc_reference_gradients(reference: GPTModel, batch: Batch) -> dict[str, torch.Tensor | None]:
    """Calculate the reference gradients of the loss summed with ``loss_mask``, by parameter."""
    (run_reference(reference, batch) * batch.loss_mask).sum().backward()
    return {name: param.grad for name, param in reference.named_parameters()}


@pytest.fixture(name="model_gradients")
def calc_model_gradients(
    model: SpeechLanguageModel, batch: Batch
) -> dict[str, torch.Tensor | None]:
    """Calculate the gradients of the model under test for the same loss, by parameter."""
    (calc_loss_at_targets(model, batch) * batch.loss_mask).sum().backward()
    return {name: param.grad for name, param in model.named_parameters()}


@WITH_TARGETS
def test_loss_has_gpt_model_shape(
    model_loss: torch.Tensor, reference_loss: torch.Tensor, batch: Batch
) -> None:
    """Check that the loss is ``[batch, sequence]``, as ``GPTModel``'s."""
    assert model_loss.shape == reference_loss.shape == batch.loss_mask.shape


@WITH_TARGETS
def test_loss_matches_gpt_model_at_targets(
    model_loss: torch.Tensor, reference_loss: torch.Tensor, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that the loss at the target positions equals ``GPTModel``'s."""
    target_mask = batch.loss_mask.bool()
    error = calc_relative_error(model_loss[target_mask], reference_loss[target_mask])
    logger.info("%s: loss max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@WITH_TARGETS
def test_loss_is_zero_off_targets(model_loss: torch.Tensor, batch: Batch) -> None:
    """Check that the positions without a target get a loss of 0."""
    assert torch.all(model_loss[~batch.loss_mask.bool()] == 0)


@WITH_TARGETS
def test_gradients_match_gpt_model(
    model_gradients: dict[str, torch.Tensor | None],
    reference_gradients: dict[str, torch.Tensor | None],
    dtype: torch.dtype,
) -> None:
    """Check that the gradients of every parameter equal ``GPTModel``'s for the masked loss."""
    errors = {
        name: calc_relative_error(grad, reference_gradients[name])
        for name, grad in model_gradients.items()
    }
    logger.info(
        "%s: gradient max relative error %s",
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


@WITH_TARGETS
def test_returns_logits_without_labels(
    model: SpeechLanguageModel, reference: GPTModel, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that without labels the model returns ``GPTModel``'s logits."""
    with torch.no_grad():
        expected = reference(
            batch.input_ids, batch.position_ids, None, packed_seq_params=batch.packed_seq_params
        )
        actual = model(
            batch.input_ids, batch.position_ids, None, packed_seq_params=batch.packed_seq_params
        )
    assert actual.shape == expected.shape
    error = calc_relative_error(actual, expected)
    logger.info("%s: logits max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@pytest.mark.parametrize("target_pattern", ["none"])
def test_loss_is_zero_without_targets(model_loss: torch.Tensor) -> None:
    """Check that a batch without target positions gives a loss of 0 at every position."""
    assert torch.all(model_loss == 0)


@pytest.mark.parametrize("target_pattern", ["none"])
def test_gradients_are_zero_without_targets(
    model_gradients: dict[str, torch.Tensor | None],
) -> None:
    """Check that a batch without target positions gives every parameter a gradient of 0."""
    for name, grad in model_gradients.items():
        assert grad is not None, name
        assert torch.all(grad == 0), name


@WITH_TARGETS
def test_returns_gpt_model_loss_without_loss_mask(
    model: SpeechLanguageModel, reference_loss: torch.Tensor, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that with labels but without a loss mask the model returns ``GPTModel``'s loss."""
    with torch.no_grad():
        actual = model(
            batch.input_ids,
            batch.position_ids,
            None,
            labels=batch.labels,
            packed_seq_params=batch.packed_seq_params,
        )
    error = calc_relative_error(actual, reference_loss)
    logger.info("%s: loss without loss mask max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def calc_reference_last_layer_output(reference: GPTModel, batch: Batch) -> torch.Tensor:
    """Return the reference's last layer output, this rank's part under sequence parallelism."""
    return reference(
        batch.input_ids,
        batch.position_ids,
        None,
        packed_seq_params=batch.packed_seq_params,
        output_processor=lambda hidden_states, **_: hidden_states,
    )


def calc_multi_stream_output(model: MultiStreamLanguageModel, batch: Batch) -> torch.Tensor:
    """Return the output of the last layer of the model under test."""
    return model(
        batch.input_ids, batch.position_ids, None, packed_seq_params=batch.packed_seq_params
    )


def calc_multi_stream_target_loss(model: MultiStreamLanguageModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss ``[batch, sequence]`` with ``calc_target_loss`` on the model's output."""
    return model.calc_target_loss(
        calc_multi_stream_output(model, batch), batch.labels, batch.loss_mask
    )


def take_rank_part(
    whole_output: torch.Tensor, sequence_parallel: bool, tensor_parallel_size: int
) -> torch.Tensor:
    """Return the part of ``[sequence, ...]`` that this rank holds under sequence parallelism."""
    if not sequence_parallel:
        return whole_output
    return whole_output.chunk(tensor_parallel_size)[parallel_state.get_tensor_model_parallel_rank()]


@pytest.mark.parametrize("target_pattern", ["some"])
def test_multi_stream_output_matches_gpt_model(
    multi_stream_model: MultiStreamLanguageModel,
    reference: GPTModel,
    batch: Batch,
    dtype: torch.dtype,
    sequence_parallel: bool,
    tensor_parallel_size: int,
) -> None:
    """Check that every rank gets the whole output of ``GPTModel``'s last layer."""
    with torch.no_grad():
        output = calc_multi_stream_output(multi_stream_model, batch)
        reference_output = calc_reference_last_layer_output(reference, batch)
    batch_size, sequence_length = batch.input_ids.shape
    assert output.shape == (sequence_length, batch_size, reference_output.size(-1))
    error = calc_relative_error(
        take_rank_part(output, sequence_parallel, tensor_parallel_size), reference_output
    )
    logger.info("%s: last layer output max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@pytest.mark.parametrize("target_pattern", ["some"])
def test_multi_stream_output_gradients_match_gpt_model(
    multi_stream_model: MultiStreamLanguageModel,
    reference: GPTModel,
    batch: Batch,
    dtype: torch.dtype,
    sequence_parallel: bool,
    tensor_parallel_size: int,
) -> None:
    """Check that a weighted sum of the output gives every parameter ``GPTModel``'s gradient.

    Under sequence parallelism, the reference sums only this rank's part of the output.
    """
    output = calc_multi_stream_output(multi_stream_model, batch)
    torch.manual_seed(2)
    output_weights = torch.randn(output.shape).cuda()
    (output.float() * output_weights).sum().backward()
    reference_output = calc_reference_last_layer_output(reference, batch)
    rank_output_weights = take_rank_part(output_weights, sequence_parallel, tensor_parallel_size)
    (reference_output * rank_output_weights).sum().backward()
    reference_gradients = dict(reference.named_parameters())
    errors = {
        name: calc_relative_error(param.grad, reference_gradients[name].grad)
        for name, param in multi_stream_model.named_parameters()
    }
    logger.info(
        "%s: gradient max relative error for the output %s",
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


@WITH_TARGETS
def test_multi_stream_target_loss_matches_gpt_model(
    multi_stream_model: MultiStreamLanguageModel,
    reference_loss: torch.Tensor,
    batch: Batch,
    dtype: torch.dtype,
) -> None:
    """Check that ``calc_target_loss`` on the output gives ``GPTModel``'s loss at the targets."""
    with torch.no_grad():
        loss = calc_multi_stream_target_loss(multi_stream_model, batch)
    target_mask = batch.loss_mask.bool()
    error = calc_relative_error(loss[target_mask], reference_loss[target_mask])
    logger.info("%s: loss from the last layer output max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@WITH_TARGETS
def test_multi_stream_target_loss_gradients_match_gpt_model(
    multi_stream_model: MultiStreamLanguageModel,
    reference_gradients: dict[str, torch.Tensor | None],
    batch: Batch,
    dtype: torch.dtype,
) -> None:
    """Check that the masked loss from ``calc_target_loss`` gives ``GPTModel``'s gradients."""
    (calc_multi_stream_target_loss(multi_stream_model, batch) * batch.loss_mask).sum().backward()
    errors = {
        name: calc_relative_error(param.grad, reference_gradients[name])
        for name, param in multi_stream_model.named_parameters()
    }
    logger.info(
        "%s: gradient max relative error for the loss %s",
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


@pytest.mark.parametrize("target_pattern", ["some"])
@pytest.mark.parametrize("target_argument", ["labels", "loss_mask"])
def test_multi_stream_model_rejects_targets(
    multi_stream_model: MultiStreamLanguageModel, batch: Batch, target_argument: str
) -> None:
    """Check that ``labels`` or ``loss_mask`` in the call raises ValueError."""
    with pytest.raises(ValueError, match="calc_target_loss"):
        multi_stream_model(
            batch.input_ids,
            batch.position_ids,
            None,
            packed_seq_params=batch.packed_seq_params,
            **{target_argument: getattr(batch, target_argument)},
        )

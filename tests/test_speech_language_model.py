import logging
from dataclasses import dataclass

import pytest
import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

from megatron_speech.models.language_model import SpeechLanguageModel
from tests.numerical_error import TOLERANCES, calc_relative_error

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

VOCAB_SIZE = 128
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16


@dataclass
class Batch:
    """Token ids, positions, labels and loss mask of a batch, each ``[batch, sequence]``."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the model under test."""
    return request.param


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


@pytest.fixture(name="target_pattern", params=["some", "all"])
def get_target_pattern(request: pytest.FixtureRequest) -> str:
    """Return which positions of the batch are targets: ``some`` of them or ``all``."""
    return request.param


@pytest.fixture(name="batch")
def build_batch(target_pattern: str) -> Batch:
    """Build a batch whose target positions ``target_pattern`` names.

    ``some`` puts the targets of the first sequence at 10 to 15 and of the second at 3 to 5,
    ``all`` makes every position a target, and ``none`` no position.
    """
    torch.manual_seed(1)
    loss_mask = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, device="cuda")
    if target_pattern == "some":
        loss_mask[0, 10:] = 1
        loss_mask[1, 3:6] = 1
    elif target_pattern == "all":
        loss_mask[:] = 1
    return Batch(
        input_ids=torch.randint(VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"),
        position_ids=torch.arange(SEQUENCE_LENGTH, device="cuda").expand(BATCH_SIZE, -1),
        labels=torch.randint(VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"),
        loss_mask=loss_mask,
    )


def run_reference(reference: GPTModel, batch: Batch) -> torch.Tensor:
    """Run ``GPTModel``, which computes the logits and the loss at every position."""
    return reference(batch.input_ids, batch.position_ids, None, labels=batch.labels)


def run_model(model: SpeechLanguageModel, batch: Batch) -> torch.Tensor:
    """Run the model under test with the labels and the loss mask of ``batch``."""
    return model(
        batch.input_ids,
        batch.position_ids,
        None,
        labels=batch.labels,
        loss_mask=batch.loss_mask,
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
        return run_model(model, batch)


def test_loss_has_gpt_model_shape(model_loss: torch.Tensor, reference_loss: torch.Tensor) -> None:
    """Check that the loss is ``[batch, sequence]``, as ``GPTModel``'s."""
    assert model_loss.shape == reference_loss.shape == (BATCH_SIZE, SEQUENCE_LENGTH)


def test_loss_matches_gpt_model_at_targets(
    model_loss: torch.Tensor, reference_loss: torch.Tensor, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that the loss at the target positions equals ``GPTModel``'s."""
    target_mask = batch.loss_mask.bool()
    error = calc_relative_error(model_loss[target_mask], reference_loss[target_mask])
    logger.info("%s: loss max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_loss_is_zero_off_targets(model_loss: torch.Tensor, batch: Batch) -> None:
    """Check that the positions without a target get a loss of 0."""
    assert torch.all(model_loss[~batch.loss_mask.bool()] == 0)


def test_gradients_match_gpt_model(
    model: SpeechLanguageModel, reference: GPTModel, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that the gradients of every parameter equal ``GPTModel``'s for the masked loss."""
    (run_reference(reference, batch) * batch.loss_mask).sum().backward()
    (run_model(model, batch) * batch.loss_mask).sum().backward()
    reference_parameters = dict(reference.named_parameters())
    errors = {
        name: calc_relative_error(param.grad, reference_parameters[name].grad)
        for name, param in model.named_parameters()
    }
    logger.info(
        "%s: gradient max relative error %s",
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


def test_returns_logits_without_labels(
    model: SpeechLanguageModel, reference: GPTModel, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that without labels the model returns ``GPTModel``'s logits."""
    with torch.no_grad():
        expected = reference(batch.input_ids, batch.position_ids, None)
        actual = model(batch.input_ids, batch.position_ids, None)
    assert actual.shape == expected.shape
    error = calc_relative_error(actual, expected)
    logger.info("%s: logits max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@pytest.mark.parametrize("target_pattern", ["none"])
def test_loss_is_zero_without_targets(model_loss: torch.Tensor) -> None:
    """Check that a batch without target positions gives a loss of 0 at every position."""
    assert torch.all(model_loss == 0)


@pytest.mark.parametrize("target_pattern", ["none"])
def test_gradients_are_zero_without_targets(model: SpeechLanguageModel, batch: Batch) -> None:
    """Check that a batch without target positions gives every parameter a gradient of 0."""
    (run_model(model, batch) * batch.loss_mask).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.all(param.grad == 0), name


def test_returns_gpt_model_loss_without_loss_mask(
    model: SpeechLanguageModel, reference_loss: torch.Tensor, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that with labels but without a loss mask the model returns ``GPTModel``'s loss."""
    with torch.no_grad():
        actual = model(batch.input_ids, batch.position_ids, None, labels=batch.labels)
    error = calc_relative_error(actual, reference_loss)
    logger.info("%s: loss without loss mask max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]

import logging
from dataclasses import dataclass, replace
from itertools import accumulate

import pytest
import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron_speech.components.projectors.mlp import MLPProjector, MLPProjectorConfig
from megatron_speech.models.multi_stream import MultiStreamMimoModel
from tests.numerical_error import TOLERANCES, WITH_DTYPES, calc_relative_error

pytestmark = [pytest.mark.gpu, WITH_DTYPES]
logger = logging.getLogger(__name__)

VOCAB_SIZE = 128
HIDDEN_SIZE = 64
STREAM_INPUT_SIZE = 16
ENCODER_NAME = "encoder"
# Two sequences packed in one row, of 9 and 6 positions.
SEQUENCE_LENGTHS = [9, 6]
# Positions of each modality in the row. Positions 2 and 8 hold all three; position 5 holds none.
MODALITY_POSITIONS = {
    "text": [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13],
    "first_stream": [2, 4, 6, 8, 10, 12, 14],
    "second_stream": [1, 2, 7, 8, 11],
}
STREAM_NAMES = [name for name in MODALITY_POSITIONS if name != "text"]
WORD_EMBEDDINGS = "language_model.embedding.word_embeddings.weight"


@dataclass
class Batch:
    """One row of packed sequences, as ``MimoModel.forward`` takes it.

    ``input_ids``, ``position_ids``, ``labels`` and ``loss_mask`` are ``[1, num_positions]``.
    ``stream_inputs`` holds the ``[num_stream_positions, STREAM_INPUT_SIZE]`` input of each
    stream, and ``modality_token_indices`` the positions of each modality.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    stream_inputs: dict[str, torch.Tensor]
    modality_token_indices: dict[str, torch.Tensor] | None


@pytest.fixture(name="model")
def build_model(dtype: torch.dtype, tensor_parallel_size: int) -> MultiStreamMimoModel:
    """Build a two-layer LLM with rotary positions and a one-layer projector for each stream."""
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    stream_spec = ModuleSpec(
        module=AudioModalitySubmodules,
        submodules={
            "encoders": {
                ENCODER_NAME: ModuleSpec(
                    module=MLPProjector,
                    params={
                        "projector_config": MLPProjectorConfig(
                            input_size=STREAM_INPUT_SIZE, output_size=HIDDEN_SIZE, num_layers=1
                        ),
                        "llm_config": transformer_config,
                        "pg_collection": pg_collection,
                    },
                )
            }
        },
    )
    mimo_config = MimoModelConfig(
        language_model_spec=ModuleSpec(
            module=GPTModel,
            params={
                "config": transformer_config,
                "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
                "vocab_size": VOCAB_SIZE,
                "max_sequence_length": 64,
                "position_embedding_type": "rope",
                "scatter_embedding_sequence_parallel": False,
                "pg_collection": pg_collection,
            },
        ),
        modality_submodules_spec=dict.fromkeys(STREAM_NAMES, stream_spec),
        kv_format="thd",
    )
    return MultiStreamMimoModel(mimo_config, cp_group=pg_collection.cp, tp_group=pg_collection.tp)


@pytest.fixture(name="batch")
def build_batch(dtype: torch.dtype) -> Batch:
    """Build a row of two sequences whose modalities sit at ``MODALITY_POSITIONS``.

    The stream inputs are in ``dtype``, the dtype of the projectors that take them.
    """
    torch.manual_seed(1)
    num_positions = sum(SEQUENCE_LENGTHS)
    return Batch(
        input_ids=torch.randint(VOCAB_SIZE, (1, num_positions), device="cuda"),
        position_ids=torch.cat([torch.arange(n) for n in SEQUENCE_LENGTHS])[None].cuda(),
        labels=torch.randint(VOCAB_SIZE, (1, num_positions), device="cuda"),
        loss_mask=torch.ones(1, num_positions, device="cuda"),
        cu_seqlens=torch.tensor(
            [0, *accumulate(SEQUENCE_LENGTHS)], dtype=torch.int32, device="cuda"
        ),
        max_seqlen=max(SEQUENCE_LENGTHS),
        stream_inputs={
            name: torch.randn(
                len(MODALITY_POSITIONS[name]), STREAM_INPUT_SIZE, device="cuda", dtype=dtype
            )
            for name in STREAM_NAMES
        },
        modality_token_indices={
            name: torch.tensor(positions, device="cuda")
            for name, positions in MODALITY_POSITIONS.items()
        },
    )


def calc_mimo_loss(model: MultiStreamMimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss ``[1, num_positions]`` of the model under test."""
    loss, _ = model(
        input_ids=batch.input_ids,
        position_ids=batch.position_ids,
        loss_mask=batch.loss_mask,
        labels=batch.labels,
        modality_inputs={
            name: {ENCODER_NAME: {"hidden_states": inputs}}
            for name, inputs in batch.stream_inputs.items()
        },
        cu_seqlens=batch.cu_seqlens,
        max_seqlen=torch.tensor(batch.max_seqlen),
        modality_token_indices=batch.modality_token_indices,
    )
    return loss


def calc_reference_llm_input(model: MultiStreamMimoModel, batch: Batch) -> torch.Tensor:
    """Sum the word embeddings and the projected streams at their positions.

    The input is ``[num_positions, 1, hidden]``. The word embedding counts at the text positions
    only; the LLM's rotary positions add nothing to it.
    """
    # The word embeddings are split over the TP ranks along the vocabulary.
    word_embeddings = gather_from_tensor_model_parallel_region(
        dict(model.named_parameters())[WORD_EMBEDDINGS].t()
    ).t()
    text_positions = batch.modality_token_indices["text"]
    llm_input = torch.zeros(batch.input_ids.size(1), HIDDEN_SIZE, device="cuda")
    llm_input[text_positions] += word_embeddings[batch.input_ids[0, text_positions]]
    for name, inputs in batch.stream_inputs.items():
        projector = model.modality_submodules[name].encoders[ENCODER_NAME]
        stream_embeddings = projector(inputs)
        llm_input[batch.modality_token_indices[name]] += stream_embeddings.float()
    return llm_input.to(word_embeddings.dtype).unsqueeze(1)


def run_reference(model: MultiStreamMimoModel, batch: Batch) -> torch.Tensor:
    """Run the projectors, sum the embeddings by hand, then run the LLM."""
    return model.language_model(
        None,
        batch.position_ids,
        None,
        decoder_input=calc_reference_llm_input(model, batch),
        labels=batch.labels,
        loss_mask=batch.loss_mask,
        packed_seq_params=PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=batch.cu_seqlens,
            cu_seqlens_kv=batch.cu_seqlens,
            max_seqlen_q=batch.max_seqlen,
            max_seqlen_kv=batch.max_seqlen,
        ),
    )


@pytest.fixture(name="model_llm_input")
def calc_model_llm_input(model: MultiStreamMimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the embeddings that the model under test passes to the LLM."""
    captured: dict[str, torch.Tensor] = {}

    def save_llm_input(
        module: torch.nn.Module,
        args: tuple[torch.Tensor | None, ...],
        kwargs: dict[str, torch.Tensor | None],
    ) -> None:
        """Keep the ``decoder_input`` that ``MimoModel`` passes to the LLM."""
        captured["decoder_input"] = kwargs["decoder_input"]

    handle = model.language_model.register_forward_pre_hook(save_llm_input, with_kwargs=True)
    try:
        with torch.no_grad():
            calc_mimo_loss(model, batch)
    finally:
        handle.remove()
    return captured["decoder_input"]


def test_llm_input_matches_reference(
    model_llm_input: torch.Tensor,
    model: MultiStreamMimoModel,
    batch: Batch,
    dtype: torch.dtype,
) -> None:
    """Check that the LLM's input sums the text and stream embeddings at their positions."""
    with torch.no_grad():
        reference_llm_input = calc_reference_llm_input(model, batch)
    assert model_llm_input.shape == reference_llm_input.shape
    error = calc_relative_error(model_llm_input, reference_llm_input)
    logger.info("%s: LLM input max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_position_without_modality_gets_zeros(model_llm_input: torch.Tensor) -> None:
    """Check that position 5, which no modality holds, gets a zero embedding."""
    assert torch.equal(model_llm_input[5], torch.zeros_like(model_llm_input[5]))


def test_loss_matches_reference(
    model: MultiStreamMimoModel, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that the loss equals the reference that sums the embeddings by hand."""
    with torch.no_grad():
        model_loss = calc_mimo_loss(model, batch)
        reference_loss = run_reference(model, batch)
    error = calc_relative_error(model_loss, reference_loss)
    logger.info("%s: loss max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def collect_gradients(model: MultiStreamMimoModel) -> dict[str, torch.Tensor | None]:
    """Copy the gradients of the parameters of ``model``, by parameter name."""
    return {
        name: None if param.grad is None else param.grad.clone()
        for name, param in model.named_parameters()
    }


@pytest.fixture(name="model_gradients")
def calc_model_gradients(
    model: MultiStreamMimoModel, batch: Batch
) -> dict[str, torch.Tensor | None]:
    """Calculate the gradients of the summed loss through the model under test."""
    model.zero_grad(set_to_none=True)
    calc_mimo_loss(model, batch).sum().backward()
    return collect_gradients(model)


@pytest.fixture(name="reference_gradients")
def calc_reference_gradients(
    model: MultiStreamMimoModel, batch: Batch
) -> dict[str, torch.Tensor | None]:
    """Calculate the gradients of the same loss through the reference."""
    model.zero_grad(set_to_none=True)
    run_reference(model, batch).sum().backward()
    return collect_gradients(model)


def test_gradients_match_reference(
    model_gradients: dict[str, torch.Tensor | None],
    reference_gradients: dict[str, torch.Tensor | None],
    dtype: torch.dtype,
) -> None:
    """Check that the gradients of every parameter, the projectors' included, match."""
    errors = {
        name: calc_relative_error(grad, reference_gradients[name])
        for name, grad in model_gradients.items()
    }
    name, error = max(errors.items(), key=lambda item: item[1])
    logger.info("%s: gradient max relative error %.2e (%s)", dtype, error, name)
    assert error < TOLERANCES[dtype], errors


def test_accepts_batch_without_streams(
    model: MultiStreamMimoModel, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that a row with text only gives the loss of the LLM on the text embeddings."""
    text_only_batch = replace(
        batch,
        stream_inputs={},
        modality_token_indices={"text": batch.modality_token_indices["text"]},
    )
    with torch.no_grad():
        model_loss = calc_mimo_loss(model, text_only_batch)
        reference_loss = run_reference(model, text_only_batch)
    error = calc_relative_error(model_loss, reference_loss)
    logger.info("%s: loss without streams max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_rejects_call_without_positions(model: MultiStreamMimoModel, batch: Batch) -> None:
    """Check that a call without ``modality_token_indices`` raises ValueError."""
    with pytest.raises(ValueError, match="needs modality_token_indices"):
        calc_mimo_loss(model, replace(batch, modality_token_indices=None))


def test_rejects_positions_of_other_modalities(model: MultiStreamMimoModel, batch: Batch) -> None:
    """Check that positions missing for a stream with inputs raise ValueError."""
    modality_token_indices = {
        name: batch.modality_token_indices[name] for name in ["text", "first_stream"]
    }
    with pytest.raises(ValueError, match="modality_token_indices has"):
        calc_mimo_loss(model, replace(batch, modality_token_indices=modality_token_indices))


def test_rejects_positions_of_another_count(model: MultiStreamMimoModel, batch: Batch) -> None:
    """Check that a stream with more inputs than positions raises ValueError."""
    stream_inputs = dict(batch.stream_inputs)
    stream_inputs["first_stream"] = stream_inputs["first_stream"][:1]
    with pytest.raises(ValueError, match="first_stream has 1 embeddings but 7 positions"):
        calc_mimo_loss(model, replace(batch, stream_inputs=stream_inputs))

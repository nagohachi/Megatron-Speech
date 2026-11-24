import logging
from dataclasses import dataclass, replace
from itertools import accumulate

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron_speech.components.projectors.mlp import MLPProjector, MLPProjectorConfig
from megatron_speech.models.language_model import MultiStreamLanguageModel
from megatron_speech.models.multi_stream import MultiStreamLosses, MultiStreamMimoModel
from tests.numerical_error import TOLERANCES, WITH_DTYPES, calc_relative_error

pytestmark = [
    pytest.mark.gpu,
    WITH_DTYPES,
    # How the batch holds its sequences: one per row, or packed into one row.
    pytest.mark.parametrize("kv_format", ["sbhd", "thd"]),
]
logger = logging.getLogger(__name__)

VOCAB_SIZE = 128
HIDDEN_SIZE = 64
STREAM_INPUT_SIZE = 16
NUM_AUDIO_FEATURES = 8
ENCODER_NAME = "encoder"
NUM_POSITIONS = 16
# The sbhd batch holds two rows of 8 positions. The thd batch packs sequences of 9 and 7
# positions into one row.
NUM_SBHD_ROWS = 2
PACKED_SEQUENCE_LENGTHS = [9, 7]
# Positions of each modality, as flat indices row_index * positions_per_row + position. Some
# positions hold all three, and some none.
MODALITY_POSITIONS = {
    "text": [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13],
    "first_stream": [2, 4, 6, 8, 10, 12, 14],
    "second_stream": [1, 2, 7, 8, 11],
}
STREAM_NAMES = [name for name in MODALITY_POSITIONS if name != "text"]
NUM_FIRST_STREAM_POSITIONS = len(MODALITY_POSITIONS["first_stream"])
EMPTY_POSITIONS = sorted(set(range(NUM_POSITIONS)).difference(*MODALITY_POSITIONS.values()))
# Weight of the cross entropy at each position, 0 where the text has no target.
TEXT_LOSS_WEIGHTS = [1, 1, 0.5, 1, 0, 0.5, 1, 1, 0, 1, 0.5, 1, 1, 0.5, 1, 0]
# Positions from which the audio decoder predicts the audio; the sbhd batch has them in both rows.
AUDIO_DECODER_POSITIONS = [1, 3, 5, 7, 9, 11, 13]
# Weights of the two losses in the total loss, different so that swapping them changes it.
TEXT_LOSS_WEIGHT = 0.5
AUDIO_LOSS_WEIGHT = 2.0
WORD_EMBEDDINGS = "language_model.embedding.word_embeddings.weight"
OUTPUT_LAYER = "language_model.output_layer.weight"
AUDIO_DECODER_LAYER = "audio_decoder.linear"


class LinearAudioDecoder(torch.nn.Module):
    """Audio decoder for the tests: a linear layer and the squared error to the audio features."""

    def __init__(self, params_dtype: torch.dtype) -> None:
        """Build the linear layer from the LLM's hidden size to the audio features."""
        super().__init__()
        self.linear = torch.nn.Linear(
            HIDDEN_SIZE, NUM_AUDIO_FEATURES, device="cuda", dtype=params_dtype
        )

    def forward(self, hidden_states: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """Return the mean squared error of the predicted ``audio_features``, or 0 for no rows."""
        squared_errors = (self.linear(hidden_states).float() - audio_features).square()
        return squared_errors.sum() / max(squared_errors.numel(), 1)


@dataclass
class Batch:
    """Inputs of ``MultiStreamMimoModel.forward`` for one batch.

    ``input_ids``, ``position_ids``, ``labels`` and ``loss_mask`` are ``[rows, positions]``.
    ``cu_seqlens`` and ``max_seqlen`` delimit the packed sequences, or are None for one sequence
    per row. ``stream_inputs`` holds the ``[num_stream_positions, STREAM_INPUT_SIZE]`` input of
    each stream, and ``modality_token_indices`` the positions of each modality.
    ``audio_features`` is the ``[num_audio_decoder_positions, NUM_AUDIO_FEATURES]`` target of
    the audio decoder at ``audio_decoder_token_indices``.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    cu_seqlens: torch.Tensor | None
    max_seqlen: int | None
    stream_inputs: dict[str, torch.Tensor]
    modality_token_indices: dict[str, torch.Tensor] | None
    audio_decoder_token_indices: torch.Tensor
    audio_features: torch.Tensor


@pytest.fixture(name="transformer_config")
def build_transformer_config(
    dtype: torch.dtype, sequence_parallel: bool, tensor_parallel_size: int
) -> TransformerConfig:
    """Build the config of a two-layer LLM without dropout."""
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_parallel_size,
        sequence_parallel=sequence_parallel,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def build_mimo_config(
    transformer_config: TransformerConfig, language_model_class: type[GPTModel], kv_format: str
) -> MimoModelConfig:
    """Build the config of an LLM with rotary positions and a one-layer projector per stream."""
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
    return MimoModelConfig(
        language_model_spec=ModuleSpec(
            module=language_model_class,
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
        kv_format=kv_format,
    )


@pytest.fixture(name="model")
def build_model(
    transformer_config: TransformerConfig, dtype: torch.dtype, kv_format: str
) -> MultiStreamMimoModel:
    """Build the model under test with ``LinearAudioDecoder`` as its audio decoder."""
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    return MultiStreamMimoModel(
        build_mimo_config(transformer_config, MultiStreamLanguageModel, kv_format),
        ModuleSpec(module=LinearAudioDecoder, params={"params_dtype": dtype}),
        TEXT_LOSS_WEIGHT,
        AUDIO_LOSS_WEIGHT,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )


@pytest.fixture(name="batch")
def build_batch(dtype: torch.dtype, kv_format: str) -> Batch:
    """Build a batch whose modalities sit at ``MODALITY_POSITIONS``.

    The stream inputs are in ``dtype``, the dtype of the projectors that take them.
    """
    torch.manual_seed(1)
    if kv_format == "thd":
        shape = (1, NUM_POSITIONS)
        position_ids = torch.cat([torch.arange(length) for length in PACKED_SEQUENCE_LENGTHS])
        cu_seqlens = torch.tensor(
            [0, *accumulate(PACKED_SEQUENCE_LENGTHS)], dtype=torch.int32, device="cuda"
        )
        max_seqlen = max(PACKED_SEQUENCE_LENGTHS)
    else:
        shape = (NUM_SBHD_ROWS, NUM_POSITIONS // NUM_SBHD_ROWS)
        position_ids = torch.arange(shape[1]).repeat(NUM_SBHD_ROWS)
        cu_seqlens = max_seqlen = None
    return Batch(
        input_ids=torch.randint(VOCAB_SIZE, shape, device="cuda"),
        position_ids=position_ids.view(shape).cuda(),
        labels=torch.randint(VOCAB_SIZE, shape, device="cuda"),
        loss_mask=torch.tensor(TEXT_LOSS_WEIGHTS, device="cuda").view(shape),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
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
        audio_decoder_token_indices=torch.tensor(AUDIO_DECODER_POSITIONS, device="cuda"),
        audio_features=torch.randn(len(AUDIO_DECODER_POSITIONS), NUM_AUDIO_FEATURES, device="cuda"),
    )


def calc_mimo_losses(model: MultiStreamMimoModel, batch: Batch) -> MultiStreamLosses:
    """Calculate the losses of the model under test."""
    return model(
        batch.input_ids,
        batch.position_ids,
        labels=batch.labels,
        loss_mask=batch.loss_mask,
        modality_inputs={
            name: {ENCODER_NAME: {"hidden_states": inputs}}
            for name, inputs in batch.stream_inputs.items()
        },
        modality_token_indices=batch.modality_token_indices,
        audio_decoder_token_indices=batch.audio_decoder_token_indices,
        audio_decoder_inputs={"audio_features": batch.audio_features},
        cu_seqlens=batch.cu_seqlens,
        max_seqlen=None if batch.max_seqlen is None else torch.tensor(batch.max_seqlen),
    )


def sum_reference_llm_input(model: MultiStreamMimoModel, batch: Batch) -> torch.Tensor:
    """Sum the word embeddings and the projected streams at their positions.

    The input is ``[NUM_POSITIONS, hidden]``, in the order of the flat indices. The word
    embedding counts at the text positions only; the LLM's rotary positions add nothing to it.
    """
    parameters = dict(model.named_parameters())
    # The TP ranks split the word embeddings along the vocabulary and each projector's linear
    # layer along its output features.
    word_embeddings = gather_from_tensor_model_parallel_region(parameters[WORD_EMBEDDINGS].t()).t()
    text_positions = batch.modality_token_indices["text"]
    llm_input = torch.zeros(NUM_POSITIONS, HIDDEN_SIZE, device="cuda")
    llm_input[text_positions] += word_embeddings[batch.input_ids.flatten()[text_positions]]
    for name, inputs in batch.stream_inputs.items():
        layer_name = f"modality_submodules.{name}.encoders.{ENCODER_NAME}.layers.0"
        weight = gather_from_tensor_model_parallel_region(
            parameters[f"{layer_name}.weight"].t()
        ).t()
        bias = gather_from_tensor_model_parallel_region(parameters[f"{layer_name}.bias"])
        stream_embeddings = torch.nn.functional.linear(inputs, weight, bias)
        llm_input[batch.modality_token_indices[name]] += stream_embeddings.float()
    return llm_input.to(word_embeddings.dtype)


def run_reference(
    model: MultiStreamMimoModel, batch: Batch, sequence_parallel: bool
) -> MultiStreamLosses:
    """Calculate the losses by hand from the parameters and the output of the model's LLM.

    The embeddings are summed by hand, and the cross entropy and the audio decoder are
    computed from the weights of the output layer and of the decoder.
    """
    num_rows, num_positions_per_row = batch.input_ids.shape
    llm_input = (
        sum_reference_llm_input(model, batch)
        .view(num_rows, num_positions_per_row, HIDDEN_SIZE)
        .transpose(0, 1)
        .contiguous()
    )
    if sequence_parallel:
        llm_input = scatter_to_sequence_parallel_region(llm_input)
    packed_seq_params = None
    if batch.cu_seqlens is not None:
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=batch.cu_seqlens,
            cu_seqlens_kv=batch.cu_seqlens,
            max_seqlen_q=batch.max_seqlen,
            max_seqlen_kv=batch.max_seqlen,
        )
    hidden_states = model.language_model(
        None, batch.position_ids, None, decoder_input=llm_input, packed_seq_params=packed_seq_params
    )
    # Rows in the order of the flat indices.
    flat_hidden_states = hidden_states.transpose(0, 1).reshape(NUM_POSITIONS, HIDDEN_SIZE)
    parameters = dict(model.named_parameters())
    output_weight = gather_from_tensor_model_parallel_region(parameters[OUTPUT_LAYER].t()).t()
    text_losses = torch.nn.functional.cross_entropy(
        (flat_hidden_states @ output_weight.t()).float(), batch.labels.flatten(), reduction="none"
    )
    text_loss_weights = batch.loss_mask.flatten()
    text_loss = (text_losses * text_loss_weights).sum() / text_loss_weights.sum()
    predicted_features = torch.nn.functional.linear(
        flat_hidden_states[batch.audio_decoder_token_indices],
        parameters[f"{AUDIO_DECODER_LAYER}.weight"],
        parameters[f"{AUDIO_DECODER_LAYER}.bias"],
    )
    audio_loss = (predicted_features.float() - batch.audio_features).square().mean()
    return MultiStreamLosses(
        text_loss=text_loss,
        audio_loss=audio_loss,
        total_loss=TEXT_LOSS_WEIGHT * text_loss + AUDIO_LOSS_WEIGHT * audio_loss,
    )


@pytest.fixture(name="model_llm_input")
def calc_model_llm_input(
    model: MultiStreamMimoModel, batch: Batch, sequence_parallel: bool
) -> torch.Tensor:
    """Calculate the embeddings ``[NUM_POSITIONS, hidden]`` that the model passes to the LLM."""
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
            calc_mimo_losses(model, batch)
    finally:
        handle.remove()
    llm_input = captured["decoder_input"]
    if sequence_parallel:
        # MimoModel passes each TP rank its part of the sequence.
        llm_input = gather_from_sequence_parallel_region(llm_input)
    return llm_input.transpose(0, 1).reshape(NUM_POSITIONS, HIDDEN_SIZE)


def test_llm_input_matches_reference(
    model_llm_input: torch.Tensor,
    model: MultiStreamMimoModel,
    batch: Batch,
    dtype: torch.dtype,
) -> None:
    """Check that the LLM's input sums the text and stream embeddings at their positions."""
    with torch.no_grad():
        reference_llm_input = sum_reference_llm_input(model, batch)
    assert model_llm_input.shape == reference_llm_input.shape
    error = calc_relative_error(model_llm_input, reference_llm_input)
    logger.info("%s: LLM input max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_positions_without_modality_get_zeros(model_llm_input: torch.Tensor) -> None:
    """Check that the positions that no modality holds get zero embeddings."""
    empty_position_inputs = model_llm_input[EMPTY_POSITIONS]
    assert empty_position_inputs.numel() > 0
    assert torch.equal(empty_position_inputs, torch.zeros_like(empty_position_inputs))


def test_losses_match_reference(
    model: MultiStreamMimoModel, batch: Batch, dtype: torch.dtype, sequence_parallel: bool
) -> None:
    """Check that the text loss, the audio loss and the total loss equal the reference."""
    with torch.no_grad():
        model_losses = calc_mimo_losses(model, batch)
        reference_losses = vars(run_reference(model, batch, sequence_parallel))
    errors = {
        name: calc_relative_error(loss, reference_losses[name])
        for name, loss in vars(model_losses).items()
    }
    logger.info(
        "%s: max relative error %s",
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


def collect_gradients(model: MultiStreamMimoModel) -> dict[str, torch.Tensor | None]:
    """Copy the gradients of the parameters of ``model``, by parameter name."""
    return {
        name: None if param.grad is None else param.grad.clone()
        for name, param in model.named_parameters()
    }


def test_gradients_match_reference(
    model: MultiStreamMimoModel, batch: Batch, dtype: torch.dtype, sequence_parallel: bool
) -> None:
    """Check that the total loss gives every parameter the reference gradient.

    The parameters include the projectors, the LLM's output layer and the audio decoder.
    """
    model.zero_grad(set_to_none=True)
    calc_mimo_losses(model, batch).total_loss.backward()
    model_gradients = collect_gradients(model)
    model.zero_grad(set_to_none=True)
    run_reference(model, batch, sequence_parallel).total_loss.backward()
    reference_gradients = collect_gradients(model)
    errors = {
        name: calc_relative_error(grad, reference_gradients[name])
        for name, grad in model_gradients.items()
    }
    name, error = max(errors.items(), key=lambda item: item[1])
    logger.info("%s: gradient max relative error %.2e (%s)", dtype, error, name)
    assert error < TOLERANCES[dtype], errors


def test_accepts_batch_without_streams(
    model: MultiStreamMimoModel, batch: Batch, dtype: torch.dtype, sequence_parallel: bool
) -> None:
    """Check that a batch with text only gives the reference total loss."""
    text_only_batch = replace(
        batch,
        stream_inputs={},
        modality_token_indices={"text": batch.modality_token_indices["text"]},
    )
    with torch.no_grad():
        model_loss = calc_mimo_losses(model, text_only_batch).total_loss
        reference_loss = run_reference(model, text_only_batch, sequence_parallel).total_loss
    error = calc_relative_error(model_loss, reference_loss)
    logger.info("%s: total loss without streams max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@pytest.fixture(name="batch_without_text_targets")
def build_batch_without_text_targets(batch: Batch) -> Batch:
    """Build ``batch`` with text loss weights of all 0."""
    return replace(batch, loss_mask=torch.zeros_like(batch.loss_mask))


def test_text_loss_is_zero_without_text_targets(
    model: MultiStreamMimoModel, batch_without_text_targets: Batch
) -> None:
    """Check that a batch whose text loss weights are all 0 gives a text loss of 0."""
    with torch.no_grad():
        losses = calc_mimo_losses(model, batch_without_text_targets)
    assert losses.text_loss.item() == 0


def test_gradients_are_finite_without_text_targets(
    model: MultiStreamMimoModel, batch_without_text_targets: Batch
) -> None:
    """Check that text loss weights of all 0 give every parameter a finite gradient."""
    calc_mimo_losses(model, batch_without_text_targets).total_loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.all(param.grad.isfinite()), name


def test_audio_decoder_gets_zero_gradients_without_audio_targets(
    model: MultiStreamMimoModel, batch: Batch
) -> None:
    """Check that a batch without audio decoder positions still runs the decoder, with no effect.

    Megatron's DDP needs a gradient for every parameter, so the decoder gets gradients of 0.
    """
    batch_without_audio_targets = replace(
        batch,
        audio_decoder_token_indices=batch.audio_decoder_token_indices[:0],
        audio_features=batch.audio_features[:0],
    )
    calc_mimo_losses(model, batch_without_audio_targets).total_loss.backward()
    for name, param in model.audio_decoder.named_parameters():
        assert param.grad is not None, name
        assert torch.all(param.grad == 0), name


def test_rejects_call_without_positions(model: MultiStreamMimoModel, batch: Batch) -> None:
    """Check that a call without ``modality_token_indices`` raises ValueError."""
    with pytest.raises(ValueError, match="needs modality_token_indices"):
        calc_mimo_losses(model, replace(batch, modality_token_indices=None))


def test_rejects_positions_of_other_modalities(model: MultiStreamMimoModel, batch: Batch) -> None:
    """Check that positions missing for a stream with inputs raise ValueError."""
    modality_token_indices = {
        name: batch.modality_token_indices[name] for name in ["text", "first_stream"]
    }
    with pytest.raises(ValueError, match="modality_token_indices has"):
        calc_mimo_losses(model, replace(batch, modality_token_indices=modality_token_indices))


# Fewer and more inputs than the positions of the first stream.
@pytest.mark.parametrize("num_inputs", [1, NUM_FIRST_STREAM_POSITIONS + 1])
def test_rejects_positions_of_another_count(
    model: MultiStreamMimoModel, batch: Batch, num_inputs: int
) -> None:
    """Check that a stream with another number of inputs than positions raises ValueError."""
    stream_inputs = dict(batch.stream_inputs)
    stream_inputs["first_stream"] = stream_inputs["first_stream"].repeat(2, 1)[:num_inputs]
    message = f"first_stream has {num_inputs} embeddings but {NUM_FIRST_STREAM_POSITIONS} positions"
    with pytest.raises(ValueError, match=message):
        calc_mimo_losses(model, replace(batch, stream_inputs=stream_inputs))


def test_rejects_language_model_of_another_class(
    transformer_config: TransformerConfig, dtype: torch.dtype, kv_format: str
) -> None:
    """Check that an LLM other than ``MultiStreamLanguageModel`` raises ValueError."""
    with pytest.raises(ValueError, match="must be MultiStreamLanguageModel"):
        MultiStreamMimoModel(
            build_mimo_config(transformer_config, GPTModel, kv_format),
            ModuleSpec(module=LinearAudioDecoder, params={"params_dtype": dtype}),
            TEXT_LOSS_WEIGHT,
            AUDIO_LOSS_WEIGHT,
        )


def test_rejects_context_parallelism(
    transformer_config: TransformerConfig, dtype: torch.dtype, kv_format: str
) -> None:
    """Check that an LLM config with context parallelism raises ValueError."""
    context_parallel_config = replace(transformer_config, context_parallel_size=2)
    with pytest.raises(ValueError, match="context_parallel_size=2"):
        MultiStreamMimoModel(
            build_mimo_config(context_parallel_config, MultiStreamLanguageModel, kv_format),
            ModuleSpec(module=LinearAudioDecoder, params={"params_dtype": dtype}),
            TEXT_LOSS_WEIGHT,
            AUDIO_LOSS_WEIGHT,
        )

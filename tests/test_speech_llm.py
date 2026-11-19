import logging
from dataclasses import dataclass, fields, replace
from itertools import accumulate

import pytest
import torch
from megatron.core.models.mimo import MimoModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron_speech.components.encoders.qwen3_asr import (
    Qwen3ASREncoderConfig,
    Qwen3ASREncoderInputs,
)
from megatron_speech.components.projectors.mlp import MLPProjectorConfig
from megatron_speech.models.speech_llm import (
    AUDIO_MODALITY,
    ENCODER_NAME,
    SpeechLLMConfig,
    build_speech_llm,
)
from tests.numerical_error import TOLERANCES, calc_relative_error

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

VOCAB_SIZE = 128
AUDIO_TOKEN_ID = VOCAB_SIZE - 1
# Mel frames of the utterances of the batch: 13 and 5 encoder frames.
MEL_LENGTHS = [100, 37]
# Text tokens before the audio of each utterance, and target tokens after it.
NUM_PROMPT_TOKENS = 3
NUM_TARGET_TOKENS = 4
WORD_EMBEDDINGS = "language_model.embedding.word_embeddings.weight"
ENCODER_CONFIG = Qwen3ASREncoderConfig(
    num_mel_bins=32,
    d_model=32,
    encoder_layers=1,
    encoder_attention_heads=4,
    encoder_ffn_dim=64,
    downsample_hidden_size=8,
    n_window_infer=200,
)


@dataclass
class Batch:
    """One row of packed utterances, as ``MimoModel.forward`` takes it.

    ``input_ids``, ``position_ids``, ``labels`` and ``loss_mask`` are ``[1, num_tokens]``;
    ``cu_seqlens`` delimits the utterances in the row.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    encoder_inputs: Qwen3ASREncoderInputs | None


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the model under test."""
    return request.param


@pytest.fixture(name="config")
def build_config(dtype: torch.dtype, tensor_parallel_size: int) -> SpeechLLMConfig:
    """Build a small speech LLM config: a two-layer LLM with rotary positions."""
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return SpeechLLMConfig(
        llm=GPTModelConfig(
            transformer=transformer,
            vocab_size=VOCAB_SIZE,
            seq_length=64,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
        ),
        encoder=ENCODER_CONFIG,
        projector=MLPProjectorConfig(input_size=32, output_size=64, hidden_size=64),
        audio_token_id=AUDIO_TOKEN_ID,
    )


@pytest.fixture(name="model")
def build_model(config: SpeechLLMConfig) -> MimoModel:
    """Build the speech LLM with random weights."""
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    return build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups()).cuda()


def build_row(num_audio_tokens: list[int]) -> Batch:
    """Build a row of utterances: prompt tokens, audio tokens, then target tokens.

    The utterances have ``num_audio_tokens`` audio tokens each; the target tokens take the loss.
    """
    input_ids, loss_mask = [], []
    for num_audio in num_audio_tokens:
        input_ids += [
            torch.randint(AUDIO_TOKEN_ID, (NUM_PROMPT_TOKENS,)),
            torch.full((num_audio,), AUDIO_TOKEN_ID),
            torch.randint(AUDIO_TOKEN_ID, (NUM_TARGET_TOKENS,)),
        ]
        loss_mask += [torch.zeros(NUM_PROMPT_TOKENS + num_audio), torch.ones(NUM_TARGET_TOKENS)]
    lengths = [NUM_PROMPT_TOKENS + n + NUM_TARGET_TOKENS for n in num_audio_tokens]
    tokens = torch.cat(input_ids)
    return Batch(
        input_ids=tokens[None].cuda(),
        position_ids=torch.cat([torch.arange(n) for n in lengths])[None].cuda(),
        labels=torch.randint(AUDIO_TOKEN_ID, (1, tokens.numel()), device="cuda"),
        loss_mask=torch.cat(loss_mask)[None].cuda(),
        cu_seqlens=torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda"),
        max_seqlen=max(lengths),
        encoder_inputs=None,
    )


@pytest.fixture(name="batch")
def build_batch() -> Batch:
    """Build a row of two utterances with audio, whose audio tokens match the encoder frames."""
    torch.manual_seed(1)
    mels = [torch.randn(ENCODER_CONFIG.num_mel_bins, n) for n in MEL_LENGTHS]
    batch = build_row([ENCODER_CONFIG.calc_num_encoder_frames(n) for n in MEL_LENGTHS])
    return replace(
        batch, encoder_inputs=Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG).to("cuda")
    )


def list_encoder_kwargs(encoder_inputs: Qwen3ASREncoderInputs) -> dict[str, torch.Tensor | int]:
    """List the keyword arguments of ``Qwen3ASREncoder.forward`` that ``encoder_inputs`` holds."""
    return {field.name: getattr(encoder_inputs, field.name) for field in fields(encoder_inputs)}


def calc_mimo_loss(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss ``[1, num_tokens]`` of the speech LLM under test."""
    modality_inputs = None
    if batch.encoder_inputs is not None:
        modality_inputs = {
            AUDIO_MODALITY: {ENCODER_NAME: list_encoder_kwargs(batch.encoder_inputs)}
        }
    loss, _ = model(
        input_ids=batch.input_ids,
        position_ids=batch.position_ids,
        loss_mask=batch.loss_mask,
        labels=batch.labels,
        modality_inputs=modality_inputs,
        cu_seqlens=batch.cu_seqlens,
        max_seqlen=torch.tensor(batch.max_seqlen),
    )
    return loss


def merge_reference_llm_input(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Merge the word embeddings and the projected frames into the LLM's input.

    The input is ``[num_tokens, 1, hidden]``. The projected encoder frames replace the LLM's
    embeddings of the audio tokens, in order.
    The LLM's rotary positions add nothing to the word embeddings.
    """
    # The word embeddings are split over the TP ranks along the vocabulary.
    word_embeddings = gather_from_tensor_model_parallel_region(
        dict(model.named_parameters())[WORD_EMBEDDINGS].t()
    ).t()
    decoder_input = torch.nn.functional.embedding(batch.input_ids, word_embeddings).transpose(0, 1)
    if batch.encoder_inputs is not None:
        audio = model.modality_submodules[AUDIO_MODALITY]
        frames = audio.encoders[ENCODER_NAME](**list_encoder_kwargs(batch.encoder_inputs))
        is_audio = (batch.input_ids == AUDIO_TOKEN_ID).transpose(0, 1)
        decoder_input[is_audio] = audio.input_projections[0](frames).to(decoder_input.dtype)
    return decoder_input


def run_reference(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Run the modules of ``model`` one by one: encoder, projector, then LLM."""
    return model.language_model(
        None,
        batch.position_ids,
        None,
        decoder_input=merge_reference_llm_input(model, batch),
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


def capture_llm_input(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Run the speech LLM under test and return the embeddings that it passes to the LLM."""
    captured = {}

    def save_llm_input(
        module: torch.nn.Module,
        args: tuple[torch.Tensor | None, ...],
        kwargs: dict[str, torch.Tensor | None],
    ) -> None:
        """Keep the ``decoder_input`` that ``MimoModel`` passes to the LLM."""
        captured["decoder_input"] = kwargs["decoder_input"]

    handle = model.language_model.register_forward_pre_hook(save_llm_input, with_kwargs=True)
    try:
        calc_mimo_loss(model, batch)
    finally:
        handle.remove()
    return captured["decoder_input"]


@pytest.fixture(name="model_llm_input")
def calc_model_llm_input(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the LLM's input in the speech LLM under test, without gradients."""
    with torch.no_grad():
        return capture_llm_input(model, batch)


@pytest.fixture(name="reference_llm_input")
def calc_reference_llm_input(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the LLM's input of the module-by-module reference, without gradients."""
    with torch.no_grad():
        return merge_reference_llm_input(model, batch)


@pytest.fixture(name="model_loss")
def calc_model_loss(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss of the speech LLM under test without gradients."""
    with torch.no_grad():
        return calc_mimo_loss(model, batch)


@pytest.fixture(name="reference_loss")
def calc_reference_loss(model: MimoModel, batch: Batch) -> torch.Tensor:
    """Calculate the loss of the module-by-module reference without gradients."""
    with torch.no_grad():
        return run_reference(model, batch)


def test_parameter_names_follow_module_tree(model: MimoModel) -> None:
    """Check that the parameters sit under the encoder, the projector and the LLM."""
    prefixes = {
        f"modality_submodules.{AUDIO_MODALITY}.encoders.{ENCODER_NAME}.",
        f"modality_submodules.{AUDIO_MODALITY}.input_projections.0.",
        "language_model.",
    }
    for name, _ in model.named_parameters():
        assert any(name.startswith(prefix) for prefix in prefixes), name
    for prefix in prefixes:
        assert any(name.startswith(prefix) for name, _ in model.named_parameters()), prefix


def test_llm_input_matches_reference(
    model_llm_input: torch.Tensor, reference_llm_input: torch.Tensor, dtype: torch.dtype
) -> None:
    """Check that the LLM's input holds the projected frames at the audio tokens, in order."""
    assert model_llm_input.shape == reference_llm_input.shape
    error = calc_relative_error(model_llm_input, reference_llm_input)
    logger.info("%s: LLM input max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_loss_matches_reference(
    model_loss: torch.Tensor, reference_loss: torch.Tensor, batch: Batch, dtype: torch.dtype
) -> None:
    """Check that the loss at the target tokens equals the module-by-module reference."""
    targets = batch.loss_mask.bool()
    error = calc_relative_error(model_loss[targets], reference_loss[targets])
    logger.info("%s: loss max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def collect_gradients(model: MimoModel) -> dict[str, torch.Tensor | None]:
    """Copy the gradients of the parameters of ``model``, by parameter name."""
    return {
        name: None if param.grad is None else param.grad.clone()
        for name, param in model.named_parameters()
    }


@pytest.fixture(name="model_gradients")
def calc_model_gradients(model: MimoModel, batch: Batch) -> dict[str, torch.Tensor | None]:
    """Calculate the gradients of the loss summed over the targets, through ``MimoModel``."""
    model.zero_grad(set_to_none=True)
    (calc_mimo_loss(model, batch) * batch.loss_mask).sum().backward()
    return collect_gradients(model)


@pytest.fixture(name="reference_gradients")
def calc_reference_gradients(model: MimoModel, batch: Batch) -> dict[str, torch.Tensor | None]:
    """Calculate the gradients of the same loss through the module-by-module reference."""
    model.zero_grad(set_to_none=True)
    (run_reference(model, batch) * batch.loss_mask).sum().backward()
    return collect_gradients(model)


def test_gradients_match_reference(
    model_gradients: dict[str, torch.Tensor | None],
    reference_gradients: dict[str, torch.Tensor | None],
    dtype: torch.dtype,
) -> None:
    """Check that the gradients of every parameter equal the module-by-module reference."""
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


def test_accepts_batch_without_audio(model: MimoModel, dtype: torch.dtype) -> None:
    """Check that a row of text-only utterances gives the LLM's loss on the text embeddings."""
    torch.manual_seed(1)
    batch = build_row([0, 0])
    with torch.no_grad():
        model_loss = calc_mimo_loss(model, batch)
        reference_loss = run_reference(model, batch)
    targets = batch.loss_mask.bool()
    error = calc_relative_error(model_loss[targets], reference_loss[targets])
    logger.info("%s: loss without audio max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_rejects_projector_input_other_than_encoder_output(config: SpeechLLMConfig) -> None:
    """Check that a projector whose input size is not the encoder's size raises ValueError."""
    config = replace(config, projector=replace(config.projector, input_size=48))
    with pytest.raises(ValueError, match=r"projector\.input_size=48 differs from encoder\.d_model"):
        build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups())


def test_rejects_projector_output_other_than_llm_hidden_size(config: SpeechLLMConfig) -> None:
    """Check that a projector whose output size is not the LLM's hidden size raises ValueError."""
    config = replace(config, projector=replace(config.projector, output_size=48))
    with pytest.raises(
        ValueError, match=r"projector\.output_size=48 differs from llm\.hidden_size"
    ):
        build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups())


def test_rejects_pipeline_parallelism(config: SpeechLLMConfig) -> None:
    """Check that pipeline parallelism, which this MimoModel layout lacks, raises ValueError."""
    transformer = replace(
        config.llm.transformer,
        pipeline_model_parallel_size=2,
        pipeline_dtype=config.llm.transformer.params_dtype,
    )
    config = replace(config, llm=replace(config.llm, transformer=transformer))
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups())


def test_rejects_llm_without_vocab_size(config: SpeechLLMConfig) -> None:
    """Check that an LLM config without a vocabulary size raises ValueError."""
    config = replace(config, llm=replace(config.llm, vocab_size=None))
    with pytest.raises(ValueError, match=r"llm\.vocab_size is not set"):
        build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups())

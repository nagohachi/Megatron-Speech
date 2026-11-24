import logging
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.components.frontends.qwen3_asr import Qwen3ASRLogMelConfig
from megatron_speech.components.projectors.mlp import MLPProjectorConfig
from megatron_speech.data.packing import AudioTextExample, PackedAudioTextBatch
from megatron_speech.models.speech_llm import SpeechLLMConfig, build_speech_llm
from megatron_speech.training.speech_llm_step import SpeechLLMForwardStep
from tests.numerical_error import TOLERANCES, WITH_DTYPES, calc_relative_error

logger = logging.getLogger(__name__)

VOCAB_SIZE = 128
AUDIO_TOKEN_ID = VOCAB_SIZE - 1
PAD_TOKEN_ID = 0
NUM_PROMPT_TOKENS = 3
# Micro-batches that each data parallel rank runs in one training step.
NUM_MICROBATCHES = 2
ENCODER_CONFIG = Qwen3ASREncoderConfig(
    num_mel_bins=32,
    d_model=32,
    encoder_layers=1,
    encoder_attention_heads=4,
    encoder_ffn_dim=64,
    downsample_hidden_size=8,
    n_window_infer=200,
)
LOG_MEL_CONFIG = Qwen3ASRLogMelConfig(feature_size=ENCODER_CONFIG.num_mel_bins)


@dataclass
class TrainingStepOutputs:
    """Mean loss over the target tokens of a training step, and the gradients by parameter name."""

    loss: torch.Tensor
    gradients: dict[str, torch.Tensor]


def build_config(dtype: torch.dtype) -> SpeechLLMConfig:
    """Build a small speech LLM config whose loss is averaged over the target tokens."""
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        finalize_model_grads_func=finalize_model_grads,
    )
    return SpeechLLMConfig(
        llm=GPTModelConfig(
            transformer=transformer_config,
            vocab_size=VOCAB_SIZE,
            seq_length=1024,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
        ),
        encoder=ENCODER_CONFIG,
        projector=MLPProjectorConfig(input_size=32, output_size=64, hidden_size=64),
        audio_token_id=AUDIO_TOKEN_ID,
    )


FORWARD_STEP = SpeechLLMForwardStep(build_config(torch.float32), LOG_MEL_CONFIG)


@pytest.fixture(name="data_parallel_size", scope="module")
def init_data_parallel(tensor_parallel_size: int) -> Iterator[int]:
    """Rebuild Megatron's process groups with DP over all processes, for the tests here.

    The groups with TP over all processes come back after the tests.
    """
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    yield parallel_state.get_data_parallel_world_size()
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)
    model_parallel_cuda_manual_seed(1234)


@pytest.fixture(name="model")
def build_model(dtype: torch.dtype, data_parallel_size: int) -> DistributedDataParallel:
    """Build the speech LLM with random weights, the same on every rank, in Megatron's DDP."""
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = build_config(dtype)
    model = build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups()).cuda()
    # Megatron's schedule reads the model type, which pretrain() sets on the model.
    model.model_type = ModelType.encoder_or_decoder
    # As in training, the gradients are accumulated in fp32.
    return DistributedDataParallel(
        config.llm.transformer, DistributedDataParallelConfig(grad_reduce_in_fp32=True), model
    )


def build_example(generator: torch.Generator) -> AudioTextExample:
    """Build an utterance of random audio whose numbers of samples and targets are random."""
    num_samples = int(torch.randint(4000, 24000, (), generator=generator))
    num_audio_tokens = ENCODER_CONFIG.calc_num_encoder_frames(
        LOG_MEL_CONFIG.calc_num_frames(num_samples)
    )
    num_target_tokens = int(torch.randint(1, 16, (), generator=generator))
    return AudioTextExample(
        waveform=0.1 * torch.randn(num_samples, generator=generator),
        input_ids=torch.cat(
            [
                torch.randint(AUDIO_TOKEN_ID, (NUM_PROMPT_TOKENS,), generator=generator),
                torch.full((num_audio_tokens,), AUDIO_TOKEN_ID),
                torch.randint(AUDIO_TOKEN_ID, (num_target_tokens,), generator=generator),
            ]
        ),
        loss_mask=torch.cat(
            [torch.zeros(NUM_PROMPT_TOKENS + num_audio_tokens), torch.ones(num_target_tokens)]
        ),
    )


@pytest.fixture(name="microbatch_examples")
def build_microbatch_examples(data_parallel_size: int) -> list[list[AudioTextExample]]:
    """Build the utterances of the micro-batches of all DP ranks, the same on every rank.

    The micro-batches hold 1 to 3 utterances, so that their numbers of target tokens differ.
    The last micro-batch has no targets: its loss mask is 0 everywhere.
    """
    generator = torch.Generator().manual_seed(1)
    microbatch_examples = [
        [build_example(generator) for _ in range(int(torch.randint(1, 4, (), generator=generator)))]
        for _ in range(data_parallel_size * NUM_MICROBATCHES)
    ]
    microbatch_examples[-1] = [
        replace(example, loss_mask=torch.zeros_like(example.loss_mask))
        for example in microbatch_examples[-1]
    ]
    return microbatch_examples


def collect_gradients(model: DistributedDataParallel) -> dict[str, torch.Tensor]:
    """Copy ``main_grad``, where Megatron's DDP accumulates the gradients, by parameter name."""
    return {name: param.main_grad.clone() for name, param in model.module.named_parameters()}


@pytest.fixture(name="model_outputs")
def run_training_step(
    model: DistributedDataParallel, microbatch_examples: list[list[AudioTextExample]]
) -> TrainingStepOutputs:
    """Run the micro-batches of this DP rank through Megatron's schedule with the forward step.

    The loss is computed from the reports of the micro-batches of all DP ranks, as Megatron's
    log computes it.
    """
    rank = parallel_state.get_data_parallel_rank()
    batches = [
        PackedAudioTextBatch.build(examples, 1, PAD_TOKEN_ID)
        for examples in microbatch_examples[rank * NUM_MICROBATCHES : (rank + 1) * NUM_MICROBATCHES]
    ]
    model.zero_grad_buffer()
    reports = get_forward_backward_func()(
        forward_step_func=FORWARD_STEP,
        data_iterator=iter(batches),
        model=[model],
        num_microbatches=NUM_MICROBATCHES,
        seq_length=0,
        micro_batch_size=1,
    )
    report = torch.stack([report["lm loss"] for report in reports]).sum(dim=0)
    torch.distributed.all_reduce(report, group=parallel_state.get_data_parallel_group())
    return TrainingStepOutputs(loss=report[0] / report[1], gradients=collect_gradients(model))


@pytest.fixture(name="reference_outputs")
def run_one_row(
    model: DistributedDataParallel, microbatch_examples: list[list[AudioTextExample]]
) -> TrainingStepOutputs:
    """Run the utterances of all micro-batches as one packed row on this rank alone.

    The loss is the mean over the target tokens of the row, and the gradients are its own,
    without the gradients of the other DP ranks.
    """
    batch = PackedAudioTextBatch.build(
        [example for examples in microbatch_examples for example in examples], 1, PAD_TOKEN_ID
    )
    model.zero_grad_buffer()
    loss, _ = FORWARD_STEP(iter([batch]), model)
    loss_mask = batch.loss_mask.cuda()
    mean_loss = (loss.float() * loss_mask).sum() / loss_mask.sum()
    mean_loss.backward()
    return TrainingStepOutputs(loss=mean_loss.detach(), gradients=collect_gradients(model))


@pytest.mark.gpu
@WITH_DTYPES
def test_loss_matches_one_row(
    model_outputs: TrainingStepOutputs,
    reference_outputs: TrainingStepOutputs,
    dtype: torch.dtype,
    data_parallel_size: int,
) -> None:
    """Check that the logged loss is the mean over the targets of all micro-batches and ranks."""
    error = calc_relative_error(model_outputs.loss, reference_outputs.loss)
    logger.info("%s, DP=%d: loss relative error %.2e", dtype, data_parallel_size, error)
    assert error < TOLERANCES[dtype]


@pytest.mark.gpu
@WITH_DTYPES
def test_gradients_match_one_row(
    model_outputs: TrainingStepOutputs,
    reference_outputs: TrainingStepOutputs,
    dtype: torch.dtype,
    data_parallel_size: int,
) -> None:
    """Check that the gradients are those of the mean loss over all targets on one rank."""
    errors = {
        name: calc_relative_error(gradient, reference_outputs.gradients[name])
        for name, gradient in model_outputs.gradients.items()
    }
    logger.info(
        "%s, DP=%d: largest gradient relative error %.2e (%s)",
        dtype,
        data_parallel_size,
        max(errors.values()),
        max(errors, key=errors.get),
    )
    assert max(errors.values()) < TOLERANCES[dtype], errors


def test_rejects_loss_averaged_per_microbatch() -> None:
    """Check that a config without ``calculate_per_token_loss`` raises ValueError."""
    config = build_config(torch.float32)
    config = replace(
        config,
        llm=replace(
            config.llm, transformer=replace(config.llm.transformer, calculate_per_token_loss=False)
        ),
    )
    with pytest.raises(ValueError, match="calculate_per_token_loss=True"):
        SpeechLLMForwardStep(config, LOG_MEL_CONFIG)


def test_rejects_log_mel_bins_other_than_encoder_input() -> None:
    """Check that a log-mel whose bins are not the encoder's input bins raises ValueError."""
    with pytest.raises(ValueError, match=r"feature_size=128 differs from encoder\.num_mel_bins"):
        SpeechLLMForwardStep(build_config(torch.float32), Qwen3ASRLogMelConfig())

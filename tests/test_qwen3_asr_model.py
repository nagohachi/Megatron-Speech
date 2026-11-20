import gc
import logging
from dataclasses import dataclass, fields
from itertools import accumulate

import numpy as np
import pytest
import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.mimo import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from transformers import (
    AutoProcessor,
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

from megatron_speech.components.encoders.qwen3_asr import (
    Qwen3ASREncoderConfig,
    Qwen3ASREncoderInputs,
)
from megatron_speech.components.frontends.qwen3_asr import Qwen3ASRLogMel, Qwen3ASRLogMelConfig
from megatron_speech.conversion.bridge import MappingBridge
from megatron_speech.conversion.qwen3_asr import build_qwen3_asr_mappings
from megatron_speech.data.qwen3_asr import Qwen3ASRExampleBuilder, SpeechExample
from megatron_speech.models.speech_llm import (
    AUDIO_MODALITY,
    ENCODER_NAME,
    SpeechLLMConfig,
    build_speech_llm,
)
from tests.hf_audio_inputs import HF_MODEL, TRANSCRIPT, WAVEFORM_LENGTHS
from tests.numerical_error import calc_norm_relative_error, calc_relative_error
from tests.qwen3_asr_config import build_speech_llm_config

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

BRIDGE = MappingBridge(build_qwen3_asr_mappings())
# The lengths of ``WAVEFORM_LENGTHS`` and 10 s of audio (1000 mel frames), whose 130 encoder
# frames the released encoder attends in two windows (104 and 26).
UTTERANCE_LENGTHS = {**WAVEFORM_LENGTHS, "longer-than-a-window": 160000}
# Pass limits of the errors of ``calc_errors``, by the dtype of the model under test.
# - fp32: the largest relative error against Hugging Face's fp32 model. The 24 encoder and 28
#   LLM layers of the released model gather more rounding than the few layers that the shared
#   1e-5 is for; measured at TP=2: logits 6.4e-5, gradients 1.1e-3 (in the encoder).
# - bf16: how many times Hugging Face's own bf16 model the bf16 model is as far from the fp32
#   one. bf16 alone moves Hugging Face's logits by up to 0.43 in the largest relative error;
#   measured: logits 2.4 times (TP=1), gradients 2.9 times (TP=2, in the encoder).
LOGITS_LIMITS = {torch.float32: 1e-4, torch.bfloat16: 3.0}
GRADIENT_LIMITS = {torch.float32: 2e-3, torch.bfloat16: 3.0}


@dataclass
class ModelOutputs:
    """Outputs of one forward and backward pass over the utterances of the test.

    - ``logits``: ``[num_tokens, vocab_size]`` float32 logits, by utterance name.
    - ``gradients``: gradients of the summed target cross entropy, by Hugging Face weight name;
      empty on the ranks other than 0 of a tensor parallel run.
    """

    logits: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]


@dataclass
class ReferenceOutputs:
    """Outputs of Hugging Face's Qwen3-ASR: in fp32, the reference, and in bf16."""

    fp32: ModelOutputs
    bf16: ModelOutputs


@pytest.fixture(name="dtype", scope="module", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the model under test, for all tests of one model."""
    return request.param


@pytest.fixture(name="processor", scope="module")
def build_processor() -> Qwen3ASRProcessor:
    """Load Qwen3-ASR's processor, downloading its files if they are not in the cache."""
    return AutoProcessor.from_pretrained(HF_MODEL)


@pytest.fixture(name="waveforms", scope="module")
def build_waveforms() -> dict[str, np.ndarray]:
    """Build noise of each length of ``UTTERANCE_LENGTHS``, by name."""
    rng = np.random.default_rng(0)
    return {
        name: (rng.standard_normal(num_samples) * 0.1).astype(np.float32)
        for name, num_samples in UTTERANCE_LENGTHS.items()
    }


@pytest.fixture(name="examples", scope="module")
def build_examples(
    processor: Qwen3ASRProcessor, waveforms: dict[str, np.ndarray]
) -> dict[str, SpeechExample]:
    """Build the training example of each waveform with ``TRANSCRIPT`` in English."""
    builder = Qwen3ASRExampleBuilder(
        processor, Qwen3ASRLogMelConfig(), Qwen3ASREncoderConfig(), "English", False
    )
    return {name: builder.build(waveform, TRANSCRIPT) for name, waveform in waveforms.items()}


def calc_target_cross_entropy(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Sum the cross entropy of the target tokens of one utterance, from ``[num_tokens, vocab]``.

    The logits at each position predict the next token; ``loss_mask`` marks the targets.
    """
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    target_log_probs = log_probs.gather(-1, input_ids[1:, None])[:, 0]
    return -(target_log_probs * loss_mask[1:]).sum()


@pytest.fixture(name="hf_pretrained", scope="module")
def build_hf_pretrained() -> PreTrainedCausalLM:
    """Load Hugging Face's Qwen3-ASR in fp32 on the GPU, for the reference and Megatron-Bridge.

    Megatron-Bridge reads the weights of the loaded model instead of reading the checkpoint
    files again, which takes minutes from a network file system. ``PreTrainedCausalLM`` reads no
    weights itself: it loads its model on first use, through ``AutoModelForCausalLM``, which has
    no Qwen3-ASR class, and the model set here comes first.
    """
    hf_pretrained = PreTrainedCausalLM.from_pretrained(HF_MODEL, device="cuda")
    hf_pretrained.model = Qwen3ASRForConditionalGeneration.from_pretrained(
        HF_MODEL, dtype=torch.float32, device_map="cuda"
    )
    return hf_pretrained


@pytest.fixture(name="reference", scope="module")
def run_reference(
    hf_pretrained: PreTrainedCausalLM,
    processor: Qwen3ASRProcessor,
    waveforms: dict[str, np.ndarray],
    examples: dict[str, SpeechExample],
) -> ReferenceOutputs:
    """Run Hugging Face's Qwen3-ASR on the processor's inputs in fp32, then in bf16.

    The released weights are bf16, so the model keeps them exactly in bf16 afterwards, for
    Megatron-Bridge to load at half the memory of fp32.
    """
    model = hf_pretrained.model
    answer = f"language English<asr_text>{TRANSCRIPT}<|im_end|>"

    def run_in_dtype(dtype: torch.dtype) -> ModelOutputs:
        """Run each utterance through the model in ``dtype`` and backpropagate its loss."""
        model.to(dtype)
        logits = {}
        for name, waveform in waveforms.items():
            prompt = processor.apply_transcription_request(waveform, tokenize=False)[0]
            inputs = processor(text=[prompt + answer], audio=[waveform], return_tensors="pt").to(
                "cuda", dtype
            )
            utterance_logits = model(**inputs).logits[0]
            calc_target_cross_entropy(
                utterance_logits, inputs.input_ids[0], examples[name].loss_mask.cuda()
            ).backward()
            logits[name] = utterance_logits.detach().float()
        # The names of the state dict are those that Megatron-Bridge exports, with the output
        # weight that is tied to the word embeddings. Only rank 0 compares gradients.
        gradients = {}
        if torch.distributed.get_rank() == 0:
            gradients = {
                name: weight.grad for name, weight in model.state_dict(keep_vars=True).items()
            }
        model.zero_grad(set_to_none=True)
        return ModelOutputs(logits=logits, gradients=gradients)

    return ReferenceOutputs(fp32=run_in_dtype(torch.float32), bf16=run_in_dtype(torch.bfloat16))


@pytest.fixture(name="config", scope="module")
def build_config(dtype: torch.dtype, tensor_parallel_size: int) -> SpeechLLMConfig:
    """Build the config of the released Qwen3-ASR as a speech LLM with ``dtype`` parameters."""
    return build_speech_llm_config(
        Qwen3ASRConfig.from_pretrained(HF_MODEL), dtype, tensor_parallel_size
    )


@pytest.fixture(name="model", scope="module")
def build_model(
    reference: ReferenceOutputs, config: SpeechLLMConfig, hf_pretrained: PreTrainedCausalLM
) -> MimoModel:
    """Build the speech LLM of the released Qwen3-ASR with its weights.

    ``reference`` runs first, so that Megatron-Bridge reads the bf16 weights.
    """
    # The hooks of Megatron's DistributedDataParallel keep the model of the previous dtype in a
    # reference cycle, which only the cycle collector frees.
    gc.collect()
    model = build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups()).cuda()
    BRIDGE.load_weights_hf_to_megatron(hf_pretrained, [model])
    return model


@pytest.fixture(name="model_outputs", scope="module")
def calc_model_outputs(
    model: MimoModel,
    config: SpeechLLMConfig,
    hf_pretrained: PreTrainedCausalLM,
    examples: dict[str, SpeechExample],
) -> ModelOutputs:
    """Run the model under test as training does: one packed row, log-mel on the GPU."""

    def export_gradients() -> dict[str, torch.Tensor]:
        """Return the gradients of ``model`` by Hugging Face weight name, on rank 0's CPU.

        The gradients are those that Megatron's ``DistributedDataParallel`` keeps in ``main_grad``,
        the only place that holds them: its hooks move each ``grad`` there and clear ``grad``, and
        ``finalize_model_grads`` and Megatron's optimizer read them there.
        The export to Hugging Face only rearranges tensors (fused QKV, tensor parallel shards),
        so exporting the gradients in place of the weights gives them Hugging Face's layout.
        """
        params = list(model.parameters())
        weights = [param.data for param in params]
        for param in params:
            param.data = param.main_grad
        try:
            return {
                name: gradient.cpu()
                for name, gradient in BRIDGE.stream_weights_megatron_to_hf(
                    [model], hf_pretrained, show_progress=False
                )
            }
        finally:
            for param, weight in zip(params, weights, strict=True):
                param.data = weight

    def build_modality_inputs() -> dict[str, dict[str, dict[str, torch.Tensor | int]]]:
        """Compute the log-mel of all utterances on the GPU and cut it for the encoder."""
        log_mel_transform = Qwen3ASRLogMel(Qwen3ASRLogMelConfig()).cuda()
        mels = log_mel_transform([example.waveform.cuda() for example in examples.values()])
        encoder_inputs = Qwen3ASREncoderInputs.build(mels, config.encoder).to("cuda")
        encoder_kwargs = {
            field.name: getattr(encoder_inputs, field.name) for field in fields(encoder_inputs)
        }
        return {AUDIO_MODALITY: {ENCODER_NAME: encoder_kwargs}}

    token_counts = [len(example.input_ids) for example in examples.values()]
    # The positions start again at 0 in each utterance.
    position_ids = torch.cat([torch.arange(token_count) for token_count in token_counts])
    # As in training: Megatron's DistributedDataParallel accumulates the gradients, in fp32 as
    # Megatron's arguments choose for bf16, and finalize_model_grads sums the gradients of the
    # LLM's QK layernorms, of which each tensor parallel rank holds the part of its own heads.
    ddp_model = DistributedDataParallel(
        config.llm.transformer, DistributedDataParallelConfig(grad_reduce_in_fp32=True), model
    )
    logits, _ = ddp_model(
        input_ids=torch.cat([example.input_ids for example in examples.values()])[None].cuda(),
        position_ids=position_ids[None].cuda(),
        modality_inputs=build_modality_inputs(),
        cu_seqlens=torch.tensor([0, *accumulate(token_counts)], dtype=torch.int32, device="cuda"),
        max_seqlen=torch.tensor(max(token_counts)),
    )
    # The logits are split over the TP ranks along the vocabulary.
    logits_by_utterance = dict(
        zip(
            examples,
            gather_from_tensor_model_parallel_region(logits)[0].split(token_counts),
            strict=True,
        )
    )
    sum(
        calc_target_cross_entropy(
            logits_by_utterance[name], example.input_ids.cuda(), example.loss_mask.cuda()
        )
        for name, example in examples.items()
    ).backward()
    finalize_model_grads([ddp_model])
    return ModelOutputs(
        logits={
            name: utterance_logits.detach().float()
            for name, utterance_logits in logits_by_utterance.items()
        },
        gradients=export_gradients(),
    )


def calc_errors(
    values: dict[str, torch.Tensor],
    reference_values: dict[str, torch.Tensor],
    reference_bf16_values: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, float]:
    """Calculate the error of each of ``values`` that the limits of ``dtype`` bound.

    - fp32: the largest relative error against Hugging Face's fp32 value.
    - bf16: the distance from Hugging Face's fp32 value, as the norm of the difference over the
      norm of the fp32 value, divided by the distance of Hugging Face's bf16 value.
    """
    if dtype == torch.float32:
        return {
            name: calc_relative_error(value.cuda(), reference_values[name])
            for name, value in values.items()
        }
    return {
        name: calc_norm_relative_error(value.cuda(), reference_values[name])
        / calc_norm_relative_error(reference_bf16_values[name], reference_values[name])
        for name, value in values.items()
    }


def test_logits_match_hf(
    model_outputs: ModelOutputs, reference: ReferenceOutputs, dtype: torch.dtype
) -> None:
    """Check that each utterance gets the logits of Hugging Face's Qwen3-ASR at every token."""
    for name, logits in model_outputs.logits.items():
        assert logits.shape == reference.fp32.logits[name].shape, name
    errors = calc_errors(model_outputs.logits, reference.fp32.logits, reference.bf16.logits, dtype)
    logger.info(
        "%s: logits error %s",
        dtype,
        ", ".join(f"{name} {error:.2e}" for name, error in errors.items()),
    )
    assert max(errors.values()) < LOGITS_LIMITS[dtype], errors


def test_gradients_match_hf(
    model_outputs: ModelOutputs, reference: ReferenceOutputs, dtype: torch.dtype
) -> None:
    """Check that every weight gets the gradient of Hugging Face's Qwen3-ASR.

    The encoder's key bias gets no gradient in exact arithmetic: it adds one value to the scores
    of all keys of a query, which the softmax ignores, so only rounding noise remains. It is
    compared together with the query and value biases, which Megatron holds in one tensor.
    """
    if torch.distributed.get_rank() != 0:  # the other ranks get no exported gradients
        return

    def join_qkv_biases(gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replace the query, key and value bias gradients of each attention with one tensor."""
        joined_gradients = dict(gradients)
        for name in gradients:
            if name.endswith("self_attn.k_proj.bias"):
                prefix = name.removesuffix("k_proj.bias")
                joined_gradients[f"{prefix}qkv_proj.bias"] = torch.cat(
                    [
                        joined_gradients.pop(f"{prefix}{projection}_proj.bias").cuda()
                        for projection in "qkv"
                    ]
                )
        return joined_gradients

    assert model_outputs.gradients.keys() == reference.fp32.gradients.keys()
    errors = calc_errors(
        join_qkv_biases(model_outputs.gradients),
        join_qkv_biases(reference.fp32.gradients),
        join_qkv_biases(reference.bf16.gradients),
        dtype,
    )
    largest_error_names = sorted(errors, key=errors.get, reverse=True)[:5]
    logger.info(
        "%s: gradient error max %.2e, median %.2e; largest: %s",
        dtype,
        max(errors.values()),
        float(np.median(list(errors.values()))),
        ", ".join(f"{name} {errors[name]:.2e}" for name in largest_error_names),
    )
    assert max(errors.values()) < GRADIENT_LIMITS[dtype], {
        name: errors[name] for name in largest_error_names
    }

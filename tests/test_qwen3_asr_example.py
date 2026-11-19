import numpy as np
import pytest
import torch
from transformers import AutoProcessor, Qwen3ASRProcessor

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.data.qwen3_asr import Qwen3ASRExampleBuilder, SpeechExample

HF_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
# Qwen3-ASR-1.7B's transcript of a LibriSpeech test-clean utterance: cased, with punctuation.
TRANSCRIPT = "Concord returned to its place amidst the tents."
# 2.3 s at 16 kHz: 230 mel frames in three chunks, and 30 encoder frames.
NUM_SAMPLES = 36800


@pytest.fixture(name="processor", scope="module")
def build_processor() -> Qwen3ASRProcessor:
    """Load Qwen3-ASR's processor from the Hugging Face cache; skip the tests without it."""
    try:
        return AutoProcessor.from_pretrained(HF_MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{HF_MODEL} is not in the Hugging Face cache")


@pytest.fixture(name="force_language", params=[False, True], ids=["language-in-target", "forced"])
def get_force_language(request: pytest.FixtureRequest) -> bool:
    """Return whether the language is forced in the prompt or predicted in the target."""
    return request.param


@pytest.fixture(name="waveform")
def build_waveform() -> np.ndarray:
    """Build noise of ``NUM_SAMPLES`` samples."""
    return (np.random.default_rng(0).standard_normal(NUM_SAMPLES) * 0.1).astype(np.float32)


@pytest.fixture(name="example")
def build_example(
    processor: Qwen3ASRProcessor, waveform: np.ndarray, force_language: bool
) -> SpeechExample:
    """Build the example of the waveform with ``TRANSCRIPT`` in English."""
    return Qwen3ASRExampleBuilder(processor, "English", force_language).build(waveform, TRANSCRIPT)


def test_input_ids_match_hf(
    example: SpeechExample, processor: Qwen3ASRProcessor, waveform: np.ndarray
) -> None:
    """Check that the token ids are Hugging Face's processor's for the whole conversation.

    Forcing the language moves where the loss starts, not the tokens.
    """
    prompt = processor.apply_transcription_request(waveform, tokenize=False)[0]
    answer = f"language English<asr_text>{TRANSCRIPT}<|im_end|>"
    expected = processor(text=[prompt + answer], audio=[waveform], return_tensors="pt")
    assert torch.equal(example.input_ids, expected.input_ids[0])


def test_loss_mask_covers_answer(
    example: SpeechExample, processor: Qwen3ASRProcessor, force_language: bool
) -> None:
    """Check that the loss mask covers the answer at the end, and only it."""
    expected = f"{TRANSCRIPT}<|im_end|>"
    if not force_language:
        expected = f"language English<asr_text>{expected}"
    num_targets = int(example.loss_mask.sum())
    assert torch.all(example.loss_mask[-num_targets:] == 1)
    assert torch.all(example.loss_mask[:-num_targets] == 0)
    assert processor.tokenizer.decode(example.input_ids[-num_targets:]) == expected


def test_mel_matches_hf_feature_extractor(
    example: SpeechExample, processor: Qwen3ASRProcessor, waveform: np.ndarray
) -> None:
    """Check that the log-mel is Hugging Face's feature extractor's for the utterance alone."""
    features = processor.feature_extractor(
        waveform,
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )
    num_mel_frames = int(features.attention_mask[0].sum())
    assert torch.equal(example.mel, features.input_features[0, :, :num_mel_frames])


def test_audio_tokens_match_encoder_frames(
    example: SpeechExample, processor: Qwen3ASRProcessor
) -> None:
    """Check that there is one audio token per encoder frame of the released Qwen3-ASR."""
    num_audio_tokens = int((example.input_ids == processor.audio_token_id).sum())
    assert num_audio_tokens == Qwen3ASREncoderConfig().calc_num_encoder_frames(example.mel.shape[1])

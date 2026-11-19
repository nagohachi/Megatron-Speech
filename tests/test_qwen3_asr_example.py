import numpy as np
import pytest
import torch
from transformers import AutoProcessor, Qwen3ASRProcessor

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.data.qwen3_asr import Qwen3ASRExampleBuilder, SpeechExample

HF_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
# Qwen3-ASR-1.7B's transcript of a LibriSpeech test-clean utterance: cased, with punctuation.
TRANSCRIPT = "Concord returned to its place amidst the tents."


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


@pytest.fixture(
    name="num_samples",
    params=[8000, 32000, 36800],
    ids=["shorter-than-a-chunk", "whole-chunks", "partial-last-chunk"],
)
def get_num_samples(request: pytest.FixtureRequest) -> int:
    """Return the length of the audio at 16 kHz: 0.5 s, 2 s or 2.3 s.

    They give 50 mel frames in one partial chunk, 200 in two whole chunks, and 230.
    """
    return request.param


@pytest.fixture(name="transcript", params=[TRANSCRIPT, ""], ids=["transcript", "empty"])
def get_transcript(request: pytest.FixtureRequest) -> str:
    """Return a transcript, or an empty one, whose answer is the end-of-turn token only."""
    return request.param


@pytest.fixture(name="waveform")
def build_waveform(num_samples: int) -> np.ndarray:
    """Build noise of ``num_samples`` samples."""
    return (np.random.default_rng(0).standard_normal(num_samples) * 0.1).astype(np.float32)


@pytest.fixture(name="example")
def build_example(
    processor: Qwen3ASRProcessor, waveform: np.ndarray, transcript: str, force_language: bool
) -> SpeechExample:
    """Build the example of the waveform with the transcript in English."""
    return Qwen3ASRExampleBuilder(processor, "English", force_language).build(waveform, transcript)


def test_input_ids_match_hf(
    example: SpeechExample, processor: Qwen3ASRProcessor, waveform: np.ndarray, transcript: str
) -> None:
    """Check that the token ids are Hugging Face's processor's for the whole conversation.

    Forcing the language moves where the loss starts, not the tokens.
    """
    prompt = processor.apply_transcription_request(waveform, tokenize=False)[0]
    answer = f"language English<asr_text>{transcript}<|im_end|>"
    expected = processor(text=[prompt + answer], audio=[waveform], return_tensors="pt")
    assert torch.equal(example.input_ids, expected.input_ids[0])


def test_loss_mask_covers_answer(
    example: SpeechExample, processor: Qwen3ASRProcessor, transcript: str, force_language: bool
) -> None:
    """Check that the float32 loss mask covers the answer at the end, and only it."""
    expected = f"{transcript}<|im_end|>"
    if not force_language:
        expected = f"language English<asr_text>{expected}"
    assert example.loss_mask.dtype == torch.float32
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

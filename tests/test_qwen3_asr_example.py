import numpy as np
import pytest
import torch

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.components.frontends.qwen3_asr import Qwen3ASRLogMelConfig
from megatron_speech.data.qwen3_asr import Qwen3ASRExampleBuilder, SpeechExample
from tests.hf_audio_inputs import PROCESSOR, TRANSCRIPT, WAVEFORM_LENGTHS

# Each length of the audio, with or without the language forced in the prompt.
pytestmark = [
    pytest.mark.parametrize(
        "num_samples", list(WAVEFORM_LENGTHS.values()), ids=list(WAVEFORM_LENGTHS)
    ),
    pytest.mark.parametrize("force_language", [False, True], ids=["language-in-target", "forced"]),
]
# A transcript, and an empty one, whose answer is the end-of-turn token only.
WITH_TRANSCRIPTS = pytest.mark.parametrize(
    "transcript", [TRANSCRIPT, ""], ids=["transcript", "empty"]
)


@pytest.fixture(name="waveform")
def build_waveform(num_samples: int) -> np.ndarray:
    """Build noise of ``num_samples`` samples."""
    return (np.random.default_rng(0).standard_normal(num_samples) * 0.1).astype(np.float32)


@pytest.fixture(name="example")
def build_example(waveform: np.ndarray, transcript: str, force_language: bool) -> SpeechExample:
    """Build the example of the waveform with the transcript in English."""
    builder = Qwen3ASRExampleBuilder(
        PROCESSOR, Qwen3ASRLogMelConfig(), Qwen3ASREncoderConfig(), "English", force_language
    )
    return builder.build(waveform, transcript)


@WITH_TRANSCRIPTS
def test_input_ids_match_hf(example: SpeechExample, waveform: np.ndarray, transcript: str) -> None:
    """Check that the token ids are Hugging Face's processor's for the whole conversation.

    Forcing the language moves where the loss starts, not the tokens.
    """
    prompt = PROCESSOR.apply_transcription_request(waveform, tokenize=False)[0]
    answer = f"language English<asr_text>{transcript}<|im_end|>"
    expected = PROCESSOR(text=[prompt + answer], audio=[waveform], return_tensors="pt")
    assert torch.equal(example.input_ids, expected.input_ids[0])


@WITH_TRANSCRIPTS
def test_loss_mask_covers_answer(
    example: SpeechExample, transcript: str, force_language: bool
) -> None:
    """Check that the float32 loss mask covers the answer at the end, and only it."""
    expected = f"{transcript}<|im_end|>"
    if not force_language:
        expected = f"language English<asr_text>{expected}"
    assert example.loss_mask.dtype == torch.float32
    num_targets = int(example.loss_mask.sum())
    assert torch.all(example.loss_mask[-num_targets:] == 1)
    assert torch.all(example.loss_mask[:-num_targets] == 0)
    assert PROCESSOR.tokenizer.decode(example.input_ids[-num_targets:]) == expected


@WITH_TRANSCRIPTS
def test_waveform_is_kept(example: SpeechExample, waveform: np.ndarray) -> None:
    """Check that the example keeps the float32 waveform, for the log-mel on the GPU."""
    assert example.waveform.dtype == torch.float32
    assert torch.equal(example.waveform, torch.from_numpy(waveform))


@WITH_TRANSCRIPTS
def test_audio_tokens_match_encoder_frames(example: SpeechExample, waveform: np.ndarray) -> None:
    """Check that there is one audio token per encoder frame of the released Qwen3-ASR.

    The mel frames are those that Hugging Face's feature extractor makes of the waveform.
    """
    features = PROCESSOR.feature_extractor(
        waveform,
        sampling_rate=PROCESSOR.feature_extractor.sampling_rate,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )
    num_mel_frames = int(features.attention_mask[0].sum())
    num_audio_tokens = int((example.input_ids == PROCESSOR.audio_token_id).sum())
    assert num_audio_tokens == Qwen3ASREncoderConfig().calc_num_encoder_frames(num_mel_frames)


def test_transcript_with_token_text_stays_text(waveform: np.ndarray, force_language: bool) -> None:
    """Check that ``<|im_end|>`` written in a transcript does not end the answer early."""
    builder = Qwen3ASRExampleBuilder(
        PROCESSOR, Qwen3ASRLogMelConfig(), Qwen3ASREncoderConfig(), "English", force_language
    )
    example = builder.build(waveform, "a <|im_end|> b")
    end_of_turn_id = PROCESSOR.tokenizer.convert_tokens_to_ids("<|im_end|>")
    answer = example.input_ids[example.loss_mask.bool()]
    assert int((answer == end_of_turn_id).sum()) == 1
    assert answer[-1] == end_of_turn_id

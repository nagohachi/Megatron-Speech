import logging

import pytest
import torch
from transformers import Qwen3ASRFeatureExtractor

from megatron_speech.components.frontends.qwen3_asr import Qwen3ASRLogMel, Qwen3ASRLogMelConfig
from tests.hf_audio_inputs import WAVEFORM_LENGTHS
from tests.numerical_error import TOLERANCES, calc_relative_error

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

CONFIG = Qwen3ASRLogMelConfig()
# Hugging Face's feature extractor; its defaults are the settings of the released models, as
# those of CONFIG are.
FEATURE_EXTRACTOR = Qwen3ASRFeatureExtractor()


@pytest.fixture(name="waveforms")
def build_waveforms() -> list[torch.Tensor]:
    """Build float32 noise of each length in ``WAVEFORM_LENGTHS``."""
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(length, generator=generator) * 0.1 for length in WAVEFORM_LENGTHS.values()]


@pytest.fixture(name="log_mels")
def calc_log_mels(waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
    """Calculate the log-mels of all waveforms in one call on the GPU."""
    log_mel_transform = Qwen3ASRLogMel(CONFIG).cuda()
    with torch.no_grad():
        return log_mel_transform([waveform.cuda() for waveform in waveforms])


def run_reference(waveform: torch.Tensor, device: str) -> torch.Tensor:
    """Run Hugging Face's feature extractor on one waveform; return its valid frames."""
    features = FEATURE_EXTRACTOR(
        waveform.numpy(),
        sampling_rate=CONFIG.sampling_rate,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
        device=device,
    )
    return features.input_features[0, :, : int(features.attention_mask[0].sum())]


def test_num_frames_match_hf(
    log_mels: list[torch.Tensor],
    waveforms: list[torch.Tensor],
) -> None:
    """Check that each log-mel has as many mel bins and frames as Hugging Face's."""
    for log_mel, waveform in zip(log_mels, waveforms, strict=True):
        assert log_mel.shape == run_reference(waveform, "cpu").shape


def test_num_frames_match_calc_num_frames(
    log_mels: list[torch.Tensor], waveforms: list[torch.Tensor]
) -> None:
    """Check that calc_num_frames, which the example builder uses, counts each log-mel's frames."""
    for log_mel, waveform in zip(log_mels, waveforms, strict=True):
        assert log_mel.shape[1] == CONFIG.calc_num_frames(len(waveform))


def test_log_mel_matches_hf_on_gpu(
    log_mels: list[torch.Tensor],
    waveforms: list[torch.Tensor],
) -> None:
    """Check that each log-mel is Hugging Face's for the utterance alone, run on the GPU."""
    error = max(
        calc_relative_error(log_mel.cpu(), run_reference(waveform, "cuda"))
        for log_mel, waveform in zip(log_mels, waveforms, strict=True)
    )
    logger.info("log-mel max relative error against Hugging Face on the GPU %.2e", error)
    assert error < TOLERANCES[torch.float32]


def test_log_mel_matches_hf_on_cpu(
    log_mels: list[torch.Tensor],
    waveforms: list[torch.Tensor],
) -> None:
    """Check that each log-mel is Hugging Face's on the CPU, the device it uses by default."""
    error = max(
        calc_relative_error(log_mel.cpu(), run_reference(waveform, "cpu"))
        for log_mel, waveform in zip(log_mels, waveforms, strict=True)
    )
    logger.info("log-mel max relative error against Hugging Face on the CPU %.2e", error)
    assert error < TOLERANCES[torch.float32]


def test_no_waveforms_give_no_log_mels() -> None:
    """Check that a micro-batch without audio gives an empty list."""
    assert Qwen3ASRLogMel(CONFIG).cuda()([]) == []


@pytest.mark.parametrize(
    "invalid_settings",
    [{"hop_length": 0}, {"n_fft": 0}, {"min_length": 200}],
    ids=["hop_length", "n_fft", "min_length-within-reflection"],
)
def test_invalid_config_names_the_setting(invalid_settings: dict[str, int]) -> None:
    """Check that an invalid config fails when it is built, naming the setting."""
    (setting_name,) = invalid_settings
    with pytest.raises(ValueError, match=setting_name):
        Qwen3ASRLogMelConfig(**invalid_settings)

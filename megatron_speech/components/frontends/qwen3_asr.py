from dataclasses import dataclass, fields

import torch
from transformers.audio_utils import mel_filter_bank

# Hugging Face's Qwen3ASRFeatureExtractor fixes the highest mel frequency at 8 kHz.
MAX_FREQUENCY = 8000.0


@dataclass
class Qwen3ASRLogMelConfig:
    """Settings of Hugging Face's ``Qwen3ASRFeatureExtractor`` that the log-mel depends on.

    The fields have the feature extractor's names, and the defaults are those of the released
    Qwen3-ASR models. ``feature_size`` is the number of mel bins.
    """

    feature_size: int = 128
    sampling_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    min_length: int = 8000

    def __post_init__(self) -> None:
        """Check that every setting is positive and that the STFT can run on ``min_length``."""
        for field in fields(self):
            if getattr(self, field.name) <= 0:
                raise ValueError(f"{field.name} must be positive, got {getattr(self, field.name)}")
        if self.min_length <= self.n_fft // 2:
            raise ValueError(
                f"min_length ({self.min_length}) must be larger than n_fft // 2 "
                f"({self.n_fft // 2}): the centered STFT reflects n_fft // 2 samples at each end "
                "of the audio"
            )

    def calc_num_frames(self, num_samples: int) -> int:
        """Count the log-mel frames of ``num_samples`` samples, as Hugging Face does.

        Audio shorter than ``min_length`` samples is padded with zeros to it, and the last STFT
        frame is dropped.
        """
        return max(num_samples, self.min_length) // self.hop_length


class Qwen3ASRLogMel(torch.nn.Module):
    """Log-mel of Hugging Face's ``Qwen3ASRFeatureExtractor``, for many utterances at once.

    For each utterance, as the feature extractor computes it alone:

    - Padding: zeros up to ``min_length`` samples.
    - STFT: Hann window, centered; each utterance is reflected at its own ends, so that the
      utterances of one call do not change one another. They share one STFT call, after zero
      padding to the longest.
    - Log-mel: power through the mel filters, log10 of at least 1e-10, a floor at the
      utterance's maximum minus 8, then ``(x + 4) / 4``.
    """

    def __init__(self, config: Qwen3ASRLogMelConfig) -> None:
        """Build the mel filters of ``config``."""
        super().__init__()
        self.config = config
        mel_filters = mel_filter_bank(
            num_frequency_bins=1 + config.n_fft // 2,
            num_mel_filters=config.feature_size,
            min_frequency=0.0,
            max_frequency=MAX_FREQUENCY,
            sampling_rate=config.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("mel_filters", torch.from_numpy(mel_filters).float(), persistent=False)

    def forward(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return the ``[feature_size, num_frames]`` float32 log-mel of each waveform.

        ``waveforms`` holds ``[num_samples]`` float32 waveforms on the module's device.
        """
        if not waveforms:
            return []
        n_fft, hop_length = self.config.n_fft, self.config.hop_length
        valid_frame_counts = [self.config.calc_num_frames(len(waveform)) for waveform in waveforms]
        stft_inputs = []
        for waveform in waveforms:
            padded_waveform = torch.nn.functional.pad(
                waveform, (0, max(self.config.min_length - len(waveform), 0))
            )
            # A centered STFT reflects n_fft // 2 samples at each end.
            stft_inputs.append(
                torch.nn.functional.pad(
                    padded_waveform[None], (n_fft // 2, n_fft // 2), mode="reflect"
                )[0]
            )
        stft = torch.stft(
            torch.nn.utils.rnn.pad_sequence(stft_inputs, batch_first=True),
            n_fft,
            hop_length,
            window=torch.hann_window(n_fft, device=self.mel_filters.device),
            center=False,
            return_complex=True,
        )
        log_spec = (self.mel_filters.T @ stft.abs().square()).clamp(min=1e-10).log10()
        frame_indices = torch.arange(log_spec.shape[-1], device=log_spec.device)
        valid_frame_mask = (
            frame_indices < torch.tensor(valid_frame_counts, device=log_spec.device)[:, None]
        )
        max_log_spec = log_spec.masked_fill(~valid_frame_mask[:, None], float("-inf")).amax(
            dim=(1, 2), keepdim=True
        )
        log_spec = (torch.maximum(log_spec, max_log_spec - 8.0) + 4.0) / 4.0
        return [
            utterance_log_spec[:, :num_utterance_frames]
            for utterance_log_spec, num_utterance_frames in zip(
                log_spec, valid_frame_counts, strict=True
            )
        ]

from abc import ABC, abstractmethod

import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.utils import compute_num_samples

from megatron_speech.data.packing import AudioTextExample, PackedAudioTextBatch


class AudioTextExampleBuilder(ABC):
    """Base class of the builders of ``AudioTextExample``; one is written for each prompt format."""

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Sampling rate of the waveforms that ``build`` takes."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Token id of the padding of packed rows."""

    @abstractmethod
    def build(self, waveform: np.ndarray, transcript: str) -> AudioTextExample:
        """Build the example of ``waveform``, sampled at ``sampling_rate``, and ``transcript``."""

    @abstractmethod
    def calc_num_tokens(self, num_samples: int, transcript: str) -> int:
        """Count the tokens of the example of ``num_samples`` samples and ``transcript``."""


def _read_transcript(cut: Cut) -> str:
    """Return the text of the only supervision of ``cut``."""
    if len(cut.supervisions) != 1:
        raise ValueError(f"cut {cut.id} has {len(cut.supervisions)} supervisions instead of one")
    return cut.supervisions[0].text


class AudioTextDataset(torch.utils.data.Dataset):
    """Dataset that is indexed by the cuts of a micro-batch, as they are given by lhotse's samplers.

    Audio at another sampling rate is resampled to that of ``example_builder``.
    """

    def __init__(self, example_builder: AudioTextExampleBuilder, pad_to_multiple: int) -> None:
        """Keep the builder of the examples and the ``pad_to_multiple`` of their packing."""
        self.example_builder = example_builder
        self.pad_to_multiple = pad_to_multiple

    def __getitem__(self, cuts: CutSet) -> PackedAudioTextBatch:
        """Load the audio and the transcripts of ``cuts`` and pack their examples into one row."""
        sampling_rate = self.example_builder.sampling_rate
        examples = [
            self.example_builder.build(
                cut.resample(sampling_rate).load_audio()[0], _read_transcript(cut)
            )
            for cut in cuts
        ]
        return PackedAudioTextBatch.build(
            examples, self.pad_to_multiple, self.example_builder.pad_token_id
        )

    def calc_num_tokens(self, cut: Cut) -> int:
        """Count the tokens that ``cut`` takes in a packed row, with its padding.

        Only the duration and the transcript are read, without the audio.
        """
        num_samples = compute_num_samples(cut.duration, self.example_builder.sampling_rate)
        return PackedAudioTextBatch.calc_segment_length(
            self.example_builder.calc_num_tokens(num_samples, _read_transcript(cut)),
            self.pad_to_multiple,
        )

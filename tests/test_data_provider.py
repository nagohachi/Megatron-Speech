import re
from dataclasses import replace
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.utils import compute_num_samples

from megatron_speech.data.dataset import AudioTextDataset, AudioTextExampleBuilder
from megatron_speech.data.packing import AudioTextExample, PackedAudioTextBatch
from megatron_speech.data.provider import (
    AudioTextDataConfig,
    AudioTextDatasetsProvider,
    iterate_epochs,
    iterate_evaluations,
)
from megatron_speech.data.sampling import TokenBucketingConfig

SAMPLING_RATE = 16000
SAMPLES_PER_AUDIO_TOKEN = 1600
PAD_TO_MULTIPLE = 4
TOKEN_BUCKETING_CONFIG = TokenBucketingConfig(
    max_tokens_per_batch=100, max_tokens_per_example=60, num_buckets=3
)
# Lengths of the utterances in samples. They all differ, so that a waveform tells its utterance.
UTTERANCE_LENGTHS = [8000 + 397 * index for index in range(40)]


class CharExampleBuilder(AudioTextExampleBuilder):
    """Builder of examples of one audio token per 0.1 s, then one token per transcript character."""

    sampling_rate = SAMPLING_RATE
    pad_token_id = 0

    def build(self, waveform: np.ndarray, transcript: str) -> AudioTextExample:
        """Build the example; the audio tokens are 1 and the characters are the targets."""
        input_ids = torch.tensor(
            [1] * (len(waveform) // SAMPLES_PER_AUDIO_TOKEN) + [ord(char) for char in transcript]
        )
        return AudioTextExample(
            waveform=torch.from_numpy(waveform),
            input_ids=input_ids,
            loss_mask=input_ids.ne(1).float(),
        )

    def calc_num_tokens(self, num_samples: int, transcript: str) -> int:
        """Count the tokens of the example of ``num_samples`` samples and ``transcript``."""
        return num_samples // SAMPLES_PER_AUDIO_TOKEN + len(transcript)


DATASET = AudioTextDataset(CharExampleBuilder(), PAD_TO_MULTIPLE)


@pytest.fixture(name="shar_dir", scope="module")
def build_shar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write noise of ``UTTERANCE_LENGTHS`` with transcripts as Shar shards of 10 utterances."""
    audio_dir = tmp_path_factory.mktemp("audio")
    rng = np.random.default_rng(0)
    cuts = []
    for index, num_samples in enumerate(UTTERANCE_LENGTHS):
        audio_path = audio_dir / f"utterance-{index}.flac"
        sf.write(audio_path, rng.standard_normal(num_samples) * 0.1, SAMPLING_RATE)
        cut = Recording.from_file(audio_path).to_cut()
        cut.supervisions = [
            SupervisionSegment(
                id=cut.recording_id,
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                text=f"UTTERANCE {index}",
            )
        ]
        cuts.append(cut)
    shar_dir = tmp_path_factory.mktemp("shar")
    CutSet.from_cuts(cuts).to_shar(shar_dir, fields={"recording": "flac"}, shard_size=10)
    return shar_dir


@pytest.fixture(name="cuts")
def read_cuts(shar_dir: Path) -> CutSet:
    """Read the first 5 utterances of the shards with their audio."""
    return CutSet.from_shar(in_dir=shar_dir).subset(first=5).to_eager()


@pytest.fixture(name="batch")
def build_batch(cuts: CutSet) -> PackedAudioTextBatch:
    """Pack the utterances of ``cuts`` into one row."""
    return DATASET[cuts]


def test_segments_take_counted_tokens(batch: PackedAudioTextBatch, cuts: CutSet) -> None:
    """Check that each utterance takes the tokens that the sampler counts for it."""
    assert torch.diff(batch.cu_seqlens).tolist() == [DATASET.calc_num_tokens(cut) for cut in cuts]


def test_waveforms_are_audio_of_cuts(batch: PackedAudioTextBatch, cuts: CutSet) -> None:
    """Check that the micro-batch holds the audio of each utterance."""
    for waveform, cut in zip(batch.waveforms, cuts, strict=True):
        assert torch.equal(waveform, torch.from_numpy(cut.load_audio()[0]))


def test_other_sampling_rate_is_resampled(cuts: CutSet) -> None:
    """Check that audio at 8 kHz is resampled to 16 kHz and takes the counted tokens."""
    cuts_at_8khz = CutSet.from_cuts(cut.resample(8000) for cut in cuts)
    batch = DATASET[cuts_at_8khz]
    for waveform, cut in zip(batch.waveforms, cuts_at_8khz, strict=True):
        assert len(waveform) == compute_num_samples(cut.duration, SAMPLING_RATE)
    assert torch.diff(batch.cu_seqlens).tolist() == [
        DATASET.calc_num_tokens(cut) for cut in cuts_at_8khz
    ]


@pytest.fixture(name="provider", scope="module")
def build_provider(shar_dir: Path) -> AudioTextDatasetsProvider:
    """Build the provider of the shards for training and validation, with 2 workers."""
    data_config = AudioTextDataConfig(
        train_shar_dirs=[shar_dir],
        valid_shar_dirs=[shar_dir],
        token_bucketing=TOKEN_BUCKETING_CONFIG,
        num_workers=2,
    )
    return AudioTextDatasetsProvider(
        data_config, CharExampleBuilder(), PAD_TO_MULTIPLE, num_microbatches_per_evaluation=3
    )


@pytest.fixture(name="batches_of_ranks", scope="module")
def build_batches_of_ranks(
    provider: AudioTextDatasetsProvider, shar_dir: Path
) -> list[list[PackedAudioTextBatch]]:
    """Load one epoch on each of two data parallel ranks; return the micro-batches by rank."""
    return [
        list(provider.build_loader([shar_dir], TOKEN_BUCKETING_CONFIG, rank, 2))
        for rank in range(2)
    ]


def test_batches_stay_within_token_limit(
    batches_of_ranks: list[list[PackedAudioTextBatch]],
) -> None:
    """Check that no packed row, padding included, is longer than ``max_tokens_per_batch``."""
    for batches in batches_of_ranks:
        for batch in batches:
            assert batch.cu_seqlens[-1] <= TOKEN_BUCKETING_CONFIG.max_tokens_per_batch


def test_ranks_take_every_utterance_once(
    batches_of_ranks: list[list[PackedAudioTextBatch]],
) -> None:
    """Check that the data parallel ranks together take each utterance once in an epoch."""
    lengths = [
        len(waveform)
        for batches in batches_of_ranks
        for batch in batches
        for waveform in batch.waveforms
    ]
    assert sorted(lengths) == UTTERANCE_LENGTHS


def test_loaders_of_one_rank_give_same_batches(
    batches_of_ranks: list[list[PackedAudioTextBatch]],
    provider: AudioTextDatasetsProvider,
    shar_dir: Path,
) -> None:
    """Check that another loader of rank 0, as on another tensor parallel rank, gives the same."""
    batches = list(provider.build_loader([shar_dir], TOKEN_BUCKETING_CONFIG, 0, 2))
    assert len(batches) == len(batches_of_ranks[0])
    for batch, batch_of_rank0 in zip(batches, batches_of_ranks[0], strict=True):
        assert torch.equal(batch.input_ids, batch_of_rank0.input_ids)


def test_epochs_take_every_utterance_in_another_order(
    provider: AudioTextDatasetsProvider, shar_dir: Path
) -> None:
    """Check that the train iterator goes on to a second epoch, in another order."""
    loader = provider.build_loader([shar_dir], TOKEN_BUCKETING_CONFIG, 0, 1)
    lengths = []
    for batch in iterate_epochs(loader):
        lengths += [len(waveform) for waveform in batch.waveforms]
        if len(lengths) >= 2 * len(UTTERANCE_LENGTHS):
            break
    first_epoch, second_epoch = lengths[: len(UTTERANCE_LENGTHS)], lengths[len(UTTERANCE_LENGTHS) :]
    assert sorted(first_epoch) == sorted(second_epoch) == UTTERANCE_LENGTHS
    assert first_epoch != second_epoch


def test_evaluations_take_same_batches(provider: AudioTextDatasetsProvider, shar_dir: Path) -> None:
    """Check that each evaluation, also one longer than an epoch, takes the same micro-batches."""
    loader = provider.build_loader([shar_dir], replace(TOKEN_BUCKETING_CONFIG, shuffle=False), 0, 1)
    num_microbatches_per_evaluation = len(list(loader)) + 2
    batches = list(
        islice(
            iterate_evaluations(loader, num_microbatches_per_evaluation),
            2 * num_microbatches_per_evaluation,
        )
    )
    for batch, batch_of_first_evaluation in zip(
        batches[num_microbatches_per_evaluation:],
        batches[:num_microbatches_per_evaluation],
        strict=True,
    ):
        assert torch.equal(batch.input_ids, batch_of_first_evaluation.input_ids)


def test_evaluations_reject_empty_data() -> None:
    """Check that validation data without micro-batches is rejected instead of read forever."""
    with pytest.raises(RuntimeError, match="no micro-batch"):
        next(iterate_evaluations([], 3))


@pytest.mark.parametrize(
    ("setting", "value"), [("pad_to_multiple", 0), ("num_microbatches_per_evaluation", 0)]
)
def test_provider_rejects_impossible_settings(shar_dir: Path, setting: str, value: int) -> None:
    """Check that an impossible padding or evaluation setting is rejected with its name."""
    data_config = AudioTextDataConfig([shar_dir], [shar_dir], TOKEN_BUCKETING_CONFIG)
    settings = {"pad_to_multiple": PAD_TO_MULTIPLE, "num_microbatches_per_evaluation": 3}
    with pytest.raises(ValueError, match=setting):
        AudioTextDatasetsProvider(data_config, CharExampleBuilder(), **{**settings, setting: value})


def test_provider_without_validation_data_needs_no_evaluations(shar_dir: Path) -> None:
    """Check that without validation data, no micro-batches of an evaluation are needed."""
    data_config = AudioTextDataConfig([shar_dir], [], TOKEN_BUCKETING_CONFIG)
    AudioTextDatasetsProvider(data_config, CharExampleBuilder(), PAD_TO_MULTIPLE, 0)


def test_loader_rejects_directories_without_shards(
    provider: AudioTextDatasetsProvider, tmp_path: Path
) -> None:
    """Check that directories without Shar shards are rejected with their paths."""
    with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
        provider.build_loader([tmp_path], TOKEN_BUCKETING_CONFIG, 0, 1)


@pytest.mark.gpu
def test_tensor_parallel_ranks_get_same_batches(
    tensor_parallel_size: int, provider: AudioTextDatasetsProvider
) -> None:
    """Check that the ranks of one data parallel group get the same train and valid batches."""
    train_iterator, valid_iterator, _ = provider((0, 0, 0))
    batches = [*islice(train_iterator, 3), *islice(valid_iterator, 3)]
    input_ids = [batch.input_ids.tolist() for batch in batches]
    input_ids_of_ranks = [None] * tensor_parallel_size
    torch.distributed.all_gather_object(input_ids_of_ranks, input_ids)
    assert all(rank_input_ids == input_ids for rank_input_ids in input_ids_of_ranks)

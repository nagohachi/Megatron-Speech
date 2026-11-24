import re
from copy import deepcopy
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
from megatron_speech.data.provider import AudioTextDataConfig, AudioTextDatasetsProvider
from megatron_speech.data.sampling import TokenBucketingConfig

SAMPLING_RATE = 16000
SAMPLES_PER_AUDIO_TOKEN = 1600
PAD_TO_MULTIPLE = 4
# The train cuts have no end; with the default buffer of 20000 cuts, the 40 utterances would be
# read 500 times before the first micro-batch.
TOKEN_BUCKETING_CONFIG = TokenBucketingConfig(
    max_tokens_per_batch=100, max_tokens_per_example=60, num_buckets=3, buffer_size=50
)
# At about 4 utterances per micro-batch, about 6 reads of the 40 utterances. An utterance can
# wait in the buffer while other buckets are sampled, and 30 micro-batches missed one.
NUM_TRAIN_BATCHES = 60
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
def build_cuts(shar_dir: Path) -> CutSet:
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


@pytest.fixture(name="train_batches_of_ranks", scope="module")
def build_train_batches_of_ranks(
    provider: AudioTextDatasetsProvider,
) -> list[list[PackedAudioTextBatch]]:
    """Load the first train micro-batches of two data parallel ranks; return them by rank.

    The micro-batches are copied, so that the shared memory of each tensor from the workers,
    which holds a file descriptor, is released.
    """
    return [
        [
            deepcopy(batch)
            for batch in islice(provider.build_train_loader(rank, 2), NUM_TRAIN_BATCHES)
        ]
        for rank in range(2)
    ]


@pytest.fixture(name="valid_batches_of_ranks", scope="module")
def build_valid_batches_of_ranks(
    provider: AudioTextDatasetsProvider,
) -> list[list[PackedAudioTextBatch]]:
    """Load one pass of the validation micro-batches of two data parallel ranks, by rank."""
    return [list(provider.build_valid_loader(rank, 2)) for rank in range(2)]


def test_batches_stay_within_token_limit(
    train_batches_of_ranks: list[list[PackedAudioTextBatch]],
    valid_batches_of_ranks: list[list[PackedAudioTextBatch]],
) -> None:
    """Check that no packed row, padding included, is longer than ``max_tokens_per_batch``."""
    for batches in [*train_batches_of_ranks, *valid_batches_of_ranks]:
        for batch in batches:
            assert batch.cu_seqlens[-1] <= TOKEN_BUCKETING_CONFIG.max_tokens_per_batch


def test_train_ranks_read_every_utterance(
    train_batches_of_ranks: list[list[PackedAudioTextBatch]],
) -> None:
    """Check that the train data are not divided: each rank reads every utterance."""
    for batches in train_batches_of_ranks:
        lengths = {len(waveform) for batch in batches for waveform in batch.waveforms}
        assert lengths == set(UTTERANCE_LENGTHS)


def test_train_ranks_read_in_different_orders(
    train_batches_of_ranks: list[list[PackedAudioTextBatch]],
) -> None:
    """Check that the two data parallel ranks get different train micro-batches."""
    batches_of_rank0, batches_of_rank1 = train_batches_of_ranks
    assert not torch.equal(batches_of_rank0[0].input_ids, batches_of_rank1[0].input_ids)


def test_train_loaders_of_one_rank_give_same_batches(
    train_batches_of_ranks: list[list[PackedAudioTextBatch]],
    provider: AudioTextDatasetsProvider,
) -> None:
    """Check that another train loader of rank 0, as on another tensor parallel rank, agrees."""
    batches = islice(provider.build_train_loader(0, 2), NUM_TRAIN_BATCHES)
    for batch, batch_of_rank0 in zip(batches, train_batches_of_ranks[0], strict=True):
        assert torch.equal(batch.input_ids, batch_of_rank0.input_ids)


def test_valid_ranks_take_every_utterance_once(
    valid_batches_of_ranks: list[list[PackedAudioTextBatch]],
) -> None:
    """Check that the data parallel ranks together take each validation utterance once."""
    lengths = [
        len(waveform)
        for batches in valid_batches_of_ranks
        for batch in batches
        for waveform in batch.waveforms
    ]
    assert sorted(lengths) == UTTERANCE_LENGTHS


def test_evaluations_take_same_batches(provider: AudioTextDatasetsProvider) -> None:
    """Check that each evaluation, also one longer than a pass, takes the same micro-batches."""
    loader = provider.build_valid_loader(0, 1)
    num_microbatches_per_evaluation = len(list(loader)) + 2
    provider_of_long_evaluations = AudioTextDatasetsProvider(
        provider.data_config, CharExampleBuilder(), PAD_TO_MULTIPLE, num_microbatches_per_evaluation
    )
    batches = list(
        islice(
            provider_of_long_evaluations.iterate_evaluations(loader),
            2 * num_microbatches_per_evaluation,
        )
    )
    for batch, batch_of_first_evaluation in zip(
        batches[num_microbatches_per_evaluation:],
        batches[:num_microbatches_per_evaluation],
        strict=True,
    ):
        assert torch.equal(batch.input_ids, batch_of_first_evaluation.input_ids)


def test_evaluations_reject_empty_data(provider: AudioTextDatasetsProvider) -> None:
    """Check that validation data without micro-batches is rejected instead of read forever."""
    with pytest.raises(RuntimeError, match="no micro-batch"):
        next(provider.iterate_evaluations([]))


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


def test_data_config_rejects_no_workers(shar_dir: Path) -> None:
    """Check that training without workers, which would sample in the training process, fails."""
    with pytest.raises(ValueError, match="num_workers"):
        AudioTextDataConfig([shar_dir], [], TOKEN_BUCKETING_CONFIG, num_workers=0)


def test_loader_rejects_directories_without_shards(tmp_path: Path) -> None:
    """Check that directories without Shar shards are rejected with their paths."""
    data_config = AudioTextDataConfig([tmp_path], [], TOKEN_BUCKETING_CONFIG)
    provider = AudioTextDatasetsProvider(data_config, CharExampleBuilder(), PAD_TO_MULTIPLE, 0)
    with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
        provider.build_train_loader(0, 1)


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

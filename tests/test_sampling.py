from bisect import bisect_left

import numpy as np
import pytest
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.cut import Cut
from lhotse.dataset import DynamicBucketingSampler

from megatron_speech.data.sampling import (
    PackedTokenConstraint,
    TokenBucketingConfig,
    build_token_bucketing_sampler,
)

SAMPLES_PER_TOKEN = 160
CONFIG = TokenBucketingConfig(
    max_tokens_per_batch=1000, max_tokens_per_example=300, num_buckets=5, seed=0
)
# Token counts of the utterances: mostly within the limit of an example, some above it.
UTTERANCE_TOKEN_COUNTS = [*np.random.default_rng(0).integers(10, 300, 2000).tolist(), 301, 900]


def build_cut(index: int, num_tokens: int) -> Cut:
    """Build the cut of an utterance of ``num_tokens`` tokens, with audio that is never read."""
    num_samples = num_tokens * SAMPLES_PER_TOKEN
    recording = Recording(
        id=f"utterance-{index}",
        sources=[AudioSource(type="file", channels=[0], source="unused.flac")],
        sampling_rate=16000,
        num_samples=num_samples,
        duration=num_samples / 16000,
    )
    cut = recording.to_cut()
    cut.supervisions = [
        SupervisionSegment(
            id=recording.id, recording_id=recording.id, start=0, duration=cut.duration
        )
    ]
    return cut


def calc_num_cut_tokens(cut: Cut) -> int:
    """Count the tokens of ``cut`` from its number of samples."""
    return cut.num_samples // SAMPLES_PER_TOKEN


@pytest.fixture(name="cuts")
def build_cuts() -> CutSet:
    """Build the cuts of the utterances of ``UTTERANCE_TOKEN_COUNTS``."""
    return CutSet.from_cuts(
        build_cut(index, num_tokens) for index, num_tokens in enumerate(UTTERANCE_TOKEN_COUNTS)
    )


@pytest.fixture(name="sampler")
def build_sampler(cuts: CutSet) -> DynamicBucketingSampler:
    """Build the sampler of the cuts on the only rank."""
    return build_token_bucketing_sampler(cuts, calc_num_cut_tokens, CONFIG, rank=0, world_size=1)


def test_batches_stay_within_token_limit(sampler: DynamicBucketingSampler) -> None:
    """Check that no micro-batch of an epoch has more than ``max_tokens_per_batch`` tokens."""
    for batch in sampler:
        assert sum(calc_num_cut_tokens(cut) for cut in batch) <= CONFIG.max_tokens_per_batch


def test_batches_hold_each_fitting_utterance_once(
    sampler: DynamicBucketingSampler, cuts: CutSet
) -> None:
    """Check that an epoch holds every utterance within the example limit once, and no other."""
    sampled_ids = sorted(cut.id for batch in sampler for cut in batch)
    fitting_ids = sorted(
        cut.id for cut in cuts if calc_num_cut_tokens(cut) <= CONFIG.max_tokens_per_example
    )
    assert sampled_ids == fitting_ids


def test_batches_hold_one_bucket(sampler: DynamicBucketingSampler) -> None:
    """Check that the utterances of each micro-batch come from one bucket."""
    for batch in sampler:
        buckets = {bisect_left(sampler.duration_bins, calc_num_cut_tokens(cut)) for cut in batch}
        assert len(buckets) == 1


def test_ranks_take_equal_steps_of_different_utterances(cuts: CutSet) -> None:
    """Check that data parallel ranks take as many steps, of different utterances, as one rank.

    Every rank has to run the same number of steps; together they cover the fitting utterances.
    """
    samplers = [
        build_token_bucketing_sampler(cuts, calc_num_cut_tokens, CONFIG, rank=rank, world_size=2)
        for rank in range(2)
    ]
    batches_rank0, batches_rank1 = (list(sampler) for sampler in samplers)
    assert len(batches_rank0) == len(batches_rank1)
    ids_rank0 = [cut.id for batch in batches_rank0 for cut in batch]
    ids_rank1 = [cut.id for batch in batches_rank1 for cut in batch]
    fitting_ids = [
        cut.id for cut in cuts if calc_num_cut_tokens(cut) <= CONFIG.max_tokens_per_example
    ]
    assert not set(ids_rank0) & set(ids_rank1)
    assert sorted(ids_rank0 + ids_rank1) == sorted(fitting_ids)


def test_constraint_closes_batch_before_limit_can_be_passed() -> None:
    """Check that a batch is closed once one more example of its bucket could pass the limit.

    In a bucket of up to 100 tokens, 90-token examples are added up to the 11th (990 tokens),
    after which a 100-token example would make 1090 tokens.
    """
    constraint = PackedTokenConstraint(1000, calc_num_cut_tokens, bucket_max_tokens=[100, 300])
    num_examples = 0
    while not constraint.close_to_exceeding():
        constraint.add(build_cut(num_examples, 90))
        num_examples += 1
    assert num_examples == 11
    assert not constraint.exceeded()


@pytest.mark.parametrize(
    ("setting", "value"),
    [("max_tokens_per_example", 1001), ("max_tokens_per_example", 0), ("num_buckets", 1)],
)
def test_config_rejects_impossible_settings(setting: str, value: int) -> None:
    """Check that the config names the setting that leaves no room for an example or bucket."""
    settings = {"max_tokens_per_batch": 1000, "max_tokens_per_example": 300, setting: value}
    with pytest.raises(ValueError, match=setting):
        TokenBucketingConfig(**settings)

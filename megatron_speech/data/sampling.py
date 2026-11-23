from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, replace

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.sampling.base import SamplingConstraint
from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets


@dataclass
class PackedTokenConstraint(SamplingConstraint):
    """Limit of the tokens in a micro-batch whose examples are packed into one row.

    Each batch is closed early enough that the largest example of its bucket would still fit,
    so that no batch passes ``max_tokens_per_batch``.
    """

    max_tokens_per_batch: int
    calc_num_tokens: Callable[[Cut], int]
    # The largest token count of each bucket of the sampler, in increasing order.
    bucket_max_tokens: list[int]
    num_tokens: int = 0
    max_tokens_of_next_example: int = 0

    def add(self, example: Cut) -> None:
        """Count the tokens of ``example``; the bucket of the batch is that of its first example."""
        num_tokens = self.measure_length(example)
        if self.num_tokens == 0:
            bucket = bisect_left(self.bucket_max_tokens, num_tokens)
            self.max_tokens_of_next_example = self.bucket_max_tokens[bucket]
        self.num_tokens += num_tokens

    def exceeded(self) -> bool:
        """Return whether the batch has more tokens than ``max_tokens_per_batch``."""
        return self.num_tokens > self.max_tokens_per_batch

    def close_to_exceeding(self) -> bool:
        """Return whether the limit could be passed with one more example of the batch's bucket."""
        return self.num_tokens + self.max_tokens_of_next_example > self.max_tokens_per_batch

    def reset(self) -> None:
        """Start counting a new batch."""
        self.num_tokens = 0
        self.max_tokens_of_next_example = 0

    def measure_length(self, example: Cut) -> int:
        """Return the number of tokens of ``example``, as lhotse's samplers measure lengths."""
        return self.calc_num_tokens(example)

    def fits(self, example: Cut) -> bool:
        """Return whether ``example`` fits in the largest bucket."""
        return self.measure_length(example) <= self.bucket_max_tokens[-1]


@dataclass
class TokenBucketingConfig:
    """Settings of the sampler that ``build_token_bucketing_sampler`` builds."""

    # Limit of the tokens of a micro-batch, whose examples are packed into one row.
    max_tokens_per_batch: int
    # Examples with more tokens are dropped.
    max_tokens_per_example: int
    num_buckets: int = 10
    # The bucket bounds are estimated from this many cuts at the start of the cuts.
    num_cuts_for_bins_estimate: int = 10000
    shuffle: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        """Check that the limits leave room for one example and that there are buckets."""
        if not 0 < self.max_tokens_per_example <= self.max_tokens_per_batch:
            raise ValueError(
                f"max_tokens_per_example ({self.max_tokens_per_example}) must be positive and "
                f"at most max_tokens_per_batch ({self.max_tokens_per_batch})"
            )
        if self.num_buckets < 2:
            raise ValueError(f"num_buckets must be at least 2, got {self.num_buckets}")


def build_token_bucketing_sampler(
    cuts: CutSet,
    calc_num_tokens: Callable[[Cut], int],
    config: TokenBucketingConfig,
    rank: int,
    world_size: int,
) -> DynamicBucketingSampler:
    """Build a sampler of micro-batches of utterances of similar token counts.

    A micro-batch has at most ``config.max_tokens_per_batch`` tokens, counted with
    ``calc_num_tokens``. ``rank`` and ``world_size`` are those of data parallelism: every rank
    gets as many micro-batches, and lhotse tries to take those of one step from the same bucket
    (``sync_buckets``).
    """
    constraint = PackedTokenConstraint(
        config.max_tokens_per_batch, calc_num_tokens, [config.max_tokens_per_example]
    )
    cuts = cuts.filter(constraint.fits)
    # The largest token count of each bucket but the last.
    bucket_bounds = [
        int(bound)
        for bound in estimate_duration_buckets(
            cuts.subset(first=config.num_cuts_for_bins_estimate), config.num_buckets, constraint
        )
    ]
    return DynamicBucketingSampler(
        cuts,
        constraint=replace(
            constraint, bucket_max_tokens=[*bucket_bounds, config.max_tokens_per_example]
        ),
        duration_bins=bucket_bounds,
        shuffle=config.shuffle,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
        sync_buckets=True,
    )

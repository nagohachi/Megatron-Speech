# Portions of this file are adapted from NVIDIA NeMo,
#   https://github.com/NVIDIA/NeMo/blob/f47ddde246380a250bd868b51b4a7f3a2f3770f3/nemo/collections/common/data/lhotse/sampling.py
#   (PackedTokenConstraint, FixedBucketBatchSizeConstraint2D, find_smallest_bucket)
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modifications: lengths are read from ``cut.custom["llm_num_tokens"]`` (set by
# megatron_speech.data.dataset.annotate_num_tokens) instead of NeMo's Formattable examples.
"""Sampling constraints for packed (THD) speech-LLM batches."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from lhotse.dataset import SamplingConstraint
from lhotse.dataset.sampling.dynamic_bucketing import FixedBucketBatchSizeConstraint

NUM_TOKENS_KEY = "llm_num_tokens"


def get_num_tokens(cut: Any) -> int:
    return int(cut.custom[NUM_TOKENS_KEY])


@dataclass
class PackedTokenConstraint(SamplingConstraint):
    """Token budget of a batch whose examples are packed into one row (no padding).

    A padded batch costs ``num_examples * longest_example``; a packed batch costs the sum of
    the example lengths, which is what this constraint counts. ``batch_tokens`` is a hard cap on
    that sum. ``quadratic_length`` adds ``length**2 / quadratic_length`` per example to account
    for attention cost growing with length. ``max_examples`` caps the number of examples.

    lhotse's batcher adds an example first and then asks ``close_to_exceeding()``, keeping the
    example even if the cap is now exceeded. With ``bucket_bins`` set (the sampler's
    ``duration_bins``), a batch, which always comes from one bucket, is closed as soon as one
    more example of the bucket's largest length would not fit, so ``batch_tokens`` is never
    exceeded. (Megatron-Speech addition; NeMo solves this with a dedicated batcher.)
    """

    batch_tokens: int | None = None
    max_examples: int | None = None
    quadratic_length: float | None = None
    bucket_bins: list[float] | None = None
    current: int = 0
    num_examples: int = 0
    _bucket_upper: float | None = None

    def __post_init__(self) -> None:
        if self.batch_tokens is not None and self.batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive or null (got {self.batch_tokens})")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError(f"max_examples must be positive or null (got {self.max_examples})")
        if self.quadratic_length is not None and self.quadratic_length <= 0:
            raise ValueError(f"quadratic_length must be positive or null (got {self.quadratic_length})")

    def budget_length(self, size: float) -> int:
        if self.quadratic_length is None:
            return math.ceil(size)
        return math.ceil(size + size**2 / self.quadratic_length)

    def add(self, example: Any) -> None:
        length = self.measure_length(example)
        if self.num_examples == 0 and self.bucket_bins is not None:
            # lhotse puts lengths in (bins[i-1], bins[i]] into bucket i = bisect_left(bins, length).
            idx = bisect_left(self.bucket_bins, length)
            self._bucket_upper = self.bucket_bins[idx] if idx < len(self.bucket_bins) else self.batch_tokens
        self.current += self.budget_length(length)
        self.num_examples += 1

    def exceeded(self) -> bool:
        if self.max_examples is not None and self.num_examples > self.max_examples:
            return True
        return self.batch_tokens is not None and self.current > self.batch_tokens

    def close_to_exceeding(self) -> bool:
        if self.max_examples is not None and self.num_examples >= self.max_examples:
            return True
        if self.num_examples == 0 or self.batch_tokens is None:
            return False
        if self._bucket_upper is not None:
            return self.current + self.budget_length(self._bucket_upper) > self.batch_tokens
        mean_length = self.current / self.num_examples
        return self.current + mean_length > self.batch_tokens

    def reset(self) -> None:
        self.current = 0
        self.num_examples = 0
        self._bucket_upper = None

    def measure_length(self, example: Any) -> float:
        return get_num_tokens(example)


@dataclass
class FixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint):
    """Fixed batch size per bucket, with 2D buckets over (duration, output tokens).

    ``max_seq_len_buckets`` is a list of ``(duration, tokens)`` upper bounds sorted by duration.
    In ``strict_2d`` mode an example only goes to sub-buckets of its duration bucket; otherwise it
    may be promoted to a larger duration bucket that fits its token count.
    """

    strict_2d: bool = True
    max_ratio: list[float] | None = None

    def __post_init__(self):
        if isinstance(self.max_seq_len_buckets[0], Sequence):
            self.max_seq_len_buckets = np.asarray(self.max_seq_len_buckets)
        if self.max_ratio is not None:
            assert len(self.max_ratio) == len(self.max_seq_len_buckets)

    @property
    def bucketing_2d_enabled(self) -> bool:
        return isinstance(self.max_seq_len_buckets, np.ndarray)

    def measure_length(self, example: Any) -> tuple[float, float] | float:
        if self.bucketing_2d_enabled:
            return example.duration, get_num_tokens(example)
        return example.duration

    def select_bucket(self, buckets: Any, example: Any = None, example_len: Any = None) -> int:
        if example_len is None:
            example_len = self.measure_length(example)
        return find_smallest_bucket(self.max_seq_len_buckets, example_len, strict=self.strict_2d, max_ratio=self.max_ratio)


def find_smallest_bucket(buckets: np.ndarray, example_lens, strict: bool = True, max_ratio: Sequence[float] | None = None) -> int | None:
    """Index of the smallest bucket that fits ``example_lens`` (1D float or 2D tuple), or None."""
    if isinstance(example_lens, (float, int)):
        idx = bisect_left(buckets, example_lens)
        return None if idx == len(buckets) else idx
    if strict:
        dim0_begin = bisect_left(buckets[:, 0], example_lens[0])
        if dim0_begin == buckets.shape[0]:
            return None
        dim0_end = dim0_begin
        while dim0_end < buckets.shape[0] and buckets[dim0_end, 0] == buckets[dim0_begin, 0]:
            dim0_end += 1
        dim1_begin = bisect_left(buckets[dim0_begin:dim0_end, 1], example_lens[1])
        if dim1_begin == dim0_end - dim0_begin:
            return None
        fit_idx = dim0_begin + dim1_begin
        if max_ratio is not None and example_lens[1] / example_lens[0] > max_ratio[fit_idx]:
            return None
        return fit_idx
    does_fit = np.all(np.asarray(example_lens) <= buckets, axis=1)
    min_fit_idx = np.argmax(does_fit)
    if min_fit_idx or does_fit[min_fit_idx]:
        return min_fit_idx.item()
    return None

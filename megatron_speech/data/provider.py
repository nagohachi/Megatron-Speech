"""Train/validation iterators for Megatron's training loop (``--dataloader-type external``).

Run config::

    data:
      train: {shar_dirs: [data/librispeech/shar/train-clean-100]}
      valid: {shar_dirs: [data/librispeech/shar/dev-clean]}
      max_tokens_per_microbatch: 8192   # packed LLM tokens per micro-batch (also --seq-length)
      max_examples_per_microbatch: null
      quadratic_factor: null            # see PackedTokenConstraint.quadratic_length
      num_buckets: 30
      shuffle_buffer_size: 10000
      min_duration: 0.1
      max_duration: 40.0
      num_workers: 4
      default_language: null            # used when a supervision has no language
      force_language: false
      context: ""                       # system prompt text
      text_transform: none              # none | lower | upper, applied to transcripts

Data parallelism: each data-parallel rank reads a disjoint subset of the Shar shards
(shard ``i`` goes to rank ``i % dp_size``). Ranks of the same data-parallel group (tensor and
context parallel peers) build identical iterators, so they receive identical batches.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets

from megatron_speech import compat
from megatron_speech.bridges.hf_io import resolve_hf_path
from megatron_speech.components.registry import parse_encoder_config
from megatron_speech.data.dataset import Qwen3ASRExampleBuilder, SpeechLLMDataset, TokenCounter
from megatron_speech.data.sampling import PackedTokenConstraint, get_num_tokens


def list_shar_shards(shar_dirs: list[str]) -> list[tuple[str, str]]:
    """``(cuts_file, recording_file)`` pairs of all shards, in a fixed order."""
    shards = []
    for d in shar_dirs:
        for cuts_file in sorted(Path(d).glob("cuts.*.jsonl.gz")):
            index = cuts_file.name.split(".")[1]
            rec = cuts_file.parent / f"recording.{index}.tar"
            if not rec.exists():
                raise FileNotFoundError(f"{rec} is missing for {cuts_file}")
            shards.append((str(cuts_file), str(rec)))
    if not shards:
        raise FileNotFoundError(f"no Shar shards in {shar_dirs}")
    return shards


def open_shar(shards: list[tuple[str, str]], seed: int, shuffle: bool, with_audio: bool = True) -> CutSet:
    fields = {"cuts": [c for c, _ in shards]}
    if with_audio:
        fields["recording"] = [r for _, r in shards]
    return CutSet.from_shar(fields=fields, shuffle_shards=shuffle, stateful_shuffle=True, seed=seed)


class _DurationFilter:
    def __init__(self, min_duration: float, max_duration: float):
        self.min_duration, self.max_duration = min_duration, max_duration

    def __call__(self, cut) -> bool:
        return self.min_duration <= cut.duration <= self.max_duration


class _TokenFilter:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def __call__(self, cut) -> bool:
        return get_num_tokens(cut) <= self.max_tokens


class CyclicIterator:
    """Yields ``length`` items from ``factory()``, then starts over from a fresh iterator, so
    every evaluation sees the same batches."""

    def __init__(self, factory: Callable[[], Iterator], length: int):
        self.factory, self.length = factory, length
        self._it, self._count = None, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None or self._count == self.length:
            self._it, self._count = iter(self.factory()), 0
        try:
            item = next(self._it)
        except StopIteration:
            self._it, self._count = iter(self.factory()), 0
            item = next(self._it)
        self._count += 1
        return item


def calc_pad_multiple(args) -> int:
    multiple = 2 * args.context_parallel_size if args.context_parallel_size > 1 else 1
    if args.sequence_parallel:
        multiple = math.lcm(multiple, args.tensor_model_parallel_size)
    return multiple


def build_loader(data_cfg: dict[str, Any], split_cfg: dict[str, Any], builder: Qwen3ASRExampleBuilder,
                 dp_rank: int, dp_size: int, seed: int, train: bool, pad_to_multiple: int):
    shards = list_shar_shards(split_cfg["shar_dirs"])
    if dp_size > 1:
        if len(shards) < dp_size:
            raise ValueError(f"{len(shards)} shards cannot be split over {dp_size} data-parallel ranks")
        shards = shards[dp_rank::dp_size]
    max_tokens = int(data_cfg["max_tokens_per_microbatch"])
    duration_filter = _DurationFilter(float(data_cfg.get("min_duration", 0.1)), float(data_cfg.get("max_duration", 40.0)))
    counter, token_filter = TokenCounter(builder), _TokenFilter(max_tokens)

    def prepare(cuts: CutSet) -> CutSet:
        return cuts.filter(duration_filter).map(counter).filter(token_filter)

    constraint = PackedTokenConstraint(
        batch_tokens=max_tokens,
        max_examples=data_cfg.get("max_examples_per_microbatch"),
        quadratic_length=data_cfg.get("quadratic_factor"),
    )
    num_buckets = int(data_cfg.get("num_buckets", 30))
    # Bucket boundaries from manifests only (no audio reads).
    sample = prepare(open_shar(shards, seed, shuffle=True, with_audio=False)).subset(first=int(data_cfg.get("num_cuts_for_bins_estimate", 5000)))
    duration_bins = estimate_duration_buckets(sample, num_buckets=num_buckets, constraint=constraint)
    constraint.bucket_bins = list(duration_bins)

    cuts = prepare(open_shar(shards, seed, shuffle=train))
    if train:
        cuts = cuts.repeat()
    sampler = DynamicBucketingSampler(
        cuts,
        constraint=constraint,
        duration_bins=duration_bins,
        shuffle=train,
        buffer_size=int(data_cfg.get("bucket_buffer_size", 10000)),
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 10000)),
        drop_last=train,
        seed=seed,
        world_size=1,
        rank=0,
    )
    num_workers = int(data_cfg.get("num_workers", 4))
    return torch.utils.data.DataLoader(
        SpeechLLMDataset(builder, pad_to_multiple=pad_to_multiple),
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=train and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def make_example_builder(data_cfg: dict[str, Any], model_section: dict[str, Any]) -> Qwen3ASRExampleBuilder:
    if model_section.get("bridge") != "qwen3_asr":
        raise NotImplementedError("the data pipeline currently supports the qwen3_asr prompt format only")
    return Qwen3ASRExampleBuilder(
        str(resolve_hf_path(model_section["hf_init"])),
        parse_encoder_config(model_section["encoder"]),
        default_language=data_cfg.get("default_language"),
        force_language=bool(data_cfg.get("force_language", False)),
        context=data_cfg.get("context", ""),
        text_transform=data_cfg.get("text_transform", "none"),
    )


def build_datasets_provider(cfg: dict[str, Any], model_section: dict[str, Any]):
    data_cfg = cfg["data"]

    def provider(_train_val_test_num_samples):
        args = compat.get_args()
        dp_rank, dp_size = compat.get_data_parallel_rank(), compat.get_data_parallel_world_size()
        builder = make_example_builder(data_cfg, model_section)
        pad = calc_pad_multiple(args)
        # A resumed run continues with a different shuffle instead of replaying the first batches.
        seed = int(args.seed) + int(getattr(args, "iteration", 0) or 0)
        train = iter(build_loader(data_cfg, data_cfg["train"], builder, dp_rank, dp_size, seed, True, pad))
        valid = None
        if data_cfg.get("valid") and args.eval_iters > 0:
            num_microbatches = args.global_batch_size // (args.micro_batch_size * dp_size)
            loader = build_loader(data_cfg, data_cfg["valid"], builder, dp_rank, dp_size, int(args.seed), False, pad)
            valid = CyclicIterator(lambda: loader, args.eval_iters * num_microbatches)
        compat.print_rank_0(f"> speech data: train {data_cfg['train']['shar_dirs']} valid {(data_cfg.get('valid') or {}).get('shar_dirs')}")
        return train, valid, None

    provider.is_distributed = True
    return provider

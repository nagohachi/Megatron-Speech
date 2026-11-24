from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import count, islice
from pathlib import Path

from lhotse import CutSet
from megatron.core import parallel_state
from torch.utils.data import DataLoader

from megatron_speech.data.dataset import AudioTextDataset, AudioTextExampleBuilder
from megatron_speech.data.packing import PackedAudioTextBatch
from megatron_speech.data.sampling import TokenBucketingConfig, build_token_bucketing_sampler


@dataclass
class AudioTextDataConfig:
    """Data of the training of a speech LLM and the settings of its micro-batches."""

    # Directories of lhotse Shar shards.
    train_shar_dirs: list[Path]
    # Without directories, there is no validation.
    valid_shar_dirs: list[Path]
    token_bucketing: TokenBucketingConfig
    # Number of worker processes on each rank, in which the audio is loaded and packed.
    num_workers: int = 1


class AudioTextDatasetsProvider:
    """Datasets provider of ``megatron.training.pretrain`` for ``--dataloader-type external``.

    Data loaders are built on each rank, and the ranks of one data parallel group are given the
    same micro-batches. A micro-batch is one packed row, so ``--micro-batch-size`` is 1.
    """

    # The provider is called on every rank, not only on the first tensor parallel rank.
    is_distributed = True

    def __init__(
        self,
        data_config: AudioTextDataConfig,
        example_builder: AudioTextExampleBuilder,
        pad_to_multiple: int,
        num_microbatches_per_evaluation: int,
    ) -> None:
        """Keep the settings; the loaders are built when the provider is called.

        Parameters
        ----------
        data_config
            Shar directories of the training and the validation, and the micro-batch settings.
        example_builder
            Builder of the examples of the utterances in the prompt format of the model.
        pad_to_multiple
            Multiple that each utterance is padded to in a packed row, as
            ``PackedAudioTextBatch.build`` takes it for sequence and context parallelism.
        num_microbatches_per_evaluation
            Micro-batches of one evaluation on each data parallel rank: ``--eval-iters`` times
            the micro-batches of an evaluation step. Without validation data, it is not used.
        """
        if pad_to_multiple < 1:
            raise ValueError(f"pad_to_multiple must be at least 1, got {pad_to_multiple}")
        if data_config.valid_shar_dirs and num_microbatches_per_evaluation < 1:
            raise ValueError(
                "num_microbatches_per_evaluation must be at least 1 with validation data, got "
                f"{num_microbatches_per_evaluation}"
            )
        self.data_config = data_config
        self.example_builder = example_builder
        self.pad_to_multiple = pad_to_multiple
        self.num_microbatches_per_evaluation = num_microbatches_per_evaluation

    def build_loader(
        self,
        shar_dirs: list[Path],
        token_bucketing_config: TokenBucketingConfig,
        data_parallel_rank: int,
        data_parallel_size: int,
    ) -> DataLoader:
        """Build the loader of one epoch of the micro-batches of ``data_parallel_rank``.

        All shards are read on every data parallel rank, in the same order, and the micro-batches
        are divided among the ranks.
        """
        fields = {"cuts": [], "recording": []}
        for shar_dir in shar_dirs:
            fields["cuts"] += sorted(shar_dir.glob("cuts.*.jsonl.gz"))
            fields["recording"] += sorted(shar_dir.glob("recording.*.tar"))
        if not fields["cuts"]:
            raise ValueError(f"no Shar shards in {[str(shar_dir) for shar_dir in shar_dirs]}")
        shuffle_shards = token_bucketing_config.shuffle
        seed = token_bucketing_config.seed
        dataset = AudioTextDataset(self.example_builder, self.pad_to_multiple)
        sampler = build_token_bucketing_sampler(
            CutSet.from_shar(fields=fields, shuffle_shards=shuffle_shards, seed=seed),
            dataset.calc_num_tokens,
            token_bucketing_config,
            rank=data_parallel_rank,
            world_size=data_parallel_size,
            # The bucket bounds need only the durations and the transcripts of the manifests.
            cuts_for_bins_estimate=CutSet.from_shar(
                fields={"cuts": fields["cuts"]}, shuffle_shards=shuffle_shards, seed=seed
            ),
        )
        return DataLoader(
            dataset, sampler=sampler, batch_size=None, num_workers=self.data_config.num_workers
        )

    def iterate_epochs(self, loader: DataLoader) -> Iterator[PackedAudioTextBatch]:
        """Yield the micro-batches of ``loader`` epoch after epoch, each epoch in another order."""
        for epoch in count():
            loader.sampler.set_epoch(epoch)
            num_microbatches_read = 0
            for batch in loader:
                num_microbatches_read += 1
                yield batch
            if not num_microbatches_read:
                raise RuntimeError("the training data have no micro-batch on this rank")

    def iterate_evaluations(
        self, loader: Iterable[PackedAudioTextBatch]
    ) -> Iterator[PackedAudioTextBatch]:
        """Yield the same first micro-batches of ``loader`` at every evaluation.

        An evaluation takes ``num_microbatches_per_evaluation`` micro-batches; when ``loader``
        has fewer, it is read again from the start within the evaluation.
        """
        while True:
            num_microbatches_left = self.num_microbatches_per_evaluation
            while num_microbatches_left:
                num_microbatches_read = 0
                for batch in islice(loader, num_microbatches_left):
                    num_microbatches_read += 1
                    yield batch
                if not num_microbatches_read:
                    raise RuntimeError("the validation data have no micro-batch on this rank")
                num_microbatches_left -= num_microbatches_read

    def __call__(
        self, train_valid_test_num_samples: tuple[int, int | None, int]
    ) -> tuple[Iterator[PackedAudioTextBatch], Iterator[PackedAudioTextBatch] | None, None]:
        """Build the iterators of the micro-batches of this rank.

        ``train_valid_test_num_samples`` is not used: the number of micro-batches of an epoch
        depends on the lengths of the utterances.

        Returns
        -------
        tuple
            The train micro-batches, epoch after epoch; the validation micro-batches, the same
            at every evaluation, or None without validation data; and None for the test.
        """
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        token_bucketing_config = self.data_config.token_bucketing
        train_iterator = self.iterate_epochs(
            self.build_loader(
                self.data_config.train_shar_dirs,
                token_bucketing_config,
                data_parallel_rank,
                data_parallel_size,
            )
        )
        if not self.data_config.valid_shar_dirs:
            return train_iterator, None, None
        valid_loader = self.build_loader(
            self.data_config.valid_shar_dirs,
            replace(token_bucketing_config, shuffle=False),
            data_parallel_rank,
            data_parallel_size,
        )
        return (
            train_iterator,
            self.iterate_evaluations(valid_loader),
            None,
        )

from abc import ABC, abstractmethod
from pathlib import Path

from lhotse import CutSet


class CorpusRecipe(ABC):
    """Base class of the corpus recipes; one subclass is written for each corpus."""

    @abstractmethod
    def build_cuts(self, corpus_dir: Path, parts: list[str], num_jobs: int) -> dict[str, CutSet]:
        """Build one cut per utterance of each of ``parts`` of the corpus, by part.

        ``num_jobs`` is the number of processes or threads that the recipe may use.
        """

    @abstractmethod
    def normalize_transcript(self, transcript: str) -> str:
        """Return ``transcript`` in the form that models trained on this corpus learn to write.

        The transcripts are normalized with it when the shards are written.
        """

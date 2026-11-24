from pathlib import Path

from lhotse import CutSet
from lhotse.recipes import prepare_librispeech

from megatron_speech.data.recipes.base import CorpusRecipe


class LibriSpeechRecipe(CorpusRecipe):
    """LibriSpeech, read with lhotse's LibriSpeech recipe."""

    def build_cuts(self, corpus_dir: Path, parts: list[str], num_jobs: int) -> dict[str, CutSet]:
        """Build one cut per utterance of each of ``parts`` of LibriSpeech, by part.

        In ``corpus_dir``, the parts are laid out as OpenSLR distributes them, such as
        ``train-clean-100`` and ``dev-clean``. The audio files are read in ``num_jobs`` threads.
        """
        manifests = prepare_librispeech(corpus_dir, dataset_parts=parts, num_jobs=num_jobs)
        return {part: CutSet.from_manifests(**manifests[part]) for part in parts}

    def normalize_transcript(self, transcript: str) -> str:
        """Return ``transcript`` in lowercase; the transcripts of LibriSpeech are in capitals."""
        return transcript.lower()

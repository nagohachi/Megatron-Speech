from pathlib import Path

from lhotse import CutSet
from lhotse.recipes import prepare_librispeech


def build_librispeech_cuts(corpus_dir: Path, parts: list[str], num_jobs: int) -> dict[str, CutSet]:
    """Build one cut per utterance of each of ``parts`` of LibriSpeech, by part.

    ``corpus_dir`` holds the parts as OpenSLR distributes them, such as ``train-clean-100`` and
    ``dev-clean``. ``num_jobs`` threads read the audio files.
    """
    manifests = prepare_librispeech(corpus_dir, dataset_parts=parts, num_jobs=num_jobs)
    return {part: CutSet.from_manifests(**manifests[part]) for part in parts}

from collections.abc import Callable
from pathlib import Path

from lhotse import CutSet

from megatron_speech.data.recipes.librispeech import build_librispeech_cuts

# Functions that build one cut per utterance of a corpus, by part, from the corpus directory, the
# part names and the number of jobs; ``megatron_speech.data.shar`` selects one by name.
RECIPES: dict[str, Callable[[Path, list[str], int], dict[str, CutSet]]] = {
    "librispeech": build_librispeech_cuts,
}

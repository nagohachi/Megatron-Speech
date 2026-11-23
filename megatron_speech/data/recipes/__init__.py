from collections.abc import Callable
from pathlib import Path

from lhotse import CutSet

from megatron_speech.data.recipes.librispeech import build_librispeech_cuts

# The recipes that ``python -m megatron_speech.data.shar`` selects by name.
RECIPES: dict[str, Callable[[Path, list[str], int], dict[str, CutSet]]] = {
    "librispeech": build_librispeech_cuts,
}

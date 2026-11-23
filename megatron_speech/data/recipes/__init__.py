from megatron_speech.data.recipes.base import CorpusRecipe
from megatron_speech.data.recipes.librispeech import LibriSpeechRecipe

# The recipes that ``python -m megatron_speech.data.shar`` selects by name.
RECIPES: dict[str, type[CorpusRecipe]] = {"librispeech": LibriSpeechRecipe}

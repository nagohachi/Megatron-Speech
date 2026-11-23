from megatron_speech.data.recipes.base import CorpusRecipe
from megatron_speech.data.recipes.librispeech import LibriSpeechRecipe

# Recipe classes by the name that is given to ``python -m megatron_speech.data.shar``.
RECIPES: dict[str, type[CorpusRecipe]] = {"librispeech": LibriSpeechRecipe}

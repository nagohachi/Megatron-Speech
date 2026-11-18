"""lhotse dataset producing packed speech-LLM micro-batches."""

from __future__ import annotations

from typing import Any

import torch
from lhotse import CutSet
from lhotse.cut import Cut

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig, calc_post_cnn_length
from megatron_speech.data.features import Qwen3ASRAudioFeaturizer
from megatron_speech.data.packing import SpeechExample, pack_examples
from megatron_speech.data.prompt import Qwen3ASRPromptFormatter
from megatron_speech.data.sampling import NUM_TOKENS_KEY


class Qwen3ASRExampleBuilder:
    """Turns a cut (audio + one supervision) into a ``SpeechExample`` and counts its tokens."""

    TEXT_TRANSFORMS = {"none": lambda t: t, "lower": str.lower, "upper": str.upper}

    def __init__(self, hf_dir: str, encoder_config: Qwen3ASREncoderConfig, default_language: str | None = None,
                 force_language: bool = False, context: str = "", text_transform: str = "none"):
        from transformers import AutoTokenizer

        self.hf_dir = hf_dir
        self.encoder_config = encoder_config
        self.featurizer = Qwen3ASRAudioFeaturizer(hf_dir, encoder_config)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        self.formatter = Qwen3ASRPromptFormatter(self.tokenizer, force_language=force_language)
        self.default_language = default_language
        self.context = context
        if text_transform not in self.TEXT_TRANSFORMS:
            raise ValueError(f"text_transform must be one of {sorted(self.TEXT_TRANSFORMS)}")
        self.text_transform = text_transform
        self.min_samples = int(getattr(self.featurizer.feature_extractor, "min_length", 0) or 0)
        self.hop_length = self.featurizer.feature_extractor.hop_length

    def _text_and_language(self, cut: Cut) -> tuple[str, str | None]:
        if len(cut.supervisions) != 1:
            raise ValueError(f"cut {cut.id} must have exactly one supervision, has {len(cut.supervisions)}")
        sup = cut.supervisions[0]
        return self.TEXT_TRANSFORMS[self.text_transform](sup.text), sup.language or self.default_language

    def count_audio_tokens(self, cut: Cut) -> int:
        num_samples = max(round(cut.duration * self.featurizer.sampling_rate), self.min_samples)
        frames = num_samples // self.hop_length
        chunk = self.encoder_config.chunk_frames
        return (frames // chunk) * self.encoder_config.tokens_per_chunk + calc_post_cnn_length(frames % chunk)

    def count_tokens(self, cut: Cut) -> int:
        text, language = self._text_and_language(cut)
        return len(self.formatter(self.count_audio_tokens(cut), text, language, self.context).input_ids)

    def __call__(self, cut: Cut) -> SpeechExample:
        if cut.sampling_rate != self.featurizer.sampling_rate:
            cut = cut.resample(self.featurizer.sampling_rate)
        audio = cut.load_audio()
        features = self.featurizer(audio.mean(axis=0) if audio.shape[0] > 1 else audio[0])
        text, language = self._text_and_language(cut)
        return SpeechExample(features, self.formatter(features.num_tokens, text, language, self.context))


class SpeechLLMDataset(torch.utils.data.Dataset):
    """Map-style lhotse dataset: ``dataset[cuts]`` returns one packed micro-batch."""

    def __init__(self, builder: Qwen3ASRExampleBuilder, pad_to_multiple: int = 1):
        self.builder = builder
        self.pad_to_multiple = pad_to_multiple
        self.pad_token_id = builder.tokenizer.pad_token_id or 0

    def __getitem__(self, cuts: CutSet) -> dict[str, Any]:
        examples = [self.builder(cut) for cut in cuts]
        return pack_examples(
            examples,
            tokens_per_chunk=self.builder.encoder_config.tokens_per_chunk,
            pad_to_multiple=self.pad_to_multiple,
            pad_token_id=self.pad_token_id,
        )


class TokenCounter:
    """Picklable ``cut -> cut`` map that stores the packed LLM length in ``cut.custom``."""

    def __init__(self, builder: Qwen3ASRExampleBuilder):
        self.builder = builder

    def __call__(self, cut: Cut) -> Cut:
        if cut.custom is None:
            cut.custom = {}
        cut.custom[NUM_TOKENS_KEY] = self.builder.count_tokens(cut)
        return cut

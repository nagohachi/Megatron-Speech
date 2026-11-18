"""Prompt formats: token ids of one training example and which of them are targets."""

from __future__ import annotations

from dataclasses import dataclass

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


@dataclass
class FormattedExample:
    input_ids: list[int]
    target_mask: list[bool]  # True where the token is predicted (loss is computed on it)


class Qwen3ASRPromptFormatter:
    """Qwen3-ASR chat format (the ``chat_template.jinja`` of Qwen/Qwen3-ASR-*-hf)::

        <|im_start|>system\\n{context}<|im_end|>\\n
        <|im_start|>user\\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\\n
        <|im_start|>assistant\\nlanguage {Language}<asr_text>{text}<|im_end|>

    The target is the assistant turn after ``assistant\\n``, i.e. what the model generates when
    the language is not forced. With ``force_language=True`` the ``language ...<asr_text>``
    prefix moves into the prompt and only ``{text}<|im_end|>`` is a target.
    """

    def __init__(self, tokenizer, force_language: bool = False):
        self.tokenizer = tokenizer
        self.force_language = force_language
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, num_audio_tokens: int, text: str, language: str | None = None, context: str = "") -> FormattedExample:
        language = LANGUAGE_NAMES.get(language, language) if language else None
        prefix = self._encode(f"<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>")
        middle = self._encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")
        lang_prefix = self._encode(f"language {language}<asr_text>") if language else []
        answer = self._encode(f"{text}<|im_end|>")
        prompt = prefix + [self.audio_token_id] * num_audio_tokens + middle
        if self.force_language:
            prompt, target = prompt + lang_prefix, answer
        else:
            target = lang_prefix + answer
        return FormattedExample(prompt + target, [False] * len(prompt) + [True] * len(target))

from dataclasses import dataclass

import numpy as np
import torch
from transformers import Qwen3ASRProcessor


@dataclass
class SpeechExample:
    """One utterance as the input of the speech LLM.

    - ``mel``: ``[num_mel_bins, num_mel_frames]`` log-mel of the utterance, without padding.
    - ``input_ids``: ``[num_tokens]`` token ids, with one audio token per encoder frame.
    - ``loss_mask``: ``[num_tokens]``, 1 on the tokens that the model learns to predict.
    """

    mel: torch.Tensor
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


class Qwen3ASRExampleBuilder:
    """Build ``SpeechExample`` from audio and its transcript, as Qwen3-ASR's processor does.

    - Prompt: Hugging Face's ``apply_transcription_request``, which expands the audio into
      one audio token per encoder frame.
    - Target: the assistant's answer, ``language <language><asr_text><transcript><|im_end|>``.
      With ``force_language``, ``language <language><asr_text>`` moves into the prompt, as
      Hugging Face writes it to force the language, and the target is the rest.
    - Log-mel: Hugging Face's feature extractor on the utterance alone.
    """

    def __init__(self, processor: Qwen3ASRProcessor, language: str, force_language: bool) -> None:
        """Keep the processor and the language, a full name such as ``English``."""
        self.processor = processor
        self.language = language
        self.force_language = force_language

    def build(self, waveform: np.ndarray, transcript: str) -> SpeechExample:
        """Build the example of ``waveform``, sampled at the feature extractor's rate."""
        prompt = self.processor.apply_transcription_request(
            waveform, language=self.language if self.force_language else None, tokenize=False
        )[0]
        target = f"{transcript}<|im_end|>"
        if not self.force_language:
            target = f"language {self.language}<asr_text>{target}"
        prompt_inputs = self.processor(text=[prompt], audio=[waveform], return_tensors="pt")
        target_ids = self.processor.tokenizer(
            target, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        prompt_ids = prompt_inputs.input_ids[0]
        num_mel_frames = int(prompt_inputs.input_features_mask[0].sum())
        return SpeechExample(
            mel=prompt_inputs.input_features[0, :, :num_mel_frames],
            input_ids=torch.cat([prompt_ids, target_ids]),
            loss_mask=torch.cat([torch.zeros(len(prompt_ids)), torch.ones(len(target_ids))]),
        )

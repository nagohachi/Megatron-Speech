from dataclasses import dataclass

import numpy as np
import torch
from transformers import Qwen3ASRProcessor

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig


@dataclass
class SpeechExample:
    """One utterance as the input of the speech LLM.

    - ``waveform``: ``[num_samples]`` float32 audio at the feature extractor's sampling rate.
      The log-mel is computed from it on the GPU, for a whole micro-batch at once.
    - ``input_ids``: ``[num_tokens]`` token ids, with one audio token per encoder frame.
    - ``loss_mask``: ``[num_tokens]`` float32, 1 on the tokens that the model learns to predict
      and 0 elsewhere; Megatron's loss is weighted by it.
    """

    waveform: torch.Tensor
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


class Qwen3ASRExampleBuilder:
    """Build ``SpeechExample`` from audio and its transcript, as Qwen3-ASR's processor does.

    - Prompt: Hugging Face's ``apply_transcription_request``, tokenized once; each example puts
      one audio token per encoder frame between its two halves.
    - Number of audio tokens: from the number of samples, as Hugging Face's feature extractor
      counts the mel frames, and from ``encoder_config`` for the encoder frames. The data
      loader computes no log-mel.
    - Target: the assistant's answer, ``language <language><asr_text><transcript><|im_end|>``.
      With ``force_language``, ``language <language><asr_text>`` moves into the prompt, as
      Hugging Face writes it to force the language, and the target is the rest.
    """

    def __init__(
        self,
        processor: Qwen3ASRProcessor,
        encoder_config: Qwen3ASREncoderConfig,
        language: str,
        force_language: bool,
    ) -> None:
        """Tokenize the prompt around the audio; ``language`` is a full name such as ``English``."""
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.encoder_config = encoder_config
        self.audio_token_id = processor.audio_token_id
        self.language = language
        self.force_language = force_language
        # The prompt holds one audio token, which each example replaces with its own count.
        # With tokenize=False, Hugging Face does not read the audio.
        prompt = processor.apply_transcription_request(
            np.zeros(1, dtype=np.float32),
            language=language if force_language else None,
            tokenize=False,
        )[0]
        before_audio, after_audio = prompt.split(processor.audio_token)
        self.prompt_ids_before_audio = self.tokenizer(
            before_audio, add_special_tokens=False
        ).input_ids
        self.prompt_ids_after_audio = self.tokenizer(
            after_audio, add_special_tokens=False
        ).input_ids

    def build(self, waveform: np.ndarray, transcript: str) -> SpeechExample:
        """Build the example of ``waveform``, sampled at the feature extractor's rate."""
        # Hugging Face pads audio shorter than min_length and drops the last STFT frame.
        num_samples = max(len(waveform), self.feature_extractor.min_length)
        num_mel_frames = num_samples // self.feature_extractor.hop_length
        num_audio_tokens = self.encoder_config.calc_num_encoder_frames(num_mel_frames)
        prompt_ids = [
            *self.prompt_ids_before_audio,
            *[self.audio_token_id] * num_audio_tokens,
            *self.prompt_ids_after_audio,
        ]
        target = f"{transcript}<|im_end|>"
        if not self.force_language:
            target = f"language {self.language}<asr_text>{target}"
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        return SpeechExample(
            waveform=torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
            input_ids=torch.tensor(prompt_ids + target_ids),
            loss_mask=torch.cat(
                [
                    torch.zeros(len(prompt_ids), dtype=torch.float32),
                    torch.ones(len(target_ids), dtype=torch.float32),
                ]
            ),
        )

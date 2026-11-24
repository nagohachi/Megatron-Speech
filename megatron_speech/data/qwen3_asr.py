import numpy as np
import torch
from transformers import Qwen3ASRProcessor

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.components.frontends.qwen3_asr import Qwen3ASRLogMelConfig
from megatron_speech.data.packing import AudioTextExample


class Qwen3ASRExampleBuilder:
    """Build ``AudioTextExample`` from audio and its transcript, as Qwen3-ASR's processor does.

    - Prompt: Hugging Face's ``apply_transcription_request``, tokenized once; each example puts
      one audio token per encoder frame between its two halves.
    - Number of audio tokens: the mel frames from ``log_mel_config``, as Hugging Face's feature
      extractor counts them, then the encoder frames from ``encoder_config``. The data loader
      computes no log-mel.
    - Target: the assistant's answer, ``language <language><asr_text><transcript><|im_end|>``.
      With ``force_language``, ``language <language><asr_text>`` moves into the prompt, as
      Hugging Face writes it to force the language, and the target is the rest.
      ``<asr_text>`` and the tokenizer's end-of-sequence token ``<|im_end|>`` are given as token
      ids. The transcript is tokenized with ``split_special_tokens=True``: the text of a special
      token such as ``<|im_end|>`` in it stays text, but ``<asr_text>``, an added token that is
      not special, still becomes its token.
    """

    def __init__(
        self,
        processor: Qwen3ASRProcessor,
        log_mel_config: Qwen3ASRLogMelConfig,
        encoder_config: Qwen3ASREncoderConfig,
        language: str,
        force_language: bool,
    ) -> None:
        """Tokenize the prompt around the audio; ``language`` is a full name such as ``English``."""
        self.tokenizer = processor.tokenizer
        self.log_mel_config = log_mel_config
        self.encoder_config = encoder_config
        self.audio_token_id = processor.audio_token_id
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
        self.answer_ids_before_transcript = []
        if not force_language:
            self.answer_ids_before_transcript = [
                *self.tokenizer(f"language {language}", add_special_tokens=False).input_ids,
                self.tokenizer.convert_tokens_to_ids("<asr_text>"),
            ]
        self.end_of_turn_id = self.tokenizer.eos_token_id

    def calc_num_audio_tokens(self, num_samples: int) -> int:
        """Count the audio tokens of ``num_samples`` samples of audio: one per encoder frame."""
        num_mel_frames = self.log_mel_config.calc_num_frames(num_samples)
        return self.encoder_config.calc_num_encoder_frames(num_mel_frames)

    def build_target_ids(self, transcript: str) -> list[int]:
        """Build the token ids of the answer that transcribes as ``transcript``."""
        transcript_ids = self.tokenizer(
            transcript, add_special_tokens=False, split_special_tokens=True
        ).input_ids
        return [*self.answer_ids_before_transcript, *transcript_ids, self.end_of_turn_id]

    def build(self, waveform: np.ndarray, transcript: str) -> AudioTextExample:
        """Build the example of ``waveform``, sampled at ``log_mel_config.sampling_rate``."""
        prompt_ids = [
            *self.prompt_ids_before_audio,
            *[self.audio_token_id] * self.calc_num_audio_tokens(len(waveform)),
            *self.prompt_ids_after_audio,
        ]
        target_ids = self.build_target_ids(transcript)
        return AudioTextExample(
            waveform=torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
            input_ids=torch.tensor(prompt_ids + target_ids),
            loss_mask=torch.cat(
                [
                    torch.zeros(len(prompt_ids), dtype=torch.float32),
                    torch.ones(len(target_ids), dtype=torch.float32),
                ]
            ),
        )

    def calc_num_tokens(self, num_samples: int, transcript: str) -> int:
        """Count the tokens of the example built of ``num_samples`` samples and ``transcript``.

        Micro-batches are limited by this count, which needs only the number of samples.
        """
        return (
            len(self.prompt_ids_before_audio)
            + self.calc_num_audio_tokens(num_samples)
            + len(self.prompt_ids_after_audio)
            + len(self.build_target_ids(transcript))
        )

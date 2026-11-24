from collections.abc import Callable, Iterator
from dataclasses import fields

import torch

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderInputs
from megatron_speech.components.frontends.qwen3_asr import Qwen3ASRLogMel, Qwen3ASRLogMelConfig
from megatron_speech.data.packing import PackedAudioTextBatch
from megatron_speech.models.speech_llm import AUDIO_MODALITY, ENCODER_NAME, SpeechLLMConfig


class SpeechLLMForwardStep:
    """Forward step that Megatron's training loop runs on each micro-batch of the speech LLM.

    The log-mel of the micro-batch is computed on the GPU. The loss function returns the loss
    summed over the target tokens and their count. With ``calculate_per_token_loss``, Megatron
    divides the gradients by the count of all micro-batches and data parallel ranks, so that
    every target token of the global batch has the same weight.
    """

    def __init__(self, config: SpeechLLMConfig, log_mel_config: Qwen3ASRLogMelConfig) -> None:
        """Check that the forward step can run with ``config`` and build the log-mel."""
        if not config.llm.transformer.calculate_per_token_loss:
            raise ValueError(
                "the forward step of the speech LLM returns the loss summed over the target "
                "tokens; set llm.transformer.calculate_per_token_loss=True "
                "(--calculate-per-token-loss)"
            )
        if log_mel_config.feature_size != config.encoder.num_mel_bins:
            raise ValueError(
                f"log_mel_config.feature_size={log_mel_config.feature_size} differs from "
                f"encoder.num_mel_bins={config.encoder.num_mel_bins}"
            )
        self.encoder_config = config.encoder
        # The forward step is built before Megatron sets the GPU of the rank, so the log-mel
        # moves to the GPU in the first call.
        self.log_mel = Qwen3ASRLogMel(log_mel_config)

    def __call__(
        self, data_iterator: Iterator[PackedAudioTextBatch], model: torch.nn.Module
    ) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]],
    ]:
        """Run the next micro-batch of ``data_iterator`` through the speech LLM ``model``.

        Returns
        -------
        loss
            ``[1, num_tokens]`` loss at the target tokens, 0 elsewhere.
        loss_func
            ``sum_target_loss``, which Megatron calls on ``loss``.
        """

        def sum_target_loss(
            loss: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
            """Sum ``loss`` over the target tokens and count them, as Megatron's loss function.

            Returns
            -------
            summed_loss
                Sum of ``loss`` where ``loss_mask`` is 1.
            num_target_tokens
                Number of these tokens, an int tensor.
            report
                ``{"lm loss": [summed_loss, num_target_tokens]}``. Megatron's log sums it over the
                micro-batches and the data parallel ranks and divides the first by the second.
            """
            summed_loss = (loss.float() * loss_mask).sum()
            num_target_tokens = loss_mask.sum().to(torch.int)
            report = {"lm loss": torch.stack([summed_loss.detach(), num_target_tokens.float()])}
            return summed_loss, num_target_tokens, report

        batch = next(data_iterator)
        # Every micro-batch has audio, since each AudioTextExample holds a waveform.
        mels = self.log_mel.cuda()([waveform.cuda() for waveform in batch.waveforms])
        encoder_inputs = Qwen3ASREncoderInputs.build(mels, self.encoder_config).to("cuda")
        encoder_kwargs = {
            field.name: getattr(encoder_inputs, field.name) for field in fields(encoder_inputs)
        }
        loss, loss_mask = model(
            input_ids=batch.input_ids.cuda(),
            position_ids=batch.position_ids.cuda(),
            loss_mask=batch.loss_mask.cuda(),
            labels=batch.labels.cuda(),
            modality_inputs={AUDIO_MODALITY: {ENCODER_NAME: encoder_kwargs}},
            cu_seqlens=batch.cu_seqlens.cuda(),
            max_seqlen=batch.max_seqlen,
        )
        return loss, sum_target_loss

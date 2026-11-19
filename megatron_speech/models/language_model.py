from collections.abc import Callable

import torch
from megatron.core.extensions.transformer_engine import TELMHeadColumnParallelLinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer import TransformerConfig


class SpeechLanguageModel(GPTModel):
    """``GPTModel`` used as the LLM of the speech LLM."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        loss_mask: torch.Tensor | None = None,
        mtp_input_mask: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        """Run ``GPTModel.forward`` with the arguments that ``MimoModel`` passes.

        The return value depends on ``labels`` and ``loss_mask``:

        - both given: the loss ``[batch, sequence]`` from ``calc_target_loss``, computed only
          where ``loss_mask`` is set and 0 elsewhere.
        - ``labels`` only: ``GPTModel``'s loss ``[batch, sequence]`` at every position.
        - no ``labels``: ``GPTModel``'s logits ``[batch, sequence, vocabulary]``, split over the TP
          ranks along the vocabulary.
        """

        def calc_target_loss(
            *,
            hidden_states: torch.Tensor,
            output_layer: ColumnParallelLinear | TELMHeadColumnParallelLinear,
            output_weight: torch.Tensor | None,
            labels: torch.Tensor,
            loss_mask: torch.Tensor,
            input_ids: torch.Tensor | None,
            position_ids: torch.Tensor | None,
            attention_mask: torch.Tensor | None,
            decoder_input: torch.Tensor | None,
            inference_context: BaseInferenceContext | None,
            packed_seq_params: PackedSeqParams | None,
            runtime_gather_output: bool | None,
            context: None,
            compute_language_model_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            scale_logits: Callable[[torch.Tensor], torch.Tensor],
            config: TransformerConfig,
        ) -> torch.Tensor:
            """Calculate the loss ``[batch, sequence]`` at the positions where ``loss_mask`` is set.

            ``GPTModel`` calls this as its ``output_processor`` with every argument it has.
            ``hidden_states`` is ``[sequence, batch, hidden]``, split along the sequence over the TP
            ranks under sequence parallelism. The other positions get a loss of 0.
            """
            sequence_parallel = output_layer.sequence_parallel
            if sequence_parallel:
                # The output layer below runs without sequence parallelism and all-reduces the
                # gradient of its input, so the backward of this gather only splits it.
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False, group=output_layer.tp_group
                )
            # (sequence, batch) indices of the targets, in the layout of hidden_states. Indexing
            # with a boolean mask instead would run nonzero again for each tensor.
            target_indices = loss_mask.transpose(0, 1).nonzero(as_tuple=True)
            output_layer.sequence_parallel = False
            try:
                logits, _ = output_layer(
                    hidden_states[target_indices].unsqueeze(1),
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
            finally:
                output_layer.sequence_parallel = sequence_parallel
            target_labels = labels.transpose(0, 1)[target_indices].unsqueeze(0)
            target_loss = compute_language_model_loss(target_labels, scale_logits(logits))
            loss = target_loss.new_zeros(hidden_states.shape[:2])
            return loss.index_put(target_indices, target_loss[0]).transpose(0, 1).contiguous()

        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            mtp_input_mask=mtp_input_mask,
            output_processor=None if labels is None or loss_mask is None else calc_target_loss,
        )

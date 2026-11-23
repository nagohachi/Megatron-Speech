from collections.abc import Callable
from typing import TypedDict, Unpack

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


class OutputProcessorArguments(TypedDict):
    """Arguments besides ``hidden_states`` that ``GPTModel`` passes to its ``output_processor``."""

    output_layer: ColumnParallelLinear | TELMHeadColumnParallelLinear
    output_weight: torch.Tensor | None
    labels: torch.Tensor | None
    loss_mask: torch.Tensor | None
    input_ids: torch.Tensor | None
    position_ids: torch.Tensor | None
    attention_mask: torch.Tensor | None
    decoder_input: torch.Tensor | None
    inference_context: BaseInferenceContext | None
    packed_seq_params: PackedSeqParams | None
    runtime_gather_output: bool | None
    context: None
    compute_language_model_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    scale_logits: Callable[[torch.Tensor], torch.Tensor]
    config: TransformerConfig


class TargetLossLanguageModel(GPTModel):
    """``GPTModel`` that computes the loss only at the target positions of the last layer."""

    def gather_last_layer_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather the ``[sequence, batch, hidden]`` output that sequence parallelism splits.

        Without sequence parallelism, return ``hidden_states`` as it is. The backward keeps this
        rank's part of the gradient without summing it over the TP ranks: the computation after
        the gather is the same on every rank, and its tensor parallel layers, such as the output
        layer without sequence parallelism, all-reduce the gradient of their input.
        """
        if not self.config.sequence_parallel:
            return hidden_states
        return gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=self.pg_collection.tp
        )

    def calc_target_loss(
        self, hidden_states: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss ``[batch, sequence]`` at the positions where ``loss_mask`` is not 0.

        ``hidden_states`` is the whole ``[sequence, batch, hidden]`` output of the last layer, as
        ``gather_last_layer_output`` returns it. The output layer runs only at these positions;
        the other positions get a loss of 0.
        """
        # (sequence, batch) indices of the targets, in the layout of hidden_states. Indexing
        # with a boolean mask instead would run nonzero again for each tensor.
        target_indices = loss_mask.transpose(0, 1).nonzero(as_tuple=True)
        output_weight = (
            self.shared_embedding_or_output_weight()
            if self.share_embeddings_and_output_weights
            else None
        )
        # The hidden states are gathered, so the output layer runs without sequence parallelism
        # and all-reduces the gradient of its input.
        sequence_parallel = self.output_layer.sequence_parallel
        self.output_layer.sequence_parallel = False
        try:
            logits, _ = self.output_layer(
                hidden_states[target_indices].unsqueeze(1), weight=output_weight
            )
        finally:
            self.output_layer.sequence_parallel = sequence_parallel
        target_labels = labels.transpose(0, 1)[target_indices].unsqueeze(0)
        # GPTModel passes _scale_logits to its output processor as scale_logits; a caller that
        # runs after forward, such as the model around MultiStreamLanguageModel, has no such
        # argument.
        target_loss = self.compute_language_model_loss(target_labels, self._scale_logits(logits))
        loss = target_loss.new_zeros(hidden_states.shape[:2])
        return loss.index_put(target_indices, target_loss[0]).transpose(0, 1).contiguous()


class SpeechLanguageModel(TargetLossLanguageModel):
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

        def calc_target_loss_from_last_layer(
            *, hidden_states: torch.Tensor, **_: Unpack[OutputProcessorArguments]
        ) -> torch.Tensor:
            """Gather the output of the last layer and pass it to ``calc_target_loss``.

            ``GPTModel`` calls this as its ``output_processor``.
            """
            return self.calc_target_loss(
                self.gather_last_layer_output(hidden_states), labels, loss_mask
            )

        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            mtp_input_mask=mtp_input_mask,
            output_processor=(
                None if labels is None or loss_mask is None else calc_target_loss_from_last_layer
            ),
        )


class MultiStreamLanguageModel(TargetLossLanguageModel):
    """LLM of a model that predicts the text and the audio from the output of the last layer.

    ``forward`` returns this output; the caller computes the text loss with
    ``calc_target_loss`` and the audio loss with its own decoder.
    """

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
        """Return the output of the last layer, ``[sequence, batch, hidden]``, after its norm.

        The output is whole on every TP rank, also under sequence parallelism. The arguments are
        those that ``MimoModel`` passes; ``labels`` and ``loss_mask`` must be None.
        """
        if labels is not None or loss_mask is not None:
            raise ValueError(
                "MultiStreamLanguageModel returns the output of the last layer; compute the "
                "text loss from it with calc_target_loss instead of passing labels or loss_mask"
            )

        def return_last_layer_output(
            *, hidden_states: torch.Tensor, **_: Unpack[OutputProcessorArguments]
        ) -> torch.Tensor:
            """Return the gathered last layer output; ``GPTModel`` calls this as its processor."""
            return self.gather_last_layer_output(hidden_states)

        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            packed_seq_params=packed_seq_params,
            mtp_input_mask=mtp_input_mask,
            output_processor=return_last_layer_output,
        )

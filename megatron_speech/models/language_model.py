"""LLM backbone of the speech LLM.

``SpeechLanguageModel`` is Megatron Core's ``GPTModel`` with one change: when labels and a
loss mask are given, the output layer and the cross entropy run only on positions with
``loss_mask == 1`` (``calc_target_only_loss``), through GPTModel's ``output_processor`` hook.

In speech recognition most positions are audio placeholders and prompt tokens; computing
``[tokens, vocab]`` logits for all of them dominates memory (8192 tokens x 151,936 vocabulary
x 4 bytes = 5.0 GB in fp32 for Qwen3). The returned loss tensor keeps the ``[batch, seq]``
shape, with zeros where ``loss_mask == 0``, so the loss function is unchanged.
"""

from __future__ import annotations

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region


def calc_target_only_loss(
    *,
    hidden_states: torch.Tensor,
    output_layer,
    output_weight,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    runtime_gather_output,
    compute_language_model_loss,
    scale_logits,
    **_,
) -> torch.Tensor:
    """``output_processor`` for ``GPTModel.forward``: per-token loss ``[b, s]`` computed only
    where ``loss_mask`` is set. ``hidden_states`` is ``[s, b, h]`` (sequence-parallel shards
    are gathered first)."""
    sequence_parallel = getattr(output_layer, "sequence_parallel", False)
    if sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, group=output_layer.tp_group)
    s, b, h = hidden_states.shape
    index = loss_mask.transpose(0, 1).reshape(-1).nonzero(as_tuple=True)[0]  # flat [s*b] positions
    if index.numel() == 0:
        return hidden_states.sum() * 0.0 + torch.zeros(b, s, device=hidden_states.device, dtype=torch.float32)
    selected = hidden_states.reshape(s * b, h).index_select(0, index).unsqueeze(1)  # [n, 1, h]
    try:
        if sequence_parallel:
            output_layer.sequence_parallel = False
        logits, _ = output_layer(selected, weight=output_weight, runtime_gather_output=runtime_gather_output)
    finally:
        if sequence_parallel:
            output_layer.sequence_parallel = True
    logits = scale_logits(logits)
    selected_labels = labels.transpose(0, 1).reshape(-1).index_select(0, index).unsqueeze(0)  # [1, n]
    selected_loss = compute_language_model_loss(selected_labels, logits).reshape(-1)
    loss = torch.zeros(s * b, device=selected_loss.device, dtype=selected_loss.dtype)
    loss = loss.index_put((index,), selected_loss)
    return loss.view(s, b).transpose(0, 1).contiguous()


class SpeechLanguageModel(GPTModel):
    def __init__(self, *args, loss_on_targets_only: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_on_targets_only = loss_on_targets_only

    def forward(self, input_ids, position_ids, attention_mask, decoder_input=None, labels=None, *args,
                loss_mask=None, output_processor=None, **kwargs):
        if (
            output_processor is None
            and self.loss_on_targets_only
            and labels is not None
            and loss_mask is not None
            and self.post_process
        ):
            output_processor = calc_target_only_loss
        return super().forward(
            input_ids, position_ids, attention_mask, decoder_input, labels, *args,
            loss_mask=loss_mask, output_processor=output_processor, **kwargs,
        )

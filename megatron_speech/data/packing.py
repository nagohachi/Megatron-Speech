import math
from dataclasses import dataclass
from itertools import accumulate
from typing import Self

import torch


@dataclass
class AudioTextExample:
    """One utterance as the input of the speech LLM.

    - ``waveform``: ``[num_samples]`` float32 audio at the log-mel's sampling rate.
      The log-mel is computed from it on the GPU, for a whole micro-batch at once.
    - ``input_ids``: ``[num_tokens]`` token ids, with one audio token per encoder frame.
    - ``loss_mask``: ``[num_tokens]`` float32, 1 on the tokens that the model learns to predict
      and 0 elsewhere.
    """

    waveform: torch.Tensor
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


def calc_segment_length(num_tokens: int, pad_to_multiple: int) -> int:
    """Return the tokens that an example of ``num_tokens`` tokens takes in a packed row."""
    return math.ceil(num_tokens / pad_to_multiple) * pad_to_multiple


@dataclass
class PackedAudioTextBatch:
    """Micro-batch whose utterances are packed into one row, in Megatron's THD format.

    Each utterance is given one segment of the row: its tokens, then padding. Attention is
    computed within each segment, and the padding comes after the tokens, so the outputs at the
    tokens do not depend on it.

    - ``waveforms``: ``[num_samples]`` float32 audio of each utterance, in order.
    - ``input_ids``: ``[1, num_tokens]`` token ids of the row.
    - ``labels``: ``[1, num_tokens]`` the next token of the same utterance at each position.
    - ``loss_mask``: ``[1, num_tokens]`` float32, 1 where the label is a token that the model
      learns to predict and 0 elsewhere, also at the last token of each utterance and on padding.
    - ``position_ids``: ``[1, num_tokens]`` positions that start at 0 in each segment.
    - ``cu_seqlens``: ``[num_utterances + 1]`` int32 cumulative segment lengths.
    - ``max_seqlen``: 0-dim tensor, the length of the longest segment.
    """

    waveforms: list[torch.Tensor]
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor

    @classmethod
    def build(
        cls, examples: list[AudioTextExample], pad_to_multiple: int, pad_token_id: int
    ) -> Self:
        """Pack ``examples`` into one row, with each padded to a multiple of ``pad_to_multiple``.

        With context parallelism, each segment is split into twice as many parts as there are
        context parallel ranks; with sequence parallelism, the row is split over the tensor
        parallel ranks. ``pad_to_multiple`` is the product of these numbers of parts for the
        parallelisms in use, and 1 for neither. Padding is given ``pad_token_id``.
        """
        if not examples:
            raise ValueError("examples is empty; a micro-batch needs at least one utterance")
        if pad_to_multiple < 1:
            raise ValueError(f"pad_to_multiple must be at least 1, got {pad_to_multiple}")
        input_ids, labels, loss_mask, segment_lengths = [], [], [], []
        for example in examples:
            num_tokens = len(example.input_ids)
            segment_length = calc_segment_length(num_tokens, pad_to_multiple)
            num_padding_tokens = segment_length - num_tokens
            input_ids += [example.input_ids, torch.full((num_padding_tokens,), pad_token_id)]
            # The last token of the utterance has no next token to predict.
            labels += [example.input_ids[1:], torch.full((num_padding_tokens + 1,), pad_token_id)]
            loss_mask += [example.loss_mask[1:], torch.zeros(num_padding_tokens + 1)]
            segment_lengths.append(segment_length)
        return cls(
            waveforms=[example.waveform for example in examples],
            input_ids=torch.cat(input_ids)[None],
            labels=torch.cat(labels)[None],
            loss_mask=torch.cat(loss_mask)[None],
            position_ids=torch.cat([torch.arange(length) for length in segment_lengths])[None],
            cu_seqlens=torch.tensor([0, *accumulate(segment_lengths)], dtype=torch.int32),
            max_seqlen=torch.tensor(max(segment_lengths)),
        )

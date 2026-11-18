"""Pack several examples into one THD row (no padding between examples)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from megatron_speech.data.features import AudioFeatures, collate_encoder_inputs
from megatron_speech.data.prompt import FormattedExample


@dataclass
class SpeechExample:
    features: AudioFeatures
    formatted: FormattedExample


def pack_examples(examples: list[SpeechExample], tokens_per_chunk: int, pad_to_multiple: int = 1, pad_token_id: int = 0) -> dict:
    """Return a packed micro-batch.

    LLM fields have shape ``[1, total_tokens]``. ``labels[t]`` is the next token within the same
    example and ``loss_mask[t]`` is 1 where that next token is a target. Each example is padded
    to a multiple of ``pad_to_multiple`` tokens (context parallelism needs ``2 * cp``);
    ``cu_seqlens`` counts the padded lengths.
    """
    input_ids, labels, loss_mask, position_ids, lengths = [], [], [], [], []
    for ex in examples:
        ids = ex.formatted.input_ids
        target = ex.formatted.target_mask
        n = len(ids)
        padded = -(-n // pad_to_multiple) * pad_to_multiple
        input_ids += ids + [pad_token_id] * (padded - n)
        labels += ids[1:] + [pad_token_id] * (padded - n + 1)
        loss_mask += [float(t) for t in target[1:]] + [0.0] * (padded - n + 1)
        position_ids += list(range(padded))
        lengths.append(padded)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.float32),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": torch.tensor(max(lengths), dtype=torch.int32),
        "num_examples": torch.tensor(len(examples)),
        "audio": collate_encoder_inputs([ex.features for ex in examples], tokens_per_chunk),
    }
    return batch

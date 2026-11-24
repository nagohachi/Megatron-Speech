from itertools import pairwise

import pytest
import torch

from megatron_speech.data.packing import PackedAudioTextBatch
from megatron_speech.data.qwen3_asr import AudioTextExample

PAD_TO_MULTIPLE = 4
PAD_TOKEN_ID = 0
# Token counts of the utterances: 8 fills its segment, the others are padded.
UTTERANCE_TOKEN_COUNTS = [5, 8, 13]
# Utterances whose second half is the target. The token ids differ between the utterances and
# from PAD_TOKEN_ID.
EXAMPLES = [
    AudioTextExample(
        waveform=torch.zeros(1000),
        input_ids=torch.arange(num_tokens) + 100 * (index + 1),
        loss_mask=(torch.arange(num_tokens) >= num_tokens // 2).float(),
    )
    for index, num_tokens in enumerate(UTTERANCE_TOKEN_COUNTS)
]


@pytest.fixture(name="batch")
def build_batch() -> PackedAudioTextBatch:
    """Pack ``EXAMPLES`` into one row."""
    return PackedAudioTextBatch.build(EXAMPLES, PAD_TO_MULTIPLE, PAD_TOKEN_ID)


def test_segments_round_up_to_multiple(batch: PackedAudioTextBatch) -> None:
    """Check that each segment is its utterance's length rounded up to ``PAD_TO_MULTIPLE``."""
    # 5, 8 and 13 tokens take 8, 8 and 16.
    assert batch.cu_seqlens.tolist() == [0, 8, 16, 32]
    assert batch.cu_seqlens.dtype == torch.int32
    assert batch.max_seqlen.item() == 16


def test_token_tensors_cover_row(batch: PackedAudioTextBatch) -> None:
    """Check that the token ids, labels, loss mask and positions have one value per token."""
    row_shape = (1, batch.cu_seqlens[-1].item())
    for tensor in [batch.input_ids, batch.labels, batch.loss_mask, batch.position_ids]:
        assert tensor.shape == row_shape


def test_input_ids_hold_utterances_then_padding(batch: PackedAudioTextBatch) -> None:
    """Check that each segment holds the tokens of its utterance, then ``PAD_TOKEN_ID``."""
    segments = pairwise(batch.cu_seqlens.tolist())
    for example, (start, end) in zip(EXAMPLES, segments, strict=True):
        num_tokens = len(example.input_ids)
        assert torch.equal(batch.input_ids[0, start : start + num_tokens], example.input_ids)
        assert (batch.input_ids[0, start + num_tokens : end] == PAD_TOKEN_ID).all()


def test_labels_are_next_tokens_of_same_utterance(batch: PackedAudioTextBatch) -> None:
    """Check that the label at each token of an utterance but its last is the next token."""
    for example, start in zip(EXAMPLES, batch.cu_seqlens[:-1].tolist(), strict=True):
        num_tokens = len(example.input_ids)
        assert torch.equal(batch.labels[0, start : start + num_tokens - 1], example.input_ids[1:])


def test_loss_mask_marks_next_targets(batch: PackedAudioTextBatch) -> None:
    """Check that the loss mask at each token is that of the next token of the utterance.

    From the last token of the utterance to the end of its padding, the loss mask is 0.
    """
    segments = pairwise(batch.cu_seqlens.tolist())
    for example, (start, end) in zip(EXAMPLES, segments, strict=True):
        num_tokens = len(example.input_ids)
        assert torch.equal(
            batch.loss_mask[0, start : start + num_tokens - 1], example.loss_mask[1:]
        )
        assert not batch.loss_mask[0, start + num_tokens - 1 : end].any()


def test_positions_restart_in_each_segment(batch: PackedAudioTextBatch) -> None:
    """Check that the positions count from 0 in each segment."""
    for start, end in pairwise(batch.cu_seqlens.tolist()):
        assert torch.equal(batch.position_ids[0, start:end], torch.arange(end - start))


def test_build_without_padding_joins_utterances() -> None:
    """Check that with ``pad_to_multiple`` 1, the row is the tokens of the utterances, joined."""
    batch = PackedAudioTextBatch.build(EXAMPLES, 1, PAD_TOKEN_ID)
    assert batch.cu_seqlens.tolist() == [0, 5, 13, 26]
    assert torch.equal(batch.input_ids[0], torch.cat([example.input_ids for example in EXAMPLES]))


def test_build_rejects_no_examples() -> None:
    """Check that an empty list of examples is rejected with the name ``examples``."""
    with pytest.raises(ValueError, match="examples"):
        PackedAudioTextBatch.build([], PAD_TO_MULTIPLE, PAD_TOKEN_ID)


def test_build_rejects_pad_to_multiple_below_one() -> None:
    """Check that a ``pad_to_multiple`` below 1 is rejected with its name."""
    with pytest.raises(ValueError, match="pad_to_multiple"):
        PackedAudioTextBatch.build(EXAMPLES, 0, PAD_TOKEN_ID)

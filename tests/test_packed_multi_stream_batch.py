from dataclasses import replace

import pytest
import torch

from megatron_speech.data.multi_stream import MultiStreamExample, PackedMultiStreamBatch

# The first window has 3 positions and is padded to 4, and the second has 2 and is not padded,
# so the row has positions 0 to 3 of the first window and 4 and 5 of the second.
PAD_TO_MULTIPLE = 2


@pytest.fixture(name="examples")
def build_examples() -> list[MultiStreamExample]:
    """Build the examples of two windows with the text and one codebook of the audio decoder.

    The codebook is absent at the first position of the first window, and at the second
    position of the second window. The audio decoder predicts the codebook at positions 0 and 2
    of the first window, where position 0 has no target, and at position 1 of the second.
    """
    first_example = MultiStreamExample(
        input_ids=torch.tensor([10, 11, 12]),
        labels=torch.tensor([11, 12, 13]),
        loss_mask=torch.tensor([1, 0.5, 1]),
        modality_positions={"text": torch.tensor([0, 1, 2]), "codebook": torch.tensor([1, 2])},
        stream_inputs={"codebook": torch.tensor([20, 21])},
        audio_decoder_positions=torch.tensor([0, 2]),
        audio_targets={"codebook": torch.tensor([20, 22])},
        audio_target_masks={"codebook": torch.tensor([False, True])},
    )
    second_example = MultiStreamExample(
        input_ids=torch.tensor([30, 31]),
        labels=torch.tensor([31, 32]),
        loss_mask=torch.tensor([1.0, 1.0]),
        modality_positions={"text": torch.tensor([0, 1]), "codebook": torch.tensor([0])},
        stream_inputs={"codebook": torch.tensor([40])},
        audio_decoder_positions=torch.tensor([1]),
        audio_targets={"codebook": torch.tensor([41])},
        audio_target_masks={"codebook": torch.tensor([True])},
    )
    return [first_example, second_example]


@pytest.fixture(name="batch")
def build_batch(examples: list[MultiStreamExample]) -> PackedMultiStreamBatch:
    """Pack ``examples`` with each window padded to a multiple of ``PAD_TO_MULTIPLE``."""
    return PackedMultiStreamBatch.build(examples, PAD_TO_MULTIPLE)


def test_packs_text_into_one_row(batch: PackedMultiStreamBatch) -> None:
    """Check that the text of the windows is in one row, with 0 and no loss at the padding."""
    assert batch.input_ids.tolist() == [[10, 11, 12, 0, 30, 31]]
    assert batch.labels.tolist() == [[11, 12, 13, 0, 31, 32]]
    assert batch.loss_mask.tolist() == [[1, 0.5, 1, 0, 1, 1]]


def test_starts_positions_of_each_segment_at_zero(batch: PackedMultiStreamBatch) -> None:
    """Check the positions of the segments and their cumulative lengths."""
    assert batch.position_ids.tolist() == [[0, 1, 2, 3, 0, 1]]
    assert batch.cu_seqlens.dtype == torch.int32
    assert batch.cu_seqlens.tolist() == [0, 4, 6]
    assert batch.max_seqlen.item() == 4


def test_moves_stream_positions_to_segments(batch: PackedMultiStreamBatch) -> None:
    """Check that the positions of each stream move to the segment of their window.

    The padding at position 3 holds no stream.
    """
    assert batch.modality_token_indices["text"].tolist() == [0, 1, 2, 4, 5]
    assert batch.modality_token_indices["codebook"].tolist() == [1, 2, 4]
    assert batch.stream_inputs.keys() == {"codebook"}
    assert batch.stream_inputs["codebook"].tolist() == [20, 21, 40]


def test_moves_audio_decoder_positions_to_segments(batch: PackedMultiStreamBatch) -> None:
    """Check that the positions of the audio decoder move to the segment of their window."""
    assert batch.audio_decoder_token_indices.tolist() == [0, 2, 5]
    assert batch.audio_targets["codebook"].tolist() == [20, 22, 41]
    assert batch.audio_target_masks["codebook"].tolist() == [False, True, True]


def test_pads_nothing_to_multiple_of_one(examples: list[MultiStreamExample]) -> None:
    """Check that ``pad_to_multiple=1`` puts the windows next to each other without padding."""
    batch = PackedMultiStreamBatch.build(examples, pad_to_multiple=1)
    assert batch.input_ids.tolist() == [[10, 11, 12, 30, 31]]
    assert batch.cu_seqlens.tolist() == [0, 3, 5]
    assert batch.modality_token_indices["codebook"].tolist() == [1, 2, 3]
    assert batch.audio_decoder_token_indices.tolist() == [0, 2, 4]


def test_packs_window_without_audio(examples: list[MultiStreamExample]) -> None:
    """Check that a window without the codebook and without audio targets packs with the others.

    The window holds only a prompt of text, as ``MultiStreamExampleBuilder`` builds it.
    """
    no_positions = torch.tensor([], dtype=torch.int64)
    prompt_example = replace(
        examples[1],
        loss_mask=torch.tensor([0.0, 0.0]),
        modality_positions={"text": torch.tensor([0, 1]), "codebook": no_positions},
        stream_inputs={"codebook": no_positions},
        audio_decoder_positions=no_positions,
        audio_targets={"codebook": no_positions},
        audio_target_masks={"codebook": torch.tensor([], dtype=torch.bool)},
    )
    batch = PackedMultiStreamBatch.build([examples[0], prompt_example], PAD_TO_MULTIPLE)
    assert batch.loss_mask.tolist() == [[1, 0.5, 1, 0, 0, 0]]
    assert batch.modality_token_indices["codebook"].tolist() == [1, 2]
    assert batch.audio_decoder_token_indices.tolist() == [0, 2]
    assert batch.audio_target_masks["codebook"].tolist() == [False, True]


def test_rejects_no_examples() -> None:
    """Check that a micro-batch without windows raises ValueError."""
    with pytest.raises(ValueError, match="needs at least one window"):
        PackedMultiStreamBatch.build([], PAD_TO_MULTIPLE)


def test_rejects_pad_to_multiple_below_one(examples: list[MultiStreamExample]) -> None:
    """Check that ``pad_to_multiple=0`` raises ValueError."""
    with pytest.raises(ValueError, match="pad_to_multiple must be at least 1, got 0"):
        PackedMultiStreamBatch.build(examples, pad_to_multiple=0)


def test_rejects_windows_of_different_streams(examples: list[MultiStreamExample]) -> None:
    """Check that windows with different streams raise ValueError that names the window."""
    example_without_codebook = replace(
        examples[1], modality_positions={"text": torch.tensor([0, 1])}, stream_inputs={}
    )
    with pytest.raises(ValueError, match=r"window 1 has the streams \['text'\]"):
        PackedMultiStreamBatch.build([examples[0], example_without_codebook], PAD_TO_MULTIPLE)

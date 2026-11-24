import math

import numpy as np
import pytest
import torch

from megatron_speech.data.aligned_streams import AlignedStreams, FrameStream, WindowSplitter

NUM_FRAMES = 10
NUM_CODEBOOKS = 8
LATENT_SIZE = 4
# Runs a test with windows cut every max_num_frames frames and with windows of even lengths.
WITH_BOTH_SPLITS = pytest.mark.parametrize("split_evenly", [False, True])


@pytest.fixture(name="streams")
def build_aligned_streams() -> AlignedStreams:
    """Build text, codec codes and codec latents of 10 frames.

    The audio is absent at frames 0 and 1, as at a prompt that holds only text.
    """
    torch.manual_seed(0)
    audio_frame_mask = torch.arange(NUM_FRAMES) >= 2
    return AlignedStreams(
        {
            "text": FrameStream(torch.arange(NUM_FRAMES), torch.ones(NUM_FRAMES, dtype=torch.bool)),
            "codes": FrameStream(
                torch.arange(NUM_FRAMES * NUM_CODEBOOKS).view(NUM_FRAMES, NUM_CODEBOOKS),
                audio_frame_mask,
            ),
            "latents": FrameStream(torch.randn(NUM_FRAMES, LATENT_SIZE), audio_frame_mask),
        }
    )


def test_counts_frames_of_streams(streams: AlignedStreams) -> None:
    """Check that ``num_frames`` is the number of frames of every stream."""
    assert streams.num_frames == NUM_FRAMES


def test_takes_frames_of_every_stream(streams: AlignedStreams) -> None:
    """Check that a window holds the values and the masks of every stream at its frames."""
    window = streams.take_frames(range(1, 5))
    assert window.num_frames == 4
    assert window.streams.keys() == streams.streams.keys()
    for name, stream in window.streams.items():
        assert torch.equal(stream.values, streams.streams[name].values[1:5]), name
        mask = streams.streams[name].present_frame_mask[1:5]
        assert torch.equal(stream.present_frame_mask, mask), name


def test_rejects_frames_with_a_step(streams: AlignedStreams) -> None:
    """Check that a range of frames with a step other than 1 raises ValueError."""
    with pytest.raises(ValueError, match="consecutive frames, got a step of 2"):
        streams.take_frames(range(0, NUM_FRAMES, 2))


# Frames past the end of the recording, and frames from before its start.
@pytest.mark.parametrize("frames", [range(5, 20), range(-3, 5)], ids=["past-end", "before-start"])
def test_rejects_frames_outside_recording(streams: AlignedStreams, frames: range) -> None:
    """Check that frames outside the recording raise ValueError instead of a shorter window."""
    with pytest.raises(ValueError, match=f"frames from 0 to {NUM_FRAMES}"):
        streams.take_frames(frames)


def test_rejects_streams_of_different_lengths() -> None:
    """Check that streams with different numbers of frames raise ValueError."""
    with pytest.raises(ValueError, match="different numbers of frames"):
        AlignedStreams(
            {
                "text": FrameStream(torch.zeros(10), torch.ones(10, dtype=torch.bool)),
                "codes": FrameStream(torch.zeros(9, 8), torch.ones(9, dtype=torch.bool)),
            }
        )


def test_rejects_mask_of_another_length() -> None:
    """Check that a mask with another number of frames than the values raises ValueError."""
    with pytest.raises(ValueError, match="frames of values but a mask"):
        AlignedStreams({"text": FrameStream(torch.zeros(10), torch.ones(9, dtype=torch.bool))})


def test_rejects_no_streams() -> None:
    """Check that a recording without streams raises ValueError."""
    with pytest.raises(ValueError, match="at least one stream"):
        AlignedStreams({})


def test_cuts_windows_every_max_num_frames() -> None:
    """Check that windows start every ``max_num_frames`` frames and only the last is shorter."""
    splitter = WindowSplitter(max_num_frames=4, split_evenly=False, min_num_frames=0)
    assert splitter.split(10) == [range(0, 4), range(4, 8), range(8, 10)]


def test_splits_windows_evenly() -> None:
    """Check that even windows are as few as fit and differ in length by at most one frame."""
    splitter = WindowSplitter(max_num_frames=4, split_evenly=True, min_num_frames=0)
    assert splitter.split(10) == [range(0, 4), range(4, 7), range(7, 10)]


def test_even_windows_match_numpy_array_split() -> None:
    """Check that even windows have the lengths of ``numpy.array_split``, as in J-Moshi."""
    max_num_frames = 7
    splitter = WindowSplitter(max_num_frames=max_num_frames, split_evenly=True, min_num_frames=0)
    for num_frames in range(1, 50):
        num_windows = math.ceil(num_frames / max_num_frames)
        lengths = [len(part) for part in np.array_split(np.arange(num_frames), num_windows)]
        windows = splitter.split(num_frames)
        assert [len(window) for window in windows] == lengths, num_frames
        assert windows[0].start == 0 and windows[-1].stop == num_frames, num_frames


@WITH_BOTH_SPLITS
def test_keeps_recording_shorter_than_window(split_evenly: bool) -> None:
    """Check that a recording shorter than ``max_num_frames`` is one window."""
    splitter = WindowSplitter(max_num_frames=4, split_evenly=split_evenly, min_num_frames=0)
    assert splitter.split(3) == [range(0, 3)]


@WITH_BOTH_SPLITS
def test_splits_no_frames_into_no_windows(split_evenly: bool) -> None:
    """Check that a recording without frames gives no windows."""
    splitter = WindowSplitter(max_num_frames=4, split_evenly=split_evenly, min_num_frames=0)
    assert splitter.split(0) == []


def test_drops_windows_shorter_than_min_num_frames() -> None:
    """Check that windows shorter than ``min_num_frames`` are dropped."""
    splitter = WindowSplitter(max_num_frames=4, split_evenly=False, min_num_frames=3)
    assert splitter.split(10) == [range(0, 4), range(4, 8)]


def test_rejects_windows_without_frames() -> None:
    """Check that ``max_num_frames=0`` raises ValueError when the splitter is made."""
    with pytest.raises(ValueError, match="max_num_frames must be at least 1"):
        WindowSplitter(max_num_frames=0, split_evenly=False, min_num_frames=0)


def test_rejects_min_num_frames_above_max_num_frames() -> None:
    """Check that ``min_num_frames`` above ``max_num_frames`` raises ValueError at construction."""
    with pytest.raises(ValueError, match="min_num_frames must be from 0 to max_num_frames"):
        WindowSplitter(max_num_frames=4, split_evenly=False, min_num_frames=5)

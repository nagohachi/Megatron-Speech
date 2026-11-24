import math

import numpy as np
import pytest
import torch

from megatron_speech.data.aligned_streams import (
    AlignedStreams,
    FrameStream,
    StreamDelay,
    StreamDelayer,
    WindowSplitter,
)

NUM_FRAMES = 10
NUM_CODEBOOKS = 8
LATENT_SIZE = 4
# Runs a test with windows cut every max_frames frames and with windows of even lengths.
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


def test_rejects_mask_that_is_not_bool() -> None:
    """Check that a mask of ints raises ValueError instead of acting as a mask later."""
    with pytest.raises(ValueError, match=r"stream text has a mask of torch\.int64, not bool"):
        AlignedStreams({"text": FrameStream(torch.zeros(10), torch.ones(10, dtype=torch.int64))})


def test_rejects_values_without_frames() -> None:
    """Check that 0-dim values raise ValueError that names the stream."""
    with pytest.raises(ValueError, match="stream text has values without a frame dimension"):
        AlignedStreams({"text": FrameStream(torch.tensor(1), torch.ones(1, dtype=torch.bool))})


def test_rejects_no_streams() -> None:
    """Check that a recording without streams raises ValueError."""
    with pytest.raises(ValueError, match="at least one stream"):
        AlignedStreams({})


def test_cuts_windows_every_max_frames() -> None:
    """Check that windows start every ``max_frames`` frames and only the last is shorter."""
    splitter = WindowSplitter(max_frames=4, split_evenly=False, min_frames=0)
    assert splitter.split(10) == [range(0, 4), range(4, 8), range(8, 10)]


def test_splits_windows_evenly() -> None:
    """Check that even windows are as few as fit and differ in length by at most one frame."""
    splitter = WindowSplitter(max_frames=4, split_evenly=True, min_frames=0)
    assert splitter.split(10) == [range(0, 4), range(4, 7), range(7, 10)]


def test_even_windows_match_numpy_array_split() -> None:
    """Check that even windows have the lengths of ``numpy.array_split``, as in J-Moshi."""
    max_frames = 7
    splitter = WindowSplitter(max_frames=max_frames, split_evenly=True, min_frames=0)
    for num_frames in range(1, 50):
        num_windows = math.ceil(num_frames / max_frames)
        lengths = [len(part) for part in np.array_split(np.arange(num_frames), num_windows)]
        windows = splitter.split(num_frames)
        assert [len(window) for window in windows] == lengths, num_frames
        assert windows[0].start == 0 and windows[-1].stop == num_frames, num_frames


@WITH_BOTH_SPLITS
def test_keeps_recording_shorter_than_window(split_evenly: bool) -> None:
    """Check that a recording shorter than ``max_frames`` is one window."""
    splitter = WindowSplitter(max_frames=4, split_evenly=split_evenly, min_frames=0)
    assert splitter.split(3) == [range(0, 3)]


@WITH_BOTH_SPLITS
def test_splits_no_frames_into_no_windows(split_evenly: bool) -> None:
    """Check that a recording without frames gives no windows."""
    splitter = WindowSplitter(max_frames=4, split_evenly=split_evenly, min_frames=0)
    assert splitter.split(0) == []


def test_drops_windows_shorter_than_min_frames() -> None:
    """Check that windows shorter than ``min_frames`` are dropped."""
    splitter = WindowSplitter(max_frames=4, split_evenly=False, min_frames=3)
    assert splitter.split(10) == [range(0, 4), range(4, 8)]


def test_rejects_windows_without_frames() -> None:
    """Check that ``max_frames=0`` raises ValueError when the splitter is made."""
    with pytest.raises(ValueError, match="max_frames must be at least 1"):
        WindowSplitter(max_frames=0, split_evenly=False, min_frames=0)


def test_rejects_min_frames_above_max_frames() -> None:
    """Check that ``min_frames`` above ``max_frames`` raises ValueError at construction."""
    with pytest.raises(ValueError, match="min_frames must be from 0 to max_frames"):
        WindowSplitter(max_frames=4, split_evenly=False, min_frames=5)


# Streams of a recording as Moshi reads it, with Moshi's initial and padding tokens: the text
# and the first codebook are not delayed, an acoustic codebook is delayed by one frame, as in
# Moshi, and by two frames, to test longer delays. A stream of codec latents is delayed by one.
STREAM_DELAYS = {
    "text": StreamDelay(num_delay_frames=0, initial_value=32000, padding_value=3),
    "semantic": StreamDelay(num_delay_frames=0, initial_value=2048, padding_value=2048),
    "acoustic": StreamDelay(num_delay_frames=1, initial_value=2048, padding_value=2048),
    "late_acoustic": StreamDelay(num_delay_frames=2, initial_value=2048, padding_value=2048),
    "latents": StreamDelay(num_delay_frames=1, initial_value=0.0, padding_value=0.0),
}
NUM_DELAYED_FRAMES = 6
MAX_DELAY = 2


@pytest.fixture(name="undelayed_streams")
def build_undelayed_streams() -> AlignedStreams:
    """Build the streams of ``STREAM_DELAYS`` with 6 frames.

    The latents are absent at frames 0 and 1, as at a prompt that holds only text.
    """
    torch.manual_seed(1)
    all_frames = torch.ones(NUM_DELAYED_FRAMES, dtype=torch.bool)
    streams = {
        name: FrameStream(torch.randint(2048, (NUM_DELAYED_FRAMES,)), all_frames)
        for name in ["text", "semantic", "acoustic", "late_acoustic"]
    }
    streams["latents"] = FrameStream(
        torch.randn(NUM_DELAYED_FRAMES, LATENT_SIZE), torch.arange(NUM_DELAYED_FRAMES) >= 2
    )
    return AlignedStreams(streams)


def delay_as_kyutai(values: torch.Tensor, stream_delay: StreamDelay) -> torch.Tensor:
    """Delay one stream as Kyutai's ``_delay_sequence``: roll it and fill its first frames."""
    delayed_values = values.roll(stream_delay.num_delay_frames, dims=0)
    delayed_values[: stream_delay.num_delay_frames] = stream_delay.initial_value
    return delayed_values


def delay_as_jmoshi(values: torch.Tensor, stream_delay: StreamDelay) -> torch.Tensor:
    """Delay one stream as J-Moshi's ``delay_and_pad_streams``, by up to ``MAX_DELAY`` frames.

    J-Moshi also puts a frame of initial tokens in front of every stream, which is the shift of
    the input by one frame and not a part of the delay.
    """
    num_frames = values.size(0)
    num_delay_frames = stream_delay.num_delay_frames
    delayed_values = torch.empty((num_frames + MAX_DELAY, *values.shape[1:]), dtype=values.dtype)
    delayed_values[:num_delay_frames] = stream_delay.initial_value
    delayed_values[num_delay_frames : num_delay_frames + num_frames] = values
    delayed_values[num_delay_frames + num_frames :] = stream_delay.padding_value
    return delayed_values


def test_keeps_length_as_kyutai(undelayed_streams: AlignedStreams) -> None:
    """Check that ``keep_length=True`` delays each stream as Kyutai's moshi-finetune does."""
    delayed_streams = StreamDelayer(STREAM_DELAYS, keep_length=True).delay_streams(
        undelayed_streams
    )
    assert delayed_streams.num_frames == NUM_DELAYED_FRAMES
    for name, stream in delayed_streams.streams.items():
        expected = delay_as_kyutai(undelayed_streams.streams[name].values, STREAM_DELAYS[name])
        assert torch.equal(stream.values, expected), name


def test_extends_length_as_jmoshi(undelayed_streams: AlignedStreams) -> None:
    """Check that ``keep_length=False`` delays each stream as J-Moshi does."""
    delayed_streams = StreamDelayer(STREAM_DELAYS, keep_length=False).delay_streams(
        undelayed_streams
    )
    assert delayed_streams.num_frames == NUM_DELAYED_FRAMES + MAX_DELAY
    for name, stream in delayed_streams.streams.items():
        expected = delay_as_jmoshi(undelayed_streams.streams[name].values, STREAM_DELAYS[name])
        assert torch.equal(stream.values, expected), name


@pytest.mark.parametrize(
    ("keep_length", "expected_mask"),
    [
        (True, [True, False, False, True, True, True]),
        (False, [True, False, False, True, True, True, True, True]),
    ],
)
def test_delays_mask_with_values(
    undelayed_streams: AlignedStreams, keep_length: bool, expected_mask: list[bool]
) -> None:
    """Check that the mask moves with the values, and that filled frames are present.

    The latents, delayed by one frame, are absent at the frames 0 and 1 before the delay.
    """
    delayed_streams = StreamDelayer(STREAM_DELAYS, keep_length).delay_streams(undelayed_streams)
    assert delayed_streams.streams["latents"].present_frame_mask.tolist() == expected_mask


@pytest.mark.parametrize(
    ("keep_length", "expected_text", "expected_codebook"),
    [(True, [10, 11, 12], [99, 20, 21]), (False, [10, 11, 12, 3], [99, 20, 21, 22])],
)
def test_matches_delay_example_of_docstring(
    keep_length: bool, expected_text: list[int], expected_codebook: list[int]
) -> None:
    """Check that the delayer gives the streams that the docstring of ``StreamDelayer`` shows.

    The text is t0 to t2 = 10 to 12 with P = 3, and the codebook c0 to c2 = 20 to 22 with I = 99.
    """
    all_frames = torch.ones(3, dtype=torch.bool)
    streams = AlignedStreams(
        {
            "text": FrameStream(torch.tensor([10, 11, 12]), all_frames),
            "codebook": FrameStream(torch.tensor([20, 21, 22]), all_frames),
        }
    )
    delays = {
        "text": StreamDelay(num_delay_frames=0, initial_value=99, padding_value=3),
        "codebook": StreamDelay(num_delay_frames=1, initial_value=99, padding_value=99),
    }
    delayed_streams = StreamDelayer(delays, keep_length).delay_streams(streams)
    assert delayed_streams.streams["text"].values.tolist() == expected_text
    assert delayed_streams.streams["codebook"].values.tolist() == expected_codebook


def test_fills_window_shorter_than_delay(undelayed_streams: AlignedStreams) -> None:
    """Check that a window shorter than a delay keeps its length, with the initial value only."""
    window = undelayed_streams.take_frames(range(0, 1))
    delayed_window = StreamDelayer(STREAM_DELAYS, keep_length=True).delay_streams(window)
    assert delayed_window.num_frames == 1
    assert delayed_window.streams["late_acoustic"].values.tolist() == [2048]


@pytest.mark.parametrize("keep_length", [False, True])
def test_rejects_streams_other_than_delays(
    undelayed_streams: AlignedStreams, keep_length: bool
) -> None:
    """Check that a recording with streams that ``delays`` does not name raises ValueError."""
    delays = {name: STREAM_DELAYS[name] for name in ["text", "semantic"]}
    with pytest.raises(ValueError, match="but delays has"):
        StreamDelayer(delays, keep_length).delay_streams(undelayed_streams)


def test_rejects_negative_delay() -> None:
    """Check that a negative delay raises ValueError when the delayer is made."""
    with pytest.raises(ValueError, match="stream text has a negative num_delay_frames of -1"):
        StreamDelayer(
            {"text": StreamDelay(num_delay_frames=-1, initial_value=0, padding_value=0)},
            keep_length=True,
        )

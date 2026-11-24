import math
from dataclasses import dataclass
from itertools import accumulate
from typing import Self

import torch


@dataclass
class FrameStream:
    """One stream of a recording, such as the text or the audio of one speaker, frame by frame.

    ``AlignedStreams`` holds the streams of one recording.

    - ``values``: ``[num_frames, ...]`` value at each audio frame: text token ids
      ``[num_frames]``, codec codes ``[num_frames, num_codebooks]`` or codec latents
      ``[num_frames, latent_size]``.
    - ``present_frame_mask``: ``[num_frames]`` bool, False at the frames where the stream has
      no value, such as the frames of a prompt that hold only text. The model gets no input
      from the stream at these frames.
    """

    values: torch.Tensor
    present_frame_mask: torch.Tensor


@dataclass
class AlignedStreams:
    """Streams of one recording, aligned on its audio frames, as Moshi and similar models read them.

    These models read the values of every stream at each audio frame, such as the text, the
    audio of the model's speaker and the audio of the user. ``streams`` maps the name of each
    stream to its values, and every stream has the same number of frames.
    """

    streams: dict[str, FrameStream]

    def __post_init__(self) -> None:
        """Check that there is a stream and that every stream covers the same frames.

        Each stream needs values with a frame dimension and a bool mask with one entry for each
        frame.
        """
        if not self.streams:
            raise ValueError("AlignedStreams needs at least one stream")
        frame_counts = {}
        for name, stream in self.streams.items():
            if stream.values.dim() == 0:
                raise ValueError(f"stream {name} has values without a frame dimension")
            if stream.present_frame_mask.dtype != torch.bool:
                raise ValueError(
                    f"stream {name} has a mask of {stream.present_frame_mask.dtype}, not bool"
                )
            if stream.present_frame_mask.shape != stream.values.shape[:1]:
                raise ValueError(
                    f"stream {name} has {stream.values.size(0)} frames of values but a mask of "
                    f"shape {tuple(stream.present_frame_mask.shape)}"
                )
            frame_counts[name] = stream.values.size(0)
        if len(set(frame_counts.values())) > 1:
            raise ValueError(f"the streams have different numbers of frames: {frame_counts}")

    @property
    def num_frames(self) -> int:
        """Number of audio frames that every stream covers."""
        return next(iter(self.streams.values())).values.size(0)

    def take_frames(self, frames: range) -> Self:
        """Return every stream at ``frames`` only, such as one window of the recording.

        ``frames`` is a range of consecutive frames of the recording, as ``WindowSplitter.split``
        returns; a step other than 1, or frames outside the recording, raise ValueError.
        """
        if frames.step != 1:
            raise ValueError(f"take_frames takes consecutive frames, got a step of {frames.step}")
        if not 0 <= frames.start <= frames.stop <= self.num_frames:
            raise ValueError(
                f"take_frames takes frames from 0 to {self.num_frames}, got {frames.start} to "
                f"{frames.stop}"
            )
        window = slice(frames.start, frames.stop)
        return type(self)(
            {
                name: FrameStream(stream.values[window], stream.present_frame_mask[window])
                for name, stream in self.streams.items()
            }
        )


@dataclass
class WindowSplitter:
    """Splitter of a long recording into windows that fit the model's sequence, for training.

    A recording longer than ``max_num_frames`` frames is trained on in windows of at most
    ``max_num_frames`` frames. ``split_evenly``, a name of this code, chooses where the
    windows start.

    - False: a window starts every ``max_num_frames`` frames from the start of the recording,
      so that only the last window can be shorter. Kyutai's moshi-finetune cuts its windows this
      way and then pads the last one to the same length; here it stays shorter, and packing
      puts windows of different lengths into one row.
    - True: the recording is split into the fewest windows of at most ``max_num_frames``
      frames, whose lengths differ by at most one frame, as J-Moshi does with
      ``numpy.array_split``.

    Windows shorter than ``min_num_frames`` frames are dropped, as J-Moshi's ``min_length``
    does. For example, 10 frames with ``max_num_frames=4`` give the windows of frames 0-3, 4-7
    and 8-9 with False, and of frames 0-3, 4-6 and 7-9 with True.
    """

    max_num_frames: int
    split_evenly: bool
    min_num_frames: int

    def __post_init__(self) -> None:
        """Check that a window can hold a frame and that ``min_num_frames`` fits in a window."""
        if self.max_num_frames < 1:
            raise ValueError(f"max_num_frames must be at least 1, got {self.max_num_frames}")
        if not 0 <= self.min_num_frames <= self.max_num_frames:
            raise ValueError(
                f"min_num_frames must be from 0 to max_num_frames={self.max_num_frames}, "
                f"got {self.min_num_frames}"
            )

    def split(self, num_frames: int) -> list[range]:
        """Return the frames of each window of a recording of ``num_frames`` frames, in order."""
        if self.split_evenly:
            num_windows = math.ceil(num_frames / self.max_num_frames)
            # As numpy.array_split: the first num_frames % num_windows windows are one longer.
            shorter_length, num_longer = divmod(num_frames, max(num_windows, 1))
            lengths = [shorter_length + (index < num_longer) for index in range(num_windows)]
        else:
            lengths = [
                min(self.max_num_frames, num_frames - start)
                for start in range(0, num_frames, self.max_num_frames)
            ]
        starts = list(accumulate(lengths, initial=0))[:-1]
        windows = [
            range(start, start + length) for start, length in zip(starts, lengths, strict=True)
        ]
        return [window for window in windows if len(window) >= self.min_num_frames]

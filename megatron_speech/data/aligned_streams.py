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

        Parameters
        ----------
        frames
            Consecutive frames of the recording (a step of 1), such as a window that
            ``WindowSplitter.split`` returns.

        Returns
        -------
        Self
            The streams whose values and masks hold only the frames of ``frames``.

        Raises
        ------
        ValueError
            If ``frames`` has a step other than 1 or reaches outside the recording.
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

    A recording longer than ``max_frames`` frames is trained on in windows of at most
    ``max_frames`` frames. ``split_evenly``, a name of this code, chooses where the
    windows start.

    - False: a window starts every ``max_frames`` frames from the start of the recording,
      so that only the last window can be shorter. Kyutai's moshi-finetune cuts its windows this
      way and then pads the last one to the same length; here it stays shorter, and packing
      puts windows of different lengths into one row.
    - True: the recording is split into the fewest windows of at most ``max_frames``
      frames, whose lengths differ by at most one frame, as J-Moshi does with
      ``numpy.array_split``.

    Windows shorter than ``min_frames`` frames are dropped, as J-Moshi's ``min_length``
    does. For example, 10 frames with ``max_frames=4`` give the windows of frames 0-3, 4-7
    and 8-9 with False, and of frames 0-3, 4-6 and 7-9 with True.
    """

    max_frames: int
    split_evenly: bool
    min_frames: int

    def __post_init__(self) -> None:
        """Check that a window can hold a frame and that ``min_frames`` fits in a window."""
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be at least 1, got {self.max_frames}")
        if not 0 <= self.min_frames <= self.max_frames:
            raise ValueError(
                f"min_frames must be from 0 to max_frames={self.max_frames}, got {self.min_frames}"
            )

    def split(self, num_frames: int) -> list[range]:
        """Return the frames of each window of a recording, in order.

        Parameters
        ----------
        num_frames
            Number of frames of the recording.

        Returns
        -------
        list[range]
            Frames of each window, numbered as the frames of the recording, which
            ``AlignedStreams.take_frames`` takes.
        """
        if self.split_evenly:
            num_windows = math.ceil(num_frames / self.max_frames)
            # As numpy.array_split: the first num_frames % num_windows windows are one longer.
            shorter_length, num_longer = divmod(num_frames, max(num_windows, 1))
            lengths = [shorter_length + (index < num_longer) for index in range(num_windows)]
        else:
            lengths = [
                min(self.max_frames, num_frames - start)
                for start in range(0, num_frames, self.max_frames)
            ]
        starts = list(accumulate(lengths, initial=0))[:-1]
        windows = [
            range(start, start + length) for start, length in zip(starts, lengths, strict=True)
        ]
        return [window for window in windows if len(window) >= self.min_frames]


@dataclass
class StreamDelay:
    """How one stream of a recording is delayed, as ``StreamDelayer`` takes it.

    - ``num_delay_frames``: number of frames by which the stream starts later than the
      recording.
    - ``initial_value``: value of the frames before the delayed stream starts, such as Moshi's
      initial token.
    - ``padding_value``: value of the frames after the stream ends, when ``StreamDelayer``
      makes the recording longer (J-Moshi's padding token).
    """

    num_delay_frames: int
    initial_value: int | float
    padding_value: int | float


@dataclass
class StreamDelayer:
    """Delayer of some streams of a recording against the others, as Moshi reads its audio.

    Moshi reads the text and the first codebook of the audio of a frame together, and the
    other codebooks one frame later. Each codebook is then a stream of its own, delayed by 0
    or by 1 frame. ``delays`` gives the ``StreamDelay`` of every stream of the recording.

    ``keep_length``, a name of this code, chooses the length of the delayed recording.

    - True: the recording keeps its length, and each delayed stream loses its last frames.
      Kyutai's moshi-finetune delays each window of a recording this way.
    - False: the recording grows by the largest delay, and each stream that is delayed less
      gets ``padding_value`` at its end, so that no frame is lost. J-Moshi delays a whole
      recording this way before it cuts the recording into windows.

    The frames that get ``initial_value`` or ``padding_value`` are present in the mask, since
    the model reads these values. For example, a text stream ``t`` (delay 0, padding value
    ``P``) and a codebook ``c`` (delay 1, initial value ``I``) of 3 frames give::

        frame                         0    1    2    3
        before the delay   text       t0   t1   t2
                           codebook   c0   c1   c2
        keep_length=True   text       t0   t1   t2
                           codebook   I    c0   c1
        keep_length=False  text       t0   t1   t2   P
                           codebook   I    c0   c1   c2
    """

    delays: dict[str, StreamDelay]
    keep_length: bool

    def __post_init__(self) -> None:
        """Check that no stream is delayed by a negative number of frames."""
        for name, stream_delay in self.delays.items():
            if stream_delay.num_delay_frames < 0:
                raise ValueError(
                    f"stream {name} has a negative num_delay_frames of "
                    f"{stream_delay.num_delay_frames}"
                )

    def delay_streams(self, streams: AlignedStreams) -> AlignedStreams:
        """Return the streams of a recording, each delayed by its ``StreamDelay``.

        Parameters
        ----------
        streams
            Streams of the recording, which must be the streams that ``delays`` names.

        Returns
        -------
        AlignedStreams
            The delayed streams, with as many frames as ``streams`` if ``keep_length`` is
            True, and with as many more as the largest delay if not.

        Raises
        ------
        ValueError
            If the streams differ from the streams that ``delays`` names.
        """
        if streams.streams.keys() != self.delays.keys():
            raise ValueError(
                f"the recording has the streams {sorted(streams.streams)}, but delays has "
                f"{sorted(self.delays)}"
            )
        max_delay = max(stream_delay.num_delay_frames for stream_delay in self.delays.values())
        num_frames = streams.num_frames

        def delay_stream(stream: FrameStream, stream_delay: StreamDelay) -> FrameStream:
            """Put ``initial_value`` before the stream, and ``padding_value`` after it."""
            if self.keep_length:
                num_initial_frames = min(stream_delay.num_delay_frames, num_frames)
                num_stream_frames = num_frames - num_initial_frames
                num_padding_frames = 0
            else:
                num_initial_frames = stream_delay.num_delay_frames
                num_stream_frames = num_frames
                num_padding_frames = max_delay - stream_delay.num_delay_frames
            frame_shape = stream.values.shape[1:]
            values = torch.cat(
                [
                    stream.values.new_full(
                        (num_initial_frames, *frame_shape), stream_delay.initial_value
                    ),
                    stream.values[:num_stream_frames],
                    stream.values.new_full(
                        (num_padding_frames, *frame_shape), stream_delay.padding_value
                    ),
                ]
            )
            present_frame_mask = torch.cat(
                [
                    stream.present_frame_mask.new_ones(num_initial_frames),
                    stream.present_frame_mask[:num_stream_frames],
                    stream.present_frame_mask.new_ones(num_padding_frames),
                ]
            )
            return FrameStream(values, present_frame_mask)

        return AlignedStreams(
            {
                name: delay_stream(stream, self.delays[name])
                for name, stream in streams.streams.items()
            }
        )

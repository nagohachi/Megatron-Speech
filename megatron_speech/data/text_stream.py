import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch


@dataclass
class TimedTokens:
    """One word of a transcript, as its tokens and the time in the audio when it is spoken.

    ``TextStreamBuilder`` takes a list of these. A word can also be split into parts, each with
    its own time, such as one part for each token.

    - ``token_ids``: the tokens of the word, in order.
    - ``start_time``: seconds from the start of the audio to the start of the word.
    """

    token_ids: list[int]
    start_time: float


@dataclass
class TextStreamBuilder:
    """Builder of the text that Moshi and similar models read, one token per audio frame.

    These models read one text token at each frame of the audio, together with the audio of
    that frame. ``build`` makes this text from the words of a transcript and the times when
    they are spoken. The tokens of a word go on the frames from the one in which the word
    starts, and the frames without a word get ``padding_id``. The last of these padding frames
    before a word gets ``end_of_padding_id`` instead, which marks that a word starts next.

    - ``frames_per_second``: frame rate of the audio codec, 12.5 for Moshi's Mimi.
    - ``padding_id``: token of a frame without text (Moshi's PAD).
    - ``end_of_padding_id``: token of the last frame of padding before text (Moshi's EPAD).
    - ``leftover_tokens``: what to do when a word has tokens left at the frame where the next
      word starts. ``"drop"`` drops the rest of the word, as Kyutai's moshi-finetune does;
      ``"delay"`` places the rest first and the next word after it, as J-Moshi does.
    """

    frames_per_second: float
    padding_id: int
    end_of_padding_id: int
    leftover_tokens: Literal["drop", "delay"]

    def __post_init__(self) -> None:
        """Check that the frame rate is positive and that ``leftover_tokens`` names a rule."""
        if self.frames_per_second <= 0:
            raise ValueError(f"frames_per_second must be positive, got {self.frames_per_second}")
        if self.leftover_tokens not in ("drop", "delay"):
            raise ValueError(
                f'leftover_tokens must be "drop" or "delay", got {self.leftover_tokens!r}'
            )

    def build(self, timed_tokens: list[TimedTokens], num_frames: int) -> torch.Tensor:
        """Make the text for ``num_frames`` frames of audio from the words of ``timed_tokens``.

        Return the ``[num_frames]`` token ids, one for each frame. The words are placed in the
        order of their start times. The tokens that do not fit before the last frame are
        dropped. A start time before 0 raises ValueError.
        """
        ordered_timed_tokens = sorted(timed_tokens, key=lambda token_group: token_group.start_time)
        start_frames = [
            math.floor(token_group.start_time * self.frames_per_second)
            for token_group in ordered_timed_tokens
        ]
        if start_frames and start_frames[0] < 0:
            raise ValueError(
                f"a start time of {ordered_timed_tokens[0].start_time} s is before the audio"
            )
        stream = [self.padding_id] * num_frames
        # Frame after the last placed token.
        next_frame = 0
        # Each start frame with the next one, or with the end of the frames for the last.
        start_frame_pairs = pairwise([*start_frames, num_frames])
        for token_group, (start_frame, next_start_frame) in zip(
            ordered_timed_tokens, start_frame_pairs, strict=True
        ):
            if self.leftover_tokens == "delay":
                first_frame, end_frame = max(start_frame, next_frame), num_frames
            else:
                first_frame, end_frame = start_frame, min(next_start_frame, num_frames)
            # Stops at the end frame, or after the last token.
            frames = range(first_frame, end_frame)
            for frame, token_id in zip(frames, token_group.token_ids, strict=False):
                if frame > 0 and stream[frame - 1] == self.padding_id:
                    stream[frame - 1] = self.end_of_padding_id
                stream[frame] = token_id
                next_frame = frame + 1
        return torch.tensor(stream, dtype=torch.int64)

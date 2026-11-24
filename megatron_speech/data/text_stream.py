import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch


@dataclass
class TimedTokens:
    """Text tokens that start at one time of the audio, such as the tokens of one word.

    - ``token_ids``: the tokens, placed on consecutive frames in this order.
    - ``start_time``: seconds from the start of the audio to the start of the first token.
    """

    token_ids: list[int]
    start_time: float


@dataclass
class TextStreamBuilder:
    """Rule that places timed text tokens on the frames of the audio, one token per frame.

    The tokens of a ``TimedTokens`` start at the frame that holds its ``start_time``, the
    frames without a token get ``padding_id``, and the frame right before a token after
    padding gets ``end_of_padding_id``.

    - ``frames_per_second``: frame rate of the audio codec, 12.5 for Moshi's Mimi.
    - ``padding_id``: token of a frame without text (Moshi's PAD).
    - ``end_of_padding_id``: token of the last frame of padding before text (Moshi's EPAD).
    - ``leftover_tokens``: what happens to the tokens of one ``TimedTokens`` that are not
      placed yet at the start frame of the next. ``"drop"`` drops them, as Kyutai's
      moshi-finetune does; ``"delay"`` places them on the next frames, before the next
      tokens, as J-Moshi does.
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
        """Place ``timed_tokens`` on ``num_frames`` frames; return the ``[num_frames]`` tokens.

        ``timed_tokens`` is placed in the order of the start times. The tokens that do not fit
        before the end of the frames are dropped. A start time before 0 raises ValueError.
        """
        ordered_tokens = sorted(timed_tokens, key=lambda tokens: tokens.start_time)
        start_frames = [
            math.floor(tokens.start_time * self.frames_per_second) for tokens in ordered_tokens
        ]
        if start_frames and start_frames[0] < 0:
            raise ValueError(
                f"a start time of {ordered_tokens[0].start_time} s is before the audio"
            )
        stream = [self.padding_id] * num_frames
        # Frame after the last placed token.
        next_frame = 0
        # Each start frame with the next one, or with the end of the frames for the last.
        start_frame_pairs = pairwise([*start_frames, num_frames])
        for tokens, (start_frame, next_start_frame) in zip(
            ordered_tokens, start_frame_pairs, strict=True
        ):
            if self.leftover_tokens == "delay":
                first_frame, end_frame = max(start_frame, next_frame), num_frames
            else:
                first_frame, end_frame = start_frame, min(next_start_frame, num_frames)
            # Stops at the end frame, or after the last token.
            frames = range(first_frame, end_frame)
            for frame, token_id in zip(frames, tokens.token_ids, strict=False):
                if frame > 0 and stream[frame - 1] == self.padding_id:
                    stream[frame - 1] = self.end_of_padding_id
                stream[frame] = token_id
                next_frame = frame + 1
        return torch.tensor(stream, dtype=torch.int64)

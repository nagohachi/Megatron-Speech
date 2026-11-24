import math
from dataclasses import dataclass
from itertools import pairwise

import torch


@dataclass
class TimedTokens:
    """Part of a transcript, as its tokens and the time in the audio when the part is spoken.

    ``TextStreamBuilder`` takes a list of these. A part is usually one word, as in Kyutai's
    moshi-finetune. J-Moshi gives each token of a word its own time, which makes each token a
    part.

    - ``token_ids``: the tokens of the part, in order.
    - ``start_time``: seconds from the start of the audio to the start of the part.
    """

    token_ids: list[int]
    start_time: float


@dataclass
class TextStreamBuilder:
    """Builder of the text that Moshi and similar models read, one token per audio frame.

    These models read one text token at each frame of the audio, together with the audio of
    that frame. ``build`` makes this text from the parts of a transcript (``TimedTokens``) and
    the times when they are spoken. The tokens of a part go on the frames from the one in
    which the part starts, one token per frame. The frames without text get ``padding_id``
    (Moshi's PAD), and the last of them before a part gets ``end_of_padding_id`` (Moshi's
    EPAD) instead, which marks that text starts next.

    A part can have more tokens than there are frames before the next part starts.
    ``keep_and_shift`` chooses what happens to the tokens that do not fit. With False, they
    are dropped, and the next part starts on time. With True, they are kept on the next
    frames, and the next part is shifted back by as many frames. Kyutai's moshi-finetune gives
    this setting the same name and trains with False; J-Moshi places the tokens as True does.

    - ``frames_per_second``: frame rate of the audio codec, 12.5 for Moshi's Mimi.
    - ``padding_id``: token of a frame without text.
    - ``end_of_padding_id``: token of the last frame without text before a part.
    - ``keep_and_shift``: True keeps the tokens that do not fit and shifts the next part
      back; False drops them.

    For example, at 12.5 frames per second, "Hello" (tokens Hel and lo) starts at 0.10 s,
    "there" (one token) at 0.20 s and "how" (one token) at 0.42 s. Only frame 1 is left for
    "Hello" before "there" starts::

        frame   time (s)    word that starts   keep_and_shift=False   keep_and_shift=True
        0       0.00-0.08                      EPAD                   EPAD
        1       0.08-0.16   Hello              Hel                    Hel
        2       0.16-0.24   there              there                  lo
        3       0.24-0.32                      PAD                    there
        4       0.32-0.40                      EPAD                   EPAD
        5       0.40-0.48   how                how                    how
    """

    frames_per_second: float
    padding_id: int
    end_of_padding_id: int
    keep_and_shift: bool

    def __post_init__(self) -> None:
        """Check that the frame rate is positive."""
        if self.frames_per_second <= 0:
            raise ValueError(f"frames_per_second must be positive, got {self.frames_per_second}")

    def build(self, timed_tokens: list[TimedTokens], num_frames: int) -> torch.Tensor:
        """Make the text for ``num_frames`` frames of audio from the parts in ``timed_tokens``.

        Return the ``[num_frames]`` token ids, one for each frame. The parts are placed in the
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
            if self.keep_and_shift:
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

import pytest
import torch

from megatron_speech.data.text_stream import TextStreamBuilder, TimedTokens

# Moshi's frame rate and its PAD and EPAD tokens.
FRAMES_PER_SECOND = 12.5
PAD = 3
EPAD = 0
# Runs a test with each value of keep_and_shift, which tells what happens to the tokens left
# at the start frame of the next part.
WITH_KEEP_AND_SHIFT = pytest.mark.parametrize("keep_and_shift", [False, True])


def calc_start_time(frame: int) -> float:
    """Return the time in seconds of the middle of ``frame``."""
    return (frame + 0.5) / FRAMES_PER_SECOND


# Tokens of three words that start at frames 0, 4 and 5, without overlap.
SEPARATE_TOKENS = [
    TimedTokens(token_ids=[10, 11], start_time=calc_start_time(0)),
    TimedTokens(token_ids=[12], start_time=calc_start_time(4)),
    TimedTokens(token_ids=[13, 14], start_time=calc_start_time(5)),
]
# Tokens of four words, of which the first runs past the start of the second and the last two
# start at the same frame.
OVERLAPPING_TOKENS = [
    TimedTokens(token_ids=[10, 11, 12], start_time=calc_start_time(1)),
    TimedTokens(token_ids=[13, 14], start_time=calc_start_time(2)),
    TimedTokens(token_ids=[15], start_time=calc_start_time(5)),
    TimedTokens(token_ids=[16], start_time=calc_start_time(5)),
]
# The example in the docstring of TextStreamBuilder: "Hello" (Hel and lo, 10 and 11) starts
# at 0.10 s, "there" (12) at 0.20 s and "how" (13) at 0.42 s.
DOCSTRING_EXAMPLE_TOKENS = [
    TimedTokens(token_ids=[10, 11], start_time=0.10),
    TimedTokens(token_ids=[12], start_time=0.20),
    TimedTokens(token_ids=[13], start_time=0.42),
]
# Tokens of a word, then a word with no tokens, such as one that the tokenizer drops.
TOKENS_BEFORE_EMPTY_WORD = [
    TimedTokens(token_ids=[10, 11, 12], start_time=calc_start_time(1)),
    TimedTokens(token_ids=[], start_time=calc_start_time(2)),
]


@pytest.fixture(name="builder")
def build_text_stream_builder(keep_and_shift: bool) -> TextStreamBuilder:
    """Build a builder with Moshi's frame rate and tokens and the given ``keep_and_shift``."""
    return TextStreamBuilder(
        frames_per_second=FRAMES_PER_SECOND,
        padding_id=PAD,
        end_of_padding_id=EPAD,
        keep_and_shift=keep_and_shift,
    )


@WITH_KEEP_AND_SHIFT
def test_places_tokens_from_start_frames(builder: TextStreamBuilder) -> None:
    """Check that tokens start at their frames, with EPAD only right after padding.

    The first word starts at frame 0, so no EPAD comes before it; the third word starts right
    after the second, so no EPAD comes between them.
    """
    stream = builder.build(SEPARATE_TOKENS, num_frames=9)
    assert stream.tolist() == [10, 11, PAD, EPAD, 12, 13, 14, PAD, PAD]


@pytest.mark.parametrize("keep_and_shift", [False])
def test_drops_leftover_tokens(builder: TextStreamBuilder) -> None:
    """Check that the tokens not placed by the next start frame are dropped.

    Only the first token of the first word fits before the second word, and the third word
    starts at the same frame as the fourth, so none of its tokens is placed.
    """
    stream = builder.build(OVERLAPPING_TOKENS, num_frames=7)
    assert stream.tolist() == [EPAD, 10, 13, 14, EPAD, 16, PAD]


@pytest.mark.parametrize("keep_and_shift", [True])
def test_shifts_leftover_tokens(builder: TextStreamBuilder) -> None:
    """Check that the tokens not placed by the next start frame come before the next tokens."""
    stream = builder.build(OVERLAPPING_TOKENS, num_frames=9)
    assert stream.tolist() == [EPAD, 10, 11, 12, 13, 14, 15, 16, PAD]


@pytest.mark.parametrize("keep_and_shift", [True])
def test_drops_tokens_shifted_past_last_frame(builder: TextStreamBuilder) -> None:
    """Check that the tokens shifted past the last frame are dropped."""
    stream = builder.build(OVERLAPPING_TOKENS, num_frames=7)
    assert stream.tolist() == [EPAD, 10, 11, 12, 13, 14, 15]


@pytest.mark.parametrize(
    ("keep_and_shift", "expected_stream"),
    [
        (False, [EPAD, 10, 12, PAD, EPAD, 13]),
        (True, [EPAD, 10, 11, 12, EPAD, 13]),
    ],
)
def test_matches_docstring_example(builder: TextStreamBuilder, expected_stream: list[int]) -> None:
    """Check that the builder gives the texts that the docstring of ``TextStreamBuilder`` shows."""
    assert builder.build(DOCSTRING_EXAMPLE_TOKENS, num_frames=6).tolist() == expected_stream


@WITH_KEEP_AND_SHIFT
def test_drops_tokens_after_last_frame(builder: TextStreamBuilder) -> None:
    """Check that the tokens past the end of the frames are dropped.

    The second word starts after the last frame, so none of its tokens is placed.
    """
    timed_tokens = [
        TimedTokens(token_ids=[10, 11, 12], start_time=calc_start_time(3)),
        TimedTokens(token_ids=[13], start_time=calc_start_time(6)),
    ]
    stream = builder.build(timed_tokens, num_frames=5)
    assert stream.tolist() == [PAD, PAD, EPAD, 10, 11]


@pytest.mark.parametrize("keep_and_shift", [False])
def test_word_without_tokens_drops_leftover_tokens(builder: TextStreamBuilder) -> None:
    """Check that a word without tokens still drops the leftover tokens of the word before."""
    stream = builder.build(TOKENS_BEFORE_EMPTY_WORD, num_frames=5)
    assert stream.tolist() == [EPAD, 10, PAD, PAD, PAD]


@pytest.mark.parametrize("keep_and_shift", [True])
def test_word_without_tokens_keeps_leftover_tokens(builder: TextStreamBuilder) -> None:
    """Check that a word without tokens leaves the tokens of the word before in place."""
    stream = builder.build(TOKENS_BEFORE_EMPTY_WORD, num_frames=5)
    assert stream.tolist() == [EPAD, 10, 11, 12, PAD]


@WITH_KEEP_AND_SHIFT
def test_pads_frames_without_tokens(builder: TextStreamBuilder) -> None:
    """Check that audio without text gets PAD on every frame, as int64 token ids."""
    stream = builder.build([], num_frames=4)
    assert stream.tolist() == [PAD] * 4
    assert stream.dtype == torch.int64


@WITH_KEEP_AND_SHIFT
def test_builds_no_frames_for_zero_frames(builder: TextStreamBuilder) -> None:
    """Check that ``num_frames=0`` gives an empty stream of int64 token ids."""
    stream = builder.build(SEPARATE_TOKENS, num_frames=0)
    assert stream.tolist() == []
    assert stream.dtype == torch.int64


@WITH_KEEP_AND_SHIFT
def test_orders_tokens_by_start_time(builder: TextStreamBuilder) -> None:
    """Check that the tokens are placed in the order of their start times, not of the list."""
    stream = builder.build(SEPARATE_TOKENS[::-1], num_frames=9)
    assert stream.tolist() == [10, 11, PAD, EPAD, 12, 13, 14, PAD, PAD]


@WITH_KEEP_AND_SHIFT
def test_rejects_start_before_audio(builder: TextStreamBuilder) -> None:
    """Check that tokens that start before the audio raise ValueError."""
    with pytest.raises(ValueError, match="before the audio"):
        builder.build([TimedTokens(token_ids=[10], start_time=-0.01)], num_frames=4)


def test_rejects_frame_rate_of_zero() -> None:
    """Check that a frame rate of 0 raises ValueError when the builder is made."""
    with pytest.raises(ValueError, match="frames_per_second must be positive"):
        TextStreamBuilder(
            frames_per_second=0, padding_id=PAD, end_of_padding_id=EPAD, keep_and_shift=False
        )

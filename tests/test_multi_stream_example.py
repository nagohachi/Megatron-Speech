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
from megatron_speech.data.multi_stream import MultiStreamExample, MultiStreamExampleBuilder

TEXT_INITIAL_TOKEN = 32000
AUDIO_INITIAL_TOKEN = 2048
PAD = 3
EPAD = 0
# Streams of a recording as Moshi reads it: the text, the semantic codebook and an acoustic
# codebook of the model's speaker, and the semantic codebook of the user, which the model only
# reads. The acoustic codebook is delayed by one frame. J-Moshi pads the text with PAD and the
# audio with the initial token.
STREAM_DELAYS = {
    "text": StreamDelay(num_delay_frames=0, initial_value=TEXT_INITIAL_TOKEN, padding_value=PAD),
    "semantic": StreamDelay(
        num_delay_frames=0, initial_value=AUDIO_INITIAL_TOKEN, padding_value=AUDIO_INITIAL_TOKEN
    ),
    "acoustic": StreamDelay(
        num_delay_frames=1, initial_value=AUDIO_INITIAL_TOKEN, padding_value=AUDIO_INITIAL_TOKEN
    ),
    "user_semantic": StreamDelay(
        num_delay_frames=0, initial_value=AUDIO_INITIAL_TOKEN, padding_value=AUDIO_INITIAL_TOKEN
    ),
}
STREAM_NAMES = list(STREAM_DELAYS)
# Delay of each stream in frames, as Kyutai's and J-Moshi's ``delays``.
DELAYS = [stream_delay.num_delay_frames for stream_delay in STREAM_DELAYS.values()]
INITIAL_TOKENS = torch.tensor([delay.initial_value for delay in STREAM_DELAYS.values()])
AUDIO_DECODER_STREAM_NAMES = ["semantic", "acoustic"]
# Kyutai's moshi-finetune weighs the text loss at PAD and EPAD by text_padding_weight = 0.5.
BUILDER = MultiStreamExampleBuilder(AUDIO_DECODER_STREAM_NAMES, {PAD: 0.5, EPAD: 0.5})
NUM_WINDOW_FRAMES = 8
# J-Moshi's recording of 13 frames grows to 15 with the delay and the initial frame, and is
# split into 3 windows of 5 frames.
NUM_RECORDING_FRAMES = 13
MAX_WINDOW_FRAMES = 5


def build_full_stream(values: torch.Tensor) -> FrameStream:
    """Build a stream that has a value at every frame, each a target."""
    all_frames = torch.ones(values.size(0), dtype=torch.bool)
    return FrameStream(values, all_frames, all_frames)


def build_recording(num_frames: int) -> AlignedStreams:
    """Build the streams of ``STREAM_DELAYS`` with random tokens.

    About a third of the text is PAD and a third EPAD. Every frame of every stream is present
    and a target.
    """
    generator = torch.Generator().manual_seed(num_frames)
    words = torch.randint(100, 200, (num_frames,), generator=generator)
    text_kinds = torch.randint(3, (num_frames,), generator=generator)
    text = torch.where(text_kinds == 0, PAD, torch.where(text_kinds == 1, EPAD, words))
    streams = {"text": build_full_stream(text)}
    for name in STREAM_NAMES[1:]:
        codes = torch.randint(AUDIO_INITIAL_TOKEN, (num_frames,), generator=generator)
        streams[name] = build_full_stream(codes)
    return AlignedStreams(streams)


def stack_values(streams: AlignedStreams) -> torch.Tensor:
    """Stack the values of the streams into ``[num_streams, num_frames]``, as Moshi's codes."""
    return torch.stack([streams.streams[name].values for name in STREAM_NAMES])


def delay_codes_as_kyutai(codes: torch.Tensor) -> torch.Tensor:
    """Delay ``codes`` and put the initial tokens in front, as Kyutai's ``LMModel.forward``.

    The temporal transformer reads ``[:, :-1]`` of the ``[num_streams, num_frames + 1]``
    result, and the depth transformer gets ``[:, 1:]`` at the same positions.
    """
    delayed_lines = []
    for index, delay in enumerate(DELAYS):
        line = codes[index].roll(delay)
        line[:delay] = INITIAL_TOKENS[index]
        delayed_lines.append(line)
    return torch.cat([INITIAL_TOKENS[:, None], torch.stack(delayed_lines)], dim=1)


def mask_targets_as_kyutai(num_frames: int) -> torch.Tensor:
    """Return the frames ``[num_streams, num_frames]`` that Kyutai's loss takes as targets.

    ``_undelay_sequence`` drops the last frames of a stream delayed by d frames, which no
    position predicts in a window.
    """
    target_mask = torch.ones(len(STREAM_NAMES), num_frames, dtype=torch.bool)
    for index, delay in enumerate(DELAYS):
        target_mask[index, num_frames - delay :] = False
    return target_mask


def split_codes_as_jmoshi(codes: torch.Tensor) -> list[torch.Tensor]:
    """Delay, pad and split ``codes`` of a recording into windows, as J-Moshi's preprocessing.

    ``delay_and_pad_streams`` delays the whole recording and puts the initial tokens in front,
    and ``split_streams`` splits it with ``numpy.array_split``.
    """
    num_streams, num_frames = codes.shape
    delayed_codes = np.zeros((num_streams, num_frames + max(DELAYS)), dtype=np.int64)
    for index, (delay, stream_delay) in enumerate(zip(DELAYS, STREAM_DELAYS.values(), strict=True)):
        delayed_codes[index, :delay] = stream_delay.initial_value
        delayed_codes[index, delay : delay + num_frames] = codes[index].numpy()
        delayed_codes[index, delay + num_frames :] = stream_delay.padding_value
    delayed_codes = np.concatenate([INITIAL_TOKENS[:, None].numpy(), delayed_codes], axis=1)
    num_windows = -(-delayed_codes.shape[1] // MAX_WINDOW_FRAMES)
    return [
        torch.from_numpy(window) for window in np.array_split(delayed_codes, num_windows, axis=1)
    ]


@pytest.fixture(name="window")
def build_window() -> AlignedStreams:
    """Build one window of a recording, before the delay, as Kyutai's moshi-finetune cuts it."""
    return build_recording(NUM_WINDOW_FRAMES)


@pytest.fixture(name="kyutai_example")
def build_kyutai_example(window: AlignedStreams) -> MultiStreamExample:
    """Build the example of ``window``, delayed and with the initial frame, as Kyutai does."""
    delayer = StreamDelayer(STREAM_DELAYS, keep_length=True)
    return BUILDER.build(delayer.prepend_initial_frame(delayer.delay_streams(window)))


@pytest.fixture(name="kyutai_codes")
def build_kyutai_codes(window: AlignedStreams) -> torch.Tensor:
    """Build the codes of ``window`` that Kyutai's ``LMModel.forward`` reads."""
    return delay_codes_as_kyutai(stack_values(window))


def test_reads_inputs_as_kyutai(
    kyutai_example: MultiStreamExample, kyutai_codes: torch.Tensor
) -> None:
    """Check that each position reads what Kyutai's temporal transformer reads there."""
    model_inputs = kyutai_codes[:, :-1]
    assert torch.equal(kyutai_example.input_ids, model_inputs[0])
    for index, name in enumerate(STREAM_NAMES):
        positions = kyutai_example.modality_positions[name]
        assert torch.equal(positions, torch.arange(NUM_WINDOW_FRAMES)), name
        if name != "text":
            assert torch.equal(kyutai_example.stream_inputs[name], model_inputs[index]), name


def test_predicts_text_as_kyutai(
    window: AlignedStreams, kyutai_example: MultiStreamExample
) -> None:
    """Check that the text targets and their weights are those of Kyutai's text loss."""
    text = window.streams["text"].values
    expected_loss_mask = torch.ones(NUM_WINDOW_FRAMES)
    expected_loss_mask[(text == PAD) | (text == EPAD)] = 0.5
    assert torch.equal(kyutai_example.labels, text)
    assert torch.equal(kyutai_example.loss_mask, expected_loss_mask)


def test_predicts_audio_as_kyutai(
    window: AlignedStreams, kyutai_example: MultiStreamExample, kyutai_codes: torch.Tensor
) -> None:
    """Check that the audio targets are those that Kyutai's depth transformer gets and learns.

    Every position predicts the semantic codebook, which is not delayed, so the audio decoder
    runs at every position, as Kyutai's depth transformer does.
    """
    positions = kyutai_example.audio_decoder_positions
    assert torch.equal(positions, torch.arange(NUM_WINDOW_FRAMES))
    codes = stack_values(window)
    target_mask = mask_targets_as_kyutai(NUM_WINDOW_FRAMES)
    for name in AUDIO_DECODER_STREAM_NAMES:
        index = STREAM_NAMES.index(name)
        audio_targets = kyutai_example.audio_targets[name]
        assert torch.equal(audio_targets, kyutai_codes[index, 1:]), name
        learned_targets = audio_targets[kyutai_example.audio_target_masks[name]]
        assert torch.equal(learned_targets, codes[index][target_mask[index]]), name


@pytest.fixture(name="jmoshi_examples")
def build_jmoshi_examples() -> list[MultiStreamExample]:
    """Build the examples of a recording, delayed, with the initial frame and split as J-Moshi.

    J-Moshi weighs the text loss in another way (#16), so the text loss weights are 1.
    """
    delayer = StreamDelayer(STREAM_DELAYS, keep_length=False)
    recording = delayer.prepend_initial_frame(
        delayer.delay_streams(build_recording(NUM_RECORDING_FRAMES))
    )
    splitter = WindowSplitter(max_frames=MAX_WINDOW_FRAMES, split_evenly=True, min_frames=0)
    builder = MultiStreamExampleBuilder(AUDIO_DECODER_STREAM_NAMES, {})
    return [
        builder.build(recording.take_frames(window))
        for window in splitter.split(recording.num_frames)
    ]


@pytest.fixture(name="jmoshi_windows")
def build_jmoshi_windows() -> list[torch.Tensor]:
    """Build the windows ``[num_streams, num_frames]`` of J-Moshi's preprocessing."""
    return split_codes_as_jmoshi(stack_values(build_recording(NUM_RECORDING_FRAMES)))


def test_reads_inputs_as_jmoshi(
    jmoshi_examples: list[MultiStreamExample], jmoshi_windows: list[torch.Tensor]
) -> None:
    """Check that each position reads what J-Moshi's model reads, ``input_ids[..., :-1]``."""
    assert len(jmoshi_examples) == len(jmoshi_windows)
    for example, window in zip(jmoshi_examples, jmoshi_windows, strict=True):
        assert torch.equal(example.input_ids, window[0, :-1])
        for index, name in enumerate(STREAM_NAMES[1:], start=1):
            assert torch.equal(example.stream_inputs[name], window[index, :-1]), name


def test_predicts_text_as_jmoshi(
    jmoshi_examples: list[MultiStreamExample], jmoshi_windows: list[torch.Tensor]
) -> None:
    """Check that the text targets are J-Moshi's labels, which drop the initial tokens.

    J-Moshi also learns the PAD that the delay puts at the last frame of the text, which is no
    target here (#43).
    """
    expected_masks = [window[0, 1:] < TEXT_INITIAL_TOKEN for window in jmoshi_windows]
    expected_masks[-1][-1] = False
    for example, window, expected_mask in zip(
        jmoshi_examples, jmoshi_windows, expected_masks, strict=True
    ):
        assert torch.equal(example.labels, window[0, 1:])
        assert torch.equal(example.loss_mask, expected_mask.float())


def test_predicts_audio_as_jmoshi(
    jmoshi_examples: list[MultiStreamExample], jmoshi_windows: list[torch.Tensor]
) -> None:
    """Check that the audio targets are J-Moshi's labels, which drop the initial tokens."""
    for example, window in zip(jmoshi_examples, jmoshi_windows, strict=True):
        positions = example.audio_decoder_positions
        for name in AUDIO_DECODER_STREAM_NAMES:
            targets = window[STREAM_NAMES.index(name), 1:]
            target_mask = torch.zeros_like(targets, dtype=torch.bool)
            target_mask[positions] = example.audio_target_masks[name]
            assert torch.equal(target_mask, targets < AUDIO_INITIAL_TOKEN), name
            assert torch.equal(example.audio_targets[name], targets[positions]), name


def test_matches_example_of_docstring() -> None:
    """Check that the builder gives the example that the docstring of the builder shows.

    The text is t0 to t2 = 10 to 12, the codebook c0 to c2 = 20 to 22, and I = 99.
    """
    recording = AlignedStreams(
        {
            "text": build_full_stream(torch.tensor([10, 11, 12])),
            "codebook": build_full_stream(torch.tensor([20, 21, 22])),
        }
    )
    delays = {
        "text": StreamDelay(num_delay_frames=0, initial_value=99, padding_value=99),
        "codebook": StreamDelay(num_delay_frames=1, initial_value=99, padding_value=99),
    }
    delayer = StreamDelayer(delays, keep_length=True)
    window = delayer.prepend_initial_frame(delayer.delay_streams(recording))
    example = MultiStreamExampleBuilder(["codebook"], {}).build(window)
    assert example.input_ids.tolist() == [99, 10, 11]
    assert example.stream_inputs["codebook"].tolist() == [99, 99, 20]
    assert example.labels.tolist() == [10, 11, 12]
    assert example.loss_mask.tolist() == [1, 1, 1]
    assert example.audio_decoder_positions.tolist() == [1, 2]
    assert example.audio_targets["codebook"].tolist() == [20, 21]


def test_reads_prompt_without_predicting_it() -> None:
    """Check that the frames of a prompt are read but not predicted, and absent audio is not read.

    Frames 0 to 2 are a prompt that holds only text, and frames 3 to 5 hold text and audio.
    """
    after_prompt_frame_mask = torch.arange(6) >= 3
    audio = FrameStream(torch.arange(6), after_prompt_frame_mask, after_prompt_frame_mask)
    window = AlignedStreams(
        {
            "text": FrameStream(
                torch.arange(6), torch.ones(6, dtype=torch.bool), audio.target_frame_mask
            ),
            "semantic": audio,
            "acoustic": audio,
        }
    )
    example = MultiStreamExampleBuilder(["semantic", "acoustic"], {}).build(window)
    assert example.modality_positions["text"].tolist() == [0, 1, 2, 3, 4]
    assert example.modality_positions["semantic"].tolist() == [3, 4]
    assert example.loss_mask.tolist() == [0, 0, 1, 1, 1]
    assert example.audio_decoder_positions.tolist() == [2, 3, 4]


def test_builds_window_without_targets() -> None:
    """Check that a window without targets, such as one of a prompt only, gives empty audio inputs.

    The text is read at every position but is no target, and the audio is absent.
    """
    no_frames = torch.zeros(6, dtype=torch.bool)
    window = AlignedStreams(
        {
            "text": FrameStream(torch.arange(6), torch.ones(6, dtype=torch.bool), no_frames),
            "semantic": FrameStream(torch.arange(6), no_frames, no_frames),
            "latents": FrameStream(torch.randn(6, 4), no_frames, no_frames),
        }
    )
    example = MultiStreamExampleBuilder(["semantic", "latents"], {}).build(window)
    assert example.loss_mask.dtype == torch.float32
    assert example.loss_mask.tolist() == [0, 0, 0, 0, 0]
    assert example.stream_inputs["latents"].shape == (0, 4)
    assert example.audio_decoder_positions.dtype == torch.int64
    assert example.audio_decoder_positions.shape == (0,)
    assert example.audio_targets["semantic"].shape == (0,)
    assert example.audio_targets["latents"].shape == (0, 4)
    assert example.audio_target_masks["latents"].dtype == torch.bool
    assert example.audio_target_masks["latents"].shape == (0,)


@pytest.mark.parametrize("missing_name", ["text", "acoustic"])
def test_rejects_window_without_stream(window: AlignedStreams, missing_name: str) -> None:
    """Check that a window without the text or a stream of the audio decoder raises ValueError."""
    streams = {name: stream for name, stream in window.streams.items() if name != missing_name}
    with pytest.raises(ValueError, match=rf"the window has no streams \['{missing_name}'\]"):
        BUILDER.build(AlignedStreams(streams))


def test_rejects_window_of_one_frame(window: AlignedStreams) -> None:
    """Check that a window of one frame, with no frame to predict, raises ValueError."""
    with pytest.raises(ValueError, match="a window needs two frames"):
        BUILDER.build(window.take_frames(range(1)))


@pytest.mark.parametrize(
    ("audio_decoder_stream_names", "message"),
    [([], "needs at least one stream"), (["text", "semantic"], "has the text stream text")],
)
def test_rejects_audio_decoder_streams(audio_decoder_stream_names: list[str], message: str) -> None:
    """Check that an audio decoder without streams or with the text raises ValueError."""
    with pytest.raises(ValueError, match=message):
        MultiStreamExampleBuilder(audio_decoder_stream_names, {})


def test_rejects_negative_text_loss_weight() -> None:
    """Check that a negative weight of the text loss raises ValueError when the builder is made."""
    with pytest.raises(ValueError, match=r"negative weight of -0\.5 for token 3"):
        MultiStreamExampleBuilder(AUDIO_DECODER_STREAM_NAMES, {PAD: -0.5})

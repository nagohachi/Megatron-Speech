from dataclasses import dataclass

import torch

from megatron_speech.data.aligned_streams import AlignedStreams

# Name that MimoModel gives the text, which the stream of text tokens takes too.
TEXT_STREAM_NAME = "text"


@dataclass
class MultiStreamExample:
    """Input of ``MultiStreamMimoModel`` for one window of a recording, before packing.

    Position p of the sequence reads frame p of the window and predicts frame p + 1, as
    ``MultiStreamExampleBuilder`` describes.

    - ``input_ids``: ``[num_positions]`` text token at each position, read only at the positions
      of ``modality_positions["text"]``.
    - ``labels``: ``[num_positions]`` text token that each position predicts, read only where
      ``loss_mask`` is not 0.
    - ``loss_mask``: ``[num_positions]`` float32 weight of the text loss at each position, 0
      where the position predicts no text.
    - ``modality_positions``: positions ``[num_modality_positions]`` of each stream, the text
      included, where the stream is present and the model reads it.
    - ``stream_inputs``: values ``[num_modality_positions, ...]`` of each stream other than the
      text, at its positions.
    - ``audio_decoder_positions``: ``[num_audio_decoder_positions]`` positions that predict a
      target of at least one stream of the audio decoder.
    - ``audio_targets``: values ``[num_audio_decoder_positions, ...]`` of each stream of the
      audio decoder at the frame that these positions predict, also where they are no target.
    - ``audio_target_masks``: ``[num_audio_decoder_positions]`` bool of each stream of the audio
      decoder, True where its value in ``audio_targets`` is a target.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    modality_positions: dict[str, torch.Tensor]
    stream_inputs: dict[str, torch.Tensor]
    audio_decoder_positions: torch.Tensor
    audio_targets: dict[str, torch.Tensor]
    audio_target_masks: dict[str, torch.Tensor]


@dataclass
class MultiStreamExampleBuilder:
    """Builder of the inputs of ``MultiStreamMimoModel`` from one window of a recording.

    Position p of the sequence reads frame p of the window and predicts frame p + 1, so a
    window of n frames gives n - 1 positions. Moshi puts a frame of initial values in front of
    the window (``StreamDelayer.prepend_initial_frame``), so that the first frame of the window
    is predicted too. A position learns to predict the value of a stream at the next frame
    where both masks of the stream are True at that frame. For example, a text stream ``t`` and
    a codebook ``c`` delayed by one frame, with the initial value ``I``, give::

        frame                0    1    2    3
        window     text      I    t0   t1   t2
                   codebook  I    I    c0   c1
        position             0    1    2
        input      text      I    t0   t1
                   codebook  I    I    c0
        target     text      t0   t1   t2
                   codebook       c0   c1

    The codebook has no target at position 0, since ``StreamDelayer`` filled frame 1 with the
    initial value.

    The stream named ``text`` holds the text tokens, and every other stream is the input of the
    modality of ``MimoModel`` with the same name.

    - ``audio_decoder_stream_names``: streams that the audio decoder predicts, such as the
      codebooks of the audio of the model's speaker. The other streams are only inputs, such
      as the audio of the user.
    - ``text_token_loss_weights``: weight of the text loss at the positions that predict each
      of these tokens, such as 0.5 for Moshi's PAD and EPAD in Kyutai's moshi-finetune. Other
      tokens have a weight of 1.
    """

    audio_decoder_stream_names: list[str]
    text_token_loss_weights: dict[int, float]

    def __post_init__(self) -> None:
        """Check that the audio decoder predicts streams other than the text, and the weights."""
        if not self.audio_decoder_stream_names:
            raise ValueError("audio_decoder_stream_names needs at least one stream")
        if TEXT_STREAM_NAME in self.audio_decoder_stream_names:
            raise ValueError(
                f"audio_decoder_stream_names has the text stream {TEXT_STREAM_NAME}, which the "
                "LLM's output layer predicts"
            )
        for token_id, weight in self.text_token_loss_weights.items():
            if weight < 0:
                raise ValueError(
                    f"text_token_loss_weights has a negative weight of {weight} for token "
                    f"{token_id}"
                )

    def build(self, window: AlignedStreams) -> MultiStreamExample:
        """Build the input of the model for one window.

        Parameters
        ----------
        window
            Streams of the window, with the stream ``text`` and the streams of
            ``audio_decoder_stream_names``, and with at least two frames.

        Returns
        -------
        MultiStreamExample
            Input with one position for each frame of ``window`` but the last.

        Raises
        ------
        ValueError
            If ``window`` lacks the text or a stream of the audio decoder, or has fewer than
            two frames.
        """
        missing_stream_names = {TEXT_STREAM_NAME, *self.audio_decoder_stream_names}.difference(
            window.streams
        )
        if missing_stream_names:
            raise ValueError(f"the window has no streams {sorted(missing_stream_names)}")
        if window.num_frames < 2:
            raise ValueError(
                f"a window needs two frames, an input and a target, but has {window.num_frames}"
            )
        input_frames = window.take_frames(range(window.num_frames - 1))
        target_frames = window.take_frames(range(1, window.num_frames))
        target_frame_masks = {
            name: stream.present_frame_mask & stream.target_frame_mask
            for name, stream in target_frames.streams.items()
        }
        text_targets = target_frames.streams[TEXT_STREAM_NAME].values
        loss_mask = target_frame_masks[TEXT_STREAM_NAME].float()
        for token_id, weight in self.text_token_loss_weights.items():
            loss_mask[text_targets == token_id] *= weight
        modality_positions = {
            name: stream.present_frame_mask.nonzero().squeeze(1)
            for name, stream in input_frames.streams.items()
        }
        audio_decoder_positions = (
            torch.stack([target_frame_masks[name] for name in self.audio_decoder_stream_names])
            .any(dim=0)
            .nonzero()
            .squeeze(1)
        )
        return MultiStreamExample(
            input_ids=input_frames.streams[TEXT_STREAM_NAME].values,
            labels=text_targets,
            loss_mask=loss_mask,
            modality_positions=modality_positions,
            stream_inputs={
                name: stream.values[modality_positions[name]]
                for name, stream in input_frames.streams.items()
                if name != TEXT_STREAM_NAME
            },
            audio_decoder_positions=audio_decoder_positions,
            audio_targets={
                name: target_frames.streams[name].values[audio_decoder_positions]
                for name in self.audio_decoder_stream_names
            },
            audio_target_masks={
                name: target_frame_masks[name][audio_decoder_positions]
                for name in self.audio_decoder_stream_names
            },
        )

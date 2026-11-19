import math
from dataclasses import dataclass, replace
from typing import Self

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron_speech.components.activations import ACTIVATIONS
from megatron_speech.components.transformer_config import build_component_transformer_config

# Qwen3-ASR has exactly three convolutions (conv2d1 to conv2d3, as Hugging Face names them), each
# with kernel 3, stride 2 and padding 1; the released weights fix the count.
NUM_CONVOLUTIONS = 3


@dataclass
class Qwen3ASREncoderConfig:
    """Shape of ``Qwen3ASREncoder``; the defaults are those of the released Qwen3-ASR models.

    Fields with a different name in Hugging Face's ``Qwen3ASREncoderConfig``:

    - ``hidden_size``: ``d_model``
    - ``num_layers``: ``encoder_layers``
    - ``num_attention_heads``: ``encoder_attention_heads``
    - ``ffn_hidden_size``: ``encoder_ffn_dim``
    - ``activation``: ``activation_function`` (the MLP activation; the convolutions use GELU)
    - ``mel_frames_per_chunk``: ``2 * n_window``
    - ``mel_frames_per_window``: ``n_window_infer``

    The encoder cuts the log-mel frames of an utterance into chunks of ``mel_frames_per_chunk``
    frames and lets the encoder frames attend within windows of ``mel_frames_per_window`` mel
    frames.
    """

    num_mel_bins: int = 128
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16
    ffn_hidden_size: int = 4096
    downsample_hidden_size: int = 480
    mel_frames_per_chunk: int = 100
    mel_frames_per_window: int = 800
    activation: str = "gelu"
    # Hugging Face builds the LayerNorms with the PyTorch default epsilon.
    layernorm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        """Raise ValueError for shapes that the encoder or its inputs cannot use."""
        for name in (
            "num_mel_bins",
            "num_layers",
            "num_attention_heads",
            "ffn_hidden_size",
            "downsample_hidden_size",
            "mel_frames_per_chunk",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if (
            self.mel_frames_per_window < self.mel_frames_per_chunk
            or self.mel_frames_per_window % self.mel_frames_per_chunk
        ):
            raise ValueError(
                f"mel_frames_per_window={self.mel_frames_per_window} must be a positive "
                f"multiple of mel_frames_per_chunk={self.mel_frames_per_chunk}"
            )
        # The position embedding puts hidden_size / 2 sines and cosines side by side and
        # divides by hidden_size / 2 - 1.
        if self.hidden_size < 4 or self.hidden_size % 2:
            raise ValueError(
                f"hidden_size must be an even number of at least 4, got {self.hidden_size}"
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size={self.hidden_size} is not a multiple of "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation={self.activation!r} is not one of {sorted(ACTIVATIONS)}")

    @property
    def encoder_frames_per_chunk(self) -> int:
        """Number of encoder frames from a full chunk (13 from 100 mel frames)."""
        return self.calc_num_encoder_frames(self.mel_frames_per_chunk)

    @property
    def encoder_frames_per_window(self) -> int:
        """Number of encoder frames in an attention window (104 for the released models)."""
        num_chunks_per_window = self.mel_frames_per_window // self.mel_frames_per_chunk
        return self.encoder_frames_per_chunk * num_chunks_per_window

    @property
    def num_conv_out_features(self) -> int:
        """Number of features of an encoder frame out of the convolutions.

        It is the number of channels times the number of frequency bins.
        """
        return self.downsample_hidden_size * self.calc_conv_output_length(self.num_mel_bins)

    @staticmethod
    def calc_conv_output_length(length: int) -> int:
        """Calculate how many of ``length`` time steps or frequency bins the convolutions leave."""
        for _ in range(NUM_CONVOLUTIONS):
            length = (length - 1) // 2 + 1 if length > 0 else 0
        return length

    def calc_num_encoder_frames(self, num_mel_frames: int) -> int:
        """Count the encoder frames that an utterance of ``num_mel_frames`` mel frames gives.

        Each chunk goes through the convolutions on its own.
        """
        num_full_chunks, rest = divmod(num_mel_frames, self.mel_frames_per_chunk)
        full_chunk_output_length = self.calc_conv_output_length(self.mel_frames_per_chunk)
        return num_full_chunks * full_chunk_output_length + self.calc_conv_output_length(rest)


@dataclass
class Qwen3ASREncoderInputs:
    """Inputs of ``Qwen3ASREncoder.forward`` for a batch of utterances, made on the CPU.

    - ``input_features``: ``[num_chunks, num_mel_bins, mel_frames_per_chunk]`` log-mel chunks
      of all utterances, utterance by utterance, zero past the frames of each chunk
    - ``encoder_frame_index``: ``[num_encoder_frames]`` positions of the valid encoder frames
      in the ``[num_chunks * encoder_frames_per_chunk]`` grid of convolution outputs
    - ``cu_seqlens``: ``[num_windows + 1]`` int32 cumulative encoder frame counts of the
      attention windows
    - ``max_seqlen``: encoder frames in the longest attention window
    """

    input_features: torch.Tensor
    encoder_frame_index: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int

    @classmethod
    def build(cls, mels: list[torch.Tensor], config: Qwen3ASREncoderConfig) -> Self:
        """Cut ``[num_mel_bins, num_frames]`` log-mels into chunks and attention windows.

        Windows start at the first encoder frame of each utterance and hold
        ``encoder_frames_per_window`` encoder frames; the last window of an utterance may be
        shorter. A batch without frames has no encoder inputs and raises ValueError.
        """
        if not any(mel.shape[1] for mel in mels):
            raise ValueError("no utterance has frames")
        mel_frames_per_chunk = config.mel_frames_per_chunk
        chunks, encoder_frame_index, window_lengths = [], [], []
        for mel in mels:
            for start in range(0, mel.shape[1], mel_frames_per_chunk):
                piece = mel[:, start : start + mel_frames_per_chunk]
                grid_offset = len(chunks) * config.encoder_frames_per_chunk
                num_encoder_frames = config.calc_num_encoder_frames(piece.shape[1])
                encoder_frame_index.append(
                    torch.arange(grid_offset, grid_offset + num_encoder_frames)
                )
                chunks.append(
                    torch.nn.functional.pad(piece, (0, mel_frames_per_chunk - piece.shape[1]))
                )
            num_full_windows, rest = divmod(
                config.calc_num_encoder_frames(mel.shape[1]), config.encoder_frames_per_window
            )
            window_lengths += [config.encoder_frames_per_window] * num_full_windows
            if rest:
                window_lengths.append(rest)
        cu_seqlens = torch.zeros(len(window_lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(window_lengths), dim=0)
        return cls(
            input_features=torch.stack(chunks),
            encoder_frame_index=torch.cat(encoder_frame_index),
            cu_seqlens=cu_seqlens,
            max_seqlen=max(window_lengths),
        )

    def copy_to(self, device: torch.device | str) -> Self:
        """Return a copy of the inputs with the tensors on ``device``."""
        return replace(
            self,
            input_features=self.input_features.to(device),
            encoder_frame_index=self.encoder_frame_index.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
        )


class Qwen3ASREncoder(MegatronModule):
    """Qwen3-ASR audio encoder (the AuT encoder of Qwen3-Omni) without its projector.

    Matches Hugging Face's ``Qwen3ASREncoder``. Each chunk goes through ``NUM_CONVOLUTIONS``
    convolutions with GELU and a linear layer to ``hidden_size``, and a sinusoidal embedding of
    the position within the chunk is added. The valid encoder frames of all chunks then go
    through pre-LayerNorm transformer layers with bidirectional attention within each window,
    and a final LayerNorm. The attention heads and MLPs are split over the tensor parallel
    ranks; the convolutions are not split.

    The layers are Megatron's ``TransformerBlock`` so that the attention over the windows runs
    in Transformer Engine's packed (THD) kernels instead of one call per window.
    """

    def __init__(
        self,
        encoder_config: Qwen3ASREncoderConfig,
        llm_config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Build the layers with the tensor parallelism and precision of ``llm_config``."""
        tp_size = llm_config.tensor_model_parallel_size

        def check_config() -> None:
            """Raise ValueError for a config that cannot be built on ``tp_size`` ranks."""
            if pg_collection is not None and pg_collection.tp.size() != tp_size:
                raise ValueError(
                    f"pg_collection.tp has {pg_collection.tp.size()} ranks, but "
                    f"llm_config.tensor_model_parallel_size is {tp_size}"
                )
            for name in ("num_attention_heads", "ffn_hidden_size"):
                if getattr(encoder_config, name) % tp_size:
                    raise ValueError(
                        f"{name}={getattr(encoder_config, name)} cannot be split over "
                        f"{tp_size} tensor parallel ranks"
                    )

        def build_position_embedding() -> torch.Tensor:
            """Build the sinusoidal position embedding that Hugging Face's encoder adds.

            The embedding is ``[encoder_frames_per_chunk, hidden_size]``.
            """
            num_channels = encoder_config.hidden_size
            log_timescale_increment = math.log(10000.0) / (num_channels // 2 - 1)
            inv_timescales = torch.exp(-log_timescale_increment * torch.arange(num_channels // 2))
            positions = torch.arange(encoder_config.encoder_frames_per_chunk)
            scaled_time = positions[:, None] * inv_timescales
            return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

        def build_transformer_config() -> TransformerConfig:
            """Build the layers' config: pre-LayerNorm, biases, the configured MLP activation."""
            return build_component_transformer_config(
                llm_config,
                num_layers=encoder_config.num_layers,
                hidden_size=encoder_config.hidden_size,
                num_attention_heads=encoder_config.num_attention_heads,
                num_query_groups=encoder_config.num_attention_heads,
                kv_channels=encoder_config.hidden_size // encoder_config.num_attention_heads,
                ffn_hidden_size=encoder_config.ffn_hidden_size,
                normalization="LayerNorm",
                layernorm_epsilon=encoder_config.layernorm_epsilon,
                add_bias_linear=True,
                add_qkv_bias=True,
                gated_linear_unit=False,
                activation_func=ACTIVATIONS[encoder_config.activation],
                # The fused bias-GELU kernel uses the tanh approximation of GELU.
                bias_activation_fusion=False,
            )

        check_config()
        config = build_transformer_config()
        super().__init__(config=config)
        num_channels = encoder_config.downsample_hidden_size
        conv_kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "dtype": config.params_dtype}
        # NUM_CONVOLUTIONS layers, one attribute each to keep Hugging Face's parameter names.
        self.conv2d1 = torch.nn.Conv2d(1, num_channels, **conv_kwargs)
        self.conv2d2 = torch.nn.Conv2d(num_channels, num_channels, **conv_kwargs)
        self.conv2d3 = torch.nn.Conv2d(num_channels, num_channels, **conv_kwargs)
        self.conv_out = torch.nn.Linear(
            encoder_config.num_conv_out_features,
            encoder_config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
        )
        self.register_buffer("position_embedding", build_position_embedding(), persistent=False)
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        layer_spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.no_mask
        self.transformer = TransformerBlock(
            config=config,
            spec=layer_spec,
            post_layer_norm=True,
            pre_process=True,
            post_process=True,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        encoder_frame_index: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Encode the utterances that ``Qwen3ASREncoderInputs`` describes.

        Returns ``[num_encoder_frames, hidden_size]``, utterance by utterance.
        """
        hidden = input_features.unsqueeze(1).to(self.conv2d1.weight.dtype)
        for conv in (self.conv2d1, self.conv2d2, self.conv2d3):
            # Hugging Face applies GELU here whatever activation_function says.
            hidden = torch.nn.functional.gelu(conv(hidden))
        num_chunks, num_channels, num_frequency_bins, num_steps = hidden.shape
        hidden = self.conv_out(
            hidden.permute(0, 3, 1, 2).reshape(
                num_chunks, num_steps, num_channels * num_frequency_bins
            )
        )
        hidden = hidden + self.position_embedding[:num_steps].to(hidden.dtype)
        # Keep the valid encoder frames: [num_encoder_frames, 1, hidden_size] in Megatron's
        # [sequence, batch].
        hidden = hidden.reshape(-1, hidden.shape[-1]).index_select(0, encoder_frame_index)
        hidden = hidden.unsqueeze(1)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        hidden = self.transformer(
            hidden_states=hidden, attention_mask=None, packed_seq_params=packed_seq_params
        )
        return hidden.squeeze(1)

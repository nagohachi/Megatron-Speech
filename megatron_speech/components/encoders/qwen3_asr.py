"""Qwen3-ASR audio encoder (the "AuT" encoder of Qwen3-Omni) on Megatron Core.

Reference: ``transformers.models.qwen3_asr.modeling_qwen3_asr.Qwen3ASREncoder``.

Computation, for each utterance:

1. The log-mel spectrogram ``[num_mel_bins, T]`` is cut into chunks of ``2 * n_window`` frames
   (the last chunk is zero padded).
2. Each chunk goes through three stride-2 3x3 convolutions with GELU (8x downsampling in time
   and frequency) and a linear layer to ``d_model``; a 100-frame chunk becomes 13 tokens.
3. A sinusoidal position embedding over the positions *within* the chunk (0..12) is added.
4. Tokens past the valid length of each chunk are dropped and the remaining tokens of all
   utterances are concatenated (THD layout).
5. Pre-LN transformer layers with bidirectional attention restricted to windows of
   ``13 * n_window_infer / (2 * n_window)`` tokens (104 for the released models), counted from
   the start of each utterance.
6. A final LayerNorm (``ln_post``).

The data pipeline computes the chunking metadata (see
``megatron_speech.data.features.Qwen3ASRAudioFeaturizer``) so the forward pass needs no
host-device synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron_speech.components.common import flatten_valid_positions, make_component_transformer_config


@dataclass
class Qwen3ASREncoderConfig:
    num_mel_bins: int = 128
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    ffn_dim: int = 4096
    downsample_hidden_size: int = 480
    n_window: int = 50
    n_window_infer: int = 800
    layernorm_epsilon: float = 1e-5

    @property
    def chunk_frames(self) -> int:
        return 2 * self.n_window

    @property
    def tokens_per_chunk(self) -> int:
        return calc_post_cnn_length(self.chunk_frames)

    @property
    def window_tokens(self) -> int:
        return self.tokens_per_chunk * (self.n_window_infer // self.chunk_frames)

    @property
    def conv_out_features(self) -> int:
        freq = self.num_mel_bins
        for _ in range(3):
            freq = (freq - 1) // 2 + 1
        return self.downsample_hidden_size * freq


def calc_post_cnn_length(num_frames):
    """Length after three (kernel 3, stride 2, padding 1) convolutions; 0 stays 0."""
    if isinstance(num_frames, torch.Tensor):
        out = num_frames
        for _ in range(3):
            out = torch.where(out > 0, (out - 1) // 2 + 1, torch.zeros_like(out))
        return out
    out = int(num_frames)
    for _ in range(3):
        out = (out - 1) // 2 + 1 if out > 0 else 0
    return out


def calc_sinusoid_position_embedding(length: int, channels: int, max_timescale: float = 10000.0):
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Qwen3ASRAudioEncoder(MegatronModule):
    """Megatron implementation of the Qwen3-ASR audio encoder (without the projector)."""

    def __init__(self, encoder_config: Qwen3ASREncoderConfig, llm_config: TransformerConfig, pg_collection=None):
        transformer_config = make_component_transformer_config(
            llm_config,
            num_layers=encoder_config.num_layers,
            hidden_size=encoder_config.d_model,
            num_attention_heads=encoder_config.num_heads,
            ffn_hidden_size=encoder_config.ffn_dim,
            kv_channels=encoder_config.d_model // encoder_config.num_heads,
            num_query_groups=encoder_config.num_heads,
            normalization="LayerNorm",
            layernorm_epsilon=encoder_config.layernorm_epsilon,
            add_bias_linear=True,
            add_qkv_bias=True,
            gated_linear_unit=False,
            activation_func="gelu",
            # The fused bias-GELU kernel uses the tanh approximation; HF uses exact GELU.
            bias_activation_fusion=False,
        )
        super().__init__(config=transformer_config)
        self.encoder_config = encoder_config
        hidden = encoder_config.downsample_hidden_size
        self.conv2d1 = torch.nn.Conv2d(1, hidden, 3, 2, padding=1)
        self.conv2d2 = torch.nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv2d3 = torch.nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv_out = torch.nn.Linear(encoder_config.conv_out_features, encoder_config.d_model, bias=False)
        self.register_buffer(
            "positional_embedding",
            calc_sinusoid_position_embedding(encoder_config.tokens_per_chunk, encoder_config.d_model),
            persistent=False,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        layer_spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.no_mask
        self.transformer = TransformerBlock(
            config=transformer_config,
            spec=layer_spec,
            post_layer_norm=True,
            pre_process=True,
            post_process=True,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        token_index: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Encode a batch of utterances.

        Args:
            input_features: ``[num_chunks, num_mel_bins, 2 * n_window]`` log-mel chunks of all
                utterances, utterance by utterance, zero padded past each chunk's valid frames.
            token_index: ``[num_tokens]`` positions of the valid post-CNN tokens in the
                ``[num_chunks * tokens_per_chunk]`` grid.
            cu_seqlens: ``[num_windows + 1]`` int32 cumulative token counts of the attention
                windows.
            max_seqlen: longest attention window.

        Returns:
            ``[num_tokens, d_model]`` encoder outputs, utterance by utterance.
        """
        x = input_features.unsqueeze(1).to(self.conv2d1.weight.dtype)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        num_chunks, channels, freq, time = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(num_chunks, time, channels * freq))
        x = x + self.positional_embedding[:time].to(x.dtype)
        hidden_states = flatten_valid_positions(x, token_index).unsqueeze(1)  # [T, 1, H]

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=int(max_seqlen),
            max_seqlen_kv=int(max_seqlen),
        )
        hidden_states = self.transformer(
            hidden_states=hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        return hidden_states.squeeze(1)

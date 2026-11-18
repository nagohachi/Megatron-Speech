"""Audio features and encoder-input metadata, computed on the CPU in data-loader workers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig, calc_post_cnn_length


@dataclass
class AudioFeatures:
    """Encoder inputs of one utterance.

    chunks:       ``[num_chunks, num_mel_bins, chunk_frames]`` log-mel chunks, zero padded
    chunk_lens:   valid mel frames per chunk
    window_lens:  token counts of the attention windows
    num_tokens:   encoder output tokens (= audio placeholder tokens in the prompt)
    num_frames:   valid mel frames
    """

    chunks: torch.Tensor
    chunk_lens: list[int]
    window_lens: list[int]
    num_tokens: int
    num_frames: int


class Qwen3ASRAudioFeaturizer:
    """Log-mel features exactly as the HF ``Qwen3ASRFeatureExtractor`` computes them for a
    single utterance, cut into encoder chunks."""

    def __init__(self, hf_dir: str, encoder_config: Qwen3ASREncoderConfig):
        from transformers import AutoProcessor

        self.feature_extractor = AutoProcessor.from_pretrained(hf_dir).feature_extractor
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.cfg = encoder_config

    def compute_mel(self, waveform: np.ndarray) -> torch.Tensor:
        """``[num_mel_bins, num_frames]`` float32 log-mel of one utterance."""
        out = self.feature_extractor(
            np.asarray(waveform, dtype=np.float32),
            sampling_rate=self.sampling_rate,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )
        num_frames = int(out["attention_mask"][0].sum())
        return out["input_features"][0, :, :num_frames].contiguous()

    def chunk(self, mel: torch.Tensor) -> AudioFeatures:
        chunk = self.cfg.chunk_frames
        num_frames = mel.shape[1]
        num_chunks = -(-num_frames // chunk)
        chunks = torch.zeros(num_chunks, mel.shape[0], chunk, dtype=mel.dtype)
        chunk_lens = []
        for i in range(num_chunks):
            piece = mel[:, i * chunk : (i + 1) * chunk]
            chunks[i, :, : piece.shape[1]] = piece
            chunk_lens.append(piece.shape[1])
        num_tokens = sum(calc_post_cnn_length(n) for n in chunk_lens)
        window = self.cfg.window_tokens
        window_lens = [window] * (num_tokens // window) + ([num_tokens % window] if num_tokens % window else [])
        return AudioFeatures(chunks, chunk_lens, window_lens, num_tokens, num_frames)

    def __call__(self, waveform: np.ndarray) -> AudioFeatures:
        return self.chunk(self.compute_mel(waveform))


def collate_encoder_inputs(features: list[AudioFeatures], tokens_per_chunk: int) -> dict[str, torch.Tensor]:
    """Concatenate the encoder inputs of several utterances (see ``Qwen3ASRAudioEncoder.forward``)."""
    chunks = torch.cat([f.chunks for f in features], dim=0)
    token_index = []
    offset = 0
    for f in features:
        for n in f.chunk_lens:
            valid = calc_post_cnn_length(n)
            token_index.append(torch.arange(offset, offset + valid))
            offset += tokens_per_chunk
    window_lens = [w for f in features for w in f.window_lens]
    cu_seqlens = torch.zeros(len(window_lens) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(window_lens, dtype=torch.int32), dim=0)
    return {
        "input_features": chunks,
        "token_index": torch.cat(token_index),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": torch.tensor(max(window_lens), dtype=torch.int32),
        # Not used by the model; lets tools split the batch back into utterances.
        "chunk_lens": torch.tensor([n for f in features for n in f.chunk_lens], dtype=torch.int32),
        "chunks_per_example": torch.tensor([len(f.chunk_lens) for f in features], dtype=torch.int32),
    }

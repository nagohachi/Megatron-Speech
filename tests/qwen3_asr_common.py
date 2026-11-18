"""Shared helpers for the Qwen3-ASR equivalence tests (HF reference vs Megatron)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron_speech.components.registry import parse_encoder_config
from megatron_speech.data.features import Qwen3ASRAudioFeaturizer
from megatron_speech.data.librispeech import iter_librispeech
from megatron_speech.data.packing import SpeechExample, pack_examples
from megatron_speech.data.prompt import Qwen3ASRPromptFormatter


def make_examples(model_section: dict, hf_dir, num: int, split: str = "test-clean"):
    """LibriSpeech utterances as packed-batch examples and as HF model inputs."""
    from transformers import AutoTokenizer

    enc_cfg = parse_encoder_config(model_section["encoder"])
    featurizer = Qwen3ASRAudioFeaturizer(str(hf_dir), enc_cfg)
    formatter = Qwen3ASRPromptFormatter(AutoTokenizer.from_pretrained(str(hf_dir)))
    examples, hf_inputs = [], []
    for _, wav, sr, text in iter_librispeech(split, limit=num):
        assert sr == featurizer.sampling_rate
        mel = featurizer.compute_mel(wav)
        feats = featurizer.chunk(mel)
        fmt = formatter(feats.num_tokens, text, language="en")
        examples.append(SpeechExample(feats, fmt))
        padded = feats.chunks.shape[0] * enc_cfg.chunk_frames
        features = torch.zeros(1, mel.shape[0], padded)
        features[0, :, : mel.shape[1]] = mel
        mask = torch.zeros(1, padded, dtype=torch.long)
        mask[0, : mel.shape[1]] = 1
        hf_inputs.append(
            {
                "input_ids": torch.tensor([fmt.input_ids]),
                "input_features": features,
                "input_features_mask": mask,
                "target_mask": torch.tensor(fmt.target_mask),
            }
        )
    batch = pack_examples(examples, tokens_per_chunk=enc_cfg.tokens_per_chunk)
    return examples, hf_inputs, batch


def load_hf_model(hf_dir, dtype=torch.bfloat16):
    from transformers import Qwen3ASRForConditionalGeneration

    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(hf_dir), dtype=dtype, attn_implementation="sdpa")
    return model.cuda().eval()


def calc_target_ce(logits: torch.Tensor, input_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean cross entropy of the target tokens given ``[T, V]`` logits of one example."""
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp[:-1].gather(-1, input_ids[1:, None]).squeeze(-1)
    mask = target_mask[1:].to(nll.device).float()
    return (nll * mask).sum() / mask.sum()

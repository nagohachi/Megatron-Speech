"""Bridge for Qwen3-ASR (``Qwen3ASRForConditionalGeneration``, transformers >= 5.13).

HF checkpoint layout              Megatron component
model.audio_tower.*          <->  modality_submodules.audio.encoders.encoder.*   (qwen3_asr encoder)
model.multi_modal_projector.* <-> modality_submodules.audio.input_projections.0.* (mlp projector)
model.language_model.*       <->  language_model.*                               (Qwen3 LLM)
lm_head.weight                    tied to the token embeddings (not stored)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from megatron_speech.bridges.base import SpeechLLMBridge
from megatron_speech.bridges.components import (
    mlp_projector_mappings,
    qwen3_asr_encoder_mappings,
    qwen3_llm_mappings,
)
from megatron_speech.bridges.mapping import MappingTable

ENCODER_PREFIX = "modality_submodules.audio.encoders.encoder."
PROJECTOR_PREFIX = "modality_submodules.audio.input_projections.0."
LLM_PREFIX = "language_model."


class Qwen3ASRBridge(SpeechLLMBridge):
    name = "qwen3_asr"

    def __init__(self, hf_path: str | Path):
        super().__init__(hf_path)
        self.text = self.hf_config["text_config"]
        self.audio = self.hf_config["audio_config"]
        if self.hf_config.get("model_type") != "qwen3_asr":
            raise ValueError(f"{hf_path} is not a transformers-format Qwen3-ASR checkpoint (model_type != qwen3_asr)")

    # -- architecture ------------------------------------------------------------------------
    def megatron_args(self) -> dict[str, Any]:
        t = self.text
        rope = t.get("rope_parameters") or {}
        head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        args = {
            "num-layers": t["num_hidden_layers"],
            "hidden-size": t["hidden_size"],
            "ffn-hidden-size": t["intermediate_size"],
            "num-attention-heads": t["num_attention_heads"],
            "group-query-attention": True,
            "num-query-groups": t["num_key_value_heads"],
            "kv-channels": head_dim,
            "normalization": "RMSNorm",
            "norm-epsilon": t["rms_norm_eps"],
            "swiglu": True,
            "disable-bias-linear": True,
            "qk-layernorm": True,
            "position-embedding-type": "rope",
            "rotary-base": int(rope.get("rope_theta", t.get("rope_theta", 10000))),
            "rotary-percent": 1.0,
            "max-position-embeddings": t["max_position_embeddings"],
            "vocab-size": t["vocab_size"],
            "attention-dropout": 0.0,
            "hidden-dropout": 0.0,
        }
        if not self.hf_config.get("tie_word_embeddings", True):
            args["untie-embeddings-and-output-weights"] = True
        return args

    def speech_defaults(self) -> dict[str, Any]:
        a = self.audio
        return {
            "audio_token_id": self.hf_config["audio_token_id"],
            "encoder": {
                "type": "qwen3_asr",
                "num_mel_bins": a["num_mel_bins"],
                "d_model": a["d_model"],
                "num_layers": a["encoder_layers"],
                "num_heads": a["encoder_attention_heads"],
                "ffn_dim": a["encoder_ffn_dim"],
                "downsample_hidden_size": a["downsample_hidden_size"],
                "n_window": a["n_window"],
                "n_window_infer": a["n_window_infer"],
            },
            "projector": {
                "type": "mlp",
                "input_size": a["d_model"],
                "hidden_size": a["d_model"],
                "output_size": a["output_dim"],
                "activation": a["activation_function"],
            },
        }

    # -- weights -----------------------------------------------------------------------------
    def mapping_table(self, speech: dict[str, Any]) -> MappingTable:
        t, enc = self.text, speech["encoder"]
        mappings = qwen3_llm_mappings(
            megatron_prefix=LLM_PREFIX,
            hf_prefix="model.language_model.",
            num_layers=t["num_hidden_layers"],
            num_heads=t["num_attention_heads"],
            num_query_groups=t["num_key_value_heads"],
            head_dim=t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"],
            tie_word_embeddings=self.hf_config.get("tie_word_embeddings", True),
        )
        mappings += qwen3_asr_encoder_mappings(
            megatron_prefix=ENCODER_PREFIX,
            hf_prefix="model.audio_tower.",
            num_layers=enc["num_layers"],
            num_heads=enc["num_heads"],
            d_model=enc["d_model"],
        )
        mappings += mlp_projector_mappings(megatron_prefix=PROJECTOR_PREFIX, hf_prefix="model.multi_modal_projector.")
        return MappingTable(mappings)

"""Recipe for speech LLMs (``model.type: speech_llm``)."""

from __future__ import annotations

import copy
from typing import Any

from megatron_speech import compat
from megatron_speech.bridges import get_bridge_for_model_section
from megatron_speech.models.speech_llm import SpeechLLMModelConfig
from megatron_speech.training.config import merge_dicts
from megatron_speech.training.recipes import Recipe

# Megatron arguments that the speech LLM recipe always uses unless the run config overrides them.
DEFAULT_MEGATRON_ARGS: dict[str, Any] = {
    # Batches come from megatron_speech.data (lhotse); every "micro-batch" is one packed row.
    "dataloader-type": "external",
    "micro-batch-size": 1,
    # Average the loss over all target tokens of the global batch, as HF Trainer does.
    "calculate-per-token-loss": True,
    "tokenizer-type": "NullTokenizer",
    # Tokenizer files come from the HF checkpoint at export time (scripts/export_hf.py).
    "no-save-tokenizer-assets": True,
    "no-create-attention-mask-in-dataloader": True,
}


def resolve_model_section(model_section: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Fill component configs from the bridge; values in the run config take precedence."""
    model_section = copy.deepcopy(model_section)
    bridge = None
    if model_section.get("bridge"):
        bridge = get_bridge_for_model_section(model_section)
        model_section = merge_dicts(bridge.speech_defaults(), model_section)
    for key in ("encoder", "audio_token_id"):
        if key not in model_section:
            raise ValueError(f"model.{key} is required (or set model.bridge/model.hf_init)")
    return model_section, bridge


def build_megatron_arg_dict(cfg: dict[str, Any], bridge) -> dict[str, Any]:
    user = {k.replace("_", "-"): v for k, v in cfg["megatron"].items()}
    arch = bridge.megatron_args() if bridge is not None else {}
    for key, value in arch.items():
        if key in user and user[key] != value:
            raise ValueError(
                f"megatron.{key}={user[key]!r} conflicts with the {bridge.name} checkpoint ({value!r})"
            )
    seq_length = cfg["data"].get("max_tokens_per_microbatch")
    derived = {"seq-length": seq_length} if seq_length else {}
    return {**DEFAULT_MEGATRON_ARGS, **derived, **arch, **user}


def build_speech_llm_model_config(args, model_section: dict[str, Any]) -> SpeechLLMModelConfig:
    transformer = compat.build_transformer_config_from_args(args)
    # Megatron's training loop reads args.padded_vocab_size (e.g. for FLOPs); the speech LLM
    # keeps the exact HF vocabulary so that weights convert one-to-one.
    args.padded_vocab_size = args.vocab_size
    return SpeechLLMModelConfig(
        transformer=transformer,
        seq_length=args.max_position_embeddings,
        vocab_size=args.vocab_size,
        should_pad_vocab=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_base=args.rotary_base,
        rotary_percent=args.rotary_percent,
        qk_layernorm=transformer.qk_layernorm,
        speech=model_section,
    )


class SpeechLLMRecipe(Recipe):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.model_section, self.bridge = resolve_model_section(cfg["model"])

    def megatron_arg_dict(self) -> dict[str, Any]:
        return build_megatron_arg_dict(self.cfg, self.bridge)

    def build_model_config(self, args):
        return build_speech_llm_model_config(args, self.model_section)

    def datasets_provider(self):
        from megatron_speech.data.provider import build_datasets_provider

        return build_datasets_provider(self.cfg, self.model_section)

    def forward_step(self):
        from megatron_speech.training.speech_llm_step import forward_step

        return forward_step

    def pretrain_kwargs(self):
        return {"build_tokenizer": False}

"""Generic speech LLM: audio encoder + projector + LLM, assembled on Megatron Core MIMO.

Module tree (names used by checkpoints and by the HF bridges)::

    MimoModel
    ├── modality_submodules.audio            AudioModalitySubmodules
    │   ├── encoders.encoder                 e.g. Qwen3ASRAudioEncoder
    │   └── input_projections.0              e.g. MLPProjector
    └── language_model                       SpeechLanguageModel (GPTModel)

MIMO replaces the audio placeholder tokens (``speech.audio_token_id``) in ``input_ids`` with
the projected encoder outputs, in order, and runs the LLM on the merged embeddings.

The run config describes the model under ``model:`` (see configs/qwen3_asr/*.yaml). A bridge
(``model.bridge``) fills the architecture from a Hugging Face checkpoint.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron_speech import compat
from megatron_speech.components.registry import build_encoder_spec, build_projector_spec
from megatron_speech.models.language_model import SpeechLanguageModel

AUDIO_MODALITY = "audio"
ENCODER_NAME = "encoder"

ModelConfig = compat.get_model_config_base()
ModelBuilder = compat.get_model_builder_base()


@dataclass(kw_only=True)
class SpeechLLMModelConfig(ModelConfig):
    """Typed model config consumed by Megatron's training loop (``cfg.model``)."""

    builder: ClassVar[str] = "megatron_speech.models.speech_llm.SpeechLLMBuilder"

    transformer: TransformerConfig
    """LLM architecture, precision and parallelism."""

    seq_length: int
    """LLM position capacity (``--max-position-embeddings``)."""

    vocab_size: int | None = None
    should_pad_vocab: bool = False
    make_vocab_size_divisible_by: int = 128
    share_embeddings_and_output_weights: bool = True
    position_embedding_type: str = "rope"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    fp16_lm_cross_entropy: bool = False
    logit_dtype: torch.dtype | None = None
    qk_layernorm: bool = False

    speech: dict[str, Any] = field(default_factory=dict)
    """``model:`` section of the run config (components, bridge, PEFT)."""


def build_language_model_spec(cfg: SpeechLLMModelConfig, pg_collection) -> ModuleSpec:
    """LLM backbone. Kept in one place so that a later move to HybridModel touches only this
    function and the LLM weight mapping."""
    layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=cfg.qk_layernorm)
    return ModuleSpec(
        module=SpeechLanguageModel,
        params={
            # Output layer and loss only at target positions (see models/language_model.py).
            "loss_on_targets_only": bool(cfg.speech.get("loss_on_targets_only", True)),
            "config": cfg.transformer,
            "transformer_layer_spec": layer_spec,
            "vocab_size": cfg.vocab_size,
            "max_sequence_length": cfg.seq_length,
            "pre_process": True,
            "post_process": True,
            "fp16_lm_cross_entropy": cfg.fp16_lm_cross_entropy,
            "logit_dtype": cfg.logit_dtype,
            "parallel_output": True,
            "share_embeddings_and_output_weights": cfg.share_embeddings_and_output_weights,
            "position_embedding_type": cfg.position_embedding_type,
            "rotary_percent": cfg.rotary_percent,
            "rotary_base": cfg.rotary_base,
            # MIMO scatters the merged embeddings itself (PartitionAdapter).
            "scatter_embedding_sequence_parallel": False,
            "pg_collection": pg_collection,
        },
    )


def build_audio_submodule_spec(speech: dict[str, Any], llm_config: TransformerConfig, pg_collection) -> ModuleSpec:
    submodules: dict[str, Any] = {
        "encoders": {ENCODER_NAME: build_encoder_spec(speech["encoder"], llm_config, pg_collection)}
    }
    if speech.get("projector") is not None:
        submodules["input_projections"] = [build_projector_spec(speech["projector"], llm_config, pg_collection)]
    return ModuleSpec(module=AudioModalitySubmodules, submodules=submodules)


def build_speech_llm(cfg: SpeechLLMModelConfig, pg_collection=None) -> MimoModel:
    if pg_collection is None:
        pg_collection = compat.get_default_pg_collection()
    if cfg.transformer.pipeline_model_parallel_size != 1:
        raise NotImplementedError(
            "speech_llm runs MIMO in colocated mode, which supports pipeline_model_parallel_size=1 only"
        )
    speech = cfg.speech
    mimo_config = MimoModelConfig(
        language_model_spec=build_language_model_spec(cfg, pg_collection),
        modality_submodules_spec={
            AUDIO_MODALITY: build_audio_submodule_spec(speech, cfg.transformer, pg_collection)
        },
        special_token_ids={AUDIO_MODALITY: int(speech["audio_token_id"])},
        kv_format="thd",
    )
    model = MimoModel(mimo_config, cp_group=pg_collection.cp, tp_group=pg_collection.tp)
    # Megatron's training loop looks up `pg_collection` on the (wrapped) model; MimoModel does
    # not define it, and without the bf16 wrapper nothing else does.
    model.pg_collection = pg_collection
    return model


class SpeechLLMBuilder(ModelBuilder):
    """Builds the speech LLM and loads Hugging Face weights before DDP wrapping."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        return build_speech_llm(self._model_config, pg_collection)

    def _pre_wrap_hooks(self) -> list[Callable]:
        from megatron_speech.bridges import load_hf_weights_hook

        hooks = list(self._model_config.pre_wrap_hooks)
        speech = self._model_config.speech
        if speech.get("hf_init"):
            hooks.append(load_hf_weights_hook(speech))
        return hooks

    def build_distributed_models(
        self,
        pg_collection,
        ddp_config=None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper=None,
        model_type=None,
        use_layer_wise_distributed_optimizer: bool = False,
        use_layer_wise_param_layout: bool = True,
    ):
        kwargs = dict(
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            # Weights come from the HF checkpoint on every rank; no broadcast needed.
            data_parallel_random_init=data_parallel_random_init,
            pre_wrap_hook=compat.compose_hooks(self._pre_wrap_hooks()),
            use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
            use_layer_wise_param_layout=use_layer_wise_param_layout,
        )
        if mixed_precision_wrapper is not None:
            kwargs["mixed_precision_wrapper"] = mixed_precision_wrapper
        if model_type is not None:
            kwargs["model_type"] = model_type
        models = compat.build_distributed_models(
            self.build_model, self._model_config.transformer, pg_collection, **kwargs
        )
        hook = compat.compose_hooks(self._model_config.post_wrap_hooks)
        out = hook(models)
        return out if out is not None else models


def make_speech_llm_model_config(transformer: TransformerConfig, model_section: dict[str, Any], **kwargs) -> SpeechLLMModelConfig:
    return SpeechLLMModelConfig(transformer=transformer, speech=copy.deepcopy(model_section), **kwargs)

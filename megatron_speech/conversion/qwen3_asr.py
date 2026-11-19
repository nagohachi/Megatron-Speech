from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge

from megatron_speech.components.encoders.qwen3_asr import NUM_CONVOLUTIONS
from megatron_speech.models.speech_llm import AUDIO_MODALITY, ENCODER_NAME

# Qwen3-ASR's encoder, with names relative to the encoder and to Hugging Face's audio tower.
# The convolutions and conv_out are not split over the tensor parallel ranks.
ENCODER_MAPPINGS = [
    *[
        ReplicatedMapping(f"conv2d{index}.{suffix}", f"conv2d{index}.{suffix}")
        for index in range(1, NUM_CONVOLUTIONS + 1)
        for suffix in ("weight", "bias")
    ],
    ReplicatedMapping("conv_out.weight", "conv_out.weight"),
    *[
        mapping
        for suffix in ("weight", "bias")
        for mapping in (
            QKVMapping(
                f"transformer.layers.*.self_attention.linear_qkv.{suffix}",
                q=f"layers.*.self_attn.q_proj.{suffix}",
                k=f"layers.*.self_attn.k_proj.{suffix}",
                v=f"layers.*.self_attn.v_proj.{suffix}",
            ),
            AutoMapping(
                f"transformer.layers.*.self_attention.linear_qkv.layer_norm_{suffix}",
                f"layers.*.self_attn_layer_norm.{suffix}",
            ),
            AutoMapping(
                f"transformer.layers.*.self_attention.linear_proj.{suffix}",
                f"layers.*.self_attn.out_proj.{suffix}",
            ),
            AutoMapping(
                f"transformer.layers.*.mlp.linear_fc1.layer_norm_{suffix}",
                f"layers.*.final_layer_norm.{suffix}",
            ),
            AutoMapping(f"transformer.layers.*.mlp.linear_fc1.{suffix}", f"layers.*.fc1.{suffix}"),
            AutoMapping(f"transformer.layers.*.mlp.linear_fc2.{suffix}", f"layers.*.fc2.{suffix}"),
            AutoMapping(f"transformer.final_layernorm.{suffix}", f"ln_post.{suffix}"),
        )
    ],
]

# Qwen3-ASR's MLP projector, with names relative to the projector and to Hugging Face's
# multi_modal_projector. Hugging Face's Qwen3-ASR projector has exactly two linear layers
# (linear_1, linear_2).
PROJECTOR_MAPPINGS = [
    AutoMapping(f"layers.{index}.{suffix}", f"linear_{index + 1}.{suffix}")
    for index in range(2)
    for suffix in ("weight", "bias")
]


def _rebase_mappings(
    mappings: list[MegatronParamMapping],
    megatron_prefix: str,
    hf_old_prefix: str,
    hf_new_prefix: str,
) -> list[MegatronParamMapping]:
    """Return copies of ``mappings`` with the names of a part inside a larger model.

    - Megatron names: ``megatron_prefix`` is added in front.
    - Hugging Face names: a leading ``hf_old_prefix`` becomes ``hf_new_prefix``; names without
      it stay, such as ``lm_head.weight``. An empty ``hf_old_prefix`` adds ``hf_new_prefix``.
    """

    def rebase_hf_name(name: str) -> str:
        """Replace the leading ``hf_old_prefix`` of ``name`` with ``hf_new_prefix``."""
        if not name.startswith(hf_old_prefix):
            return name
        return hf_new_prefix + name.removeprefix(hf_old_prefix)

    rebased = []
    for mapping in mappings:
        megatron_name = megatron_prefix + mapping.megatron_param
        # QKVMapping (q, k, v) and GatedMLPMapping (gate, up) hold several HF names.
        if isinstance(mapping.hf_param, dict):
            hf_names = {key: rebase_hf_name(name) for key, name in mapping.hf_param.items()}
            rebased.append(type(mapping)(megatron_name, **hf_names))
        else:
            rebased.append(type(mapping)(megatron_name, rebase_hf_name(mapping.hf_param)))
    return rebased


def build_qwen3_asr_mappings() -> list[MegatronParamMapping]:
    """Map the speech LLM built as Qwen3-ASR to Hugging Face's ``Qwen3ASRForConditionalGeneration``.

    - Encoder: ``ENCODER_MAPPINGS`` under ``model.audio_tower``.
    - Projector: ``PROJECTOR_MAPPINGS`` under ``model.multi_modal_projector``.
    - LLM: Megatron-Bridge's table of Qwen3, under ``model.language_model``.
    """
    audio = f"modality_submodules.{AUDIO_MODALITY}."
    return [
        *_rebase_mappings(
            ENCODER_MAPPINGS, f"{audio}encoders.{ENCODER_NAME}.", "", "model.audio_tower."
        ),
        *_rebase_mappings(
            PROJECTOR_MAPPINGS, f"{audio}input_projections.0.", "", "model.multi_modal_projector."
        ),
        *_rebase_mappings(
            Qwen3Bridge().mapping_registry().get_all_mappings(),
            "language_model.",
            "model.",
            "model.language_model.",
        ),
    ]

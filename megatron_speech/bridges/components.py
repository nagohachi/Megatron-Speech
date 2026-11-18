"""Weight mappings of individual components.

Each function returns the mappings of one component given the Megatron prefix of the component
inside the model and the HF prefix of the same component in the HF checkpoint, so that model
bridges (and user-composed models) can combine them.
"""

from __future__ import annotations

from megatron_speech.bridges.mapping import DirectMapping, GatedMLPMapping, Mapping, QKVMapping


def qwen3_llm_mappings(
    *,
    megatron_prefix: str,
    hf_prefix: str,
    num_layers: int,
    num_heads: int,
    num_query_groups: int,
    head_dim: int,
    tie_word_embeddings: bool,
    lm_head_name: str = "lm_head.weight",
) -> list[Mapping]:
    """Qwen3 (and Llama/Qwen2-like with QK-norm) decoder on Megatron GPTModel + TE layer spec."""
    m, h = megatron_prefix, hf_prefix
    out: list[Mapping] = [
        DirectMapping(f"{m}embedding.word_embeddings.weight", f"{h}embed_tokens.weight"),
        DirectMapping(f"{m}decoder.final_layernorm.weight", f"{h}norm.weight"),
    ]
    if not tie_word_embeddings:
        out.append(DirectMapping(f"{m}output_layer.weight", lm_head_name))
    for i in range(num_layers):
        ml, hl = f"{m}decoder.layers.{i}.", f"{h}layers.{i}."
        out += [
            DirectMapping(f"{ml}self_attention.linear_qkv.layer_norm_weight", f"{hl}input_layernorm.weight"),
            QKVMapping(
                f"{ml}self_attention.linear_qkv.weight",
                f"{hl}self_attn.q_proj.weight",
                f"{hl}self_attn.k_proj.weight",
                f"{hl}self_attn.v_proj.weight",
                num_heads,
                num_query_groups,
                head_dim,
            ),
            DirectMapping(f"{ml}self_attention.q_layernorm.weight", f"{hl}self_attn.q_norm.weight"),
            DirectMapping(f"{ml}self_attention.k_layernorm.weight", f"{hl}self_attn.k_norm.weight"),
            DirectMapping(f"{ml}self_attention.linear_proj.weight", f"{hl}self_attn.o_proj.weight"),
            DirectMapping(f"{ml}mlp.linear_fc1.layer_norm_weight", f"{hl}post_attention_layernorm.weight"),
            GatedMLPMapping(f"{ml}mlp.linear_fc1.weight", f"{hl}mlp.gate_proj.weight", f"{hl}mlp.up_proj.weight"),
            DirectMapping(f"{ml}mlp.linear_fc2.weight", f"{hl}mlp.down_proj.weight"),
        ]
    return out


def qwen3_asr_encoder_mappings(*, megatron_prefix: str, hf_prefix: str, num_layers: int, num_heads: int, d_model: int) -> list[Mapping]:
    """Qwen3-ASR / Qwen3-Omni audio encoder (``Qwen3ASREncoder``)."""
    m, h = megatron_prefix, hf_prefix
    out: list[Mapping] = []
    for conv in ("conv2d1", "conv2d2", "conv2d3"):
        out += [
            DirectMapping(f"{m}{conv}.weight", f"{h}{conv}.weight"),
            DirectMapping(f"{m}{conv}.bias", f"{h}{conv}.bias"),
        ]
    out.append(DirectMapping(f"{m}conv_out.weight", f"{h}conv_out.weight"))
    head_dim = d_model // num_heads
    for i in range(num_layers):
        ml, hl = f"{m}transformer.layers.{i}.", f"{h}layers.{i}."
        for suffix in ("weight", "bias"):
            out += [
                DirectMapping(f"{ml}self_attention.linear_qkv.layer_norm_{suffix}", f"{hl}self_attn_layer_norm.{suffix}"),
                QKVMapping(
                    f"{ml}self_attention.linear_qkv.{suffix}",
                    f"{hl}self_attn.q_proj.{suffix}",
                    f"{hl}self_attn.k_proj.{suffix}",
                    f"{hl}self_attn.v_proj.{suffix}",
                    num_heads,
                    num_heads,
                    head_dim,
                ),
                DirectMapping(f"{ml}self_attention.linear_proj.{suffix}", f"{hl}self_attn.out_proj.{suffix}"),
                DirectMapping(f"{ml}mlp.linear_fc1.layer_norm_{suffix}", f"{hl}final_layer_norm.{suffix}"),
                DirectMapping(f"{ml}mlp.linear_fc1.{suffix}", f"{hl}fc1.{suffix}"),
                DirectMapping(f"{ml}mlp.linear_fc2.{suffix}", f"{hl}fc2.{suffix}"),
            ]
    out += [
        DirectMapping(f"{m}transformer.final_layernorm.weight", f"{h}ln_post.weight"),
        DirectMapping(f"{m}transformer.final_layernorm.bias", f"{h}ln_post.bias"),
    ]
    return out


def mlp_projector_mappings(
    *, megatron_prefix: str, hf_prefix: str, fc1: str = "linear_1", fc2: str = "linear_2", bias: bool = True
) -> list[Mapping]:
    """``MLPProjector`` <-> an HF two-layer MLP (e.g. ``Qwen3ASRMultiModalProjector``)."""
    m, h = f"{megatron_prefix}projector.encoder.", hf_prefix
    suffixes = ("weight", "bias") if bias else ("weight",)
    out: list[Mapping] = []
    for suffix in suffixes:
        out += [
            DirectMapping(f"{m}linear_fc1.{suffix}", f"{h}{fc1}.{suffix}"),
            DirectMapping(f"{m}linear_fc2.{suffix}", f"{h}{fc2}.{suffix}"),
        ]
    return out

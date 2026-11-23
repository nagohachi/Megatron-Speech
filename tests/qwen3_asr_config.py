import torch
from megatron.core.transformer import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig
from transformers import Qwen3ASRConfig

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.components.projectors.mlp import MLPProjectorConfig
from megatron_speech.models.speech_llm import SpeechLLMConfig


def build_speech_llm_config(
    hf_config: Qwen3ASRConfig, dtype: torch.dtype, tensor_parallel_size: int
) -> SpeechLLMConfig:
    """Build the speech LLM config of the shape of ``hf_config``, with ``dtype`` parameters."""
    audio_config, text_config = hf_config.audio_config, hf_config.text_config
    transformer = TransformerConfig(
        num_layers=text_config.num_hidden_layers,
        hidden_size=text_config.hidden_size,
        num_attention_heads=text_config.num_attention_heads,
        num_query_groups=text_config.num_key_value_heads,
        kv_channels=text_config.head_dim,
        ffn_hidden_size=text_config.intermediate_size,
        normalization="RMSNorm",
        layernorm_epsilon=text_config.rms_norm_eps,
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    return SpeechLLMConfig(
        llm=GPTModelConfig(
            transformer=transformer,
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
            position_embedding_type="rope",
            rotary_base=text_config.rope_parameters["rope_theta"],
        ),
        encoder=Qwen3ASREncoderConfig(
            num_mel_bins=audio_config.num_mel_bins,
            d_model=audio_config.d_model,
            encoder_layers=audio_config.encoder_layers,
            encoder_attention_heads=audio_config.encoder_attention_heads,
            encoder_ffn_dim=audio_config.encoder_ffn_dim,
            downsample_hidden_size=audio_config.downsample_hidden_size,
            n_window=audio_config.n_window,
            n_window_infer=audio_config.n_window_infer,
            activation_function=audio_config.activation_function,
        ),
        projector=MLPProjectorConfig(
            input_size=audio_config.d_model,
            output_size=audio_config.output_dim,
            hidden_size=audio_config.d_model,
            activation=audio_config.activation_function,
        ),
        audio_token_id=hf_config.audio_token_id,
    )

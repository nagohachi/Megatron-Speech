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
    audio, text = hf_config.audio_config, hf_config.text_config

    def build_llm_config() -> GPTModelConfig:
        """Build the config of the LLM of the shape of ``text``."""
        transformer = TransformerConfig(
            num_layers=text.num_hidden_layers,
            hidden_size=text.hidden_size,
            num_attention_heads=text.num_attention_heads,
            num_query_groups=text.num_key_value_heads,
            kv_channels=text.head_dim,
            ffn_hidden_size=text.intermediate_size,
            normalization="RMSNorm",
            layernorm_epsilon=text.rms_norm_eps,
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
        return GPTModelConfig(
            transformer=transformer,
            vocab_size=text.vocab_size,
            seq_length=text.max_position_embeddings,
            share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
            position_embedding_type="rope",
            rotary_base=text.rope_parameters["rope_theta"],
        )

    return SpeechLLMConfig(
        llm=build_llm_config(),
        encoder=Qwen3ASREncoderConfig(
            num_mel_bins=audio.num_mel_bins,
            d_model=audio.d_model,
            encoder_layers=audio.encoder_layers,
            encoder_attention_heads=audio.encoder_attention_heads,
            encoder_ffn_dim=audio.encoder_ffn_dim,
            downsample_hidden_size=audio.downsample_hidden_size,
            n_window=audio.n_window,
            n_window_infer=audio.n_window_infer,
            activation_function=audio.activation_function,
        ),
        projector=MLPProjectorConfig(
            input_size=audio.d_model,
            output_size=audio.output_dim,
            hidden_size=audio.d_model,
            activation=audio.activation_function,
        ),
        audio_token_id=hf_config.audio_token_id,
    )

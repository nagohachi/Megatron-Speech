import torch
from megatron.core.transformer import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoderConfig
from megatron_speech.components.projectors.mlp import MLPProjectorConfig
from megatron_speech.models.speech_llm import SpeechLLMConfig

VOCAB_SIZE = 128
AUDIO_TOKEN_ID = VOCAB_SIZE - 1
ENCODER_CONFIG = Qwen3ASREncoderConfig(
    num_mel_bins=32,
    d_model=32,
    encoder_layers=1,
    encoder_attention_heads=4,
    encoder_ffn_dim=64,
    downsample_hidden_size=8,
    n_window_infer=200,
)


def build_small_speech_llm_config(dtype: torch.dtype, tensor_parallel_size: int) -> SpeechLLMConfig:
    """Build the config of the small speech LLM that the speech LLM tests are run with.

    The LLM has two layers and rotary positions, and the encoder has one layer.
    """
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return SpeechLLMConfig(
        llm_config=GPTModelConfig(
            transformer=transformer_config,
            vocab_size=VOCAB_SIZE,
            seq_length=64,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
        ),
        encoder_config=ENCODER_CONFIG,
        projector_config=MLPProjectorConfig(input_size=32, output_size=64, hidden_size=64),
        audio_token_id=AUDIO_TOKEN_ID,
    )

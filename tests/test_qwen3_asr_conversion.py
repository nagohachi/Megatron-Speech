import logging
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.models.mimo import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

from megatron_speech.components.encoders.qwen3_asr import (
    Qwen3ASREncoderConfig,
    Qwen3ASREncoderInputs,
)
from megatron_speech.components.projectors.mlp import MLPProjectorConfig
from megatron_speech.conversion.bridge import MappingBridge
from megatron_speech.conversion.qwen3_asr import build_qwen3_asr_mappings
from megatron_speech.models.speech_llm import (
    AUDIO_MODALITY,
    ENCODER_NAME,
    SpeechLLMConfig,
    build_speech_llm,
)
from tests.hf_audio_inputs import build_hf_audio_inputs
from tests.hf_checkpoint import check_export_restores_hf_weights, save_hf_checkpoint
from tests.numerical_error import TOLERANCES, calc_relative_error

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)

HF_CONFIG = Qwen3ASRConfig(
    audio_config={
        "num_mel_bins": 32,
        "d_model": 32,
        "encoder_layers": 1,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 64,
        "downsample_hidden_size": 8,
        "output_dim": 64,
        "n_window_infer": 200,
    },
    text_config={
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
    },
    audio_token_id=127,
)
SEQUENCE_LENGTH = 16
# Mel frames of the utterances: 33 encoder frames in two attention windows (26 and 7), and 5.
MEL_LENGTHS = [250, 37]
BRIDGE = MappingBridge(build_qwen3_asr_mappings())


def build_speech_llm_config(dtype: torch.dtype, tensor_parallel_size: int) -> SpeechLLMConfig:
    """Build the speech LLM config of the shape of ``HF_CONFIG``, with ``dtype`` parameters."""
    audio, text = HF_CONFIG.audio_config, HF_CONFIG.text_config

    def build_transformer_config() -> TransformerConfig:
        """Build the LLM's layer config: Qwen3's RMSNorm, query and key norms, gated SiLU MLP."""
        return TransformerConfig(
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

    def build_encoder_config() -> Qwen3ASREncoderConfig:
        """Build the encoder config from the fields of the same names in the audio config."""
        return Qwen3ASREncoderConfig(
            num_mel_bins=audio.num_mel_bins,
            d_model=audio.d_model,
            encoder_layers=audio.encoder_layers,
            encoder_attention_heads=audio.encoder_attention_heads,
            encoder_ffn_dim=audio.encoder_ffn_dim,
            downsample_hidden_size=audio.downsample_hidden_size,
            n_window=audio.n_window,
            n_window_infer=audio.n_window_infer,
            activation_function=audio.activation_function,
        )

    return SpeechLLMConfig(
        llm=GPTModelConfig(
            transformer=build_transformer_config(),
            vocab_size=text.vocab_size,
            seq_length=text.max_position_embeddings,
            share_embeddings_and_output_weights=HF_CONFIG.tie_word_embeddings,
            position_embedding_type="rope",
            rotary_base=text.rope_parameters["rope_theta"],
        ),
        encoder=build_encoder_config(),
        projector=MLPProjectorConfig(
            input_size=audio.d_model,
            output_size=audio.output_dim,
            hidden_size=audio.d_model,
            activation=audio.activation_function,
        ),
        audio_token_id=HF_CONFIG.audio_token_id,
    )


@pytest.fixture(name="hf_dir")
def build_hf_checkpoint(tmp_path: Path, tensor_parallel_size: int) -> Path:
    """Save a Qwen3-ASR with random weights as a Hugging Face checkpoint, the same on every rank.

    ``tensor_parallel_size`` makes torch.distributed start first.
    """
    torch.manual_seed(0)
    model = Qwen3ASRForConditionalGeneration(HF_CONFIG)
    with torch.no_grad():  # norms away from (1, 0), so that a missed copy shows
        for name, param in model.named_parameters():
            if "norm" not in name and "ln_post" not in name:
                continue
            param.normal_(mean=1.0 if name.endswith("weight") else 0.0, std=0.1)
    return save_hf_checkpoint(model, tmp_path)


@pytest.fixture(name="reference")
def build_reference(hf_dir: Path) -> Qwen3ASRForConditionalGeneration:
    """Load the Hugging Face Qwen3-ASR in fp32."""
    return Qwen3ASRForConditionalGeneration.from_pretrained(hf_dir, dtype=torch.float32).cuda()


@pytest.fixture(name="model")
def build_model(hf_dir: Path, dtype: torch.dtype, tensor_parallel_size: int) -> MimoModel:
    """Build the speech LLM in ``dtype`` and load the Hugging Face weights into it."""
    config = build_speech_llm_config(dtype, tensor_parallel_size)
    model = build_speech_llm(config, ProcessGroupCollection.use_mpu_process_groups()).cuda()
    BRIDGE.load_weights_hf_to_megatron(PreTrainedCausalLM.from_pretrained(hf_dir), [model])
    return model


def test_export_restores_hf_weights(model: MimoModel, hf_dir: Path, dtype: torch.dtype) -> None:
    """Check that writing the loaded weights back gives every Hugging Face weight exactly."""
    check_export_restores_hf_weights(BRIDGE, model, hf_dir, dtype)


def test_text_logits_match_hf(
    model: MimoModel, reference: Qwen3ASRForConditionalGeneration, dtype: torch.dtype
) -> None:
    """Check that a text-only sequence gives the logits of Hugging Face's Qwen3-ASR."""
    torch.manual_seed(1)
    input_ids = torch.randint(HF_CONFIG.audio_token_id, (1, SEQUENCE_LENGTH), device="cuda")
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        logits, _ = model(
            input_ids=input_ids,
            position_ids=torch.arange(SEQUENCE_LENGTH, device="cuda")[None],
            cu_seqlens=torch.tensor([0, SEQUENCE_LENGTH], dtype=torch.int32, device="cuda"),
            max_seqlen=torch.tensor(SEQUENCE_LENGTH),
        )
        # The logits are split over the TP ranks along the vocabulary.
        actual = gather_from_tensor_model_parallel_region(logits)
    error = calc_relative_error(actual, expected)
    logger.info("%s: text logits max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_audio_features_match_hf(
    model: MimoModel,
    reference: Qwen3ASRForConditionalGeneration,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> None:
    """Check that the encoder and the projector give Hugging Face's audio features."""
    encoder_config = build_speech_llm_config(dtype, tensor_parallel_size).encoder
    torch.manual_seed(1)
    mels = [torch.randn(encoder_config.num_mel_bins, length) for length in MEL_LENGTHS]
    encoder_inputs = Qwen3ASREncoderInputs.build(mels, encoder_config).to("cuda")
    with torch.no_grad():
        expected = reference.get_audio_features(
            **build_hf_audio_inputs(mels, encoder_config)
        ).pooler_output
        actual = model.modality_submodules[AUDIO_MODALITY](
            encoder_inputs={
                ENCODER_NAME: {
                    field.name: getattr(encoder_inputs, field.name)
                    for field in fields(encoder_inputs)
                }
            }
        )
    assert actual.shape == expected.shape
    error = calc_relative_error(actual, expected)
    logger.info("%s: audio features max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]

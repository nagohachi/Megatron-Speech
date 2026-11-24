import logging
from pathlib import Path

import pytest
import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from megatron_speech.conversion.bridge import MappingBridge, SafeTensorsCheckpoint
from megatron_speech.models.language_model import SpeechLanguageModel
from tests.hf_reference import (
    check_export_restores_hf_weights,
    randomize_norms,
    save_hf_checkpoint,
)
from tests.numerical_error import TOLERANCES, WITH_DTYPES, calc_relative_error

pytestmark = [
    pytest.mark.gpu,
    WITH_DTYPES,
    # Which checkpoint the model loads: the Hugging Face one, or the same weights in a
    # checkpoint that transformers cannot load.
    pytest.mark.parametrize("checkpoint_format", ["hf", "safetensors"]),
]
logger = logging.getLogger(__name__)

HF_CONFIG = Qwen3Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    tie_word_embeddings=True,
)
SEQUENCE_LENGTH = 16
# Megatron-Bridge's mapping table of Qwen3.
BRIDGE = MappingBridge(Qwen3Bridge().mapping_registry().get_all_mappings())
# Weights that the model does not hold, as those of a codec in the same checkpoint, by the
# file that holds them: one in the file of the model's weights and one in a file of its own.
CODEC_WEIGHT_NAMES = {
    "model.safetensors": "codec.encoder.weight",
    "codec.safetensors": "codec.decoder.weight",
}
# A file besides the weights, as a tokenizer's.
TOKENIZER_FILE = "tokenizer.model"


@pytest.fixture(name="hf_dir")
def build_hf_checkpoint(tmp_path: Path, tensor_parallel_size: int) -> Path:
    """Save a Qwen3 with random weights as a Hugging Face checkpoint, the same on every rank.

    ``tensor_parallel_size`` makes torch.distributed start first.
    """
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(HF_CONFIG)
    randomize_norms(model, ("norm",))
    return save_hf_checkpoint(model, tmp_path)


@pytest.fixture(name="safetensors_dir")
def build_safetensors_checkpoint(hf_dir: Path) -> Path:
    """Write the weights of ``hf_dir`` as a checkpoint that transformers cannot load.

    The checkpoint has no config.json. Its safetensors files hold the weights of ``hf_dir`` and
    the weights of ``CODEC_WEIGHT_NAMES``, and ``TOKENIZER_FILE`` holds random bytes.
    """
    checkpoint_dir = hf_dir / "safetensors"
    if torch.distributed.get_rank() == 0:
        checkpoint_dir.mkdir()
        torch.manual_seed(2)
        weights_by_file = {"model.safetensors": load_file(hf_dir / "model.safetensors")}
        for file_name, weight_name in CODEC_WEIGHT_NAMES.items():
            weights_by_file.setdefault(file_name, {})[weight_name] = torch.randn(4, 4)
        for file_name, weights in weights_by_file.items():
            save_file(weights, checkpoint_dir / file_name)
        tokenizer_bytes = torch.randint(256, (64,), dtype=torch.uint8).numpy().tobytes()
        (checkpoint_dir / TOKENIZER_FILE).write_bytes(tokenizer_bytes)
    torch.distributed.barrier()
    return checkpoint_dir


@pytest.fixture(name="model")
def build_model(
    hf_dir: Path,
    safetensors_dir: Path,
    checkpoint_format: str,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> SpeechLanguageModel:
    """Build a SpeechLanguageModel of the reference's shape and load the weights.

    The weights come from ``hf_dir`` or ``safetensors_dir``, as ``checkpoint_format`` says.
    """
    config = TransformerConfig(
        num_layers=HF_CONFIG.num_hidden_layers,
        hidden_size=HF_CONFIG.hidden_size,
        num_attention_heads=HF_CONFIG.num_attention_heads,
        num_query_groups=HF_CONFIG.num_key_value_heads,
        kv_channels=HF_CONFIG.head_dim,
        ffn_hidden_size=HF_CONFIG.intermediate_size,
        normalization="RMSNorm",
        layernorm_epsilon=HF_CONFIG.rms_norm_eps,
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
    model = SpeechLanguageModel(
        config,
        get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        HF_CONFIG.vocab_size,
        HF_CONFIG.max_position_embeddings,
        position_embedding_type="rope",
        rotary_base=HF_CONFIG.rope_parameters["rope_theta"],
        share_embeddings_and_output_weights=HF_CONFIG.tie_word_embeddings,
    ).cuda()
    checkpoint = (
        PreTrainedCausalLM.from_pretrained(hf_dir)
        if checkpoint_format == "hf"
        else SafeTensorsCheckpoint(safetensors_dir)
    )
    BRIDGE.load_weights_hf_to_megatron(checkpoint, [model])
    return model


def test_logits_match_hf(
    model: SpeechLanguageModel, hf_dir: Path, dtype: torch.dtype, checkpoint_format: str
) -> None:
    """Check that the model with the loaded weights gives the Hugging Face model's logits."""
    reference = Qwen3ForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).cuda().eval()
    torch.manual_seed(1)
    input_ids = torch.randint(HF_CONFIG.vocab_size, (1, SEQUENCE_LENGTH), device="cuda")
    position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda")[None]
    with torch.no_grad():
        expected = reference(input_ids).logits
        # The logits are split over the TP ranks along the vocabulary.
        actual = gather_from_tensor_model_parallel_region(model(input_ids, position_ids, None))
    error = calc_relative_error(actual, expected)
    logger.info("%s, %s: logits max relative error %.2e", dtype, checkpoint_format, error)
    assert error < TOLERANCES[dtype]


def test_export_restores_hf_weights(
    model: SpeechLanguageModel, hf_dir: Path, dtype: torch.dtype
) -> None:
    """Check that writing the loaded weights back gives every Hugging Face weight exactly."""
    check_export_restores_hf_weights(BRIDGE, model, hf_dir, dtype)


def test_save_restores_safetensors_checkpoint(
    model: SpeechLanguageModel, safetensors_dir: Path, dtype: torch.dtype
) -> None:
    """Check that saving the model in the layout of ``safetensors_dir`` writes back every file.

    The model's weights come back as the source weights rounded to ``dtype``; the codec weights
    and the tokenizer file come back unchanged, and no config.json is added.
    """
    saved_dir = safetensors_dir.with_name("saved")
    BRIDGE.save_safetensors_checkpoint([model], SafeTensorsCheckpoint(safetensors_dir), saved_dir)
    assert sorted(path.name for path in saved_dir.iterdir()) == sorted(
        path.name for path in safetensors_dir.iterdir()
    )
    tokenizer_bytes = (saved_dir / TOKENIZER_FILE).read_bytes()
    assert tokenizer_bytes == (safetensors_dir / TOKENIZER_FILE).read_bytes()
    codec_weight_names = set(CODEC_WEIGHT_NAMES.values())
    for file_name in CODEC_WEIGHT_NAMES:
        source_weights = load_file(safetensors_dir / file_name)
        saved_weights = load_file(saved_dir / file_name)
        assert saved_weights.keys() == source_weights.keys(), file_name
        for name, tensor in saved_weights.items():
            expected = source_weights[name]
            if name not in codec_weight_names:
                expected = expected.to(dtype)
            assert tensor.dtype == expected.dtype, name
            assert torch.equal(tensor, expected), name

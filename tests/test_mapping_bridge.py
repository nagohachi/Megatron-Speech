import logging
from pathlib import Path

import pytest
import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig
from transformers import Qwen3Config, Qwen3ForCausalLM

from megatron_speech.conversion.bridge import MappingBridge
from megatron_speech.models.language_model import SpeechLanguageModel
from tests.numerical_error import TOLERANCES, calc_relative_error

pytestmark = pytest.mark.gpu
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
NUM_TOKENS = 16


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the model under test."""
    return request.param


@pytest.fixture(name="hf_dir")
def build_hf_checkpoint(tmp_path: Path, tensor_parallel_size: int) -> Path:
    """Save a Qwen3 with random weights as a Hugging Face checkpoint that every rank reads.

    Under torch.distributed, ``save_pretrained`` writes only on rank 0, so every rank takes
    rank 0's directory. ``tensor_parallel_size`` makes torch.distributed start first.
    """
    paths = [tmp_path]
    torch.distributed.broadcast_object_list(paths, src=0)
    if torch.distributed.get_rank() == 0:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(HF_CONFIG)
        with torch.no_grad():  # RMSNorm weights away from 1, so that a missed copy shows
            for name, param in model.named_parameters():
                if "norm" in name:
                    param.normal_(mean=1.0, std=0.1)
        model.save_pretrained(paths[0])
    torch.distributed.barrier()
    return paths[0]


@pytest.fixture(name="reference")
def build_reference(hf_dir: Path) -> Qwen3ForCausalLM:
    """Load the Hugging Face Qwen3 in fp32."""
    return Qwen3ForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).cuda().eval()


@pytest.fixture(name="bridge")
def build_bridge() -> MappingBridge:
    """Build a bridge with Megatron-Bridge's mapping table of Qwen3."""
    return MappingBridge(Qwen3Bridge().mapping_registry().get_all_mappings())


@pytest.fixture(name="model")
def build_model(
    hf_dir: Path, bridge: MappingBridge, dtype: torch.dtype, tensor_parallel_size: int
) -> SpeechLanguageModel:
    """Build a SpeechLanguageModel of the reference's shape and load the Hugging Face weights."""
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
    bridge.load_weights_hf_to_megatron(PreTrainedCausalLM.from_pretrained(hf_dir), [model])
    return model


def test_logits_match_hf(
    model: SpeechLanguageModel, reference: Qwen3ForCausalLM, dtype: torch.dtype
) -> None:
    """Check that the model with the loaded weights gives the Hugging Face model's logits."""
    torch.manual_seed(1)
    input_ids = torch.randint(HF_CONFIG.vocab_size, (1, NUM_TOKENS), device="cuda")
    position_ids = torch.arange(NUM_TOKENS, device="cuda")[None]
    with torch.no_grad():
        expected = reference(input_ids).logits
        # The logits are split over the TP ranks along the vocabulary.
        actual = gather_from_tensor_model_parallel_region(model(input_ids, position_ids, None))
    error = calc_relative_error(actual, expected)
    logger.info("%s: logits max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


def test_export_restores_hf_weights(
    model: SpeechLanguageModel, bridge: MappingBridge, hf_dir: Path, dtype: torch.dtype
) -> None:
    """Check that writing the loaded weights back gives every Hugging Face weight exactly.

    A bf16 model holds the fp32 Hugging Face weights rounded to bf16, so its export is compared
    with the rounded weights.
    """
    hf_pretrained = PreTrainedCausalLM.from_pretrained(hf_dir)
    exported = dict(
        bridge.stream_weights_megatron_to_hf([model], hf_pretrained, show_progress=False)
    )
    if torch.distributed.get_rank() == 0:
        assert exported.keys() == set(hf_pretrained.state.keys())
        for name, tensor in exported.items():
            assert tensor.dtype == dtype, name
            assert torch.equal(tensor, hf_pretrained.state[name].to(dtype)), name

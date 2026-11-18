import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from transformers.models.qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASREncoderConfig as HFEncoderConfig,
)
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder as HFEncoder
from transformers.models.qwen3_asr.modeling_qwen3_asr import get_audio_cu_seqlens

from megatron_speech.components.encoders.qwen3_asr import (
    Qwen3ASREncoder,
    Qwen3ASREncoderConfig,
    Qwen3ASREncoderInputs,
)
from tests.comparison import calc_relative_error, take_shard

logger = logging.getLogger(__name__)

# Largest relative error allowed against the fp32 reference, by parameter dtype.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 5e-2}
# Several chunks and windows, a short last chunk, one chunk, a partial chunk and one frame.
UTTERANCE_FRAMES = [250, 100, 37, 1]
# Windows of 200 frames (26 tokens) so that the 250-frame utterance spans two of them.
ENCODER_CONFIG = Qwen3ASREncoderConfig(
    num_mel_bins=32,
    hidden_size=64,
    num_layers=2,
    num_attention_heads=4,
    ffn_hidden_size=128,
    downsample_hidden_size=16,
    n_window_infer=200,
)


@dataclass
class WeightPair:
    """One of our parameters, the reference parameters it is made of, and its TP split dim."""

    ours: torch.nn.Parameter
    references: list[torch.nn.Parameter]
    combine: Callable[[list[torch.Tensor]], torch.Tensor]
    split_dim: int | None


def take_first(tensors: list[torch.Tensor]) -> torch.Tensor:
    """The only tensor of a one-to-one weight pair."""
    return tensors[0]


def interleave_qkv(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate q, k and v head by head, as Megatron's fused ``linear_qkv`` stores them."""
    heads = [t.reshape(ENCODER_CONFIG.num_attention_heads, -1, *t.shape[1:]) for t in tensors]
    return torch.cat(heads, dim=1).reshape(-1, *tensors[0].shape[1:])


def list_weight_pairs(encoder: Qwen3ASREncoder, reference: HFEncoder) -> list[WeightPair]:
    """Pair every parameter of ``encoder`` with the reference parameters it is made of."""
    pairs = [
        WeightPair(
            getattr(encoder, name).weight, [getattr(reference, name).weight], take_first, None
        )
        for name in ("conv2d1", "conv2d2", "conv2d3", "conv_out")
    ] + [
        WeightPair(getattr(encoder, name).bias, [getattr(reference, name).bias], take_first, None)
        for name in ("conv2d1", "conv2d2", "conv2d3")
    ]
    for ours, theirs in zip(encoder.transformer.layers, reference.layers, strict=True):
        attention, mlp, hf_attention = ours.self_attention, ours.mlp, theirs.self_attn
        qkv = [hf_attention.q_proj, hf_attention.k_proj, hf_attention.v_proj]
        pairs += [
            WeightPair(attention.linear_qkv.weight, [p.weight for p in qkv], interleave_qkv, 0),
            WeightPair(attention.linear_qkv.bias, [p.bias for p in qkv], interleave_qkv, 0),
            WeightPair(
                attention.linear_qkv.layer_norm_weight,
                [theirs.self_attn_layer_norm.weight],
                take_first,
                None,
            ),
            WeightPair(
                attention.linear_qkv.layer_norm_bias,
                [theirs.self_attn_layer_norm.bias],
                take_first,
                None,
            ),
            WeightPair(attention.linear_proj.weight, [hf_attention.out_proj.weight], take_first, 1),
            WeightPair(attention.linear_proj.bias, [hf_attention.out_proj.bias], take_first, None),
            WeightPair(mlp.linear_fc1.weight, [theirs.fc1.weight], take_first, 0),
            WeightPair(mlp.linear_fc1.bias, [theirs.fc1.bias], take_first, 0),
            WeightPair(
                mlp.linear_fc1.layer_norm_weight, [theirs.final_layer_norm.weight], take_first, None
            ),
            WeightPair(
                mlp.linear_fc1.layer_norm_bias, [theirs.final_layer_norm.bias], take_first, None
            ),
            WeightPair(mlp.linear_fc2.weight, [theirs.fc2.weight], take_first, 1),
            WeightPair(mlp.linear_fc2.bias, [theirs.fc2.bias], take_first, None),
        ]
    final_norm = encoder.transformer.final_layernorm
    pairs += [
        WeightPair(final_norm.weight, [reference.ln_post.weight], take_first, None),
        WeightPair(final_norm.bias, [reference.ln_post.bias], take_first, None),
    ]
    return pairs


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Parameter dtype of the encoder under test."""
    return request.param


@pytest.fixture(name="reference")
def build_reference() -> HFEncoder:
    """Hugging Face encoder in fp32 with random weights, the same on every rank."""
    torch.manual_seed(0)
    hf_config = HFEncoderConfig(
        num_mel_bins=ENCODER_CONFIG.num_mel_bins,
        d_model=ENCODER_CONFIG.hidden_size,
        encoder_layers=ENCODER_CONFIG.num_layers,
        encoder_attention_heads=ENCODER_CONFIG.num_attention_heads,
        encoder_ffn_dim=ENCODER_CONFIG.ffn_hidden_size,
        downsample_hidden_size=ENCODER_CONFIG.downsample_hidden_size,
        n_window=ENCODER_CONFIG.n_window,
        n_window_infer=ENCODER_CONFIG.n_window_infer,
    )
    hf_config._attn_implementation = "eager"
    reference = HFEncoder(hf_config).cuda()
    with torch.no_grad():  # LayerNorms away from (1, 0), so that a missed copy shows
        for name, param in reference.named_parameters():
            if "layer_norm" in name or "ln_post" in name:
                param.normal_(mean=1.0 if name.endswith("weight") else 0.0, std=0.1)
    return reference


@pytest.fixture(name="encoder")
def build_encoder(
    reference: HFEncoder, dtype: torch.dtype, tensor_parallel_size: int
) -> Qwen3ASREncoder:
    """Qwen3ASREncoder that holds this rank's shards of the reference weights."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    encoder = Qwen3ASREncoder(ENCODER_CONFIG, llm_config, pg_collection).cuda()
    with torch.no_grad():
        for pair in list_weight_pairs(encoder, reference):
            value = take_shard(pair.combine(pair.references), pair.split_dim)
            assert pair.ours.shape == value.shape
            pair.ours.copy_(value)
    return encoder


@pytest.fixture(name="mels")
def build_mels() -> list[torch.Tensor]:
    """fp32 log-mels of UTTERANCE_FRAMES frames, the same on every rank."""
    torch.manual_seed(1)
    return [torch.randn(ENCODER_CONFIG.num_mel_bins, frames) for frames in UTTERANCE_FRAMES]


def run_reference(reference: HFEncoder, mels: list[torch.Tensor]) -> torch.Tensor:
    """Pad the log-mels to a multiple of the chunk length with zeros and run Hugging Face's."""
    chunk_frames = ENCODER_CONFIG.chunk_frames
    padded_frames = -(-max(UTTERANCE_FRAMES) // chunk_frames) * chunk_frames
    features = torch.zeros(len(mels), ENCODER_CONFIG.num_mel_bins, padded_frames)
    mask = torch.zeros(len(mels), padded_frames, dtype=torch.long)
    for index, mel in enumerate(mels):
        features[index, :, : mel.shape[1]] = mel
        mask[index, : mel.shape[1]] = 1
    return reference(
        input_features=features.cuda(), input_features_mask=mask.cuda()
    ).last_hidden_state


def test_windows_match_hf() -> None:
    """The attention windows equal the ones Hugging Face computes for the padded batch."""
    inputs = Qwen3ASREncoderInputs.build(
        [torch.zeros(1, n) for n in UTTERANCE_FRAMES], ENCODER_CONFIG
    )
    chunk_frames = ENCODER_CONFIG.chunk_frames
    chunk_lengths = [
        min(chunk_frames, n - start)
        for n in UTTERANCE_FRAMES
        for start in range(0, n, chunk_frames)
    ]
    expected = get_audio_cu_seqlens(
        torch.tensor(chunk_lengths),
        torch.tensor(UTTERANCE_FRAMES),
        ENCODER_CONFIG.n_window_infer,
        ENCODER_CONFIG.n_window,
    )
    assert inputs.cu_seqlens.tolist() == expected.tolist()
    assert inputs.input_features.shape == (len(chunk_lengths), 1, chunk_frames)


@pytest.mark.gpu
def test_output_matches_hf(
    encoder: Qwen3ASREncoder, reference: HFEncoder, mels: list[torch.Tensor], dtype: torch.dtype
) -> None:
    """The encoder output equals Hugging Face's for the same utterances."""
    inputs = Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG).to("cuda")
    with torch.no_grad():
        expected = run_reference(reference, mels)
        actual = encoder(
            inputs.input_features, inputs.token_index, inputs.cu_seqlens, inputs.max_seqlen
        )
    error = calc_relative_error(actual, expected)
    logger.info("%s: output max relative error %.2e", dtype, error)
    assert actual.shape == expected.shape
    assert error < TOLERANCES[dtype]


@pytest.mark.gpu
def test_gradients_match_hf(
    encoder: Qwen3ASREncoder, reference: HFEncoder, mels: list[torch.Tensor], dtype: torch.dtype
) -> None:
    """Gradients of every weight equal the reference ones, combined and split as the weights."""
    inputs = Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG).to("cuda")
    expected = run_reference(reference, mels)
    actual = encoder(
        inputs.input_features, inputs.token_index, inputs.cu_seqlens, inputs.max_seqlen
    )
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    errors = [
        calc_relative_error(
            pair.ours.grad,
            take_shard(pair.combine([p.grad for p in pair.references]), pair.split_dim),
        )
        for pair in list_weight_pairs(encoder, reference)
    ]
    logger.info("%s: weight gradient max relative error %.2e", dtype, max(errors))
    assert max(errors) < TOLERANCES[dtype]


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_attention_heads", 3), ("n_window_infer", 150)],
    ids=["hidden-not-multiple-of-heads", "window-not-multiple-of-chunk"],
)
def test_rejects_invalid_config(field: str, value: int) -> None:
    """Head counts that do not divide hidden_size and windows that cut chunks raise ValueError."""
    llm_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=1)
    with pytest.raises(ValueError, match=field):
        Qwen3ASREncoder(replace(ENCODER_CONFIG, **{field: value}), llm_config)


def test_rejects_heads_not_divisible_by_tp() -> None:
    """Attention heads that do not split evenly over the TP ranks raise ValueError."""
    llm_config = TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=3, tensor_model_parallel_size=3
    )
    with pytest.raises(ValueError, match="num_attention_heads=4 cannot be split"):
        Qwen3ASREncoder(ENCODER_CONFIG, llm_config)

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from transformers.models.qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASREncoderConfig as HFEncoderConfig,
)
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder as HFEncoder
from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    _get_feat_extract_output_lengths,
    get_audio_cu_seqlens,
)

from megatron_speech.components.encoders.qwen3_asr import (
    Qwen3ASREncoder,
    Qwen3ASREncoderConfig,
    Qwen3ASREncoderInputs,
)
from tests.comparison import calc_relative_error, take_shard

logger = logging.getLogger(__name__)

# Largest relative error allowed against the fp32 reference, by parameter dtype.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 5e-2}
# Several chunks and windows, no frames, one chunk, a partial chunk and one frame.
UTTERANCE_FRAMES = [250, 0, 100, 37, 1]
# Names of the convolutions and the linear layer after them, the same on both sides.
CONVOLUTIONS = ("conv2d1", "conv2d2", "conv2d3", "conv_out")
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
    """Pair every parameter of ``encoder`` with the reference parameters it is made of.

    Both sides are looked up by name in ``named_parameters()``. The names on our side come
    from Megatron's layers (``transformer.layers.0.self_attention.linear_qkv.weight``); only
    they tell which fused weight holds which Hugging Face weights.
    """
    ours = dict(encoder.named_parameters())
    theirs = dict(reference.named_parameters())

    def pair(our_name: str, their_names: list[str], split_dim: int | None) -> WeightPair:
        combine = interleave_qkv if len(their_names) == 3 else take_first
        return WeightPair(ours[our_name], [theirs[n] for n in their_names], combine, split_dim)

    pairs = [pair(f"{name}.weight", [f"{name}.weight"], None) for name in CONVOLUTIONS]
    pairs += [pair(f"{name}.bias", [f"{name}.bias"], None) for name in CONVOLUTIONS[:3]]
    for index in range(ENCODER_CONFIG.num_layers):
        attention = f"transformer.layers.{index}.self_attention."
        mlp = f"transformer.layers.{index}.mlp."
        hf = f"layers.{index}."
        for suffix in ("weight", "bias"):
            qkv = [f"{hf}self_attn.{p}_proj.{suffix}" for p in "qkv"]
            pairs += [
                pair(f"{attention}linear_qkv.{suffix}", qkv, 0),
                pair(
                    f"{attention}linear_qkv.layer_norm_{suffix}",
                    [f"{hf}self_attn_layer_norm.{suffix}"],
                    None,
                ),
                pair(
                    f"{attention}linear_proj.{suffix}",
                    [f"{hf}self_attn.out_proj.{suffix}"],
                    1 if suffix == "weight" else None,
                ),
                pair(f"{mlp}linear_fc1.{suffix}", [f"{hf}fc1.{suffix}"], 0),
                pair(
                    f"{mlp}linear_fc1.layer_norm_{suffix}", [f"{hf}final_layer_norm.{suffix}"], None
                ),
                pair(
                    f"{mlp}linear_fc2.{suffix}",
                    [f"{hf}fc2.{suffix}"],
                    1 if suffix == "weight" else None,
                ),
            ]
    pairs += [
        pair(f"transformer.final_layernorm.{s}", [f"ln_post.{s}"], None) for s in ("weight", "bias")
    ]
    assert len(pairs) == len(ours)
    assert {id(p.ours) for p in pairs} == {id(p) for p in ours.values()}, "unpaired parameters"
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


@pytest.fixture(name="cpu_inputs")
def build_cpu_inputs(mels: list[torch.Tensor]) -> Qwen3ASREncoderInputs:
    """Encoder inputs of the test utterances, on the CPU as the data loader makes them."""
    return Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG)


@pytest.fixture(name="inputs")
def copy_inputs_to_gpu(cpu_inputs: Qwen3ASREncoderInputs) -> Qwen3ASREncoderInputs:
    """Encoder inputs of the test utterances on the GPU."""
    return cpu_inputs.to("cuda")


@pytest.fixture(name="hf_window_bounds")
def calc_hf_window_bounds() -> list[int]:
    """Cumulative token counts of the attention windows that Hugging Face computes."""
    chunk_frames = ENCODER_CONFIG.chunk_frames
    chunk_lengths = [
        min(chunk_frames, n - start)
        for n in UTTERANCE_FRAMES
        for start in range(0, n, chunk_frames)
    ]
    return get_audio_cu_seqlens(
        torch.tensor(chunk_lengths),
        torch.tensor(UTTERANCE_FRAMES),
        ENCODER_CONFIG.n_window_infer,
        ENCODER_CONFIG.n_window,
    ).tolist()


@pytest.fixture(name="hf_token_counts")
def calc_hf_token_counts() -> list[int]:
    """Encoder tokens per utterance by Hugging Face's length formula."""
    frames = torch.tensor(UTTERANCE_FRAMES)
    return _get_feat_extract_output_lengths(frames, ENCODER_CONFIG.n_window).tolist()


def test_window_bounds_match_hf(
    cpu_inputs: Qwen3ASREncoderInputs, hf_window_bounds: list[int]
) -> None:
    """The attention windows start and end where Hugging Face's do."""
    assert cpu_inputs.cu_seqlens.tolist() == hf_window_bounds


def test_longest_window_matches_hf(
    cpu_inputs: Qwen3ASREncoderInputs, hf_window_bounds: list[int]
) -> None:
    """max_seqlen is the longest of Hugging Face's windows."""
    longest = max(end - start for start, end in pairwise(hf_window_bounds))
    assert cpu_inputs.max_seqlen == longest


def test_token_counts_match_hf(hf_token_counts: list[int]) -> None:
    """calc_num_tokens agrees with Hugging Face's length formula for every utterance."""
    assert [ENCODER_CONFIG.calc_num_tokens(n) for n in UTTERANCE_FRAMES] == hf_token_counts


def test_token_index_selects_hf_token_count(
    cpu_inputs: Qwen3ASREncoderInputs, hf_token_counts: list[int]
) -> None:
    """token_index keeps as many tokens as Hugging Face's encoder outputs."""
    assert cpu_inputs.token_index.numel() == sum(hf_token_counts)


def test_chunks_hold_the_frames(
    cpu_inputs: Qwen3ASREncoderInputs, mels: list[torch.Tensor]
) -> None:
    """The chunks hold every frame once, in order, and zeros after the last frame."""
    chunks = iter(cpu_inputs.input_features)
    for mel in mels:
        num_chunks = -(-mel.shape[1] // ENCODER_CONFIG.chunk_frames)
        # The empty tensor lets an utterance without frames take no chunks.
        frames = torch.cat(
            [next(chunks) for _ in range(num_chunks)] + [mel.new_zeros(mel.shape[0], 0)], dim=1
        )
        assert torch.equal(frames[:, : mel.shape[1]], mel)
        assert not frames[:, mel.shape[1] :].any()
    assert next(chunks, None) is None


@pytest.mark.parametrize("frames", [[], [0, 0]], ids=["no-utterances", "empty-utterances"])
def test_rejects_batch_without_frames(frames: list[int]) -> None:
    """A batch without frames has no encoder inputs; MIMO skips the encoder for it."""
    mels = [torch.zeros(ENCODER_CONFIG.num_mel_bins, n) for n in frames]
    with pytest.raises(ValueError, match="no utterance has frames"):
        Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG)


@pytest.mark.gpu
def test_output_matches_hf(
    encoder: Qwen3ASREncoder,
    reference: HFEncoder,
    mels: list[torch.Tensor],
    inputs: Qwen3ASREncoderInputs,
    dtype: torch.dtype,
) -> None:
    """The encoder output equals Hugging Face's for the same utterances."""
    with torch.no_grad():
        expected = run_reference(reference, mels)
        actual = encoder(**vars(inputs))
    error = calc_relative_error(actual, expected)
    logger.info("%s: output max relative error %.2e", dtype, error)
    assert actual.shape == expected.shape
    assert error < TOLERANCES[dtype]


@pytest.mark.gpu
def test_gradients_match_hf(
    encoder: Qwen3ASREncoder,
    reference: HFEncoder,
    mels: list[torch.Tensor],
    inputs: Qwen3ASREncoderInputs,
    dtype: torch.dtype,
) -> None:
    """Gradients of every weight equal the reference ones, combined and split as the weights."""
    expected = run_reference(reference, mels)
    actual = encoder(**vars(inputs))
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


@dataclass
class InvalidConfig:
    """Changes to ENCODER_CONFIG and the error message they cause."""

    changes: dict[str, int | str]
    message: str


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(
            InvalidConfig({"num_attention_heads": 3}, "hidden_size=64 is not a multiple"),
            id="hidden-not-multiple-of-heads",
        ),
        pytest.param(
            InvalidConfig({"n_window": 0}, "n_window must be at least 1"), id="no-chunk-frames"
        ),
        pytest.param(
            InvalidConfig({"n_window_infer": 150}, "n_window_infer=150 must be a positive"),
            id="window-not-multiple-of-chunk",
        ),
        pytest.param(
            InvalidConfig({"n_window_infer": 0}, "n_window_infer=0 must be a positive"),
            id="no-window",
        ),
        pytest.param(
            InvalidConfig({"activation": "swish"}, "activation='swish' is not one of"),
            id="unknown-activation",
        ),
        pytest.param(
            InvalidConfig({"hidden_size": 65}, "hidden_size must be an even number"),
            id="odd-hidden-size",
        ),
        pytest.param(
            InvalidConfig({"hidden_size": 2}, "hidden_size must be an even number"),
            id="hidden-size-two",
        ),
    ],
)
def test_config_rejects_invalid_values(invalid: InvalidConfig) -> None:
    """Values that the encoder or its inputs cannot use raise ValueError at construction."""
    with pytest.raises(ValueError, match=re.escape(invalid.message)):
        replace(ENCODER_CONFIG, **invalid.changes)


@dataclass
class UnsplittableConfig:
    """A TP size, changes to ENCODER_CONFIG that it cannot split, and the error message."""

    tp_size: int
    changes: dict[str, int]
    message: str


@pytest.mark.parametrize(
    "unsplittable",
    [
        pytest.param(
            UnsplittableConfig(3, {}, "num_attention_heads=4 cannot be split"), id="heads"
        ),
        pytest.param(
            UnsplittableConfig(2, {"ffn_hidden_size": 129}, "ffn_hidden_size=129 cannot be split"),
            id="ffn",
        ),
    ],
)
def test_rejects_sizes_not_divisible_by_tp(unsplittable: UnsplittableConfig) -> None:
    """Heads and MLP features that do not split evenly over the TP ranks raise ValueError."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=unsplittable.tp_size,
        tensor_model_parallel_size=unsplittable.tp_size,
    )
    with pytest.raises(ValueError, match=re.escape(unsplittable.message)):
        Qwen3ASREncoder(replace(ENCODER_CONFIG, **unsplittable.changes), llm_config)


@pytest.mark.gpu
def test_rejects_pg_collection_of_another_tp_size(tensor_parallel_size: int) -> None:
    """The TP group in pg_collection must have as many ranks as llm_config says."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size + 1,
        tensor_model_parallel_size=tensor_parallel_size + 1,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    with pytest.raises(ValueError, match=r"pg_collection\.tp has"):
        Qwen3ASREncoder(ENCODER_CONFIG, llm_config, pg_collection)

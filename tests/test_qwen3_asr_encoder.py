import logging
import re
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
from tests.reference_comparison import TOLERANCES, calc_relative_error, take_shard

logger = logging.getLogger(__name__)

# Mel frames of each utterance: several chunks and windows, no frames, one chunk, a partial
# chunk and one frame.
UTTERANCE_LENGTHS = [250, 0, 100, 37, 1]
# Names of the convolutions and of the linear layer after them, the same on both sides. The
# linear layer has no bias.
CONVOLUTIONS = ("conv2d1", "conv2d2", "conv2d3")
CONV_OUT = "conv_out"
# Windows of 200 mel frames (26 encoder frames) so that the 250-frame utterance spans two.
ENCODER_CONFIG = Qwen3ASREncoderConfig(
    num_mel_bins=32,
    hidden_size=64,
    num_layers=2,
    num_attention_heads=4,
    ffn_hidden_size=128,
    downsample_hidden_size=16,
    mel_frames_per_window=200,
)
# Hugging Face's n_window, which is half a chunk.
HF_N_WINDOW = ENCODER_CONFIG.mel_frames_per_chunk // 2


def map_hf_tensors(hf_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Hugging Face weights or gradients to the encoder's parameter names, before splitting.

    The encoder's names come from Megatron's layers (``transformer.layers.0.self_attention.
    linear_qkv.weight``); only they tell which fused weight holds which Hugging Face weights.
    """

    def interleave_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Concatenate q, k and v head by head, as Megatron's fused ``linear_qkv`` stores them."""
        num_heads = ENCODER_CONFIG.num_attention_heads
        heads = [t.reshape(num_heads, -1, *t.shape[1:]) for t in (query, key, value)]
        return torch.cat(heads, dim=1).reshape(-1, *query.shape[1:])

    mapped = {f"{name}.weight": hf_tensors[f"{name}.weight"] for name in (*CONVOLUTIONS, CONV_OUT)}
    mapped |= {f"{name}.bias": hf_tensors[f"{name}.bias"] for name in CONVOLUTIONS}
    for index in range(ENCODER_CONFIG.num_layers):
        layer, hf_layer = f"transformer.layers.{index}.", f"layers.{index}."
        for suffix in ("weight", "bias"):
            qkv = [hf_tensors[f"{hf_layer}self_attn.{p}_proj.{suffix}"] for p in "qkv"]
            mapped |= {
                f"{layer}self_attention.linear_qkv.{suffix}": interleave_qkv(*qkv),
                f"{layer}self_attention.linear_qkv.layer_norm_{suffix}": hf_tensors[
                    f"{hf_layer}self_attn_layer_norm.{suffix}"
                ],
                f"{layer}self_attention.linear_proj.{suffix}": hf_tensors[
                    f"{hf_layer}self_attn.out_proj.{suffix}"
                ],
                f"{layer}mlp.linear_fc1.{suffix}": hf_tensors[f"{hf_layer}fc1.{suffix}"],
                f"{layer}mlp.linear_fc1.layer_norm_{suffix}": hf_tensors[
                    f"{hf_layer}final_layer_norm.{suffix}"
                ],
                f"{layer}mlp.linear_fc2.{suffix}": hf_tensors[f"{hf_layer}fc2.{suffix}"],
            }
    for suffix in ("weight", "bias"):
        mapped[f"transformer.final_layernorm.{suffix}"] = hf_tensors[f"ln_post.{suffix}"]
    return mapped


def calc_split_dim(name: str) -> int | None:
    """Calculate the dim of the encoder parameter ``name`` that TP splits, if any."""
    if name.endswith(
        ("linear_qkv.weight", "linear_qkv.bias", "linear_fc1.weight", "linear_fc1.bias")
    ):
        return 0  # column parallel: output features
    if name.endswith(("linear_proj.weight", "linear_fc2.weight")):
        return 1  # row parallel: input features
    return None


@pytest.fixture(name="dtype", params=[torch.float32, torch.bfloat16], ids=str)
def get_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Return the parameter dtype of the encoder under test."""
    return request.param


@pytest.fixture(name="reference")
def build_reference() -> HFEncoder:
    """Build the Hugging Face encoder in fp32 with random weights, the same on every rank."""
    torch.manual_seed(0)
    hf_config = HFEncoderConfig(
        num_mel_bins=ENCODER_CONFIG.num_mel_bins,
        d_model=ENCODER_CONFIG.hidden_size,
        encoder_layers=ENCODER_CONFIG.num_layers,
        encoder_attention_heads=ENCODER_CONFIG.num_attention_heads,
        encoder_ffn_dim=ENCODER_CONFIG.ffn_hidden_size,
        downsample_hidden_size=ENCODER_CONFIG.downsample_hidden_size,
        n_window=HF_N_WINDOW,
        n_window_infer=ENCODER_CONFIG.mel_frames_per_window,
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
    """Build a Qwen3ASREncoder that holds this rank's shards of the reference weights."""
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
    encoder_parameters = dict(encoder.named_parameters())
    hf_weights = map_hf_tensors({name: p.detach() for name, p in reference.named_parameters()})
    assert hf_weights.keys() == encoder_parameters.keys()
    with torch.no_grad():
        for name, weight in hf_weights.items():
            shard = take_shard(weight, calc_split_dim(name))
            assert encoder_parameters[name].shape == shard.shape
            encoder_parameters[name].copy_(shard)
    return encoder


@pytest.fixture(name="mels")
def build_mels() -> list[torch.Tensor]:
    """Build fp32 log-mels of UTTERANCE_LENGTHS frames, the same on every rank."""
    torch.manual_seed(1)
    return [torch.randn(ENCODER_CONFIG.num_mel_bins, length) for length in UTTERANCE_LENGTHS]


def run_hf_encoder(reference: HFEncoder, mels: list[torch.Tensor]) -> torch.Tensor:
    """Pad the log-mels to a multiple of the chunk length with zeros and run Hugging Face's."""
    mel_frames_per_chunk = ENCODER_CONFIG.mel_frames_per_chunk
    longest_length = max(mel.shape[1] for mel in mels)
    padded_length = -(-longest_length // mel_frames_per_chunk) * mel_frames_per_chunk
    features = torch.zeros(len(mels), ENCODER_CONFIG.num_mel_bins, padded_length)
    mask = torch.zeros(len(mels), padded_length, dtype=torch.long)
    for index, mel in enumerate(mels):
        features[index, :, : mel.shape[1]] = mel
        mask[index, : mel.shape[1]] = 1
    return reference(
        input_features=features.cuda(), input_features_mask=mask.cuda()
    ).last_hidden_state


@pytest.fixture(name="cpu_inputs")
def build_cpu_inputs(mels: list[torch.Tensor]) -> Qwen3ASREncoderInputs:
    """Build the encoder inputs of the test utterances on the CPU, as the data loader does."""
    return Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG)


@pytest.fixture(name="inputs")
def copy_inputs_to_gpu(cpu_inputs: Qwen3ASREncoderInputs) -> Qwen3ASREncoderInputs:
    """Copy the encoder inputs of the test utterances to the GPU."""
    return cpu_inputs.to("cuda")


@pytest.fixture(name="hf_window_bounds")
def calc_hf_window_bounds(mels: list[torch.Tensor]) -> list[int]:
    """Calculate the cumulative encoder frame counts of the windows as Hugging Face does."""
    mel_frames_per_chunk = ENCODER_CONFIG.mel_frames_per_chunk
    lengths = [mel.shape[1] for mel in mels]
    chunk_lengths = [
        min(mel_frames_per_chunk, n - start)
        for n in lengths
        for start in range(0, n, mel_frames_per_chunk)
    ]
    return get_audio_cu_seqlens(
        torch.tensor(chunk_lengths),
        torch.tensor(lengths),
        ENCODER_CONFIG.mel_frames_per_window,
        HF_N_WINDOW,
    ).tolist()


@pytest.fixture(name="hf_encoder_frame_counts")
def calc_hf_encoder_frame_counts(mels: list[torch.Tensor]) -> list[int]:
    """Calculate the encoder frames of each utterance with Hugging Face's length formula."""
    lengths = torch.tensor([mel.shape[1] for mel in mels])
    return _get_feat_extract_output_lengths(lengths, HF_N_WINDOW).tolist()


def test_window_bounds_match_hf(
    cpu_inputs: Qwen3ASREncoderInputs, hf_window_bounds: list[int]
) -> None:
    """Check that the attention windows start and end where Hugging Face's do."""
    assert cpu_inputs.cu_seqlens.tolist() == hf_window_bounds


def test_longest_window_matches_hf(
    cpu_inputs: Qwen3ASREncoderInputs, hf_window_bounds: list[int]
) -> None:
    """Check that max_seqlen is the length of the longest of Hugging Face's windows."""
    longest_window_length = max(end - start for start, end in pairwise(hf_window_bounds))
    assert cpu_inputs.max_seqlen == longest_window_length


def test_encoder_frame_counts_match_hf(hf_encoder_frame_counts: list[int]) -> None:
    """Check calc_num_encoder_frames against Hugging Face's length formula for each utterance."""
    num_encoder_frames = [ENCODER_CONFIG.calc_num_encoder_frames(n) for n in UTTERANCE_LENGTHS]
    assert num_encoder_frames == hf_encoder_frame_counts


def test_encoder_frame_index_selects_hf_count(
    cpu_inputs: Qwen3ASREncoderInputs, hf_encoder_frame_counts: list[int]
) -> None:
    """Check that encoder_frame_index keeps as many encoder frames as Hugging Face outputs."""
    assert cpu_inputs.encoder_frame_index.numel() == sum(hf_encoder_frame_counts)


def test_chunks_hold_the_frames(
    cpu_inputs: Qwen3ASREncoderInputs, mels: list[torch.Tensor]
) -> None:
    """Check that the chunks hold every frame once, in order, and zeros after the last frame."""
    chunks = iter(cpu_inputs.input_features)
    for mel in mels:
        num_chunks = -(-mel.shape[1] // ENCODER_CONFIG.mel_frames_per_chunk)
        # The empty tensor lets an utterance without frames take no chunks.
        frames = torch.cat(
            [next(chunks) for _ in range(num_chunks)] + [mel.new_zeros(mel.shape[0], 0)], dim=1
        )
        assert torch.equal(frames[:, : mel.shape[1]], mel)
        assert not frames[:, mel.shape[1] :].any()
    assert next(chunks, None) is None


@pytest.mark.parametrize("lengths", [[], [0, 0]], ids=["no-utterances", "empty-utterances"])
def test_rejects_batch_without_frames(lengths: list[int]) -> None:
    """Check that a batch without frames raises ValueError."""
    mels = [torch.zeros(ENCODER_CONFIG.num_mel_bins, n) for n in lengths]
    with pytest.raises(ValueError, match="no utterance has frames"):
        Qwen3ASREncoderInputs.build(mels, ENCODER_CONFIG)


@pytest.fixture(name="hf_output")
def calc_hf_output(reference: HFEncoder, mels: list[torch.Tensor]) -> torch.Tensor:
    """Calculate the Hugging Face encoder's output for the test utterances without gradients."""
    with torch.no_grad():
        return run_hf_encoder(reference, mels)


@pytest.fixture(name="encoder_output")
def calc_encoder_output(encoder: Qwen3ASREncoder, inputs: Qwen3ASREncoderInputs) -> torch.Tensor:
    """Calculate the output of the encoder under test for the test utterances without gradients."""
    with torch.no_grad():
        return encoder(**vars(inputs))


@pytest.mark.gpu
def test_output_shape_matches_hf(encoder_output: torch.Tensor, hf_output: torch.Tensor) -> None:
    """Check that the encoder output has the shape of Hugging Face's."""
    assert encoder_output.shape == hf_output.shape


@pytest.mark.gpu
def test_output_matches_hf(
    encoder_output: torch.Tensor, hf_output: torch.Tensor, dtype: torch.dtype
) -> None:
    """Check that the encoder output equals Hugging Face's for the same utterances."""
    error = calc_relative_error(encoder_output, hf_output)
    logger.info("%s: output max relative error %.2e", dtype, error)
    assert error < TOLERANCES[dtype]


@pytest.mark.gpu
def test_gradients_match_hf(
    encoder: Qwen3ASREncoder,
    reference: HFEncoder,
    mels: list[torch.Tensor],
    inputs: Qwen3ASREncoderInputs,
    dtype: torch.dtype,
) -> None:
    """Check that weight gradients equal the reference ones, combined and split as the weights."""
    expected = run_hf_encoder(reference, mels)
    actual = encoder(**vars(inputs))
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output.to(dtype))

    hf_gradients = map_hf_tensors({name: p.grad for name, p in reference.named_parameters()})
    errors = [
        calc_relative_error(param.grad, take_shard(hf_gradients[name], calc_split_dim(name)))
        for name, param in encoder.named_parameters()
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
        *[
            pytest.param(InvalidConfig({name: 0}, f"{name} must be at least 1"), id=f"no-{name}")
            for name in (
                "num_mel_bins",
                "num_layers",
                "num_attention_heads",
                "ffn_hidden_size",
                "downsample_hidden_size",
                "mel_frames_per_chunk",
            )
        ],
        pytest.param(
            InvalidConfig({"mel_frames_per_window": 150}, "mel_frames_per_window=150 must be"),
            id="window-not-multiple-of-chunk",
        ),
        pytest.param(
            InvalidConfig({"mel_frames_per_window": 0}, "mel_frames_per_window=0 must be"),
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
    """Check that values the encoder or its inputs cannot use raise ValueError at construction."""
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
    """Check that heads and MLP features that TP cannot split evenly raise ValueError."""
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
    """Check that a pg_collection TP group of another size than llm_config raises ValueError."""
    llm_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=tensor_parallel_size + 1,
        tensor_model_parallel_size=tensor_parallel_size + 1,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    with pytest.raises(ValueError, match=r"pg_collection\.tp has"):
        Qwen3ASREncoder(ENCODER_CONFIG, llm_config, pg_collection)

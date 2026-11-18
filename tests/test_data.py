"""Data pipeline: Shar -> bucketing sampler -> packed batches (CPU only).

    apptainer/run.sh python -m pytest -s tests/test_data.py

Needs data/librispeech/shar/dev-clean (scripts/prepare_librispeech.py).
"""

import time
from pathlib import Path

import pytest
import torch

SHAR_DIR = "data/librispeech/shar/dev-clean"
MAX_TOKENS = 4096


@pytest.fixture(scope="module")
def loader_and_builder():
    if not Path(SHAR_DIR).exists():
        pytest.skip(f"{SHAR_DIR} not prepared")
    from megatron_speech.data.provider import build_loader, make_example_builder
    from megatron_speech.testing import QWEN3_ASR_HF, make_run_config
    from megatron_speech.training.speech_llm_recipe import resolve_model_section

    section, _ = resolve_model_section(make_run_config({"bridge": "qwen3_asr", "hf_init": QWEN3_ASR_HF})["model"])
    data_cfg = {"max_tokens_per_microbatch": MAX_TOKENS, "num_workers": 4, "num_buckets": 10}
    builder = make_example_builder(data_cfg, section)
    loader = build_loader(data_cfg, {"shar_dirs": [SHAR_DIR]}, builder, dp_rank=0, dp_size=1, seed=1, train=True, pad_to_multiple=1)
    return loader, builder, section


def test_packed_batches_are_consistent(loader_and_builder):
    loader, builder, _ = loader_and_builder
    audio_id = builder.formatter.audio_token_id
    start, audio_seconds, n_batches = time.time(), 0.0, 20
    for i, batch in zip(range(n_batches), loader):
        ids = batch["input_ids"][0]
        cu = batch["cu_seqlens"]
        assert cu[-1].item() == ids.numel() <= MAX_TOKENS
        assert batch["labels"].shape == batch["input_ids"].shape
        # one audio placeholder per encoder output token
        assert (ids == audio_id).sum().item() == batch["audio"]["token_index"].numel()
        assert batch["audio"]["cu_seqlens"][-1].item() == batch["audio"]["token_index"].numel()
        # labels are next tokens within each example; targets only after the prompt
        for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
            assert torch.equal(batch["labels"][0, s : e - 1], ids[s + 1 : e])
            assert batch["loss_mask"][0, e - 1] == 0
        audio_seconds += batch["audio"]["input_features"].shape[0]  # 1 s per 100-frame chunk (upper bound)
        if i == 0:
            print(f"\nfirst batch: {batch['num_examples'].item()} utterances, {ids.numel()} tokens")
    elapsed = time.time() - start
    print(f"{n_batches} batches in {elapsed:.1f} s ({n_batches / elapsed:.1f} batches/s, ~{audio_seconds / elapsed:.0f} s of audio per second)")


def test_token_count_matches_packed_length(loader_and_builder):
    """The sampler's length estimate equals the packed length of each example."""
    from lhotse import CutSet

    from megatron_speech.data.provider import list_shar_shards
    from megatron_speech.data.sampling import get_num_tokens

    _, builder, _ = loader_and_builder
    cuts = CutSet.from_shar(fields={"cuts": [list_shar_shards([SHAR_DIR])[0][0]], "recording": [list_shar_shards([SHAR_DIR])[0][1]]})
    for cut in cuts.subset(first=20):
        estimated = builder.count_tokens(cut)
        actual = len(builder(cut).formatted.input_ids)
        assert estimated == actual, (cut.id, estimated, actual)

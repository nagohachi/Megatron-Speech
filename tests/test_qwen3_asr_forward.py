"""Megatron Qwen3-ASR computes the same outputs as the HF model.

* fp32 (TF32 off, see conftest.py): equal up to float32 rounding.
* bf16: differences stay at the level of bf16 rounding (measured against HF fp32).

    apptainer/run.sh python -m pytest -s tests/test_qwen3_asr_forward.py
"""

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

NUM_UTTERANCES = 4


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _setup(dtype: torch.dtype):
    from megatron_speech.bridges import get_bridge_for_model_section
    from megatron_speech.testing import QWEN3_ASR_HF, build_speech_llm_for_eval, make_run_config, setup_speech_llm

    from tests.qwen3_asr_common import make_examples

    run_cfg = make_run_config(
        {"bridge": "qwen3_asr", "hf_init": QWEN3_ASR_HF},
        tensor_parallel=int(os.environ.get("WORLD_SIZE", "1")),
        megatron={"bf16": dtype == torch.bfloat16},
    )
    _, model_cfg = setup_speech_llm(run_cfg)
    hf_dir = get_bridge_for_model_section(model_cfg.speech).hf_dir
    model = build_speech_llm_for_eval(model_cfg, dtype=dtype)
    _, hf_inputs, batch = make_examples(model_cfg.speech, hf_dir, NUM_UTTERANCES)
    return model, hf_inputs, batch, hf_dir


@torch.no_grad()
def _run_both(model, hf_model, hf_inputs, batch, dtype):
    """Per-utterance (megatron logits, hf logits) and the projected audio embeddings."""
    from megatron_speech.testing import gather_vocab_parallel_logits
    from megatron_speech.training.speech_llm_step import batch_to_model_inputs

    inputs = batch_to_model_inputs(batch, with_labels=False)
    logits = gather_vocab_parallel_logits(model(**inputs)[0]).squeeze(0)
    audio = model.modality_submodules["audio"](encoder_inputs=inputs["modality_inputs"]["audio"])
    ref_audio = torch.cat(
        [
            hf_model.model.get_audio_features(x["input_features"].cuda().to(dtype), x["input_features_mask"].cuda()).pooler_output
            for x in hf_inputs
        ]
    )
    cu = batch["cu_seqlens"].tolist()
    pairs = []
    for i, x in enumerate(hf_inputs):
        ref = hf_model(
            input_ids=x["input_ids"].cuda(),
            input_features=x["input_features"].cuda().to(dtype),
            input_features_mask=x["input_features_mask"].cuda(),
        ).logits[0]
        pairs.append((logits[cu[i] : cu[i + 1]], ref))
    return pairs, audio, ref_audio


def test_prompt_matches_hf_processor():
    """Audio token count, prompt tokens and log-mel features equal the HF processor's."""
    from transformers import AutoProcessor

    from megatron_speech.data.librispeech import iter_librispeech
    from megatron_speech.testing import QWEN3_ASR_HF, make_run_config
    from megatron_speech.training.speech_llm_recipe import resolve_model_section

    from tests.qwen3_asr_common import make_examples

    section, bridge = resolve_model_section(make_run_config({"bridge": "qwen3_asr", "hf_init": QWEN3_ASR_HF})["model"])
    _, hf_inputs, _ = make_examples(section, bridge.hf_dir, NUM_UTTERANCES)
    processor = AutoProcessor.from_pretrained(str(bridge.hf_dir))
    for (_, wav, sr, _), ours in zip(iter_librispeech("test-clean", limit=NUM_UTTERANCES), hf_inputs):
        ref = processor.apply_transcription_request(wav, language="English", sampling_rate=sr)
        ref_ids = ref["input_ids"][0].tolist()
        assert ours["input_ids"][0, : len(ref_ids)].tolist() == ref_ids
        num_frames = int(ref["input_features_mask"][0].sum())
        assert torch.equal(ours["input_features"][0, :, :num_frames], ref["input_features"][0, :, :num_frames])


def test_forward_matches_hf_fp32():
    from tests.qwen3_asr_common import calc_target_ce, load_hf_model

    model, hf_inputs, batch, hf_dir = _setup(torch.float32)
    hf_model = load_hf_model(hf_dir, torch.float32)
    pairs, audio, ref_audio = _run_both(model, hf_model, hf_inputs, batch, torch.float32)
    print(f"\n[fp32] projected audio embeddings: relative error {_rel_err(audio, ref_audio):.2e}")
    assert _rel_err(audio, ref_audio) < 1e-4
    for i, ((ours, ref), x) in enumerate(zip(pairs, hf_inputs)):
        ids = x["input_ids"][0].cuda()
        ce = calc_target_ce(ours, ids, x["target_mask"]).item(), calc_target_ce(ref, ids, x["target_mask"]).item()
        print(f"[fp32] utt {i}: logits relative error {_rel_err(ours, ref):.2e}; target CE megatron {ce[0]:.5f} hf {ce[1]:.5f}")
        assert _rel_err(ours, ref) < 1e-4
        assert abs(ce[0] - ce[1]) < 1e-4
    del model, hf_model
    torch.cuda.empty_cache()


def test_forward_close_to_hf_bf16():
    """bf16 Megatron vs bf16 HF; both are compared with HF fp32 to bound the rounding error."""
    from tests.qwen3_asr_common import calc_target_ce, load_hf_model

    model, hf_inputs, batch, hf_dir = _setup(torch.bfloat16)
    hf16 = load_hf_model(hf_dir, torch.bfloat16)
    hf32 = load_hf_model(hf_dir, torch.float32)
    pairs, audio, ref16 = _run_both(model, hf16, hf_inputs, batch, torch.bfloat16)
    pairs32, _, ref32 = _run_both(model, hf32, hf_inputs, batch, torch.float32)
    err_ours, err_hf = _rel_err(audio, ref32), _rel_err(ref16, ref32)
    print(f"\n[bf16] audio embeddings vs HF fp32: megatron {err_ours:.4f}, hf-bf16 {err_hf:.4f}")
    assert err_ours < 2 * err_hf + 1e-3
    for i, ((ours, r16), (_, r32), x) in enumerate(zip(pairs, pairs32, hf_inputs)):
        ids = x["input_ids"][0].cuda()
        ce = [calc_target_ce(l, ids, x["target_mask"]).item() for l in (ours, r16, r32)]
        print(f"[bf16] utt {i}: target CE megatron {ce[0]:.4f} hf-bf16 {ce[1]:.4f} hf-fp32 {ce[2]:.4f}")
        assert abs(ce[0] - ce[2]) < 0.05

"""HF -> Megatron -> HF round trip of Qwen3-ASR weights must be bit-exact.

Single GPU:   apptainer/run.sh python -m pytest tests/test_qwen3_asr_roundtrip.py
Two-way TP:   apptainer/run.sh torchrun --nproc-per-node 2 -m pytest tests/test_qwen3_asr_roundtrip.py
"""

import os

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_roundtrip_is_exact():
    from megatron_speech import compat
    from megatron_speech.bridges import get_bridge_for_model_section
    from megatron_speech.bridges.hf_io import SafetensorsReader
    from megatron_speech.testing import QWEN3_ASR_HF, build_speech_llm_for_eval, make_run_config, setup_speech_llm

    tp = int(os.environ.get("WORLD_SIZE", "1"))
    run_cfg = make_run_config({"bridge": "qwen3_asr", "hf_init": QWEN3_ASR_HF}, tensor_parallel=tp)
    _, model_cfg = setup_speech_llm(run_cfg)
    model = build_speech_llm_for_eval(model_cfg)

    bridge = get_bridge_for_model_section(model_cfg.speech)
    exported = bridge.export_state_dict(model, model_cfg.speech)
    reader = SafetensorsReader(bridge.hf_dir)

    assert set(exported) == reader.keys()
    mismatched = [k for k in sorted(exported) if not torch.equal(exported[k], reader.get(k))]
    assert not mismatched, f"{len(mismatched)} tensors differ, e.g. {mismatched[:5]}"

    n_params = sum(p.numel() for p in model.parameters())
    tp_size = compat.get_tensor_model_parallel_world_size()
    print(f"TP={tp_size}: {len(exported)} tensors exact; {n_params:,} parameters on this rank")

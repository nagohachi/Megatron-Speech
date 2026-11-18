"""LoRA on the Megatron speech LLM agrees with HF + PEFT (fp32, TF32 off).

Model: Qwen3-ASR with the encoder frozen, the projector trained in full and LoRA on the LLM,
for both LoRA variants (canonical: one pair per HF projection; fused: A shared by q/k/v).

    apptainer/run.sh python -m pytest -s tests/test_lora.py
    apptainer/run.sh torchrun --nproc-per-node 2 -m pytest -s tests/test_lora.py   # TP=2

Checks:
1. Right after LoRA is added (B = 0) the output equals the HF base model.
2. With random adapters (and a perturbed projector) the output equals HF + the exported PEFT
   adapter (``PeftModel.from_pretrained``).
3. LoRA gradients equal the HF + PEFT gradients (fused variant: the gradient of the shared
   ``lora_A`` equals the sum over the HF projections that share it).
4. After merging, the output is unchanged and equals HF with the exported merged weights.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

NUM_UTTERANCES = 3
TOL = 1e-4


def make_peft_cfg(variant: str) -> dict:
    return {
        "encoder": {"mode": "frozen"},
        "projector": {"mode": "full"},
        "llm": {"mode": "lora", "rank": 8, "alpha": 16, "variant": variant},
    }


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _megatron_logits(model, batch):
    from megatron_speech.training.speech_llm_step import batch_to_model_inputs

    from megatron_speech.testing import gather_vocab_parallel_logits

    with torch.no_grad():
        logits = gather_vocab_parallel_logits(model(**batch_to_model_inputs(batch, with_labels=False))[0]).squeeze(0)
    cu = batch["cu_seqlens"].tolist()
    return [logits[cu[i] : cu[i + 1]] for i in range(len(cu) - 1)]


def _hf_logits(hf_model, hf_inputs):
    with torch.no_grad():
        return [
            hf_model(
                input_ids=x["input_ids"].cuda(),
                input_features=x["input_features"].cuda(),
                input_features_mask=x["input_features_mask"].cuda(),
            ).logits[0]
            for x in hf_inputs
        ]


def _hf_loss(hf_model, hf_inputs, reverse: bool = False):
    """Mean target-token CE over all utterances (same definition as the Megatron loss)."""
    total, count = 0.0, 0
    for x in reversed(hf_inputs) if reverse else hf_inputs:
        logits = hf_model(
            input_ids=x["input_ids"].cuda(),
            input_features=x["input_features"].cuda(),
            input_features_mask=x["input_features_mask"].cuda(),
        ).logits[0].float()
        ids = x["input_ids"][0].cuda()
        mask = x["target_mask"][1:].cuda().float()
        nll = torch.nn.functional.cross_entropy(logits[:-1], ids[1:], reduction="none")
        total = total + (nll * mask).sum()
        count += mask.sum()
    return total / count


def _check_gradients(model, peft_model, hf_inputs, loss, ours_grads):
    """Compare Megatron LoRA gradients with HF + PEFT (step 3 of the test)."""
    # Control: HF gradients with the utterances summed in reverse order differ from the forward
    # order only by float32 rounding; their difference is the floor for the comparison below.
    _hf_loss(peft_model, hf_inputs, reverse=True).backward()
    control = {n: p.grad.clone() for n, p in peft_model.named_parameters() if p.grad is not None and "lora_" in n}
    peft_model.zero_grad(set_to_none=True)
    ref_loss = _hf_loss(peft_model, hf_inputs)
    ref_loss.backward()
    print(f"[3] loss Megatron {loss.item():.6f} HF+PEFT {ref_loss.item():.6f}")
    assert abs(loss.item() - ref_loss.item()) < TOL
    ref_named = dict(peft_model.named_parameters())
    c_diff2 = sum(((control[n] - ref_named[n].grad) ** 2).sum().item() for n in control)
    c_ref2 = sum((ref_named[n].grad ** 2).sum().item() for n in control)
    control_err = (c_diff2 / c_ref2) ** 0.5
    pairs = []  # (name, ours, ref)
    for hf_module, g in ours_grads.items():
        pairs.append((f"{hf_module}.lora_B", g["lora_B"], ref_named[f"base_model.model.{hf_module}.lora_B.default.weight"].grad))
    shared_a: dict[int, list[str]] = {}
    for hf_module, g in ours_grads.items():
        shared_a.setdefault(id(g["lora_A"]), []).append(hf_module)
    for modules in shared_a.values():
        ref_a = sum(ref_named[f"base_model.model.{m}.lora_A.default.weight"].grad for m in modules)
        pairs.append((f"{'+'.join(modules)}.lora_A", ours_grads[modules[0]]["lora_A"], ref_a))
    diff2 = sum(((a.float() - b.float()) ** 2).sum().item() for _, a, b in pairs)
    ref2 = sum((b.float() ** 2).sum().item() for _, _, b in pairs)
    global_err = (diff2 / ref2) ** 0.5
    name, ours_t, ref_t = max(pairs, key=lambda p: _rel_err(p[1], p[2]))
    print(f"[3] HF vs HF with reversed summation order (control): relative error {control_err:.2e}")
    print(f"[3] LoRA gradients: relative error over all {len(pairs)} tensors {global_err:.2e}; "
          f"worst tensor {_rel_err(ours_t, ref_t):.2e} ({name}, norm {ref_t.norm().item():.2e} "
          f"vs RMS norm {(ref2 / len(pairs)) ** 0.5:.2e})")
    # The worst single tensor can be one whose gradient nearly cancels (fused variant: a sum
    # over q/k/v), so the tighter bound is on all gradients together.
    assert global_err < max(1e-4, 10 * control_err)
    assert _rel_err(ours_t, ref_t) < 1e-2


def _is_rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


@pytest.mark.parametrize("variant", ["canonical", "fused"])
def test_lora_matches_hf_peft(variant):
    from peft import PeftModel

    from megatron_speech.bridges import get_bridge_for_model_section
    from megatron_speech.bridges.convert import export_hf_state_dict
    from megatron_speech.peft.export import export_lora_tensors, export_peft_adapter, merge_lora_weights, save_peft_adapter
    from megatron_speech.peft.lora import LoRALinear
    from megatron_speech.testing import QWEN3_ASR_HF, build_speech_llm_for_eval, make_run_config, setup_speech_llm
    from megatron_speech.training.speech_llm_step import batch_to_model_inputs

    from tests.qwen3_asr_common import load_hf_model, make_examples

    tp = int(os.environ.get("WORLD_SIZE", "1"))
    run_cfg = make_run_config(
        {"bridge": "qwen3_asr", "hf_init": QWEN3_ASR_HF, "peft": make_peft_cfg(variant)},
        tensor_parallel=tp,
        megatron={"bf16": False, "no-gradient-accumulation-fusion": True},
    )
    _, model_cfg = setup_speech_llm(run_cfg)
    bridge = get_bridge_for_model_section(model_cfg.speech)
    table = bridge.mapping_table(model_cfg.speech)
    model = build_speech_llm_for_eval(model_cfg, dtype=torch.float32)
    _, hf_inputs, batch = make_examples(model_cfg.speech, bridge.hf_dir, NUM_UTTERANCES)

    # 1. Fresh adapters do not change the output.
    hf_base = load_hf_model(bridge.hf_dir, torch.float32)
    for i, (ours, ref) in enumerate(zip(_megatron_logits(model, batch), _hf_logits(hf_base, hf_inputs))):
        err = _rel_err(ours, ref)
        print(f"\n[1] utt {i}: fresh LoRA vs HF base, logits relative error {err:.2e}")
        assert err < TOL

    # 2. Random adapters and a perturbed projector; same values on every TP rank. The scale
    # keeps the model near the base model (with std 0.02 each layer changed by ~30% and the
    # gradients became sensitive to float32 summation order).
    torch.manual_seed(0)
    with torch.no_grad():
        for _, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.adapter.lora_A.weight.normal_(0, 0.005)
                for b in module.adapter.lora_B:
                    b.weight.normal_(0, 0.005)
        for name, p in model.named_parameters():
            if name.startswith("modality_submodules.audio.input_projections."):
                p.add_(torch.randn_like(p) * 0.01)
    ours_lora = _megatron_logits(model, batch)

    out_dir = Path(f"outputs/test_tmp/lora_{variant}_tp{tp}")
    adapter_sd, adapter_cfg = export_peft_adapter(model, table, make_peft_cfg(variant), str(bridge.hf_dir))
    if _is_rank0():
        save_peft_adapter(adapter_sd, adapter_cfg, out_dir / "adapter")
    _barrier()
    print(f"[2] adapter: {len(adapter_sd)} tensors, modules_to_save={adapter_cfg['modules_to_save']}")
    peft_model = PeftModel.from_pretrained(hf_base, str(out_dir / "adapter"), is_trainable=True).cuda()
    for i, (ours, ref) in enumerate(zip(ours_lora, _hf_logits(peft_model, hf_inputs))):
        err = _rel_err(ours, ref)
        print(f"[2] utt {i}: Megatron LoRA vs HF+PEFT, logits relative error {err:.2e}")
        assert err < TOL

    # 3. Gradients of the LoRA parameters.
    model.train()
    output, loss_mask = model(**batch_to_model_inputs(batch))
    loss = (output.float() * loss_mask).sum() / loss_mask.sum()
    loss.backward()
    ours_grads = export_lora_tensors(model, table, use_grad=True)
    _check_gradients(model, peft_model, hf_inputs, loss, ours_grads)
    del peft_model, hf_base
    torch.cuda.empty_cache()

    # 4. Merge and export merged weights.
    model.eval()
    merge_lora_weights(model)
    merged = _megatron_logits(model, batch)
    merged_sd = export_hf_state_dict(model, table)  # keep fp32 for the comparison
    hf_merged = load_hf_model(bridge.hf_dir, torch.float32)
    missing, unexpected = hf_merged.load_state_dict({k: v.float() for k, v in merged_sd.items()}, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}
    for i, (ours, before, ref) in enumerate(zip(merged, ours_lora, _hf_logits(hf_merged, hf_inputs))):
        e1, e2 = _rel_err(ours, before), _rel_err(ours, ref)
        print(f"[4] utt {i}: merged vs unmerged {e1:.2e}, merged Megatron vs HF merged export {e2:.2e}")
        assert e1 < TOL and e2 < TOL

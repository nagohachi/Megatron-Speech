"""Train the same steps with Megatron-Speech and with HF + PEFT and compare the loss curves.

    apptainer/run.sh torchrun --nproc-per-node 1 tests/compare_training_with_hf.py --phase megatron --out outputs/test_tmp/loss_curve
    apptainer/run.sh python tests/compare_training_with_hf.py --phase hf --out outputs/test_tmp/loss_curve
    apptainer/run.sh python tests/compare_training_with_hf.py --phase compare --out outputs/test_tmp/loss_curve

Setup (both sides): Qwen3-ASR-1.7B, encoder frozen, projector trained in full, LoRA (r=16,
alpha=32) on the LLM; the same fixed packed batches from LibriSpeech dev-clean, one batch per
step; Adam (beta 0.9/0.999, eps 1e-8), no weight decay, constant learning rate, gradient
clipping at 1.0; bf16 compute.

* Megatron: the real training loop (``pretrain()``, distributed optimizer with fp32 main
  parameters). A post-wrap hook exports the initial adapter, so HF starts from the same values.
* HF: ``Qwen3ASRForConditionalGeneration`` (fp32 parameters) + PEFT, ``torch.autocast(bf16)``,
  (``MS_COMPARE_PRECISION=fp32``: no autocast on HF, no --bf16 on Megatron; set
  ``NVIDIA_TF32_OVERRIDE=0`` so that both sides compute in true fp32)
  ``torch.optim.AdamW``. The loss is the mean cross entropy over all target tokens of the batch,
  as in Megatron with ``--calculate-per-token-loss``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUM_STEPS = 20
LR = 1.0e-4
MAX_TOKENS = 2048
SHAR_DIR = "data/librispeech/shar/dev-clean"
PEFT_CFG = {
    "encoder": {"mode": "frozen"},
    "projector": {"mode": "full"},
    "llm": {"mode": "lora", "rank": 16, "alpha": 32, "dropout": 0.0},
}


PRECISION = os.environ.get("MS_COMPARE_PRECISION", "bf16")  # bf16 | fp32 (fp32 needs NVIDIA_TF32_OVERRIDE=0)


def make_run_cfg():
    from megatron_speech.testing import QWEN3_ASR_HF

    return {
        "model": {"type": "speech_llm", "bridge": "qwen3_asr", "hf_init": QWEN3_ASR_HF, "peft": PEFT_CFG},
        "data": {"max_tokens_per_microbatch": MAX_TOKENS, "text_transform": "lower", "num_workers": 0},
        "megatron": {
            "bf16": PRECISION == "bf16",
            "use-distributed-optimizer": True,
            "global-batch-size": 1,
            "train-iters": NUM_STEPS,
            "lr": LR,
            "lr-decay-style": "constant",
            "lr-warmup-iters": 0,
            "weight-decay": 0.0,
            "clip-grad": 1.0,
            "adam-beta1": 0.9,
            "adam-beta2": 0.999,
            "adam-eps": 1.0e-8,
            "eval-iters": 0,
            "log-interval": 1,
            "seed": 1234,
        },
    }


def build_batches(run_cfg):
    from megatron_speech.data.provider import build_loader, make_example_builder
    from megatron_speech.training.speech_llm_recipe import resolve_model_section

    section, _ = resolve_model_section(run_cfg["model"])
    data_cfg = dict(run_cfg["data"], num_buckets=5)
    builder = make_example_builder(data_cfg, section)
    loader = build_loader(data_cfg, {"shar_dirs": [SHAR_DIR]}, builder, 0, 1, seed=1, train=False, pad_to_multiple=1)
    return [b for _, b in zip(range(NUM_STEPS), loader)]


def run_megatron(out: Path) -> None:
    from megatron_speech import compat
    from megatron_speech.bridges import get_bridge_for_model_section
    from megatron_speech.peft.export import export_peft_adapter, save_peft_adapter
    from megatron_speech.training.speech_llm_recipe import SpeechLLMRecipe
    from megatron_speech.training.speech_llm_step import forward_step as speech_forward_step

    run_cfg = make_run_cfg()
    batches = build_batches(run_cfg)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(batches, out / "batches.pt")
    losses: list[float] = []

    class FixedBatchRecipe(SpeechLLMRecipe):
        def build_model_config(self, args):
            cfg = super().build_model_config(args)
            speech = cfg.speech

            def export_initial_adapter(models):
                model = compat.unwrap_model(models[0])
                bridge = get_bridge_for_model_section(speech)
                sd, adapter_cfg = export_peft_adapter(model, bridge.mapping_table(speech), PEFT_CFG, str(bridge.hf_dir))
                save_peft_adapter(sd, adapter_cfg, out / "initial_adapter")
                return models

            cfg.post_wrap_hooks = [export_initial_adapter]
            return cfg

        def datasets_provider(self):
            def provider(_):
                return iter(batches), None, None

            provider.is_distributed = True
            return provider

        def forward_step(self):
            def step(data_iterator, model):
                output, loss_fn = speech_forward_step(data_iterator, model)

                def recorded(output_tensor):
                    loss, num_tokens, report = loss_fn(output_tensor)
                    losses.append(loss.item() / num_tokens.item())
                    return loss, num_tokens, report

                return output, recorded

            return step

    FixedBatchRecipe(run_cfg).run()
    with open(out / "megatron_losses.json", "w") as f:
        json.dump(losses, f)
    print("megatron losses:", " ".join(f"{x:.4f}" for x in losses))


def split_examples(batch):
    """Per-utterance HF inputs from a packed batch."""
    cu = batch["cu_seqlens"].tolist()
    audio = batch["audio"]
    chunk_lens = audio["chunk_lens"].tolist()
    c = 0
    for i, n_chunks in enumerate(audio["chunks_per_example"].tolist()):
        s, e = cu[i], cu[i + 1]
        chunks = audio["input_features"][c : c + n_chunks]  # [n, mel, 100]
        lens = chunk_lens[c : c + n_chunks]
        c += n_chunks
        features = chunks.permute(1, 0, 2).reshape(chunks.shape[1], -1)[None]
        mask = torch.zeros(1, features.shape[-1], dtype=torch.long)
        for j, n in enumerate(lens):
            mask[0, j * chunks.shape[-1] : j * chunks.shape[-1] + n] = 1
        yield batch["input_ids"][:, s:e], features, mask, batch["loss_mask"][0, s : e - 1]


def run_hf(out: Path, reverse: bool = False) -> None:
    """HF + PEFT training. ``reverse=True`` accumulates the utterances of each batch in reverse
    order: a control run whose only difference is floating-point summation order."""
    from peft import PeftModel
    from transformers import Qwen3ASRForConditionalGeneration

    from megatron_speech.bridges.hf_io import resolve_hf_path
    from megatron_speech.testing import QWEN3_ASR_HF

    batches = torch.load(out / "batches.pt", weights_only=False)
    base = Qwen3ASRForConditionalGeneration.from_pretrained(str(resolve_hf_path(QWEN3_ASR_HF)), dtype=torch.float32)
    model = PeftModel.from_pretrained(base.cuda(), str(out / "initial_adapter"), is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    losses = []
    for batch in batches:
        # Backward per utterance (loss divided by the batch's target count) to bound memory;
        # the accumulated gradient equals that of the batch-mean loss.
        count = batch["loss_mask"].sum().item()
        total = 0.0
        examples = list(split_examples(batch))
        for ids, features, mask, target in reversed(examples) if reverse else examples:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=PRECISION == "bf16"):
                logits = model(input_ids=ids.cuda(), input_features=features.cuda(), input_features_mask=mask.cuda()).logits[0]
            nll = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[0, 1:].cuda(), reduction="none")
            part = (nll * target.cuda()).sum() / count
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total)
    name = "hf_reversed_losses.json" if reverse else "hf_losses.json"
    with open(out / name, "w") as f:
        json.dump(losses, f)
    print(f"{name}:", " ".join(f"{x:.4f}" for x in losses))


def compare(out: Path, first: str = "megatron_losses.json", second: str = "hf_losses.json") -> dict:
    meg = json.load(open(out / first))
    hf = json.load(open(out / second))
    assert len(meg) == len(hf) == NUM_STEPS
    rel = [abs(a - b) / b for a, b in zip(meg, hf)]
    result = {
        "max_relative_diff": max(rel),
        "mean_relative_diff": sum(rel) / len(rel),
        "megatron_first_last": (meg[0], meg[-1]),
        "hf_first_last": (hf[0], hf[-1]),
    }
    na, nb = first.removesuffix("_losses.json"), second.removesuffix("_losses.json")
    for i, (a, b) in enumerate(zip(meg, hf)):
        print(f"step {i + 1:2d}: {na} {a:.4f}  {nb} {b:.4f}  relative diff {abs(a - b) / b:.4f}")
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["megatron", "hf", "hf_reversed", "compare", "compare_control"], required=True)
    parser.add_argument("--out", default=f"outputs/test_tmp/loss_curve_{PRECISION}")
    ns = parser.parse_args()
    out = Path(ns.out)
    phases = {
        "megatron": lambda: run_megatron(out),
        "hf": lambda: run_hf(out),
        "hf_reversed": lambda: run_hf(out, reverse=True),
        "compare": lambda: compare(out),
        # HF vs HF with reversed summation order: the divergence floor of the comparison.
        "compare_control": lambda: compare(out, "hf_reversed_losses.json", "hf_losses.json"),
    }
    phases[ns.phase]()


if __name__ == "__main__":
    main()

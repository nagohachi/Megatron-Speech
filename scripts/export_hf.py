"""Export a Megatron-Speech checkpoint to Hugging Face format.

    apptainer/run.sh python scripts/export_hf.py --config configs/qwen3_asr/lora_librispeech.yaml \
        --load outputs/qwen3_asr_librispeech_lora --out outputs/hf/qwen3_asr_lora --mode both

Run under ``torchrun --nproc-per-node N`` to load with N-way tensor parallelism (for models
that do not fit one GPU); the distributed checkpoint is resharded on load.

Modes:
  merged   full HF checkpoint (LoRA folded into the base weights) in <out>/merged
  adapter  PEFT adapter in <out>/adapter (LoRA weights plus fully trained components as
           modules_to_save); load with PeftModel.from_pretrained(base, <out>/adapter)
  both     both of the above
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from megatron_speech import compat
from megatron_speech.bridges import get_bridge_for_model_section
from megatron_speech.bridges.hf_io import copy_non_weight_files
from megatron_speech.peft.export import export_peft_adapter, iter_lora_modules, merge_lora_weights, save_peft_adapter
from megatron_speech.testing import build_speech_llm_for_eval, setup_speech_llm
from megatron_speech.training.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config used for training")
    parser.add_argument("--load", required=True, help="Megatron checkpoint directory (megatron.save of the run)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["merged", "adapter", "both"], default="both")
    parser.add_argument("overrides", nargs="*")
    ns = parser.parse_args()

    tp = int(os.environ.get("WORLD_SIZE", "1"))
    run_cfg = load_config(ns.config, ns.overrides)
    run_cfg["megatron"].update({"tensor-model-parallel-size": tp, "load": ns.load, "save": None, "tensorboard-dir": None})
    args, model_cfg = setup_speech_llm(run_cfg)
    compat.build_pretrain_config(args, model_cfg)
    model = build_speech_llm_for_eval(model_cfg, load_hf=False, apply_peft=True, dtype=torch.bfloat16)
    iteration = compat.load_megatron_checkpoint([model], ns.load)
    compat.print_rank_0(f"> loaded {ns.load} (iteration {iteration})")

    bridge = get_bridge_for_model_section(model_cfg.speech)
    rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    out = Path(ns.out)
    has_lora = any(True for _ in iter_lora_modules(model))
    if ns.mode in ("adapter", "both"):
        if not has_lora:
            raise ValueError("the run has no LoRA layers; use --mode merged")
        state_dict, config = export_peft_adapter(
            model, bridge.mapping_table(model_cfg.speech), model_cfg.speech.get("peft") or {},
            base_model_name_or_path=model_cfg.speech["hf_init"], dtype=bridge.hf_torch_dtype(),
        )
        if rank0:
            save_peft_adapter(state_dict, config, out / "adapter")
            copy_non_weight_files(bridge.hf_dir, out / "adapter" / "processor")
            print(f"adapter: {len(state_dict)} tensors -> {out / 'adapter'}")
    if ns.mode in ("merged", "both"):
        if has_lora:
            merge_lora_weights(model)
        state_dict = bridge.export_state_dict(model, model_cfg.speech)
        if rank0:
            bridge.save_hf(state_dict, out / "merged")
            print(f"merged: {len(state_dict)} tensors -> {out / 'merged'}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()

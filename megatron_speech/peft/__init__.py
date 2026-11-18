"""Per-component training modes: full fine-tuning, frozen, or LoRA.

Run config::

    model:
      peft:
        encoder:   {mode: frozen}
        projector: {mode: full}
        llm:       {mode: lora, rank: 16, alpha: 32, dropout: 0.0,
                    target_modules: [linear_qkv, linear_proj, linear_fc1, linear_fc2],
                    variant: canonical}   # canonical (= HF PEFT) | fused (see peft/lora.py)

Components not listed are trained in full.
"""

from megatron_speech.peft.apply import apply_peft, apply_peft_hook  # noqa: F401

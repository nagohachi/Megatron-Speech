"""Forward step of the speech LLM for Megatron's training loop."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from megatron_speech import compat
from megatron_speech.models.speech_llm import AUDIO_MODALITY, ENCODER_NAME


def batch_to_model_inputs(batch: dict[str, Any], with_labels: bool = True) -> dict[str, Any]:
    """Move a packed batch (``megatron_speech.data.packing.pack_examples``) to the GPU and
    arrange it as ``MimoModel.forward`` keyword arguments."""
    audio = dict(batch["audio"])
    encoder_inputs = {
        "input_features": audio["input_features"].cuda(non_blocking=True),
        "token_index": audio["token_index"].cuda(non_blocking=True),
        "cu_seqlens": audio["cu_seqlens"].cuda(non_blocking=True),
        "max_seqlen": int(audio["max_seqlen"]),
    }
    inputs = {
        "input_ids": batch["input_ids"].cuda(non_blocking=True),
        "position_ids": batch["position_ids"].cuda(non_blocking=True),
        "attention_mask": None,
        "modality_inputs": {AUDIO_MODALITY: {ENCODER_NAME: encoder_inputs}},
        "cu_seqlens": batch["cu_seqlens"].cuda(non_blocking=True),
        # Read on the host by Megatron (.item()); keep it on the CPU.
        "max_seqlen": batch["max_seqlen"],
    }
    if with_labels:
        inputs["labels"] = batch["labels"].cuda(non_blocking=True)
        inputs["loss_mask"] = batch["loss_mask"].cuda(non_blocking=True)
    return inputs


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Sum of target-token losses and the number of target tokens (``--calculate-per-token-loss``)."""
    losses = output_tensor.view(-1).float()
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().clone().detach().to(torch.int)
    cfg = compat.get_run_config()
    if cfg.rerun_state_machine.check_for_nan_in_loss:
        rsm = compat.get_rerun_state_machine()
        rsm.validate_result(result=loss, rejection_func=torch.isnan, message="found NaN in local forward loss calculation", tolerance=0.0, fatal=True)
        rsm.validate_result(result=loss, rejection_func=torch.isinf, message="found Inf in local forward loss calculation", tolerance=0.0, fatal=True)
    report = {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
    return loss, num_tokens, report


def forward_step(data_iterator, model):
    timers = compat.get_timers()
    timers("batch-generator", log_level=2).start()
    batch = next(data_iterator)
    inputs = batch_to_model_inputs(batch)
    timers("batch-generator").stop()
    output_tensor, loss_mask = model(**inputs)
    return output_tensor, partial(loss_func, loss_mask)

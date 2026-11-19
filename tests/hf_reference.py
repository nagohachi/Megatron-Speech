from pathlib import Path

import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.transformer import MegatronModule
from transformers import PreTrainedModel

from megatron_speech.conversion.bridge import MappingBridge


def take_rank0_path(path: Path) -> Path:
    """Return rank 0's ``path`` on every rank; each rank of a run has its own ``tmp_path``."""
    paths = [path]
    torch.distributed.broadcast_object_list(paths, src=0)
    return paths[0]


def save_hf_checkpoint(model: PreTrainedModel, tmp_path: Path) -> Path:
    """Save ``model`` as a Hugging Face checkpoint and return the directory that every rank reads.

    Under torch.distributed, ``save_pretrained`` writes only on rank 0.
    """
    hf_dir = take_rank0_path(tmp_path)
    if torch.distributed.get_rank() == 0:
        model.save_pretrained(hf_dir)
    torch.distributed.barrier()
    return hf_dir


def check_export_restores_hf_weights(
    bridge: MappingBridge, model: MegatronModule, hf_dir: Path, dtype: torch.dtype
) -> None:
    """Check that writing ``model``'s weights back gives every weight of ``hf_dir`` exactly.

    A bf16 model holds the fp32 Hugging Face weights rounded to bf16, so its export is compared
    with the rounded weights.
    """
    hf_pretrained = PreTrainedCausalLM.from_pretrained(hf_dir)
    exported = dict(
        bridge.stream_weights_megatron_to_hf([model], hf_pretrained, show_progress=False)
    )
    if torch.distributed.get_rank() == 0:  # the other ranks get no tensors
        assert exported.keys() == set(hf_pretrained.state.keys())
        for name, tensor in exported.items():
            assert tensor.dtype == dtype, name
            assert torch.equal(tensor, hf_pretrained.state[name].to(dtype)), name


def randomize_norms(model: torch.nn.Module, name_parts: tuple[str, ...]) -> None:
    """Draw the norm weights of ``model`` around 1 and the norm biases around 0 (std 0.1).

    A norm parameter is one whose name contains one of ``name_parts``. A conversion or copy that
    misses a norm then changes the outputs, which it would not with the initial 1 and 0.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not any(part in name for part in name_parts):
                continue
            param.normal_(mean=1.0 if name.endswith("weight") else 0.0, std=0.1)

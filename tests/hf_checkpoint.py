from pathlib import Path

import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.transformer import MegatronModule
from transformers import PreTrainedModel

from megatron_speech.conversion.bridge import MappingBridge


def save_hf_checkpoint(model: PreTrainedModel, tmp_path: Path) -> Path:
    """Save ``model`` as a Hugging Face checkpoint and return the directory that every rank reads.

    Under torch.distributed, ``save_pretrained`` writes only on rank 0, so every rank takes rank
    0's ``tmp_path``.
    """
    paths = [tmp_path]
    torch.distributed.broadcast_object_list(paths, src=0)
    if torch.distributed.get_rank() == 0:
        model.save_pretrained(paths[0])
    torch.distributed.barrier()
    return paths[0]


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

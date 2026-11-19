import torch
from megatron.core import parallel_state

# Largest relative error allowed against an fp32 reference, by the dtype of the module's
# parameters. bf16 keeps 8 bits of mantissa, and errors grow through the layers.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 5e-2}


def take_shard(tensor: torch.Tensor, dim: int | None) -> torch.Tensor:
    """Cut out the part of an unsplit reference ``tensor`` that this tensor parallel rank holds.

    The tests use it to copy reference weights into a module split over the TP ranks and to
    compare its gradients with the reference ones. ``dim`` is the dim that tensor parallelism
    splits; for None, which marks a parameter that every rank holds whole, return ``tensor``.
    """
    if dim is None:
        return tensor
    size = parallel_state.get_tensor_model_parallel_world_size()
    return tensor.chunk(size, dim)[parallel_state.get_tensor_model_parallel_rank()]


def calc_relative_error(value: torch.Tensor, target: torch.Tensor) -> float:
    """Divide the largest absolute difference by the largest absolute value of ``target``."""
    return ((value.float() - target).abs().max() / target.abs().max()).item()

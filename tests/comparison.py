import torch
from megatron.core import parallel_state


def take_shard(tensor: torch.Tensor, dim: int | None) -> torch.Tensor:
    """This rank's part of ``tensor`` split along ``dim``, or all of it for None."""
    if dim is None:
        return tensor
    size = parallel_state.get_tensor_model_parallel_world_size()
    return tensor.chunk(size, dim)[parallel_state.get_tensor_model_parallel_rank()]


def calc_relative_error(value: torch.Tensor, target: torch.Tensor) -> float:
    """Largest absolute difference divided by the largest absolute target value."""
    return ((value.float() - target).abs().max() / target.abs().max()).item()

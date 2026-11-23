import torch

# Largest relative error allowed against an fp32 reference, by the dtype of the module's
# parameters. bf16 keeps 8 bits of mantissa, and errors grow through the layers.
TOLERANCES = {torch.float32: 1e-5, torch.bfloat16: 2e-2}


def calc_relative_error(value: torch.Tensor, target: torch.Tensor) -> float:
    """Divide the largest absolute difference by the largest absolute value of ``target``."""
    return ((value.float() - target).abs().max() / target.abs().max()).item()


def calc_norm_relative_error(value: torch.Tensor, target: torch.Tensor) -> float:
    """Divide the norm of the difference by the norm of ``target``."""
    return ((value.float() - target).norm() / target.norm()).item()

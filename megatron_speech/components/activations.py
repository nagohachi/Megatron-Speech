from collections.abc import Callable
from functools import partial

import torch

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": torch.nn.functional.gelu,  # exact (erf) GELU, as torch.nn.GELU() and HF ACT2FN["gelu"]
    "gelu_tanh": partial(torch.nn.functional.gelu, approximate="tanh"),
    "relu": torch.nn.functional.relu,
    "silu": torch.nn.functional.silu,
}

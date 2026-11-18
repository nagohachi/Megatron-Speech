"""Fixtures shared by the tests.

GPU tests run in one process (``python -m pytest``) or with tensor parallelism over several
processes (``python -m torch.distributed.run --nproc-per-node 2 -m pytest``).
"""

import os

# Tests compare against fp32 references; TF32 GEMMs, which the NGC container enables in cuBLAS
# and cuDNN, would add about 1e-3 relative error. This must be set before CUDA initializes.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import socket
from collections.abc import Iterator

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


@pytest.fixture(scope="session")
def tensor_parallel_size() -> Iterator[int]:
    """Set up torch.distributed and Megatron's process groups with TP over all processes."""
    if "MASTER_PORT" not in os.environ:  # one process started by pytest, not by torch.distributed
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    model_parallel_cuda_manual_seed(1234)
    yield world_size
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

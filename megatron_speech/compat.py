"""Thin layer over the Megatron-LM APIs that Megatron-Speech depends on.

Every import from ``megatron.*`` outside of model-building code goes through this module, so
that an upstream rename or signature change is fixed in one place. Model code may still import
Megatron Core building blocks (``TransformerConfig``, ``ModuleSpec``, layers) directly; those
are stable public APIs of Megatron Core.

Pinned upstream: third_party/Megatron-LM (see ``git submodule status``).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import torch

# --------------------------------------------------------------------------------------------
# Argument parsing and run configuration
# --------------------------------------------------------------------------------------------


def convert_dict_to_argv(arg_dict: dict[str, Any]) -> list[str]:
    """Convert ``{"micro-batch-size": 1, "bf16": True}`` into Megatron CLI tokens.

    Keys may use ``-`` or ``_``. ``True`` becomes a bare flag, ``False``/``None`` are dropped
    (write the explicit ``no-...`` flag to turn a default-on option off), lists become
    space-separated values.
    """
    argv: list[str] = []
    for key, value in arg_dict.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    return argv


@contextmanager
def _patched_argv(argv: list[str]):
    saved = sys.argv
    sys.argv = [saved[0]] + argv
    try:
        yield
    finally:
        sys.argv = saved


def parse_megatron_args(
    arg_dict: dict[str, Any],
    extra_args_provider: Callable | None = None,
    args_defaults: dict[str, Any] | None = None,
):
    """Parse, validate and register Megatron's global ``args`` from a dict.

    Returns the ``argparse.Namespace`` that ``megatron.training.get_args()`` will return.
    """
    from megatron.training.arguments import parse_and_validate_args

    with _patched_argv(convert_dict_to_argv(arg_dict)):
        return parse_and_validate_args(
            extra_args_provider=extra_args_provider, args_defaults=args_defaults or {}
        )


def build_transformer_config_from_args(args):
    """Return the LLM ``TransformerConfig`` that Megatron derives from ``args``."""
    from megatron.training.argument_utils import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def build_pretrain_config(args, model_cfg, data_cfg=None):
    """Build the ``PretrainConfigContainer`` that ``pretrain()`` consumes and register it."""
    from megatron.training.argument_utils import pretrain_cfg_container_from_args
    from megatron.training.global_vars import set_run_config

    cfg = pretrain_cfg_container_from_args(args, model_cfg, data_cfg=data_cfg)
    set_run_config(cfg)
    return cfg


def get_args():
    from megatron.training import get_args as _get_args

    return _get_args()


def get_run_config():
    from megatron.training.global_vars import get_run_config as _get_run_config

    return _get_run_config()


def run_pretrain(cfg, datasets_provider, forward_step_func, **kwargs):
    """Run Megatron's training loop (``megatron.training.pretrain``)."""
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    return pretrain(cfg, datasets_provider, ModelType.encoder_or_decoder, forward_step_func, **kwargs)


def print_rank_0(message: str) -> None:
    from megatron.training import print_rank_0 as _print_rank_0

    _print_rank_0(message)


def get_timers():
    from megatron.training import get_timers as _get_timers

    return _get_timers()


# --------------------------------------------------------------------------------------------
# GPT sanity recipe (upstream GPT model, datasets and forward step)
# --------------------------------------------------------------------------------------------


def build_gpt_model_config(args):
    from megatron.training.argument_utils import gpt_config_from_args

    return gpt_config_from_args(args, vocab_size_from_tokenizer=True)


def build_gpt_data_config(args):
    from megatron.training.argument_utils import gpt_data_config_from_args

    return gpt_data_config_from_args(args)


def get_upstream_gpt_step_functions():
    """Return ``(forward_step, datasets_provider, get_embedding_ranks)`` of ``pretrain_gpt.py``."""
    import pretrain_gpt  # third_party/Megatron-LM is on sys.path via the venv .pth file

    provider = pretrain_gpt.train_valid_test_datasets_provider
    provider.is_distributed = True
    return pretrain_gpt.forward_step, provider, pretrain_gpt.get_embedding_ranks


# --------------------------------------------------------------------------------------------
# Typed model configs and builders
# --------------------------------------------------------------------------------------------


def get_model_config_base():
    from megatron.training.models.base import ModelConfig

    return ModelConfig


def get_model_builder_base():
    from megatron.training.models.base import ModelBuilder

    return ModelBuilder


def build_distributed_models(build_model_func, transformer_config, pg_collection, **kwargs):
    """Build model chunks, run ``pre_wrap_hook``, move to GPU and wrap with Float16Module/DDP."""
    from megatron.training.models.dist_utils import unimodal_build_distributed_models

    return unimodal_build_distributed_models(
        build_model_func, transformer_config, pg_collection, **kwargs
    )


def compose_hooks(hooks: Iterable[Callable]):
    from megatron.training.models.base import compose_hooks as _compose_hooks

    return _compose_hooks(list(hooks))


def unwrap_model(model):
    from megatron.core.utils import unwrap_model as _unwrap_model

    return _unwrap_model(model)


# --------------------------------------------------------------------------------------------
# Parallel state
# --------------------------------------------------------------------------------------------


def get_tensor_model_parallel_group():
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_group()


def get_tensor_model_parallel_rank() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_rank()


def get_tensor_model_parallel_world_size() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_world_size()


def get_data_parallel_rank() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_data_parallel_rank()


def get_data_parallel_world_size() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_data_parallel_world_size()


def get_pipeline_model_parallel_world_size() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_pipeline_model_parallel_world_size()


def get_context_parallel_world_size() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_context_parallel_world_size()


def initialize_model_parallel_for_tests(
    tensor_model_parallel_size: int = 1, seed: int = 1234
) -> None:
    """Initialize torch.distributed and Megatron parallel state outside of ``pretrain()``.

    Used by tests and conversion scripts. Reads the usual torchrun environment variables and
    falls back to a single-process group.
    """
    import os

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size
        )
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)


def destroy_model_parallel_for_tests() -> None:
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_default_pg_collection():
    from megatron.core.process_groups_config import ProcessGroupCollection

    return ProcessGroupCollection.use_mpu_process_groups()


# --------------------------------------------------------------------------------------------
# Tensor-parallel autograd regions (used by LoRA adapters)
# --------------------------------------------------------------------------------------------


def copy_to_tensor_model_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Identity in forward, all-reduce over the TP group in backward."""
    from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region as f

    return f(x)


def reduce_from_tensor_model_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """All-reduce over the TP group in forward, identity in backward."""
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region as f

    return f(x)


def make_sharded_tensors_for_checkpoint(state_dict, prefix, tp_axis_map, sharded_offsets):
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint as f

    return f(state_dict, prefix, tp_axis_map, sharded_offsets)


# --------------------------------------------------------------------------------------------
# Training-step helpers
# --------------------------------------------------------------------------------------------


def get_rerun_state_machine():
    from megatron.core.rerun_state_machine import get_rerun_state_machine as f

    return f()


def build_packed_seq_params(cu_seqlens: torch.Tensor, max_seqlen: int, cu_seqlens_padded=None):
    """``PackedSeqParams`` for a THD (packed, no padding between sequences) batch."""
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def load_megatron_checkpoint(model_chunks: list, load_dir: str) -> int:
    """Load model weights (no optimizer/RNG state) from a Megatron checkpoint; returns the
    iteration. Requires ``build_pretrain_config`` to have registered the run config."""
    from megatron.training.checkpointing import load_checkpoint

    out = load_checkpoint(
        model_chunks, None, None, load_dir=load_dir, finetune=False, load_optim=False, load_rng=False
    )
    return out[0] if isinstance(out, tuple) else out

"""Recipes tie a run config to a model config, a data provider and a forward step.

``model.type`` in the run config selects the recipe:

* ``gpt``         upstream Megatron GPT model and datasets (sanity check of the entry point)
"""

import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from typing import TypedDict

import torch
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.training.argument_utils import (
    gpt_config_from_args,
    gpt_data_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_and_validate_args
from megatron.training.config import DataConfig
from megatron.training.global_vars import set_run_config
from megatron.training.models.base import ModelConfig
from torch.utils.data import Dataset

from megatron_speech.training.config import ConfigDict

type LossFunc = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]
type ForwardStep = Callable[..., tuple[torch.Tensor, LossFunc]]
type DatasetsProvider = Callable[..., tuple[Dataset | None, Dataset | None, Dataset | None]]


class PretrainKwargs(TypedDict, total=False):
    """Optional keyword arguments of ``megatron.training.pretrain`` that recipes set."""

    get_embedding_ranks: Callable[[list[int]], list[int]]


class Recipe:
    """Base recipe. Subclasses override the hooks that differ between model families."""

    def __init__(self, cfg: ConfigDict) -> None:
        """Keep the run config that ``RunConfigLoader.load`` returned."""
        self.cfg = cfg

    # -- hooks -------------------------------------------------------------------------------
    def build_megatron_arg_dict(self) -> ConfigDict:
        """Build the Megatron CLI arguments; recipes add the ones derived from the model."""
        return dict(self.cfg["megatron"])

    def get_extra_args_provider(self) -> Callable[[ArgumentParser], ArgumentParser] | None:
        """Return a function that adds recipe-specific arguments to Megatron's parser, if any."""
        return None

    def build_model_config(self, args: Namespace) -> ModelConfig:
        """Build the model section of the ``PretrainConfigContainer`` from Megatron's args."""
        raise NotImplementedError

    def build_data_config(self, args: Namespace) -> DataConfig | None:
        """Build the data section; None keeps only Megatron's data loader settings."""
        return None

    def get_datasets_provider(self) -> DatasetsProvider:
        """Return the function that ``pretrain()`` calls to build the train, valid and test data."""
        raise NotImplementedError

    def get_forward_step(self) -> ForwardStep:
        """Return the function that ``pretrain()`` calls to run a micro-batch through the model."""
        raise NotImplementedError

    def build_pretrain_kwargs(self) -> PretrainKwargs:
        """Return the optional keyword arguments that this recipe passes to ``pretrain()``."""
        return {}

    # -- driver ------------------------------------------------------------------------------
    def run(self) -> None:
        """Parse Megatron's args, build the pretrain config and run Megatron's training loop.

        ``build_megatron_arg_dict()`` becomes Megatron CLI tokens: keys may use ``-`` or ``_``,
        ``True`` is a bare flag, ``False`` and ``None`` drop the flag (write the explicit
        ``no-...`` flag to turn a default-on option off), and lists become space-separated values.
        """
        argv: list[str] = []
        for key, value in self.build_megatron_arg_dict().items():
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

        # parse_and_validate_args reads sys.argv and registers the result for get_args().
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], *argv]
        try:
            args = parse_and_validate_args(extra_args_provider=self.get_extra_args_provider())
        finally:
            sys.argv = saved_argv

        pretrain_config = pretrain_cfg_container_from_args(
            args, self.build_model_config(args), data_cfg=self.build_data_config(args)
        )
        set_run_config(pretrain_config)
        pretrain(
            pretrain_config,
            self.get_datasets_provider(),
            ModelType.encoder_or_decoder,
            self.get_forward_step(),
            **self.build_pretrain_kwargs(),
        )


class GPTRecipe(Recipe):
    """Upstream GPT model through the Megatron-Speech entry point."""

    def build_model_config(self, args: Namespace) -> ModelConfig:
        """Build the upstream ``GPTModelConfig`` from Megatron's args."""
        return gpt_config_from_args(args)

    def build_data_config(self, args: Namespace) -> DataConfig | None:
        """Build the upstream GPT data section (``--data-path``, ``--mock-data``, ...)."""
        return gpt_data_config_from_args(args)

    def get_datasets_provider(self) -> DatasetsProvider:
        """Return the datasets provider of ``pretrain_gpt.py``."""
        import pretrain_gpt  # third_party/Megatron-LM is on sys.path via the venv .pth file

        provider = pretrain_gpt.train_valid_test_datasets_provider
        provider.is_distributed = True
        return provider

    def get_forward_step(self) -> ForwardStep:
        """Return the forward step of ``pretrain_gpt.py``."""
        import pretrain_gpt

        return pretrain_gpt.forward_step

    def build_pretrain_kwargs(self) -> PretrainKwargs:
        """Pass ``get_embedding_ranks`` of ``pretrain_gpt.py``, as its ``__main__`` block does."""
        import pretrain_gpt

        return {"get_embedding_ranks": pretrain_gpt.get_embedding_ranks}


def get_recipe(model_type: str) -> type[Recipe]:
    """Return the recipe class that ``model.type`` names."""
    if model_type == "gpt":
        return GPTRecipe
    raise ValueError(f"unknown model.type {model_type!r} (expected 'gpt')")

"""Recipes tie a run config to a model config, a data provider and a forward step.

``model.type`` in the run config selects the recipe:

* ``gpt``         upstream Megatron GPT model and datasets (sanity check of the entry point)
"""

from __future__ import annotations

from typing import Any, Callable

from megatron_speech import compat


class Recipe:
    """Base recipe. Subclasses override the hooks that differ between model families."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    # -- hooks -------------------------------------------------------------------------------
    def megatron_arg_dict(self) -> dict[str, Any]:
        """Megatron CLI arguments; recipes add the ones derived from the model."""
        return dict(self.cfg["megatron"])

    def extra_args_provider(self) -> Callable | None:
        return None

    def build_model_config(self, args):
        raise NotImplementedError

    def build_data_config(self, args):
        return None

    def datasets_provider(self):
        raise NotImplementedError

    def forward_step(self):
        raise NotImplementedError

    def pretrain_kwargs(self) -> dict[str, Any]:
        return {}

    # -- driver ------------------------------------------------------------------------------
    def run(self) -> None:
        args = compat.parse_megatron_args(
            self.megatron_arg_dict(), extra_args_provider=self.extra_args_provider()
        )
        model_cfg = self.build_model_config(args)
        run_cfg = compat.build_pretrain_config(args, model_cfg, self.build_data_config(args))
        compat.run_pretrain(
            run_cfg, self.datasets_provider(), self.forward_step(), **self.pretrain_kwargs()
        )


class GPTRecipe(Recipe):
    """Upstream GPT model through the Megatron-Speech entry point."""

    def build_model_config(self, args):
        return compat.build_gpt_model_config(args)

    def build_data_config(self, args):
        return compat.build_gpt_data_config(args)

    def datasets_provider(self):
        return compat.get_upstream_gpt_step_functions()[1]

    def forward_step(self):
        return compat.get_upstream_gpt_step_functions()[0]

    def pretrain_kwargs(self):
        return {"get_embedding_ranks": compat.get_upstream_gpt_step_functions()[2]}


def get_recipe(model_type: str) -> type[Recipe]:
    if model_type == "gpt":
        return GPTRecipe
    raise ValueError(f"unknown model.type {model_type!r} (expected 'gpt')")

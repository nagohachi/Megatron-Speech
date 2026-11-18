"""Training entry point.

    torchrun --nproc-per-node N -m megatron_speech.training.train \
        --config configs/qwen3_asr/finetune_librispeech.yaml [key.subkey=value ...]
"""

import argparse

from megatron_speech.training.config import load_config
from megatron_speech.training.recipes import get_recipe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, help="run config (YAML)")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. megatron.lr=1e-5")
    ns = parser.parse_args(argv)

    cfg = load_config(ns.config, ns.overrides)
    model_type = cfg["model"].get("type")
    if model_type is None:
        raise ValueError("model.type is required in the run config")
    get_recipe(model_type)(cfg).run()


if __name__ == "__main__":
    main()

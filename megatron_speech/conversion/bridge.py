import shutil
from collections.abc import Iterator
from pathlib import Path

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict
from megatron.core.transformer import MegatronModule

# Files of a Hugging Face checkpoint besides the weights: config, tokenizer, processor.
NON_WEIGHT_FILES = ["*.json", "*.txt", "*.jinja"]
# Index of the weights of a checkpoint sharded over several safetensors files.
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"


class SafeTensorsCheckpoint:
    """Checkpoint directory of safetensors files that transformers cannot load.

    Kyutai's Moshi checkpoints, for example, have no config.json. ``MappingBridge`` takes this
    class where it takes Megatron-Bridge's ``PreTrainedCausalLM``, which loads the config with
    transformers. The weights are those of every safetensors file in the directory. Without a
    config that gives the vocabulary size, the export keeps the rows that Megatron adds to pad
    the vocabulary.
    """

    def __init__(self, path: Path) -> None:
        """Index the weights of the safetensors files in ``path``; a tensor loads when read."""
        # The attributes of a Hugging Face checkpoint that Megatron-Bridge's conversion reads.
        self.model_name_or_path = path
        self.state = StateDict(SafeTensorsStateSource(path))


class MappingBridge(MegatronModelBridge):
    """Megatron-Bridge's weight conversion with a mapping table that the caller gives.

    - Table: each mapping relates Megatron parameter names to Hugging Face weight names
      (``*`` matches a layer number).
    - Loading: the base class's ``load_weights_hf_to_megatron`` reads Hugging Face weights
      into a model that Megatron-Speech builds.
    - Writing back: the base class's ``stream_weights_megatron_to_hf``, and
      ``save_hf_checkpoint`` and ``save_safetensors_checkpoint`` for a whole checkpoint
      directory.
    - Tensor parallelism: the mappings split the weights over the ranks and gather them back.
    """

    def __init__(self, mappings: list[MegatronParamMapping]) -> None:
        """Keep the mappings that relate the model's parameters to Hugging Face weights."""
        super().__init__()
        self.mappings = mappings

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the registry of the mappings, through which the base class converts."""
        return MegatronMappingRegistry(*self.mappings)

    def save_hf_checkpoint(
        self, models: list[MegatronModule], hf_pretrained: PreTrainedCausalLM, path: Path
    ) -> None:
        """Write ``models`` as a Hugging Face checkpoint in ``path``, laid out as ``hf_pretrained``.

        - Weights: the models' values, in the safetensors files of ``hf_pretrained``.
        - Other files: the config, tokenizer and processor files of ``hf_pretrained``.

        Every rank calls this, since gathering the weights over the tensor parallel ranks is
        collective; rank 0 writes the files, and every rank returns once they are written.
        """
        if torch.distributed.get_rank() == 0:
            hf_pretrained.save_artifacts(path, additional_files=NON_WEIGHT_FILES)
        # The writer that Megatron-Bridge's AutoBridge.save_hf_weights also calls. AutoBridge
        # itself picks Megatron-Bridge's own table for the architecture, not this bridge's.
        hf_pretrained.state.source.save_generator(
            self.stream_weights_megatron_to_hf(models, hf_pretrained, show_progress=False), path
        )
        torch.distributed.barrier()

    def save_safetensors_checkpoint(
        self, models: list[MegatronModule], checkpoint: SafeTensorsCheckpoint, path: Path
    ) -> None:
        """Write ``models`` in ``path``, laid out as ``checkpoint``.

        Every rank calls this, as ``save_hf_checkpoint``; rank 0 writes the files.

        Parameters
        ----------
        models
            Models whose parameters the mappings write.
        checkpoint
            Checkpoint whose files ``path`` gets. The weights that the mappings write take the
            models' values; the other weights, such as those of a codec that the models do not
            hold, are copied from ``checkpoint``, and so are the files besides the safetensors
            files and their index.
        path
            Directory to write.
        """
        rank = torch.distributed.get_rank()

        def stream_weights() -> Iterator[tuple[str, torch.Tensor]]:
            """Yield the models' weights, then on rank 0 the other weights of ``checkpoint``."""
            converted_names = set()
            for name, tensor in self.stream_weights_megatron_to_hf(
                models, checkpoint, show_progress=False
            ):
                converted_names.add(name)
                yield name, tensor
            if rank == 0:
                for name in checkpoint.state:
                    if name not in converted_names:
                        yield name, checkpoint.state[name]

        if rank == 0:
            path.mkdir(parents=True, exist_ok=True)
            for source_file in checkpoint.model_name_or_path.iterdir():
                # save_generator writes the safetensors files and, for a sharded checkpoint, an
                # index of the files that it writes.
                if (
                    source_file.is_file()
                    and source_file.suffix != ".safetensors"
                    and source_file.name != SAFETENSORS_INDEX_FILE
                ):
                    shutil.copy2(source_file, path / source_file.name)
        checkpoint.state.source.save_generator(stream_weights(), path)
        torch.distributed.barrier()

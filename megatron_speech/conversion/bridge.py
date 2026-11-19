from pathlib import Path

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.transformer import MegatronModule

# Files of a Hugging Face checkpoint besides the weights: config, tokenizer, processor.
NON_WEIGHT_FILES = ["*.json", "*.txt", "*.jinja"]


class MappingBridge(MegatronModelBridge):
    """Megatron-Bridge's weight conversion with a mapping table that the caller gives.

    - Table: each mapping relates Megatron parameter names to Hugging Face weight names
      (``*`` matches a layer number).
    - Loading: the base class's ``load_weights_hf_to_megatron`` reads Hugging Face weights
      into a model that Megatron-Speech builds.
    - Writing back: the base class's ``stream_weights_megatron_to_hf``, and
      ``save_hf_checkpoint`` for a whole checkpoint directory.
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
        collective; rank 0 writes the files.
        """
        if torch.distributed.get_rank() == 0:
            hf_pretrained.save_artifacts(path, additional_files=NON_WEIGHT_FILES)
        hf_pretrained.state.source.save_generator(
            self.stream_weights_megatron_to_hf(models, hf_pretrained, show_progress=False), path
        )

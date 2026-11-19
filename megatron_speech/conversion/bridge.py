from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping


class MappingBridge(MegatronModelBridge):
    """Megatron-Bridge's weight conversion, driven by a mapping table that the caller gives.

    The base class's ``load_weights_hf_to_megatron`` and ``stream_weights_megatron_to_hf``
    read Hugging Face weights into a model that Megatron-Speech builds, and write them back.
    Each mapping relates Megatron parameter names (``*`` matches a layer number) to Hugging
    Face weight names, and splits the weights over the tensor parallel ranks.
    """

    def __init__(self, mappings: list[MegatronParamMapping]) -> None:
        """Keep the mappings that relate the model's parameters to Hugging Face weights."""
        super().__init__()
        self.mappings = mappings

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the registry of the mappings, through which the base class converts."""
        return MegatronMappingRegistry(*self.mappings)

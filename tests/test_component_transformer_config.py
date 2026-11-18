import torch
from megatron.core.transformer import TransformerConfig

from megatron_speech.components.transformer_config import (
    INHERITED_FIELDS,
    build_component_transformer_config,
)


def test_takes_parallelism_and_precision_from_llm() -> None:
    """TP size and precision come from the LLM; sequence, context and pipeline parallelism not."""
    llm_config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        context_parallel_size=2,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        init_method_std=0.01,
    )
    config = build_component_transformer_config(
        llm_config, num_layers=1, hidden_size=32, num_attention_heads=2
    )
    assert {name: getattr(config, name) for name in INHERITED_FIELDS} == {
        name: getattr(llm_config, name) for name in INHERITED_FIELDS
    }
    assert config.sequence_parallel is False
    assert config.context_parallel_size == 1
    assert config.pipeline_model_parallel_size == 1

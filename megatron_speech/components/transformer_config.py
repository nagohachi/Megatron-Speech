from megatron.core.transformer import TransformerConfig

# Parallelism and precision settings that every component takes from the LLM config.
INHERITED_FIELDS = (
    "tensor_model_parallel_size",
    "params_dtype",
    "bf16",
    "fp16",
    "use_cpu_initialization",
    "perform_initialization",
    "gradient_accumulation_fusion",
    "deterministic_mode",
    "attention_backend",
    "init_method_std",
    "masked_softmax_fusion",
    "bias_dropout_fusion",
    "use_te_rng_tracker",
)


def build_component_transformer_config(
    llm_config: TransformerConfig, **arch: int | bool
) -> TransformerConfig:
    """Build the ``TransformerConfig`` of a component.

    The component takes tensor parallelism and precision from the LLM and runs without
    sequence, context or pipeline parallelism: it sits on the first pipeline stage and sees
    whole sequences. ``arch`` sets the architecture fields (``num_layers``, ``hidden_size``, ...).
    """
    return TransformerConfig(
        **{name: getattr(llm_config, name) for name in INHERITED_FIELDS},
        sequence_parallel=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        tp_comm_overlap=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **arch,
    )

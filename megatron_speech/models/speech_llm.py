from dataclasses import dataclass

from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training.models.gpt import GPTModelConfig, default_layer_spec
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron_speech.components.encoders.qwen3_asr import Qwen3ASREncoder, Qwen3ASREncoderConfig
from megatron_speech.components.projectors.mlp import MLPProjector, MLPProjectorConfig
from megatron_speech.models.language_model import SpeechLanguageModel

# Names of the audio modality and of its encoder inside MimoModel. Parameter names, and so
# checkpoints, contain them: modality_submodules.audio.encoders.encoder.*.
AUDIO_MODALITY = "audio"
ENCODER_NAME = "encoder"


@dataclass
class SpeechLLMConfig:
    """Shape of the speech LLM: the LLM, the audio encoder and the projector between them.

    Each ``audio_token_id`` in the LLM's input takes one projected encoder frame, in order.
    """

    llm_config: GPTModelConfig
    encoder_config: Qwen3ASREncoderConfig
    projector_config: MLPProjectorConfig
    audio_token_id: int


def build_speech_llm(config: SpeechLLMConfig, pg_collection: ProcessGroupCollection) -> MimoModel:
    """Build the speech LLM as a ``MimoModel`` that holds every module on every rank.

    The encoder and the projector take the tensor parallelism and precision of the LLM.
    """
    llm_config = config.llm_config
    transformer_config = llm_config.transformer

    def check_config() -> None:
        """Raise ValueError for a config that the speech LLM cannot be built from."""
        if transformer_config.pipeline_model_parallel_size != 1:
            raise ValueError(
                "a MimoModel with every module on every rank needs pipeline_model_parallel_size=1, "
                f"got {transformer_config.pipeline_model_parallel_size}"
            )
        # The target loss over packed rows is tested without context parallelism only.
        if transformer_config.context_parallel_size != 1:
            raise ValueError(
                "the speech LLM does not support context parallelism; it needs "
                f"context_parallel_size=1, got {transformer_config.context_parallel_size}"
            )
        if config.projector_config.input_size != config.encoder_config.d_model:
            raise ValueError(
                f"projector_config.input_size={config.projector_config.input_size} differs from "
                f"encoder_config.d_model={config.encoder_config.d_model}"
            )
        if config.projector_config.output_size != transformer_config.hidden_size:
            raise ValueError(
                f"projector_config.output_size={config.projector_config.output_size} differs from "
                f"llm_config.hidden_size={transformer_config.hidden_size}"
            )
        if llm_config.vocab_size is None:
            raise ValueError("llm_config.vocab_size is not set")
        tp_size = transformer_config.tensor_model_parallel_size
        if not llm_config.should_pad_vocab and llm_config.vocab_size % tp_size:
            raise ValueError(
                f"llm_config.vocab_size={llm_config.vocab_size} cannot be split over {tp_size} "
                "tensor parallel ranks; set llm_config.should_pad_vocab to pad it"
            )
        if not 0 <= config.audio_token_id < llm_config.vocab_size:
            raise ValueError(
                f"audio_token_id={config.audio_token_id} is outside the vocabulary of "
                f"llm_config.vocab_size={llm_config.vocab_size} tokens"
            )

    def build_audio_spec() -> ModuleSpec:
        """Build the spec of the audio modality: the encoder and the projector."""
        component_params = {"llm_config": transformer_config, "pg_collection": pg_collection}
        return ModuleSpec(
            module=AudioModalitySubmodules,
            submodules={
                "encoders": {
                    ENCODER_NAME: ModuleSpec(
                        module=Qwen3ASREncoder,
                        params={"encoder_config": config.encoder_config, **component_params},
                    )
                },
                "input_projections": [
                    ModuleSpec(
                        module=MLPProjector,
                        params={"projector_config": config.projector_config, **component_params},
                    )
                ],
            },
        )

    def build_language_model_spec() -> ModuleSpec:
        """Build the spec of the LLM from the fields of ``config.llm_config``."""
        vocab_size = llm_config.vocab_size
        if llm_config.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size,
                llm_config.make_vocab_size_divisible_by,
                transformer_config.tensor_model_parallel_size,
            )
        return ModuleSpec(
            module=SpeechLanguageModel,
            params={
                "config": transformer_config,
                "transformer_layer_spec": default_layer_spec(llm_config, vp_stage=None),
                "vocab_size": vocab_size,
                "max_sequence_length": llm_config.seq_length,
                "fp16_lm_cross_entropy": llm_config.fp16_lm_cross_entropy,
                "logit_dtype": llm_config.logit_dtype,
                "parallel_output": llm_config.parallel_output,
                "share_embeddings_and_output_weights": (
                    llm_config.share_embeddings_and_output_weights
                ),
                "position_embedding_type": llm_config.position_embedding_type,
                "rotary_percent": llm_config.rotary_percent,
                "rotary_base": llm_config.rotary_base,
                "rope_scaling": llm_config.rope_scaling,
                "rope_scaling_factor": llm_config.rope_scaling_factor,
                "seq_len_interpolation_factor": llm_config.seq_len_interpolation_factor,
                # MimoModel scatters the merged embeddings over the sequence parallel ranks.
                "scatter_embedding_sequence_parallel": False,
                "pg_collection": pg_collection,
            },
        )

    check_config()
    mimo_config = MimoModelConfig(
        language_model_spec=build_language_model_spec(),
        modality_submodules_spec={AUDIO_MODALITY: build_audio_spec()},
        special_token_ids={AUDIO_MODALITY: config.audio_token_id},
        kv_format="thd",
    )
    return MimoModel(mimo_config, cp_group=pg_collection.cp, tp_group=pg_collection.tp)

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

    llm: GPTModelConfig
    encoder: Qwen3ASREncoderConfig
    projector: MLPProjectorConfig
    audio_token_id: int


def build_speech_llm(config: SpeechLLMConfig, pg_collection: ProcessGroupCollection) -> MimoModel:
    """Build the speech LLM as a ``MimoModel`` that holds every module on every rank.

    The encoder and the projector take the tensor parallelism and precision of the LLM.
    """
    llm = config.llm
    transformer = llm.transformer

    def check_config() -> None:
        """Raise ValueError for a config that the speech LLM cannot be built from."""
        if transformer.pipeline_model_parallel_size != 1:
            raise ValueError(
                "a MimoModel with every module on every rank needs "
                f"pipeline_model_parallel_size=1, got {transformer.pipeline_model_parallel_size}"
            )
        if config.projector.input_size != config.encoder.d_model:
            raise ValueError(
                f"projector.input_size={config.projector.input_size} differs from "
                f"encoder.d_model={config.encoder.d_model}"
            )
        if config.projector.output_size != transformer.hidden_size:
            raise ValueError(
                f"projector.output_size={config.projector.output_size} differs from "
                f"llm.hidden_size={transformer.hidden_size}"
            )
        if llm.vocab_size is None:
            raise ValueError("llm.vocab_size is not set")

    def build_audio_spec() -> ModuleSpec:
        """Build the spec of the audio modality: the encoder and the projector."""
        component_params = {"llm_config": transformer, "pg_collection": pg_collection}
        return ModuleSpec(
            module=AudioModalitySubmodules,
            submodules={
                "encoders": {
                    ENCODER_NAME: ModuleSpec(
                        module=Qwen3ASREncoder,
                        params={"encoder_config": config.encoder, **component_params},
                    )
                },
                "input_projections": [
                    ModuleSpec(
                        module=MLPProjector,
                        params={"projector_config": config.projector, **component_params},
                    )
                ],
            },
        )

    def build_language_model_spec() -> ModuleSpec:
        """Build the spec of the LLM from the fields of ``config.llm``."""
        vocab_size = llm.vocab_size
        if llm.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size, llm.make_vocab_size_divisible_by, transformer.tensor_model_parallel_size
            )
        return ModuleSpec(
            module=SpeechLanguageModel,
            params={
                "config": transformer,
                "transformer_layer_spec": default_layer_spec(llm, vp_stage=None),
                "vocab_size": vocab_size,
                "max_sequence_length": llm.seq_length,
                "fp16_lm_cross_entropy": llm.fp16_lm_cross_entropy,
                "logit_dtype": llm.logit_dtype,
                "parallel_output": llm.parallel_output,
                "share_embeddings_and_output_weights": llm.share_embeddings_and_output_weights,
                "position_embedding_type": llm.position_embedding_type,
                "rotary_percent": llm.rotary_percent,
                "rotary_base": llm.rotary_base,
                "rope_scaling": llm.rope_scaling,
                "rope_scaling_factor": llm.rope_scaling_factor,
                "seq_len_interpolation_factor": llm.seq_len_interpolation_factor,
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

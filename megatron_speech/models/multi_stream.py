from dataclasses import dataclass

import torch
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from megatron_speech.models.language_model import MultiStreamLanguageModel


@dataclass
class MultiStreamLosses:
    """Losses of one batch of ``MultiStreamMimoModel``, each a scalar.

    ``total_loss`` is ``text_loss_weight * text_loss + audio_loss_weight * audio_loss``.
    """

    text_loss: torch.Tensor
    audio_loss: torch.Tensor
    total_loss: torch.Tensor


class MultiStreamMimoModel(MimoModel):
    """``MimoModel`` that sums the embeddings at each position and predicts the text and the audio.

    Every call of ``forward`` passes the positions of each modality in
    ``modality_token_indices``; a position can hold several modalities. From the output of the
    LLM's last layer, the LLM's output layer predicts the text and the audio decoder the audio.
    """

    def __init__(
        self,
        mimo_config: MimoModelConfig,
        audio_decoder_spec: ModuleSpec,
        text_loss_weight: float,
        audio_loss_weight: float,
        tp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        """Build the modules of ``mimo_config`` and the audio decoder of ``audio_decoder_spec``.

        Parameters
        ----------
        mimo_config
            Specs of the LLM and of the encoders of the streams. The module of
            ``language_model_spec`` must be ``MultiStreamLanguageModel`` or its subclass, and
            the ``params`` of ``language_model_spec`` hold its ``config``, which ``MimoModel``
            reads as well.
        audio_decoder_spec
            Spec of the audio decoder. ``forward`` passes the decoder the LLM's last layer
            output at the positions that predict the audio, and the decoder returns the audio
            loss.
        text_loss_weight
            Weight of the text loss in the total loss.
        audio_loss_weight
            Weight of the audio loss in the total loss.
        tp_group
            Tensor parallel group of the LLM, which ``MimoModel`` needs for sequence
            parallelism. None takes the group of Megatron's ``parallel_state``.
        """
        language_model_spec = mimo_config.language_model_spec
        language_model_class = language_model_spec.module
        if not (
            isinstance(language_model_class, type)
            and issubclass(language_model_class, MultiStreamLanguageModel)
        ):
            raise ValueError(
                "language_model_spec.module must be MultiStreamLanguageModel or its subclass, "
                f"got {language_model_class}"
            )
        # MimoModel would split the sequence over the CP ranks, while the text loss and the
        # audio decoder take the positions of the whole sequence.
        context_parallel_size = language_model_spec.params["config"].context_parallel_size
        if context_parallel_size != 1:
            raise ValueError(
                "MultiStreamMimoModel does not support context parallelism, but the LLM's config "
                f"has context_parallel_size={context_parallel_size}"
            )
        super().__init__(mimo_config, tp_group=tp_group)
        self.audio_decoder = build_module(audio_decoder_spec)
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight

    def align_embeddings_by_token_positions(
        self,
        modality_embeddings: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        special_token_ids: dict[str, int],
        modality_token_indices: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Add the embeddings of each modality at its positions; return ``[seq, batch, hidden]``.

        ``modality_token_indices`` maps every modality of ``modality_embeddings``, ``"text"``
        included, to the flat indices ``batch_index * seq_length + position`` of its rows. The
        positions of different modalities may be the same; a position that no modality holds
        gets zeros. ``special_token_ids`` is not used.
        """
        if modality_token_indices is None:
            raise ValueError(
                "MultiStreamMimoModel needs modality_token_indices, the positions of each modality"
            )
        if modality_token_indices.keys() != modality_embeddings.keys():
            raise ValueError(
                f"modality_token_indices has {sorted(modality_token_indices)}, but the "
                f"embeddings are of {sorted(modality_embeddings)}"
            )
        for name, embeddings in modality_embeddings.items():
            num_positions = modality_token_indices[name].numel()
            if num_positions != embeddings.size(0):
                raise ValueError(
                    f"{name} has {embeddings.size(0)} embeddings but {num_positions} positions "
                    "in modality_token_indices"
                )
        text_embeddings = modality_embeddings["text"]
        batch_size, seq_length = input_ids.shape
        combined_embeddings = text_embeddings.new_zeros(
            batch_size * seq_length, text_embeddings.size(-1)
        )
        for name, embeddings in modality_embeddings.items():
            combined_embeddings.index_add_(
                0, modality_token_indices[name], embeddings.to(combined_embeddings.dtype)
            )
        return combined_embeddings.view(batch_size, seq_length, -1).transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        modality_inputs: dict[str, dict[str, dict[str, torch.Tensor]]],
        modality_token_indices: dict[str, torch.Tensor],
        audio_decoder_token_indices: torch.Tensor,
        audio_decoder_inputs: dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_padded: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> MultiStreamLosses:
        """Calculate the text loss, the audio loss and their weighted sum.

        ``input_ids``, ``position_ids``, ``labels`` and ``loss_mask`` are ``[batch, sequence]``.
        ``modality_inputs``, ``modality_token_indices``, ``cu_seqlens``, ``cu_seqlens_padded``
        and ``max_seqlen`` are those of ``MimoModel.forward``. ``loss_mask`` holds the weight of
        each position's cross entropy, and the text loss is their weighted sum divided by the
        sum of the weights, or 0 where the weights sum to 0.

        ``audio_decoder_token_indices`` holds the flat indices ``batch_index * seq_length +
        position`` of the positions from which the audio decoder predicts the audio. The audio
        decoder takes the LLM's last layer output at these positions, ``[num_indices, hidden]``,
        with ``audio_decoder_inputs`` as keyword arguments, and returns the audio loss.
        """
        hidden_states, _ = super().forward(
            input_ids,
            position_ids,
            modality_inputs=modality_inputs,
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            max_seqlen=max_seqlen,
            modality_token_indices=modality_token_indices,
        )
        text_losses = self.language_model.calc_target_loss(hidden_states, labels, loss_mask)
        loss_mask_sum = loss_mask.sum()
        # Where the weights sum to 0, the weighted sum is 0 too; dividing it by 1 instead of 0
        # keeps the loss and its gradients at 0 instead of NaN.
        text_loss = (text_losses * loss_mask).sum() / torch.where(
            loss_mask_sum > 0, loss_mask_sum, 1
        )
        # hidden_states is [sequence, batch, hidden].
        seq_length = input_ids.size(1)
        audio_loss = self.audio_decoder(
            hidden_states[
                audio_decoder_token_indices % seq_length, audio_decoder_token_indices // seq_length
            ],
            **audio_decoder_inputs,
        )
        return MultiStreamLosses(
            text_loss=text_loss,
            audio_loss=audio_loss,
            total_loss=self.text_loss_weight * text_loss + self.audio_loss_weight * audio_loss,
        )

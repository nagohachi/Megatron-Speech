import torch
from megatron.core.models.mimo import MimoModel


class MultiStreamMimoModel(MimoModel):
    """``MimoModel`` whose LLM input at a position is the sum of the embeddings placed there.

    Every call of ``forward`` passes the positions of each modality in
    ``modality_token_indices``; a position can hold several modalities.
    """

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

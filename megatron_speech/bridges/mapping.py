"""Parameter mappings between Hugging Face and Megatron layouts.

A mapping relates one Megatron parameter (full, unsharded) to one or more HF tensors:

* ``DirectMapping``     same tensor
* ``QKVMapping``        HF q/k/v projections <-> Megatron fused ``linear_qkv`` with the
                        per-query-group interleaving [q heads of group g, k_g, v_g]
* ``GatedMLPMapping``   HF gate/up projections <-> Megatron ``linear_fc1`` = [gate; up]

Tensor-parallel sharding of the full tensor is decided per parameter: a parameter whose local
shape equals the full shape is replicated, otherwise it is split along the one dimension that
differs. ``GatedMLPMapping`` splits gate and up separately because each TP rank holds
[gate_shard; up_shard].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Mapping:
    megatron_name: str
    hf_names: list[str] = field(default_factory=list)

    def to_megatron(self, hf: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def to_hf(self, full: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    # -- tensor parallelism -----------------------------------------------------------------
    def shard(self, full: torch.Tensor, local_shape: torch.Size, tp_rank: int, tp_size: int) -> torch.Tensor:
        if tuple(full.shape) == tuple(local_shape):
            return full
        dim = _find_partition_dim(full.shape, local_shape, tp_size, self.megatron_name)
        return full.chunk(tp_size, dim=dim)[tp_rank]

    def unshard(self, shards: list[torch.Tensor], partition_dim: int | None) -> torch.Tensor:
        if partition_dim is None or len(shards) == 1:
            return shards[0]
        return torch.cat(shards, dim=partition_dim)


def _find_partition_dim(full_shape, local_shape, tp_size: int, name: str) -> int:
    dims = [d for d, (f, l) in enumerate(zip(full_shape, local_shape)) if f != l]
    if len(full_shape) != len(local_shape) or len(dims) != 1 or full_shape[dims[0]] != local_shape[dims[0]] * tp_size:
        raise ValueError(
            f"{name}: cannot shard full shape {tuple(full_shape)} to local shape {tuple(local_shape)} "
            f"with tensor parallel size {tp_size}"
        )
    return dims[0]


@dataclass
class DirectMapping(Mapping):
    def __init__(self, megatron_name: str, hf_name: str):
        super().__init__(megatron_name, [hf_name])

    def to_megatron(self, hf):
        return hf[self.hf_names[0]]

    def to_hf(self, full):
        return {self.hf_names[0]: full}


@dataclass
class QKVMapping(Mapping):
    num_heads: int = 0
    num_query_groups: int = 0
    head_dim: int = 0

    def __init__(self, megatron_name, q, k, v, num_heads, num_query_groups, head_dim):
        super().__init__(megatron_name, [q, k, v])
        self.num_heads, self.num_query_groups, self.head_dim = num_heads, num_query_groups, head_dim

    def to_megatron(self, hf):
        q, k, v = (hf[n] for n in self.hf_names)
        is_bias = q.dim() == 1
        groups, hd = self.num_query_groups, self.head_dim
        heads_per_group = self.num_heads // groups
        rest = () if is_bias else (q.shape[-1],)
        q = q.reshape(groups, heads_per_group * hd, *rest)
        k = k.reshape(groups, hd, *rest)
        v = v.reshape(groups, hd, *rest)
        return torch.cat([q, k, v], dim=1).reshape(groups * (heads_per_group + 2) * hd, *rest)

    def to_hf(self, full):
        groups, hd = self.num_query_groups, self.head_dim
        heads_per_group = self.num_heads // groups
        rest = tuple(full.shape[1:])
        grouped = full.reshape(groups, (heads_per_group + 2) * hd, *rest)
        q, k, v = torch.split(grouped, [heads_per_group * hd, hd, hd], dim=1)
        q = q.reshape(self.num_heads * hd, *rest)
        k = k.reshape(groups * hd, *rest)
        v = v.reshape(groups * hd, *rest)
        return dict(zip(self.hf_names, (q.contiguous(), k.contiguous(), v.contiguous())))


@dataclass
class GatedMLPMapping(Mapping):
    def __init__(self, megatron_name, gate, up):
        super().__init__(megatron_name, [gate, up])

    def to_megatron(self, hf):
        return torch.cat([hf[self.hf_names[0]], hf[self.hf_names[1]]], dim=0)

    def to_hf(self, full):
        gate, up = full.chunk(2, dim=0)
        return {self.hf_names[0]: gate.contiguous(), self.hf_names[1]: up.contiguous()}

    def shard(self, full, local_shape, tp_rank, tp_size):
        if tuple(full.shape) == tuple(local_shape):
            return full
        gate, up = full.chunk(2, dim=0)
        return torch.cat([gate.chunk(tp_size, dim=0)[tp_rank], up.chunk(tp_size, dim=0)[tp_rank]], dim=0)

    def unshard(self, shards, partition_dim):
        if len(shards) == 1:
            return shards[0]
        assert partition_dim == 0, f"{self.megatron_name}: gated fc1 must be split on dim 0"
        halves = [s.chunk(2, dim=0) for s in shards]
        return torch.cat([h[0] for h in halves] + [h[1] for h in halves], dim=0)


class MappingTable:
    """All mappings of a model, looked up by Megatron parameter name."""

    def __init__(self, mappings: list[Mapping]):
        self.by_megatron: dict[str, Mapping] = {}
        for m in mappings:
            if m.megatron_name in self.by_megatron:
                raise ValueError(f"duplicate mapping for {m.megatron_name}")
            self.by_megatron[m.megatron_name] = m

    def __iter__(self):
        return iter(self.by_megatron.values())

    def get(self, megatron_name: str) -> Mapping | None:
        return self.by_megatron.get(megatron_name)

    def hf_names(self) -> set[str]:
        return {n for m in self for n in m.hf_names}

"""LoRA for Megatron (Transformer Engine) linear layers under tensor parallelism.

``LoRALinear`` replaces a linear submodule (e.g. ``linear_qkv``) and keeps the original module
as ``to_wrap``; the low-rank update lives in ``adapter``::

    y = base(x) + (alpha / r) * B(A(dropout(x')))

``x'`` is the input of the base matmul: for TE layers with a fused input norm
(``TELayerNormColumnParallelLinear``) it is ``norm(x)``, recomputed here in PyTorch so that the
LoRA path does not depend on TE's internal handling of the norm output.

Variants for fused layers (Megatron fuses q/k/v into ``linear_qkv`` and gate/up into
``linear_fc1``; HF keeps them separate):

* ``canonical`` (default): one (A, B) pair per HF projection, exactly the parameterization of
  HF PEFT with ``target_modules`` q/k/v/o/gate/up/down. Training follows the same trajectory
  as HF PEFT, and adapters convert one-to-one.
* ``fused``: one (A, B) pair for the fused layer; A is shared by q/k/v (gate/up). Fewer
  parameters; the export to PEFT repeats A per projection (exact for inference).

Tensor parallelism (no sequence parallelism):

* column-parallel base (output split over TP): A is replicated, B is split like the base
  weight. ``A x`` passes through ``copy_to_tensor_model_parallel_region`` (identity forward,
  all-reduce backward) so that the gradients of A and x are complete on every rank.
* row-parallel base (input split over TP): A is split like the base weight, ``A x`` is
  all-reduced in forward, B is replicated.

Checkpoints keep the base parameters under their original names (``...linear_qkv.weight``) and
add ``...linear_qkv.adapter.lora_A.weight`` and ``...adapter.lora_B.{i}.weight``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes

from megatron_speech import compat

VARIANTS = ("canonical", "fused")


def detect_parallel_mode(module: torch.nn.Module) -> str:
    """'column', 'row' or 'none' for a Megatron/TE linear layer."""
    mode = getattr(module, "parallel_mode", None)
    if mode in ("column", "row"):
        return mode
    if mode in (None, "duplicated"):
        name = type(module).__name__
        if "ColumnParallel" in name:
            return "column"
        if "RowParallel" in name:
            return "row"
        return "none"
    raise ValueError(f"unsupported parallel_mode {mode!r} on {type(module).__name__}")


def describe_fused_layout(name: str, module: torch.nn.Module) -> tuple[str, list[int]] | None:
    """For fused layers, ``(kind, local output rows per HF projection)``; else None.

    ``kind`` is ``"qkv"`` (rows interleaved per query group: [q_g, k_g, v_g] for each group g)
    or ``"concat"`` (rows are [gate; up]).
    """
    config = getattr(module, "config", None)
    tp = compat.get_tensor_model_parallel_world_size()
    out_local = module.weight.shape[0]
    if name == "linear_qkv" and config is not None:
        head_dim = config.kv_channels
        groups = config.num_query_groups // tp
        heads_per_group = config.num_attention_heads // config.num_query_groups
        assert groups * (heads_per_group + 2) * head_dim == out_local, "unexpected linear_qkv shape"
        return "qkv", [groups * heads_per_group * head_dim, groups * head_dim, groups * head_dim]
    if name == "linear_fc1" and config is not None and config.gated_linear_unit:
        return "concat", [out_local // 2, out_local // 2]
    return None


class LoRAAdapter(torch.nn.Module):
    """Low-rank update with ``len(split_rows)`` (A_i, B_i) pairs whose outputs are combined
    into the base layer's (possibly fused) output layout."""

    def __init__(self, in_features: int, split_rows: list[int], rank: int, alpha: float, dropout: float,
                 parallel_mode: str, dtype: torch.dtype, device: torch.device, kind: str = "single",
                 qkv_shape: tuple[int, int, int] | None = None):
        super().__init__()
        self.parallel_mode, self.kind, self.qkv_shape = parallel_mode, kind, qkv_shape
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.split_rows = list(split_rows)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        n = len(split_rows)
        self.lora_A = torch.nn.Linear(in_features, n * rank, bias=False, dtype=dtype, device=device)
        self.lora_B = torch.nn.ModuleList(
            torch.nn.Linear(rank, rows, bias=False, dtype=dtype, device=device) for rows in split_rows
        )
        # Same initialization as HF PEFT (each A_i is a [rank, in] block of lora_A). Replicated
        # parameters must be identical on all TP ranks: Megatron seeds the default generators
        # identically on every rank.
        for i in range(n):
            torch.nn.init.kaiming_uniform_(self.lora_A.weight[i * rank : (i + 1) * rank], a=math.sqrt(5))
        for b in self.lora_B:
            torch.nn.init.zeros_(b.weight)
            if parallel_mode == "column":
                set_tensor_model_parallel_attributes(b.weight, True, 0, 1)
        if parallel_mode == "row":
            set_tensor_model_parallel_attributes(self.lora_A.weight, True, 1, 1)

    def combine(self, outs: list[torch.Tensor]) -> torch.Tensor:
        """Arrange per-projection outputs (last dim = rows) into the fused layout."""
        if len(outs) == 1:
            return outs[0]
        if self.kind == "qkv":
            groups, heads_per_group, head_dim = self.qkv_shape
            lead = outs[0].shape[:-1]
            q = outs[0].reshape(*lead, groups, heads_per_group * head_dim)
            k = outs[1].reshape(*lead, groups, head_dim)
            v = outs[2].reshape(*lead, groups, head_dim)
            return torch.cat([q, k, v], dim=-1).reshape(*lead, -1)
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(self.dropout(x), self.lora_A.weight)
        if self.parallel_mode == "column":
            h = compat.copy_to_tensor_model_parallel_region(h)
        elif self.parallel_mode == "row":
            h = compat.reduce_from_tensor_model_parallel_region(h)
        if len(self.lora_B) == 1:
            out = F.linear(h, self.lora_B[0].weight)
        else:
            out = self.combine([F.linear(p, b.weight) for p, b in zip(h.split(self.rank, dim=-1), self.lora_B)])
        return out * self.scale

    def calc_delta_weight(self) -> torch.Tensor:
        """``scale * B A`` in the base weight's local layout (fp32)."""
        a = self.lora_A.weight.float()
        deltas = [b.weight.float() @ a[i * self.rank : (i + 1) * self.rank] for i, b in enumerate(self.lora_B)]
        return self.combine([d.t() for d in deltas]).t() * self.scale

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        state = self.state_dict(prefix="", keep_vars=True)
        if self.parallel_mode == "column":
            axis = {f"lora_B.{i}.weight": 0 for i in range(len(self.lora_B))}
        elif self.parallel_mode == "row":
            axis = {"lora_A.weight": 1}
        else:
            axis = {}
        return compat.make_sharded_tensors_for_checkpoint(state, prefix, axis, sharded_offsets)


class LoRALinear(torch.nn.Module):
    """Wraps a Megatron linear layer (returns ``(output, bias)``) with a LoRA adapter."""

    def __init__(self, to_wrap: torch.nn.Module, name: str, rank: int, alpha: float, dropout: float = 0.0,
                 variant: str = "canonical"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"LoRA variant must be one of {VARIANTS}, got {variant!r}")
        if getattr(getattr(to_wrap, "config", None), "sequence_parallel", False):
            raise NotImplementedError("LoRA with sequence parallelism is not implemented")
        self.to_wrap = to_wrap
        weight = to_wrap.weight
        mode = detect_parallel_mode(to_wrap)
        out_local, in_local = weight.shape
        layout = describe_fused_layout(name, to_wrap) if variant == "canonical" else None
        kind, split_rows, qkv_shape = "single", [out_local], None
        if layout is not None:
            kind, split_rows = layout
            if kind == "qkv":
                c = to_wrap.config
                tp = compat.get_tensor_model_parallel_world_size()
                qkv_shape = (c.num_query_groups // tp, c.num_attention_heads // c.num_query_groups, c.kv_channels)
        self.adapter = LoRAAdapter(
            in_features=in_local, split_rows=split_rows, rank=rank, alpha=alpha, dropout=dropout,
            parallel_mode=mode, dtype=weight.dtype, device=weight.device, kind=kind, qkv_shape=qkv_shape,
        )
        self.has_input_norm = hasattr(to_wrap, "layer_norm_weight")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("to_wrap"), name)

    def _input_norm(self, x: torch.Tensor) -> torch.Tensor:
        m = self.to_wrap
        weight = m.layer_norm_weight
        if getattr(m, "zero_centered_gamma", False):
            weight = weight + 1
        shape = (x.shape[-1],)
        if getattr(m, "normalization", "LayerNorm") == "RMSNorm":
            return F.rms_norm(x, shape, weight, m.eps)
        return F.layer_norm(x, shape, weight, getattr(m, "layer_norm_bias", None), m.eps)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.to_wrap(x, *args, **kwargs)
        base_out, bias = out if isinstance(out, tuple) else (out, None)
        lora_in = self._input_norm(x) if self.has_input_norm else x
        return base_out + self.adapter(lora_in).to(base_out.dtype), bias

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        sd = self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata)
        sd.update(self.adapter.sharded_state_dict(f"{prefix}adapter.", sharded_offsets, metadata))
        return sd

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Add ``scale * B @ A`` to the local base weight shard (in place)."""
        self.to_wrap.weight.add_(self.adapter.calc_delta_weight().to(self.to_wrap.weight.dtype))

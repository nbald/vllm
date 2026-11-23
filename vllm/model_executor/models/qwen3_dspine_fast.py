# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fast draft path for the DSpine drafter (qwen3_dspine.py).

The eager draft step is ~635 small kernels at a few rows per GEMM, so on a
fast target it costs as much as the verify it feeds. Each flag below removes
one source of that overhead; the eager code in qwen3_dspine.py stays the
reference (tests/v1/spec_decode/test_dspine_fast.py).

- ``OLALA_DSPINE_FUSED=1``: the adjacent injection after every draft layer and
  the transition scoring each run as one ``torch.compile``'d function of
  tensors (the weights are arguments, so one graph serves every layer and
  batch size) instead of ~50 eager elementwise kernels per layer.
  ``OLALA_DSPINE_FUSED=2`` also compiles every draft decoder layer: with custom
  ops off, RMSNorm, rotary and SiluAndMul are native torch chains that only a
  compile fuses.
- ``OLALA_DSPINE_GEMV=1``: the drafter's bf16 linears and head run through
  Olala's skinny GEMV (``olala_linear``, up to 64 rows) instead of cuBLAS.
- ``OLALA_DSPINE_CUDA_NORM=1``: the drafter's RMSNorms call vLLM's CUDA
  ``rms_norm`` / ``fused_add_rms_norm`` kernels instead of the native chain
  (~8 kernels each, opaque to the compile above). The target is untouched.

The fused functions keep the eager casts and order of operations; they are
not bit-identical to it (different GEMM splits and fusion), so they are
opt-in. Draft quality is unaffected in practice: the target verifies every
token, and accepted lengths matched the eager drafter on the validation set.
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

FUSED_MODE = os.environ.get("OLALA_DSPINE_FUSED", "0")
FUSED = FUSED_MODE in ("1", "2")
COMPILE_LAYERS = FUSED_MODE == "2"
GEMV = os.environ.get("OLALA_DSPINE_GEMV", "0") == "1"
CUDA_NORM = os.environ.get("OLALA_DSPINE_CUDA_NORM", "0") == "1"

_TINY = torch.finfo(torch.float32).tiny


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if GEMV and x.dim() == 2 and x.dtype == torch.bfloat16 == weight.dtype:
        from vllm.model_executor.layers.mamba.olala.decode_gemv import olala_linear

        return olala_linear(x, weight)
    return F.linear(x, weight)


def inject_fn(
    pre: torch.Tensor,
    anchor_features: torch.Tensor,
    w_read: torch.Tensor,
    w_recv: torch.Tensor,
    w_pred_msg: torch.Tensor,
    w_write: torch.Tensor,
    gate_bias: torch.Tensor,
    block_size: int,
    transfer_dim: int,
    eps: float,
) -> torch.Tensor:
    """AdjacentCausalInjection.forward for one layer, same casts and order of
    operations; ``w_pred_msg = cat(predecessor_gate, message)`` so the
    predecessor features go through one GEMM."""
    n = pre.shape[0]
    pre_f = pre.float()
    rms = pre_f.square().mean(-1, keepdim=True).clamp_min(_TINY).sqrt()
    projected = _linear((pre_f / (rms + eps)).to(pre.dtype), w_read).float()
    features = (
        math.sqrt(transfer_dim)
        * projected
        / (projected.norm(dim=-1, keepdim=True) + eps)
    ).to(pre.dtype)
    pos = torch.arange(n, device=pre.device) % block_size
    predecessor = torch.where(
        (pos == 1)[:, None],
        anchor_features.repeat_interleave(block_size, dim=0),
        features.roll(1, dims=0),
    )
    m = w_recv.shape[0]
    pm = _linear(predecessor, w_pred_msg)
    gate = torch.sigmoid(
        _linear(features, w_recv).float() + pm[:, :m].float() + gate_bias.float()
    )
    gated = (gate * pm[:, m:].float()).to(pre.dtype)
    update = _linear(gated, w_write).float()
    written = (pre_f + rms * update).to(pre.dtype)
    return torch.where((pos == 0)[:, None], pre, written)


def transitions_fn(
    final_pre_injection: torch.Tensor,
    candidate_ids: torch.Tensor,
    predecessor_ids: torch.Tensor,
    head_w: torch.Tensor,
    head_scale: torch.Tensor | None,
    codebook: torch.Tensor,
    w_read: torch.Tensor,
    w_recv: torch.Tensor,
    w_pred: torch.Tensor,
    w_msg: torch.Tensor,
    w_write: torch.Tensor,
    gate_bias: torch.Tensor,
    norm_w: torch.Tensor,
    transfer_dim: int,
    eps: float,
    norm_eps: float,
) -> torch.Tensor:
    """qwen3_dspine._score_transitions (tp_size == 1) as one function of
    tensors."""
    dtype = final_pre_injection.dtype
    pre_f = final_pre_injection.float()
    rms = pre_f.square().mean(-1, keepdim=True).clamp_min(_TINY).sqrt()
    projected = F.linear((pre_f / (rms + eps)).to(dtype), w_read).float()
    receiver_features = (
        math.sqrt(transfer_dim)
        * projected
        / (projected.norm(dim=-1, keepdim=True) + eps)
    ).to(dtype)
    ids = candidate_ids.long()
    successor_weights = head_w[ids].float()
    if head_scale is not None:
        # INT8 W8A16 head (OLALA_DSPINE_SHARED_HEAD): dequantized rows, per
        # channel or per group (qwen3_dspine.head_rows).
        s = head_scale[ids].float()
        if head_scale.dim() == 1:
            successor_weights = successor_weights * s.unsqueeze(-1)
        else:
            group = head_w.shape[1] // head_scale.shape[1]
            successor_weights = successor_weights * s.repeat_interleave(group, dim=-1)
    exact = codebook[predecessor_ids].float() * math.sqrt(transfer_dim)
    gate = torch.sigmoid(
        F.linear(receiver_features[:, :, None, :], w_recv).float()
        + F.linear(exact, w_pred.float())
        + gate_bias.float()
    )
    update = F.linear(gate * F.linear(exact, w_msg.float()), w_write.float())
    refined = pre_f[:, :, None, :] + rms[:, :, None, :] * update
    normalized = (
        refined
        / (refined.square().mean(-1, keepdim=True) + norm_eps).sqrt()
        * norm_w.float()
    )
    return torch.einsum("bnkh,bnjh->bnkj", normalized, successor_weights)


inject_compiled = (
    torch.compile(inject_fn, dynamic=True, fullgraph=True) if FUSED else None
)
transitions_compiled = (
    torch.compile(transitions_fn, dynamic=True, fullgraph=True) if FUSED else None
)


def refresh_fused_weights(injection: nn.Module) -> None:
    """(Re)build the concatenated predecessor-gate / message weight of
    inject_fn. An existing buffer is updated in place, so CUDA graphs captured
    before a weight reload keep reading the current values."""
    w = torch.cat(
        (injection.predecessor_gate.weight, injection.message.weight), 0
    ).contiguous()
    buf = getattr(injection, "fused_pred_msg_weight", None)
    if buf is not None and buf.shape == w.shape and buf.dtype == w.dtype:
        buf.copy_(w)
    else:
        injection.fused_pred_msg_weight = w


def inject(
    injection: nn.Module,
    layer_idx: int,
    pre_injection_states: torch.Tensor,
    anchor_features: torch.Tensor,
    fn=None,
) -> torch.Tensor:
    if getattr(injection, "fused_pred_msg_weight", None) is None:
        refresh_fused_weights(injection)
    fn = fn or inject_compiled or inject_fn
    return fn(
        pre_injection_states,
        anchor_features,
        injection.readouts[layer_idx].weight,
        injection.receiver_gate.weight,
        injection.fused_pred_msg_weight,
        injection.writes[layer_idx].weight,
        injection.gate_bias,
        injection.block_size,
        injection.transfer_dim,
        injection.eps,
    )


def score_transitions(
    injection: nn.Module,
    norm: nn.Module,
    lm_head: nn.Module,
    transfer_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    final_pre_injection: torch.Tensor,
    predecessor_ids: torch.Tensor,
    final_layer_idx: int,
    fn=None,
) -> torch.Tensor:
    fn = fn or transitions_compiled or transitions_fn
    return fn(
        final_pre_injection,
        candidate_ids,
        predecessor_ids,
        lm_head.weight,
        None if lm_head.weight.is_floating_point() else lm_head.weight_scale,
        transfer_codebook,
        injection.readouts[final_layer_idx].weight,
        injection.receiver_gate.weight,
        injection.predecessor_gate.weight,
        injection.message.weight,
        injection.writes[final_layer_idx].weight,
        injection.gate_bias,
        norm.weight,
        injection.transfer_dim,
        injection.eps,
        norm.variance_epsilon,
    )


def bind_cuda_rms_norm(module: nn.Module) -> bool:
    """Route one vLLM RMSNorm module to the CUDA kernels; False when its
    configuration is not covered (variance-size override, no weight, non-bf16
    weight)."""
    from vllm import _custom_ops as ops

    weight = getattr(module, "weight", None)
    if (
        getattr(module, "variance_size_override", None) is not None
        or not getattr(module, "has_weight", False)
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
    ):
        return False
    eps = module.variance_epsilon

    def forward(x, residual=None):
        if residual is None:
            x = x.contiguous()
            out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            ops.rms_norm(out, x, module.weight.data, eps)
            return out
        # In place on x and residual, as with custom ops on.
        ops.fused_add_rms_norm(x, residual, module.weight.data, eps)
        return x, residual

    module.forward = forward
    return True


class GemvLinearMethod:
    """Unquantized linear / LM-head apply through olala_linear (which falls
    back to F.linear above 64 rows)."""

    def __init__(self, base):
        self.base = base

    def apply(self, layer, x, bias=None):
        if bias is not None or x.dim() != 2:
            return self.base.apply(layer, x, bias)
        from vllm.model_executor.layers.mamba.olala.decode_gemv import olala_linear

        return olala_linear(x, layer.weight)

    def __getattr__(self, name):
        if name == "base":  # not set yet (copy / unpickle)
            raise AttributeError(name)
        return getattr(self.base, name)


def apply_fast_path(model: nn.Module) -> None:
    """Install the flagged fast paths on a loaded DSpineQwen3ForCausalLM.
    Called after every load_weights; idempotent."""
    if FUSED:
        refresh_fused_weights(model.model.injection)
    if GEMV:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        for name, module in model.named_modules():
            method = getattr(module, "quant_method", None)
            weight = getattr(module, "weight", None)
            if (
                "embed" not in name
                and type(method)
                in (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)
                and isinstance(weight, torch.Tensor)
                and weight.dim() == 2
                and weight.dtype == torch.bfloat16
            ):
                module.quant_method = GemvLinearMethod(method)
    if CUDA_NORM:
        from vllm.model_executor.layers.layernorm import RMSNorm

        for module in model.model.modules():
            if type(module) is RMSNorm:
                bind_cuda_rms_norm(module)
    if COMPILE_LAYERS:
        for layer in model.model.layers:
            if not getattr(layer, "dspine_compiled", False):
                layer.forward = torch.compile(layer.forward, dynamic=True)
                layer.dspine_compiled = True

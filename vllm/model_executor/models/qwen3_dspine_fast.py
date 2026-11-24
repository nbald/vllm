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
  ``OLALA_DSPINE_FUSED=3`` is level 2 plus the hand-fused kernels of
  qwen3_dspine_kernels.py (one launch per chain): q/k RMSNorm + rotary in
  every draft layer.
  ``OLALA_DSPINE_FUSED=2`` also compiles every draft decoder layer: with custom
  ops off, RMSNorm, rotary and SiluAndMul are native torch chains that only a
  compile fuses.
- ``OLALA_DSPINE_GEMV=1``: the drafter's bf16 linears and head run through
  Olala's skinny GEMV (``olala_linear``, up to 64 rows) instead of cuBLAS.
- ``OLALA_DSPINE_CUDA_NORM=1``: the drafter's RMSNorms call vLLM's CUDA
  ``rms_norm`` / ``fused_add_rms_norm`` kernels instead of the native chain
  (~8 kernels each, opaque to the compile above). The target is untouched.
- ``OLALA_DSPINE_INT8`` (on by default on the GPUs where it was measured:
  SM70, the SM80 family, SM120; ``0`` turns it off, ``1`` forces it on
  elsewhere): the drafter's decoder linears (qkv, o, gate_up,
  down) and its context projection ``fc`` are quantized at load time to INT8
  with one fp32 scale per output channel (symmetric RTN of the bf16 weights,
  as the Q8 target) and run through the target's W8A16 linear method
  (``ModelOptW8A16LinearMethod``), as does the fused context K/V projection. The
  injection and transition weights, the codebook and the norms stay bf16
  (they are read directly by the fused functions). Halves the bytes of those
  linears per draft step (about 0.15 ms on a V100); weight reloads after the
  quantization are not supported. On sm_86-class GPUs (GeForce / workstation
  Ampere and Ada) the draft steps of 33-256 rows (6 to 42 requests at K = 5)
  take the Triton W8A16 GEMM with tiles tuned for them (_mid_rows_cfg): there
  the generic W8A16 dispatch, tuned on an RTX 5090, ran 1.2-1.4x the bf16
  drafter's linears.

The fused functions keep the eager casts and order of operations; they are
not bit-identical to it (different GEMM splits and fusion), so they are
opt-in. Draft quality is unaffected in practice: the target verifies every
token, and accepted lengths matched the eager drafter on the validation set.
"""

import math
import os
import re
import types

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.quantization.modelopt_w8a16 import (
    ModelOptW8A16LinearMethod,
)
from vllm.utils.torch_utils import direct_register_custom_op

FUSED_MODE = os.environ.get("OLALA_DSPINE_FUSED", "0")
FUSED = FUSED_MODE in ("1", "2", "3")
COMPILE_LAYERS = FUSED_MODE in ("2", "3")
HAND_FUSED = FUSED_MODE == "3"
GEMV = os.environ.get("OLALA_DSPINE_GEMV", "0") == "1"
CUDA_NORM = os.environ.get("OLALA_DSPINE_CUDA_NORM", "0") == "1"
# OLALA_DSPINE_INT8 unset: on for these compute-capability majors (V100, A100 / RTX 30-40, RTX 50),
# where offline acceptance and served throughput were measured against the bf16 drafter.
_INT8_DEFAULT_MAJORS = (7, 8, 12)


def int8_enabled() -> bool:
    """OLALA_DSPINE_INT8: ``1`` on, ``0`` off, unset = on for _INT8_DEFAULT_MAJORS."""
    raw = os.environ.get("OLALA_DSPINE_INT8", "")
    if raw in ("0", "1"):
        return raw == "1"
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability()
    return cap is not None and cap.major in _INT8_DEFAULT_MAJORS


# The linears OLALA_DSPINE_INT8 quantizes: decoder projections and the context projection.
_INT8_NAME = re.compile(r"(^|\.)layers\.\d+\.(self_attn\.(qkv|o)_proj|mlp\.(gate_up|down)_proj)$|(^|\.)fc$")

# Rows of a drafter INT8 linear that take the Triton W8A16 GEMM with _mid_rows_cfg's tiles on sm_86-class GPUs: 6 per
# request at K = 5, so 6 to 42 requests; the range of the config search. Outside it the target's dispatch stays (the
# tiles lost to it at 384 rows for the long-K weights).
_MID_ROWS = (33, 256)


def _mid_rows_gpu() -> bool:
    """GeForce / workstation Ampere and Ada (sm_86 - sm_89): bf16 MMA with fp32 accumulation at half rate and
    ~100 KB of shared memory per SM, so from a few dozen rows the draft linears are compute bound and the generic
    W8A16 dispatch (tuned on an RTX 5090: the skinny decode kernels up to 64 / 128 rows, dequantize + cuBLAS for
    narrow weights) lost to bf16 cuBLAS. Not sm_80 (A100: full-rate tensor cores, HBM) and not a V100 through
    volta-forever (it reports 8.0): neither was measured."""
    from vllm.model_executor.layers.mamba.olala.arch import (
        GpuFamily,
        capability,
        gpu_family,
    )

    return (
        gpu_family() is GpuFamily.SM80
        and capability()[1] >= 6
        and not os.environ.get("VOLTA_FOREVER")
    )


def _mid_rows_cfg(M: int, N: int, K: int) -> dict | None:
    """Triton W8A16 GEMM tiles for a draft linear of M rows on an sm_86-class GPU, or None (the generic W8A16
    dispatch). From a config search on the drafter's shapes (RTX 3080, M = 48-256, scratch/spec_state/tools/
    tune_w8a16_drafter.py): per draft step at 192 rows (32 requests) 1.19 -> 0.88 ms against 0.84 ms for bf16;
    at 48 / 96 rows 0.61 -> 0.42 / 0.88 -> 0.59 ms against 0.50 / 0.72 ms."""
    if not (_MID_ROWS[0] <= M <= _MID_ROWS[1]) or K % 128 or not _mid_rows_gpu():
        return None
    if K >= 3072:  # narrow, long-K weights (down 1536 x 4096, fc 1536 x 7680): more column tiles
        if M <= 64:
            return dict(BM=64, BN=32, BK=128, num_warps=4, num_stages=4)
        if M <= 128:
            return dict(BM=64, BN=64, BK=128, num_warps=8, num_stages=3)
        return dict(BM=32, BN=64, BK=128, num_warps=4, num_stages=3)
    if N >= 4096:  # wide (gate_up 8192, the fused context K/V 5120)
        if M <= 64:
            return dict(BM=64, BN=128, BK=128, num_warps=8, num_stages=3)
        if M <= 128:
            return dict(BM=32, BN=128, BK=64, num_warps=4, num_stages=3)
        return dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=3)
    if M <= 64:  # qkv 2560, o 1536
        return dict(BM=32, BN=64, BK=128, num_warps=4, num_stages=3)
    if M <= 128:
        return dict(BM=64, BN=64, BK=128, num_warps=8, num_stages=3)
    if N >= 2560:  # qkv
        return dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=3)
    return dict(BM=32, BN=64, BK=64, num_warps=4, num_stages=3)


def _draft_int8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ (scale * weight).T`` for the drafter's INT8 linears: the Triton W8A16 GEMM with _mid_rows_cfg's tiles
    where it gives some, the target's W8A16 dispatch (w8a16_utils.int8_mm) everywhere else."""
    from vllm.model_executor.layers.quantization.utils import w8a16_utils as wu

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    w = wu.Int8Weight(weight, scale)
    cfg = _mid_rows_cfg(x2.shape[0], w.n, w.k) if scale.dim() == 1 and x.dtype == torch.bfloat16 else None
    if cfg is not None:
        y = wu.w8a16_gemm(x2 if x2.stride(-1) == 1 else x2.contiguous(), w, cfg=cfg)
    else:
        y = wu.int8_mm(x2, w)
    return y.view(*shape[:-1], w.n)


def _draft_int8_linear_fake(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


# Opaque to torch.compile (OLALA_DSPINE_FUSED=2 compiles the draft layers): the row count is dispatched at run time.
direct_register_custom_op(
    op_name="dspine_int8_linear",
    op_func=_draft_int8_linear,
    mutates_args=[],
    fake_impl=_draft_int8_linear_fake,
)


def draft_int8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.dspine_int8_linear(x, weight, scale)

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


def quantize_int8_rtn(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel RTN of a bf16 ``[N, K]`` weight: int8
    ``q`` and fp32 ``scale [N]`` with ``weight ~= scale[:, None] * q``."""
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


class DraftW8A16LinearMethod(ModelOptW8A16LinearMethod):
    """The target's W8A16 linear method with the drafter's row-count dispatch (draft_int8_linear)."""

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.weight_scale.dim() != 1:  # per-group scales: the target's dispatch
            return super().apply(layer, x, bias)
        y = draft_int8_linear(x, layer.weight, layer.weight_scale)
        return y if bias is None else y + bias


def install_int8(model: nn.Module) -> int:
    """Quantize the drafter linears of _INT8_NAME in place (idempotent) and
    rebuild the fused context K/V projection from the INT8 rows. Returns the
    number of linears quantized by this call."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    n = 0
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if (
            _INT8_NAME.search(name)
            and type(getattr(module, "quant_method", None)) is UnquantizedLinearMethod
            and isinstance(weight, torch.Tensor)
            and weight.dim() == 2
            and weight.dtype == torch.bfloat16
            and getattr(module, "bias", None) is None
        ):
            q, scale = quantize_int8_rtn(weight.data)
            module.weight = nn.Parameter(q, requires_grad=False)
            module.weight_scale = nn.Parameter(scale, requires_grad=False)
            # The target's W8A16 linear method: its post-load step (the SM70 family's fp16
            # tensor-core copy where that edition exists); its apply, except for the row counts
            # draft_int8_linear dispatches itself.
            method = DraftW8A16LinearMethod()
            method.process_weights_after_loading(module)
            module.quant_method = method
            n += 1
    inner = model.model
    attn = [layer.self_attn for layer in inner.layers]
    if attn and all(isinstance(a.qkv_proj.quant_method, ModelOptW8A16LinearMethod) for a in attn):
        inner._fused_kv_weight = torch.cat([a.qkv_proj.weight[a.q_size :] for a in attn], 0)
        inner._fused_kv_scale = torch.cat([a.qkv_proj.weight_scale[a.q_size :] for a in attn], 0)
        inner._fused_kv_bias = None
        inner._fused_kv_int8_linear = draft_int8_linear
        _register_volta_tc(inner._fused_kv_weight, "dspine.fused_kv")
    return n


def _register_volta_tc(weight: torch.Tensor, name: str) -> None:
    """An INT8 weight built outside a quant method: the fp16 tensor-core copy that
    ModelOptW8A16LinearMethod makes for its own weights on the SM70 family (no-op
    elsewhere, and on trees without that edition)."""
    try:
        from vllm.model_executor.layers.quantization.utils.w8a16_utils import volta_tc
    except ImportError:
        return
    if (tc := volta_tc()) is not None:
        tc.register_int8(weight, 1, name)


def _fused_attention_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """DFlashQwen3Attention.forward with the q/k RMSNorms and the rotary in one
    kernel (qwen3_dspine_kernels.qk_norm_rope)."""
    from vllm.model_executor.models.qwen3_dspine_kernels import qk_norm_rope

    qkv, _ = self.qkv_proj(hidden_states)
    q, k = qk_norm_rope(
        qkv, positions, self.rotary_emb.cos_sin_cache, self.q_norm.weight, self.k_norm.weight,
        self.q_norm.variance_epsilon, self.num_heads, self.num_kv_heads, self.head_dim,
    )
    v = qkv[:, self.q_size + self.kv_size :]
    if self.v_scale is not None:
        v = v * self.v_scale
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def _hand_fuse_attention(attn: nn.Module) -> bool:
    """Install _fused_attention_forward on one drafter attention module; False
    when its configuration is not covered (partial or GPT-J rotary, norms
    without weights)."""
    rope = attn.rotary_emb
    if (
        getattr(rope, "rotary_dim", None) != attn.head_dim
        or not getattr(rope, "is_neox_style", False)
        or getattr(attn.q_norm, "weight", None) is None
        or getattr(attn.k_norm, "weight", None) is None
        or attn.q_norm.variance_epsilon != attn.k_norm.variance_epsilon
        or attn.head_dim % 2
    ):
        return False
    attn.forward = types.MethodType(_fused_attention_forward, attn)
    return True


def apply_fast_path(model: nn.Module) -> None:
    """Install the flagged fast paths on a loaded DSpineQwen3ForCausalLM.
    Called after every load_weights; idempotent."""
    if FUSED:
        refresh_fused_weights(model.model.injection)
    if int8_enabled():
        install_int8(model)
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
    if HAND_FUSED:
        for layer in model.model.layers:
            if not getattr(layer.self_attn, "dspine_hand_fused", False):
                layer.self_attn.dspine_hand_fused = _hand_fuse_attention(layer.self_attn)
    if COMPILE_LAYERS:
        for layer in model.model.layers:
            if not getattr(layer, "dspine_compiled", False):
                layer.forward = torch.compile(layer.forward, dynamic=True)
                layer.dspine_compiled = True

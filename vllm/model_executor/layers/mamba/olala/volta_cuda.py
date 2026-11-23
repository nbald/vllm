# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA editions of Triton kernels for Volta (V100, natively or through volta-forever) that return the Triton
kernels' bits: the geodesic norm (csrc/geodesic_volta.cu) and the weight-only latent-MoE small path from 4 tokens
(csrc/moe_small_volta.cu). Each reproduces its Triton kernels' per-output arithmetic (the same fp32 chains, the same
reduction tree as the Triton layout, the same rounding instructions) with another work decomposition, so they are
on by default on the SM70 family; OLALA_GEODESIC_CUDA=0 / OLALA_VOLTA_MOE_CUDA=0 keep the Triton kernels (part of
the compile-cache key like every OLALA_* switch)."""

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.olala.arch import (
    GpuFamily,
    gencode,
    gpu_family,
    has_bf16_mma,
    sm_arch,
)

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def _on(name: str) -> bool:
    """The switch, on the SM70 family told sm_80+ (through volta-forever: the kernels use sm_80 bf16
    conversions, which volta-forever lowers to Volta)."""
    return os.environ.get(name, "1") != "0" and gpu_family() is GpuFamily.SM70 and has_bf16_mma()


@lru_cache(maxsize=1)
def _geodesic_ext():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(f"olala_geodesic_volta_sm{sm_arch()}", "geodesic_volta.cu", ["-O3", "-std=c++20", gencode()])


@lru_cache(maxsize=1)
def _moe_ext():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(f"olala_moe_small_volta_sm{sm_arch()}", "moe_small_volta.cu", ["-O3", "-std=c++20", gencode()])


def _aligned(*ts: torch.Tensor) -> bool:
    return all(t.data_ptr() % 16 == 0 for t in ts)


def geodesic_ok(x: torch.Tensor, g: torch.Tensor, out: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> bool:
    """The CUDA edition takes rows of 1024 < D <= 2048 bf16 elements (D % 8 == 0, 16-byte aligned rows) and bf16
    scale / bias: the shapes and dtypes of the Triton kernel's BLOCK = 2048 specialization it reproduces."""
    if not _on("OLALA_GEODESIC_CUDA"):
        return False
    D = x.shape[-1]
    if not (1024 < D <= 2048 and D % 8 == 0):
        return False
    if any(t.dtype != torch.bfloat16 for t in (x, g, out, scale, bias)):
        return False
    if any(t.stride(-1) != 1 or (t.dim() > 1 and t.stride(-2) % 8) for t in (x, g, out)):
        return False
    return _aligned(x, g, out) and g.shape == x.shape and out.shape == x.shape


def geodesic(x, g, out, scale, bias, inv_depth: float, clamp: float) -> torch.Tensor:
    logger.info_once("Olala Volta: geodesic norm through its CUDA edition (OLALA_GEODESIC_CUDA=0: Triton)")
    _geodesic_ext().geodesic(x, g, out, scale, bias, inv_depth, clamp)
    return out


def _int8_row(w, n: int, k: int, experts: int = 0) -> bool:
    shape = (experts, n, k) if experts else (n, k)
    return (
        w.weight.dtype == torch.int8
        and tuple(w.weight.shape) == shape
        and w.scale is not None
        and w.scale.dtype == torch.float32
        and tuple(w.scale.shape) == shape[:-1]
        and w.weight.is_contiguous()
        and w.scale.is_contiguous()
    )


def _int4_affine(w, n: int, k: int, experts: int) -> bool:
    return (
        w.weight.dtype == torch.uint8
        and tuple(w.weight.shape) == (experts, n, k // 2)
        and w.scale is not None
        and w.scale.dtype == torch.int32
        and tuple(w.scale.shape) == (experts, n, k // 32)
        and w.weight.is_contiguous()
        and w.scale.is_contiguous()
    )


# Token counts from which the CUDA small path is dispatched (its kernels cover 3..32 tokens: the Triton tiles of 3..32
# tokens share their per-output structure; 1-2 tokens keep the Triton path). V100 through volta-forever, graphed,
# rotated weight sets, us per layer, Triton -> CUDA: INT8 experts 3 tokens 70.8 -> 77.4, 4: 73.6 -> 72.3, 8: 103.7 ->
# 99.0, 16: 164.9 -> 125.0, 32: 307.6 -> 183.2; INT4 affine experts 8: 114.6 -> 127.5, 9: 143.6 -> 137.1, 16: 201.1 ->
# 173.4, 32: 407.3 -> 256.8.
MOE_SMALL_MIN_TOKENS = {"int8": 4, "int4": 9}


def moe_small_ok(M, hidden, x_in, router_logits, router_logits_lo, e_score_bias, w13, w2, shared_up, shared_down, fc2,
                 top_k: int) -> bool:
    """The whole CUDA small path (route, up, down, exit) from MOE_SMALL_MIN_TOKENS to 32 tokens: INT8 per-row (Q8) or
    INT4 affine (Q4A2) routed experts, INT8 per-row shared expert and fc2_latent, in the model's shapes; x_in /
    logits as the entry launch writes them (fp32, unit column stride, 16-byte aligned rows)."""
    if not (min(MOE_SMALL_MIN_TOKENS.values()) <= M <= 32) or not _on("OLALA_VOLTA_MOE_CUDA") or top_k != 6:
        return False
    if _int8_row(w13, 768, 384, 256) and _int8_row(w2, 384, 768, 256):
        fmt = "int8"
    elif _int4_affine(w13, 768, 384, 256) and _int4_affine(w2, 384, 768, 256):
        fmt = "int4"
    else:
        return False
    if M < MOE_SMALL_MIN_TOKENS[fmt] or not (
        _int8_row(shared_up, 1536, 1536) and _int8_row(shared_down, 1536, 1536) and _int8_row(fc2, 1536, 384)
    ):
        return False
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (M, 1536) or hidden.stride(1) != 1:
        return False
    for t in (x_in, router_logits) + ((router_logits_lo,) if router_logits_lo is not None else ()):
        if t.dtype != torch.float32 or t.stride(1) != 1 or t.stride(0) % 4 or t.data_ptr() % 16:
            return False
    if router_logits_lo is not None and router_logits_lo.stride(0) != router_logits.stride(0):
        return False
    return (
        x_in.shape[1] >= 384
        and tuple(router_logits.shape) == (M, 256)
        and e_score_bias.dtype == torch.float32
        and e_score_bias.is_contiguous()
        and e_score_bias.numel() == 256
        and _aligned(e_score_bias, w13.weight, w2.weight, shared_up.weight, shared_down.weight, fc2.weight)
    )


def moe_small(hidden, x_in, router_logits, router_logits_lo, e_score_bias, w13, w2, shared_up, shared_down, fc2,
              top_k: int, routed_scale: float, bufs: dict | None = None) -> torch.Tensor:
    """olala_moe_small_int8_volta's output through the CUDA small path, bit for bit. ``bufs`` (tests) receives the
    intermediates."""
    M, dev = hidden.shape[0], hidden.device
    logger.info_once("Olala Volta: weight-only MoE small path through its CUDA editions (route, up, down, exit) "
                     "from %d tokens (INT8 experts), %d (INT4); OLALA_VOLTA_MOE_CUDA=0: Triton",
                     MOE_SMALL_MIN_TOKENS["int8"], MOE_SMALL_MIN_TOKENS["int4"])
    e = _moe_ext()
    mbp = 8 if M <= 8 else 16 if M <= 16 else 32
    MK = M * top_k
    ids = torch.empty(MK, dtype=torch.int32, device=dev)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    meta = torch.empty(e.route_meta_ints(), dtype=torch.int32, device=dev)
    hT = torch.empty(1536 * mbp, dtype=torch.bfloat16, device=dev)
    h = torch.empty(MK, 768, dtype=torch.bfloat16, device=dev)
    part = torch.empty(MK, 384, dtype=torch.float32, device=dev)
    sT = torch.empty(1536 * mbp, dtype=torch.bfloat16, device=dev)
    rT = torch.empty(384 * mbp, dtype=torch.bfloat16, device=dev)
    out = torch.empty(M, 1536, dtype=torch.bfloat16, device=dev)
    e.small(hidden, x_in, router_logits, router_logits_lo, e_score_bias, w13.weight, w13.scale, w2.weight, w2.scale,
            shared_up.weight, shared_up.scale, shared_down.weight, shared_down.scale, fc2.weight, fc2.scale, ids, tw,
            meta, hT, h, part, sT, rT, out, M, routed_scale)
    if bufs is not None:
        bufs.update(ids=ids, tw=tw, meta=meta, h=h, part=part, sT=sT, rT=rT, mbp=mbp)
    return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA editions of Triton kernels for Volta (V100, natively or through volta-forever) that return the Triton
kernels' bits: the geodesic norm (csrc/geodesic_volta.cu) and the weight-only latent-MoE small path from 17 tokens
(csrc/moe_small_volta.cu). Each reproduces its Triton kernels' per-output arithmetic (the same fp32 chains, the same
reduction tree as the Triton layout, the same rounding instructions) with another work decomposition, so they are
on by default on the SM70 family; OLALA_GEODESIC_CUDA=0 / OLALA_VOLTA_MOE_CUDA=0 keep the Triton kernels (part of
the compile-cache key like every OLALA_* switch).

The multi-row MoE (csrc/moe_rows_volta.cu, moe_rows) is not a bit-for-bit edition: for up to 32 tokens it computes the
whole weight-only MoE (input projections and routing, routed and shared experts, output projections) in three launches
with its own summation orders, deterministic but rounding-level different from the small path. Opt-in:
OLALA_VOLTA_MOE_ROWS=1, from 3 to 8 tokens with INT8 routed experts (Q8)."""

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
# tokens share their per-output structure; below the threshold the Triton path runs). V100 through volta-forever,
# graphed, rotated weight sets, us per layer, Triton -> CUDA: INT8 experts 4 tokens 73.6 -> 72.3, 8: 103.7 -> 99.0,
# 16: 164.9 -> 125.0, 32: 307.6 -> 183.2; INT4 affine experts 9: 143.6 -> 137.1, 16: 201.1 -> 173.4, 32: 407.3 ->
# 256.8. In the server's decode graph both paths cost more, the CUDA one most (nsys, Q8, per layer: 4 tokens 76.8 ->
# 100.6, 8: 102.1 -> 119.7; the routing launch alone 18.5-19.8), and steady decode at 4 / 8 / 16 requests was 11.7 /
# 6.9 % slower / equal with this path; from 17 tokens the Triton path runs the 32-token tile (300-400 us per layer).
MOE_SMALL_MIN_TOKENS = {"int8": 17, "int4": 17}


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


# -- multi-row MoE (csrc/moe_rows_volta.cu) ------------------------------------------------------------------------
# From 3 to 8 tokens, INT8 per-row routed experts (Q8). The kernels take 1 to 32 tokens; at 2 tokens the small path is
# 6 % faster; in the server's decode graphs (nsys, Q8, MoE per layer) the rows path saves 17 % at 4 tokens and 12 % at 8
# but nothing at 16 (210.6 against 211.3 us, where the graphed layer benchmark gives 15 %: its unrolled kernels run
# with warm instruction caches there), and from 17 tokens its 32-row tile loses to the CUDA small path. The kernels also
# take INT4 affine experts (Q4A2, tested), which the dispatch leaves to the small path: no clear gain (+17 % at 4
# tokens, 1-3 % at 6-8, slower at 2 and 16). The 1-token path is untouched.
MOE_ROWS_MIN_TOKENS, MOE_ROWS_MAX_TOKENS = 3, 8
_ROWS_BUFS: dict = {}


def moe_rows_enabled() -> bool:
    """OLALA_VOLTA_MOE_ROWS=1 on the SM70 family told sm_80+ (read at each call: tests flip it)."""
    return os.environ.get("OLALA_VOLTA_MOE_ROWS", "0") == "1" and gpu_family() is GpuFamily.SM70 and has_bf16_mma()


@lru_cache(maxsize=1)
def _rows_ext():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(f"olala_moe_rows_volta_sm{sm_arch()}", "moe_rows_volta.cu", ["-O3", "-std=c++20", gencode()])


def moe_rows_buffers(device) -> tuple[torch.Tensor, torch.Tensor]:
    """The workspace (sized for 32 tokens) and the counters of the rows path, one pair per device. The counters are
    zeroed once and every kernel resets those it completes, so layers and graphs share them on one stream. Built by
    the first call, which the layer makes in its profiling run (before CUDA graph capture: graphs bake addresses)."""
    dev = torch.device(device)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _ROWS_BUFS:
        e = _rows_ext()
        ws = torch.empty(e.workspace_floats(32), dtype=torch.float32, device=dev)
        ctr = torch.zeros(e.counter_ints(), dtype=torch.int32, device=dev)
        _ROWS_BUFS[key] = (ws, ctr)
    return _ROWS_BUFS[key]


def moe_rows_ok(M: int, hidden, router, fc1, bias, w13, w2, shared_up, shared_down, fc2, top_k: int) -> bool:
    """The rows path takes MOE_ROWS_MIN_TOKENS..MOE_ROWS_MAX_TOKENS tokens of the model's shapes: bf16 hidden rows (unit
    column stride, 16-byte aligned rows), the bf16 router rows [hi ; lo] (bf16x2), INT8 per-row routed experts (Q8),
    fc1_latent, shared expert and fc2_latent, fp32 correction bias."""
    if not (MOE_ROWS_MIN_TOKENS <= M <= MOE_ROWS_MAX_TOKENS) or top_k != 6 or not moe_rows_enabled():
        return False
    if not (_int8_row(w13, 768, 384, 256) and _int8_row(w2, 384, 768, 256)):
        return False
    if not (_int8_row(fc1, 384, 1536) and _int8_row(shared_up, 1536, 1536) and _int8_row(shared_down, 1536, 1536)
            and _int8_row(fc2, 1536, 384)):
        return False
    if (hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (M, 1536) or hidden.stride(1) != 1
            or hidden.stride(0) % 8):
        return False
    if router.dtype != torch.bfloat16 or tuple(router.shape) != (512, 1536) or not router.is_contiguous():
        return False
    if bias.dtype != torch.float32 or not bias.is_contiguous() or bias.numel() != 256:
        return False
    return _aligned(hidden, router, bias, fc1.weight, w13.weight, w2.weight, shared_up.weight, shared_down.weight,
                    fc2.weight)


def moe_rows(hidden, router, fc1, bias, w13, w2, shared_up, shared_down, fc2, routed_scale: float) -> torch.Tensor:
    """The MoE output [M, 1536] bf16 of 1..32 tokens through the rows path (moe_rows_ok's inputs; the kernels take any
    count up to 32, the dispatch uses MOE_ROWS_MIN_TOKENS..MOE_ROWS_MAX_TOKENS)."""
    logger.info_once("Olala Volta: weight-only MoE from %d to %d tokens in three launches (OLALA_VOLTA_MOE_ROWS=1)",
                     MOE_ROWS_MIN_TOKENS, MOE_ROWS_MAX_TOKENS)
    M = hidden.shape[0]
    ws, ctr = moe_rows_buffers(hidden.device)
    out = torch.empty(M, 1536, dtype=torch.bfloat16, device=hidden.device)
    _rows_ext().moe_rows(hidden, router, fc1.weight, fc1.scale, shared_up.weight, shared_up.scale, bias, w13.weight,
                         w13.scale, w2.weight, w2.scale, shared_down.weight, shared_down.scale, fc2.weight, fc2.scale,
                         ws, ctr, out, M, routed_scale, 7, 0)
    return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch path for Olala's latent MoE (decode).

The generic modular MoE path (permute / align / two grouped gemms / unpermute
/ reduce) launches ~15 kernels per layer, which at batch <= 16 costs more in
launch floor than in bytes moved: per token only ``top_k`` experts of
384x768 are touched. This path computes the routed experts with two
gather-GEMV kernels (up + relu^2, down + routed weight + scale) and keeps
the shared expert and latent projections as plain GEMVs. Same math as the
runner: ``shared(x) + fc2(scale * sum_k w_k * E_k(fc1(x)))``.

The whole MoE call is the custom op ``vllm::olala_latent_moe`` (layer
looked up by name, like the mixers): the small/generic dispatch on the
token count then happens inside the op, where torch.compile cannot
specialize it away.
"""

import functools
import os

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.olala.arch import (
    GpuFamily,
    gencode,
    gpu_family,
    sm80_class,
)
from vllm.model_executor.layers.mamba.olala.pdl import pdl_kwargs
from vllm.triton_utils import tl, triton
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.olala.decode_gemv import decode_gemv
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

SMALL_BATCH_MAX_TOKENS = 8
# With the medium path available (bf16 experts) it wins from 6 tokens on B300.
SMALL_BATCH_MAX_TOKENS_WITH_MEDIUM = 5
# Volta (V100, natively or through volta-forever): the medium path's grouped
# GEMMs are tl.dot, whose bf16 mma.sync is emulated with scalar code there, and
# pad each expert's rows to 16; the small path's gather-GEMV kernels are plain
# FMA. Per layer, small vs medium (FMA projections): 8 tokens 110 vs 364 us,
# 16 tokens 168 vs 459 us, 32 tokens 452 vs 751 us.
SMALL_BATCH_MAX_TOKENS_VOLTA = 32

# The CUDA small path (int8 experts) beats the generic path up to 32 tokens on an A4500
# (realistic routing, decode16: -0.6 ms/step): no align/sort/sum launches.
SMALL_BATCH_MAX_TOKENS_CUDA = int(os.environ.get("OLALA_MOE_SMALL_MAX", "32"))
# Above this many tokens the concatenated-projection path costs more in
# activation traffic than it saves in launches (measured on 24k prefills).
FUSED_PROJ_MAX_TOKENS = int(os.environ.get("OLALA_FUSED_PROJ_MAX_TOKENS", "512"))
# int8 experts on sm8x+: the routed part as two CUDA kernels (csrc/moe_small_int8.cu)
_SMALL_CUDA = os.environ.get("OLALA_MOE_SMALL_CUDA", "1") == "1"
_SMALL_EXT = None


@functools.cache
def _small_cuda_arch() -> bool:
    # Ampere / Ada / Hopper (mma.sync), and Volta through volta-forever; Blackwell keeps the
    # Triton small path tuned for it
    return sm80_class() or gpu_family() is GpuFamily.SM90


def small_path_max_tokens(w13: torch.Tensor) -> int:
    if _SMALL_CUDA and w13.dtype == torch.int8 and w13.is_cuda and _small_cuda_arch():
        return SMALL_BATCH_MAX_TOKENS_CUDA
    return SMALL_BATCH_MAX_TOKENS


def _small_ext():
    global _SMALL_EXT
    if _SMALL_EXT is None:
        from vllm.model_executor.layers.mamba.olala.jit import load_extension

        _SMALL_EXT = load_extension(
            "olala_moe_small_int8",
            "moe_small_int8.cu",
            ["-O3", "-std=c++20", gencode()],
        )
    return _SMALL_EXT


@triton.jit
def _latent_moe_up_kernel(
    x_ptr,       # [M, D] bf16 latent input
    logits_ptr,  # [M, E] fp32 router logits
    bias_ptr,    # [E] fp32 e_score_correction_bias
    ids_ptr,     # [M*K] int32 out (written by the nb == 0 programs)
    tw_ptr,      # [M*K] fp32 out routing weights (normalized)
    w1_ptr,      # [E, N, D] bf16, or int8 with per-row scales
    w1s_ptr,     # [E, N] fp32 scales (int8 weights only)
    h_ptr,       # [M*K, N] bf16 out
    stride_xm,   # row stride of x (fp32 or bf16; rounded to bf16 on load)
    stride_lm,   # row stride of the logits
    E: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Expert up-projection + relu^2 for one (token, slot k, N-block).

    The router top-k (sigmoid scores, ranked by score + bias, weights
    normalized over the k winners — topk_sigmoid semantics) is recomputed by
    every program from the 256 logits: cheaper than a separate launch. The
    nb == 0 program of each (token, k) publishes the id and weight for the
    down kernel."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)  # m * K + k
    nb = tl.program_id(1)
    m = pid // K
    k_slot = pid % K
    offs_e = tl.arange(0, E)
    sc = 1.0 / (1.0 + tl.exp(-tl.load(logits_ptr + m * stride_lm + offs_e)))
    sb = sc + tl.load(bias_ptr + offs_e)
    # Padded batch rows carry garbage/NaN logits; NaN never compares equal to
    # the max, which would select index E and read past the expert weights.
    sb = tl.where(sb != sb, float("-inf"), sb)
    tot = 0.0
    e = 0
    wsel = 0.0
    for kk in range(K):
        mx = tl.max(sb, axis=0)
        idx = tl.minimum(tl.min(tl.where(sb == mx, offs_e, E), axis=0), E - 1)
        w = tl.sum(tl.where(offs_e == idx, sc, 0.0), axis=0)
        tot += w
        if kk == k_slot:
            e = idx
            wsel = w
        sb = tl.where(offs_e == idx, float("-inf"), sb)
    if nb == 0:
        tl.store(ids_ptr + pid, e.to(tl.int32))
        tl.store(tw_ptr + pid, wsel / tot)
    e64 = e.to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for d0 in range(0, D, BD):
        offs_d = d0 + tl.arange(0, BD)
        x = tl.load(x_ptr + m * stride_xm + offs_d).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w1_ptr + e64 * N * D + offs_n[:, None] * D + offs_d[None, :])
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    if HAS_SCALE:
        acc *= tl.load(w1s_ptr + e64 * N + offs_n)
    h = acc.to(tl.bfloat16).to(tl.float32)
    h = tl.maximum(h, 0.0)
    h = h * h
    tl.store(h_ptr + pid * N + offs_n, h.to(tl.bfloat16))


@triton.jit
def _latent_moe_down_partial_kernel(
    h_ptr,      # [M*K, N1] bf16
    ids_ptr,    # [M*K] int32
    tw_ptr,     # [M*K] fp32 routing weights
    w2_ptr,     # [E, D2, N1] bf16, or int8 with per-row scales
    w2s_ptr,    # [E, D2] fp32 scales (int8 weights only)
    part_ptr,   # [M*K, D2] fp32 out (routing weight applied)
    N1: tl.constexpr,
    D2: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)  # m * K + k
    nb = tl.program_id(1)
    e = tl.load(ids_ptr + row).to(tl.int64)
    wk = tl.load(tw_ptr + row)
    offs_n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for d0 in range(0, N1, BD):
        offs_d = d0 + tl.arange(0, BD)
        h = tl.load(h_ptr + row * N1 + offs_d).to(tl.float32)
        w = tl.load(w2_ptr + e * D2 * N1 + offs_n[:, None] * N1 + offs_d[None, :])
        acc += tl.sum(w.to(tl.float32) * h[None, :], axis=1)
    if HAS_SCALE:
        acc *= tl.load(w2s_ptr + e * D2 + offs_n)
    tl.store(part_ptr + row * D2 + offs_n, acc * wk)


@triton.jit
def _latent_moe_epilogue_kernel(
    part_ptr,   # [M*K, D2] fp32 weighted expert partials
    s_ptr,      # [M, S] bf16 shared-expert pre-activation (from the fused GEMV)
    cat_ptr,    # [M, D2 + S] bf16 out: [scale * sum_k part | relu2(s)]
    scale,
    S_LD,       # leading dim of s rows
    D2: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    m = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BS + tl.arange(0, BS)
    if nb * BS < D2:
        acc = tl.zeros([BS], dtype=tl.float32)
        for k in range(K):
            acc += tl.load(part_ptr + (m * K + k) * D2 + offs)
        tl.store(cat_ptr + m * (D2 + S) + offs, (acc * scale).to(tl.bfloat16))
    else:
        so = offs - D2
        v = tl.load(s_ptr + m * S_LD + so).to(tl.bfloat16).to(tl.float32)
        v = tl.maximum(v, 0.0)
        tl.store(cat_ptr + m * (D2 + S) + offs, (v * v).to(tl.bfloat16))


@triton.jit
def _moe_cat_kernel(
    routed_ptr,  # [M, D2] bf16 (already routed-scaled)
    s_ptr,       # [M, S] bf16 shared pre-activation (row stride S_LD)
    cat_ptr,     # [M, D2 + S] bf16 out: [routed | relu2(s)]
    S_LD, R_LD,
    D2: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
):
    m = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BS + tl.arange(0, BS)
    if nb * BS < D2:
        r = tl.load(routed_ptr + m * R_LD + offs)
        tl.store(cat_ptr + m * (D2 + S) + offs, r)
    else:
        so = offs - D2
        a = tl.load(s_ptr + m * S_LD + so).to(tl.float32)
        a = tl.maximum(a, 0.0)
        tl.store(cat_ptr + m * (D2 + S) + offs, (a * a).to(tl.bfloat16))


def moe_cat_shared(routed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[routed | relu2(s)] in one launch (generic-M path of the latent MoE)."""
    M, D2 = routed.shape
    S = s.shape[1]
    cat = torch.empty(M, D2 + S, dtype=routed.dtype, device=routed.device)
    _moe_cat_kernel[(M, (D2 + S) // 128)](
        routed, s, cat, s.stride(0), routed.stride(0), D2=D2, S=S, BS=128, num_warps=1
    )
    return cat


def olala_moe_small(
    hidden: torch.Tensor,         # [M, hidden] bf16
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,   # [E] fp32
    w_in: torch.Tensor,           # [latent + shared_inter, hidden]  (fc1 ‖ shared_up)
    w_out: torch.Tensor,          # [hidden, latent + shared_inter]  (fc2 ‖ shared_down)
    w13: torch.Tensor,            # [E, inter, latent]
    w2: torch.Tensor,             # [E, latent, inter]
    top_k: int,
    routed_scale: float,
    w13_scale: torch.Tensor | None = None,  # [E, inter] fp32 (int8 w13)
    w2_scale: torch.Tensor | None = None,   # [E, latent] fp32 (int8 w2)
    x_in: torch.Tensor | None = None,       # precomputed hidden @ w_in.T
) -> torch.Tensor:
    """Small-batch latent MoE. ``w_in``/``w_out`` are the cached concatenations
    of the latent and shared-expert projections (see OlalaLatentMoE). int8
    expert weights (experts_int8) are dequantized per row in the kernels.
    ``x_in`` (fp32 or bf16, any row stride) replaces the input GEMV; it is
    rounded to bf16 on load, like a bf16 GEMM output."""
    M = hidden.shape[0]
    dev = hidden.device
    E, N1, D1 = w13.shape
    D2 = w2.shape[1]
    S = w_in.shape[0] - D1

    topk_w = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=dev)

    if x_in is None:
        x_in = decode_gemv(hidden, w_in)  # [M, D1 + S]: latent | shared pre-act
    q13 = w13.dtype == torch.int8
    q2 = w2.dtype == torch.int8
    if (
        _SMALL_CUDA and q13 and q2 and (E, N1, D1, D2) == (256, 768, 384, 384) and top_k <= 7
        and router_logits.dtype == torch.float32 and _small_cuda_arch()
        # bf16 rows only (x_in from geo_linear is fp32)
        and x_in.dtype == torch.bfloat16 and x_in.stride(1) == 1 and x_in.stride(0) % 8 == 0
    ):
        cat = _small_ext().moe_small_int8(
            x_in[:, :D1], router_logits.contiguous(), e_score_bias.float(), w13, w13_scale, w2, w2_scale,
            x_in[:, D1:], top_k, routed_scale,
        )
        return decode_gemv(cat, w_out)
    h = torch.empty(M * top_k, N1, dtype=hidden.dtype, device=dev)
    _latent_moe_up_kernel[(M * top_k, N1 // 16)](
        x_in, router_logits, e_score_bias, topk_ids, topk_w, w13,
        w13_scale if q13 else topk_w, h, x_in.stride(0), router_logits.stride(0),
        E=E, D=D1, N=N1, BN=16, BD=64, K=top_k, HAS_SCALE=q13, num_warps=4,
        **pdl_kwargs(),
    )
    part = torch.empty(M * top_k, D2, dtype=torch.float32, device=dev)
    _latent_moe_down_partial_kernel[(M * top_k, D2 // 8)](
        h, topk_ids, topk_w, w2, w2_scale if q2 else topk_w, part,
        N1=N1, D2=D2, BN=8, BD=256, HAS_SCALE=q2, num_warps=4,
        **pdl_kwargs(),
    )
    cat = torch.empty(M, D2 + S, dtype=hidden.dtype, device=dev)
    _latent_moe_epilogue_kernel[(M, (D2 + S) // 128)](
        part, x_in[:, D1:], cat, routed_scale, x_in.stride(0),
        D2=D2, S=S, BS=128, K=top_k, num_warps=1,
        **pdl_kwargs(),
    )
    return decode_gemv(cat, w_out)                    # shared + fc2(routed), one GEMV


def _olala_latent_moe(
    hidden: torch.Tensor,
    router_logits: torch.Tensor | None,
    layer_name: LayerNameType,
    g: torch.Tensor | None = None,
    h_out: torch.Tensor | None = None,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    if g is not None:
        # hidden is the residual; h_out receives geodesic(hidden, g).
        return layer.forward_dispatch_geo(hidden, g, h_out)
    return layer.forward_dispatch(hidden, router_logits)


def _olala_latent_moe_fake(
    hidden: torch.Tensor,
    router_logits: torch.Tensor | None,
    layer_name: LayerNameType,
    g: torch.Tensor | None = None,
    h_out: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden)


direct_register_custom_op(
    op_name="olala_latent_moe",
    op_func=_olala_latent_moe,
    mutates_args=["h_out"],
    fake_impl=_olala_latent_moe_fake,
)

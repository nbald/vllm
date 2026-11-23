# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token GEMV for Olala's decode projections.

cuBLAS's batch-1 kernels sit on a ~4.2 us floor whatever the matrix size,
so the many small projections of a decode step (latent MoE, in_proj_dyn,
TPA factors) run at well under 1 TB/s. This streaming kernel — each program
owns BN output rows and walks K in BK chunks — is 2.5-3 us on those shapes
and also ahead of cuBLAS on the large ones (measured under CUDA graphs on
GH200). For 2..64 rows the skinny kernel puts the weight tile on the MMA's M
side so every weight byte is still read once (cuBLAS picks 55-95 % of the
copy bandwidth there on an A4500, the skinny kernel 80-97 %).
"""

import os

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


_DISABLED = os.environ.get("OLALA_GEMV", "1") == "0"


@triton.jit
def _gemv_m1_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        x = tl.load(x_ptr + offs_k)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _gemv_m1_full_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """Whole (padded) K extent per program: every load is in flight at once
    instead of walking K in a dependent loop."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    k_mask = offs_k < K
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
        mask=n_mask[:, None] & k_mask[None, :], other=0.0,
    )
    x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)
    acc = tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def _decode_gemv_blackwell(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """B300 (weights streamed from HBM, CUDA graphs): the full-K kernel is
    2.7 us on the latent MoE projections where the K loop takes 4-12 us, and
    3.8 / 5.1 us on mixer_proj (K = 3072 / 4608) vs 4.5 / 9.0 us. Wide outputs
    (N >= 4096: in_proj, the TPA projections) are faster on cuBLAS."""
    N, K = weight.shape
    bk = triton.next_power_of_2(K)
    if N >= 4096 or bk > 8192:
        return F.linear(x, weight)
    if bk <= 2048:
        bn, nw = 2, 8
    elif bk <= 4096:
        bn, nw = 1, 4
    else:
        bn, nw = 2, 2
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    _gemv_m1_full_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, N, K, weight.stride(0), BN=bn, BK=bk, num_warps=nw,
    )
    return y


@triton.jit
def _skinny_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """``y = x @ w.T`` for a few rows of x: the BN weight rows are the MMA's M side and the
    BM <= 64 rows of x its N side, so every weight byte is read once."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    n_ok = offs_n[:, None] < N
    m_ok = offs_m[None, :] < M
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for _ in range(0, K, BK):
        w = tl.load(w_ptrs, mask=n_ok, other=0.0)
        x = tl.load(x_ptrs, mask=m_ok, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptrs += BK
        x_ptrs += BK
    y_ptrs = y_ptr + offs_m[None, :] * stride_ym + offs_n[:, None]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_ok & m_ok)


def skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, cfg: tuple[int, int, int, int] | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for M <= 64 rows (K % BK == 0); ``cfg`` = (BN, BK, stages, warps)."""
    M, K = x.shape
    N = weight.shape[0]
    bn, bk, stages, warps = cfg or _skinny_cfg(M, N, K)
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _skinny_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, M, N, K, x.stride(0), weight.stride(0), y.stride(0),
        BM=16 if M <= 16 else (32 if M <= 32 else 64), BN=bn, BK=bk,
        num_stages=stages, num_warps=warps,
    )
    return y


def _skinny_cfg(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    # Sweep of the decode shapes on an RTX A4500 (graphed, weights from DRAM): 88-97 % of the
    # copy bandwidth up to 32 rows, 80-88 % at 64; split-K never won.
    if M > 32 and N > 1536 and K < 3072:
        return 64, 64, 3, 4
    return 32, 128, 4, (8 if M > 32 else 4)


SKINNY_MAX_M = 64
_SKINNY = os.environ.get("OLALA_SKINNY", "1") == "1"


def _skinny_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _SKINNY and x.dtype == torch.bfloat16 and weight.shape[1] % 128 == 0
        and x.stride(-1) == 1 and weight.stride(-1) == 1
    )


def _decode_gemv_sm120(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RTX 5090 sweep of the decode shapes (graphed, weights from DRAM). One row: the full-K
    GEMV on narrow outputs (latent MoE 4.7 vs 5.8-6.7 us cuBLAS, mixer_proj 6.7 vs 8.3 us), the
    K-loop GEMV on wide ones (in_proj 14.7 vs 16.2 us). 2..32 rows, and up to 64 on wide
    outputs: the skinny kernel, where cuBLAS picks sm80 wmma kernels (mixer_proj 8.0 vs
    17.2 us, latent MoE 5.2 vs 11.3 us); beyond, cuBLAS."""
    M = x.shape[0]
    N, K = weight.shape
    if M == 1:
        if N < 4096 and triton.next_power_of_2(K) <= 8192:
            return _decode_gemv_blackwell(x, weight)
        if K % 128 == 0:
            return _gemv_m1(x, weight)
    elif (M <= 32 or (M <= SKINNY_MAX_M and N >= 4096)) and _skinny_ok(x, weight):
        return skinny_linear(x, weight)
    return F.linear(x, weight)


# -- Volta (V100), natively or through volta-forever ------------------------------------------
# volta-forever tells the process sm_80 (so bf16 stays on), and on sm_70 tl.dot's bf16 mma.sync is
# emulated with scalar code (~200 instructions per m16n8k16, 64 of them FMAs) while plain FMA code
# runs natively. The skinny kernel pays that for 16 padded rows per real row (~170 GB/s whatever
# M), and cuBLAS has no Volta bf16 GEMM kernels (F.linear: 70-300 us per decode projection). So on
# Volta decode GEMVs take FMA kernels: one row the GEMV below at any K; 2..32 rows the kernels
# below, which load each weight tile once for all rows and keep fp32 partial sums per lane until a
# single reduction. V100 through volta-forever, graphed, weights from DRAM, us (skinny -> FMA):
# 1 row: in_proj 7344x1536 124 -> 30, mixer_proj 1536x3072 121 -> 14, TPA mixer 1536x4608
# 178 -> 20, MoE w_out 1536x1920 77 -> 14; 8 rows: in_proj 134 -> 50, mixer_proj 130 -> 28,
# TPA mixer 193 -> 40, w_in 1920x1536 67 -> 16; 32 rows: mixer_proj 197 -> 94, in_proj 175 -> 165.
VOLTA_ROWS_MAX_M = 32

# Weight scales of the Volta GEMVs (constexpr SCALE): none (bf16 / fp32 weights), one fp32 scale
# per output row (INT8 W8A16 per channel: applied to the fp32 row sum), one scale per G weights
# along K (INT8 W8A16 per group: applied to the fp32 weight before the FMAs).
VOLTA_SCALE_NONE, VOLTA_SCALE_ROW, VOLTA_SCALE_GROUP = 0, 1, 2


@triton.jit
def _volta_w_tile(
    w_ptr, s_ptr, offs_n, n_mask, offs_k, mask, stride_wn, stride_sn,
    SCALE: tl.constexpr, G: tl.constexpr,
):
    """A [BN, BK] weight tile as fp32 (INT8 codes are exact), group scales applied."""
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=mask, other=0.0,
    ).to(tl.float32)
    if SCALE == 2:
        # one scale per G codes, gathered per element (L1 hits): no reshape of the tile
        s = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + (offs_k // G)[None, :], mask=mask, other=0.0,
        ).to(tl.float32)
        w = w * s
    return w


@triton.jit
def _volta_row_sum(acc, s_ptr, offs_n, n_mask, SCALE: tl.constexpr):
    """Row sums of the per-lane partial sums, per-row scales applied."""
    y = tl.sum(acc, axis=1)
    if SCALE == 1:
        y = y * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    return y


@triton.jit
def _gemv_m1_volta_kernel(
    x_ptr, w_ptr, y_ptr, s_ptr, N, K, stride_wn, stride_sn,
    BN: tl.constexpr, BK: tl.constexpr, KMASK: tl.constexpr,
    SCALE: tl.constexpr = 0, G: tl.constexpr = 1,
):
    """One-row GEMV for Volta: partial sums per (n, k) lane, reduced once after the K loop; the
    optional K mask lets K = 1920 (MoE w_out) use 512-wide blocks."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        if KMASK:
            k_mask = offs_k < K
            w = _volta_w_tile(
                w_ptr, s_ptr, offs_n, n_mask, offs_k, n_mask[:, None] & k_mask[None, :],
                stride_wn, stride_sn, SCALE, G,
            )
            x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)
        else:
            w = _volta_w_tile(
                w_ptr, s_ptr, offs_n, n_mask, offs_k, n_mask[:, None], stride_wn, stride_sn,
                SCALE, G,
            )
            x = tl.load(x_ptr + offs_k)
        acc += w * x.to(tl.float32)[None, :]
    y = _volta_row_sum(acc, s_ptr, offs_n, n_mask, SCALE)
    tl.store(y_ptr + offs_n, y.to(y_ptr.dtype.element_ty), mask=n_mask)


def _volta_scale_args(weight: torch.Tensor, scale: torch.Tensor | None) -> tuple:
    """(scale pointer, its row stride, SCALE, G) of a Volta GEMV launch."""
    if scale is None:
        return weight, 0, VOLTA_SCALE_NONE, 1
    if scale.dim() == 1:
        return scale, 0, VOLTA_SCALE_ROW, 1
    return scale, scale.stride(0), VOLTA_SCALE_GROUP, weight.shape[1] // scale.shape[1]


def _gemv_m1_volta(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # V100 through volta-forever, graphed, weights from DRAM: one weight row per 4-warp
    # program, 512-wide K blocks, against the GH200 configs of _gemv_m1, us: in_proj
    # 7344x1536 30.5 -> 28.8, mixer_proj 1536x3072 14.5 -> 14.0, TPA gate 4608x1536
    # 23.5 -> 19.2, MoE w_out 1536x1920 13.7 -> 9.3 (512-wide blocks need the K mask),
    # fp32 router 256x1536 5.0 -> 3.9.
    N, K = weight.shape
    y = torch.empty(1, N, dtype=out_dtype or x.dtype, device=x.device) if out is None else out
    s, ss, mode, g = _volta_scale_args(weight, scale)
    _gemv_m1_volta_kernel[(N,)](
        x, weight, y, s, N, K, weight.stride(0), ss, BN=1, BK=512, KMASK=K % 512 != 0,
        SCALE=mode, G=g, num_warps=4,
    )
    return y


@triton.jit
def _rows_acc(acc, w, x_ptr, m: tl.constexpr, stride_xm, offs_k, M):
    x = tl.load(x_ptr + m * stride_xm + offs_k, mask=(offs_k * 0 + m) < M, other=0.0)
    return acc + w * x.to(tl.float32)[None, :]


@triton.jit
def _rows_store(
    y_ptr, acc, m: tl.constexpr, stride_ym, offs_n, n_mask, M, s_ptr, SCALE: tl.constexpr,
):
    y = _volta_row_sum(acc, s_ptr, offs_n, n_mask, SCALE)
    tl.store(y_ptr + m * stride_ym + offs_n, y.to(y_ptr.dtype.element_ty), mask=n_mask & (m < M))


@triton.jit
def _gemv_rows_kernel(
    x_ptr, w_ptr, y_ptr, s_ptr, M, N, K, stride_xm, stride_wn, stride_ym, stride_sn,
    MB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SCALE: tl.constexpr = 0, G: tl.constexpr = 1,
):
    """``y = x @ w.T`` for M <= MB <= 8 rows with FMAs (Volta): each [BN, BK] weight tile is
    loaded once and used for every row; per-row partial sums stay per (n, k) lane across the K
    loop and are reduced once at the end."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    a0 = tl.zeros([BN, BK], dtype=tl.float32)
    a1 = tl.zeros([BN, BK], dtype=tl.float32)
    a2 = tl.zeros([BN, BK], dtype=tl.float32)
    a3 = tl.zeros([BN, BK], dtype=tl.float32)
    a4 = tl.zeros([BN, BK], dtype=tl.float32)
    a5 = tl.zeros([BN, BK], dtype=tl.float32)
    a6 = tl.zeros([BN, BK], dtype=tl.float32)
    a7 = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = _volta_w_tile(
            w_ptr, s_ptr, offs_n, n_mask, offs_k, n_mask[:, None], stride_wn, stride_sn, SCALE, G,
        )
        a0 = _rows_acc(a0, w, x_ptr, 0, stride_xm, offs_k, M)
        a1 = _rows_acc(a1, w, x_ptr, 1, stride_xm, offs_k, M)
        if MB > 2:
            a2 = _rows_acc(a2, w, x_ptr, 2, stride_xm, offs_k, M)
            a3 = _rows_acc(a3, w, x_ptr, 3, stride_xm, offs_k, M)
        if MB > 4:
            a4 = _rows_acc(a4, w, x_ptr, 4, stride_xm, offs_k, M)
            a5 = _rows_acc(a5, w, x_ptr, 5, stride_xm, offs_k, M)
            a6 = _rows_acc(a6, w, x_ptr, 6, stride_xm, offs_k, M)
            a7 = _rows_acc(a7, w, x_ptr, 7, stride_xm, offs_k, M)
    _rows_store(y_ptr, a0, 0, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
    _rows_store(y_ptr, a1, 1, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
    if MB > 2:
        _rows_store(y_ptr, a2, 2, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
        _rows_store(y_ptr, a3, 3, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
    if MB > 4:
        _rows_store(y_ptr, a4, 4, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
        _rows_store(y_ptr, a5, 5, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
        _rows_store(y_ptr, a6, 6, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)
        _rows_store(y_ptr, a7, 7, stride_ym, offs_n, n_mask, M, s_ptr, SCALE)


@triton.jit
def _gemv_rows_many_kernel(
    x_ptr, w_ptr, y_ptr, s_ptr, M, N, K, stride_xm, stride_wn, stride_ym, stride_sn,
    MB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SCALE: tl.constexpr = 0, G: tl.constexpr = 1,
):
    """The same for 9..32 rows (Volta): one [MB, BN, BK] product per K step, partial sums kept
    across the loop (more separate row accumulators would spill)."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, MB)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros([MB, BN, BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = _volta_w_tile(
            w_ptr, s_ptr, offs_n, n_mask, offs_k, n_mask[:, None], stride_wn, stride_sn, SCALE, G,
        )
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :], mask=m_mask[:, None], other=0.0,
        )
        acc += x.to(tl.float32)[:, None, :] * w[None, :, :]
    y = tl.sum(acc, axis=2)
    if SCALE == 1:
        y = y * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], y.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _decode_gemv_volta(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """1..VOLTA_ROWS_MAX_M rows on Volta (K % 128 == 0, unit inner strides); x and weight may be
    any float dtypes, or an INT8 weight with ``scale`` (fp32 [N] per row, or [N, K // G] per
    group); fp32 accumulation, the output is ``out_dtype`` (default x's), or written into
    ``out`` (an (M, N) view with unit column stride, e.g. a column slice of a wider buffer)."""
    M, K = x.shape
    N = weight.shape[0]
    if M == 1:
        return _gemv_m1_volta(x, weight, out_dtype, scale, out)
    y = torch.empty(M, N, dtype=out_dtype or x.dtype, device=x.device) if out is None else out
    if M <= 8:
        kernel, bn, bk, nw = _gemv_rows_kernel, 16, 128, 8
    else:
        kernel, bn, bk, nw = _gemv_rows_many_kernel, 8, 64, 4
    s, ss, mode, g = _volta_scale_args(weight, scale)
    kernel[(triton.cdiv(N, bn),)](
        x, weight, y, s, M, N, K, x.stride(0), weight.stride(0), y.stride(0), ss,
        MB=triton.next_power_of_2(M), BN=bn, BK=bk, SCALE=mode, G=g, num_warps=nw,
    )
    return y


def volta_gemv_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """The Volta FMA GEMV applies (decode-sized x, K % 128 == 0, unit inner strides)."""
    return (
        x.is_cuda and not _DISABLED and x.dim() == 2 and x.shape[0] <= VOLTA_ROWS_MAX_M
        and weight.shape[1] % 128 == 0 and x.stride(-1) == 1 and weight.stride(-1) == 1
        and gpu_family() is GpuFamily.SM70
    )


def volta_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` in fp32 FMAs, stored as ``out_dtype``, for any float x / weight dtypes
    (e.g. bf16 activations against the fp32 router weight) or an INT8 weight with its W8A16
    ``scale`` (per row: on the fp32 row sum; per group: on the weight); ``out``: write into this
    (M, N) view instead. Requires volta_gemv_ok()."""
    return _decode_gemv_volta(x, weight, out_dtype, scale, out)


def decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for ``x: (M, K)``, ``weight: (N, K)``: the skinny kernel up to
    SKINNY_MAX_M rows (bf16), except single rows with K <= 1536 (triton GEMV, as fast there);
    F.linear otherwise. At one row the GEMV's 128-wide K blocks are slow for K = 1920 (w_out:
    53 % of the copy bandwidth vs 89 % on an A4500). Datacenter Blackwell keeps its own dispatch
    (the full-K GEMV at one row, F.linear otherwise), and so does consumer Blackwell. Volta takes
    FMA kernels up to VOLTA_ROWS_MAX_M rows."""
    M = x.shape[0]
    family = gpu_family() if x.is_cuda else None
    if family in (GpuFamily.SM100, GpuFamily.SM120):
        if x.dtype != weight.dtype or _DISABLED:
            return F.linear(x, weight)
        if family is GpuFamily.SM120:
            return _decode_gemv_sm120(x, weight)
        return _decode_gemv_blackwell(x, weight) if M == 1 else F.linear(x, weight)
    N, K = weight.shape
    if _DISABLED or not x.is_cuda or x.dtype != weight.dtype or K % 128 != 0 or M > SKINNY_MAX_M:
        return F.linear(x, weight)
    if (
        family is GpuFamily.SM70 and M <= VOLTA_ROWS_MAX_M
        and x.stride(-1) == 1 and weight.stride(-1) == 1
    ):
        return _decode_gemv_volta(x, weight)
    if (M != 1 or K > 1536) and _skinny_ok(x, weight):
        return skinny_linear(x, weight)
    if M != 1:
        return F.linear(x, weight)
    return _gemv_m1(x, weight)


def _gemv_m1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N, K = weight.shape
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    # HBM-streaming configs measured on GH200 (36 distinct weights, graphed):
    # wide outputs like 8 rows/program; narrow outputs want more programs.
    if N >= 4096:
        bn, bk, nw = 8, 512, 4
    elif K % 1024 == 0 and K >= 3072:
        bn, bk, nw = 4, 1024, 8
    else:
        bn, bk, nw = 4, (512 if K % 512 == 0 else 128), 8
    _gemv_m1_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, N, K, weight.stride(0), BN=bn, BK=bk, num_warps=nw,
    )
    return y


def _olala_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return decode_gemv(x, weight)


def _olala_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


# Opaque to torch.compile so the M == 1 dispatch is re-evaluated per cudagraph
# capture instead of being specialized by the first (large) compile.
direct_register_custom_op(
    op_name="olala_linear",
    op_func=_olala_linear,
    mutates_args=[],
    fake_impl=_olala_linear_fake,
)


def olala_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.olala_linear(x, weight)


def cat_rows_shared(params: list[torch.nn.Parameter], pad_to: int = 1) -> torch.Tensor:
    """Row-concatenate ``params`` and rebind each one to its slice of the result.

    The concatenated decode weight then costs no extra memory (a separate copy
    is ~0.8 GB over the 36 layers, a fifth of the KV budget on a 20 GB GPU).
    Row slices of a contiguous tensor keep the parameters' shapes and strides,
    and in-place weight reloads (``param.data.copy_``) land in the shared
    storage, so the concatenation never goes stale.

    With ``pad_to`` > 1 the buffer gets zero rows up to a multiple of ``pad_to`` (the returned
    tensor includes them): cuBLAS picks Hopper kernels only for aligned output widths.
    """
    with torch.no_grad():
        n_rows = sum(p.shape[0] for p in params)
        n_pad = -(-n_rows // pad_to) * pad_to
        ref = params[0].data
        w = torch.zeros(n_pad, *ref.shape[1:], dtype=ref.dtype, device=ref.device)
        torch.cat([p.data for p in params], 0, out=w[:n_rows])
        off = 0
        for p in params:
            n = p.shape[0]
            p.data = w[off : off + n]
            off += n
    return w

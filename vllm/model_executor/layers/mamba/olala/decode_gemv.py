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


def decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for ``x: (M, K)``, ``weight: (N, K)``: the skinny kernel up to
    SKINNY_MAX_M rows (bf16), except single rows with K <= 1536 (triton GEMV, as fast there);
    F.linear otherwise. At one row the GEMV's 128-wide K blocks are slow for K = 1920 (w_out:
    53 % of the copy bandwidth vs 89 % on an A4500). Blackwell keeps its own dispatch: the full-K
    GEMV at one row, F.linear otherwise."""
    M = x.shape[0]
    if x.is_cuda and gpu_family() in (GpuFamily.SM100, GpuFamily.SM120):
        if M != 1 or x.dtype != weight.dtype or _DISABLED:
            return F.linear(x, weight)
        return _decode_gemv_blackwell(x, weight)
    N, K = weight.shape
    if _DISABLED or not x.is_cuda or x.dtype != weight.dtype or K % 128 != 0 or M > SKINNY_MAX_M:
        return F.linear(x, weight)
    if (
        (M != 1 or K > 1536) and _SKINNY and x.dtype == torch.bfloat16
        and x.stride(-1) == 1 and weight.stride(-1) == 1
    ):
        return skinny_linear(x, weight)
    if M != 1:
        return F.linear(x, weight)
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

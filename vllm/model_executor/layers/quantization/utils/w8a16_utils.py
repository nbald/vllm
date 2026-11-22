# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W8A16 kernels: symmetric INT8 weights, BF16 activations.

Weights are row-major ``int8 [N, K]`` with one fp32 scale per output row, or
one fp16 / fp32 scale per group of G input channels (``[N, K // G]``, G = 32 in
practice). The reconstructed weight is ``scale * q``; accumulation is fp32.

- Decode, M = 1: a GEMV. M = 2..128: a skinny GEMM whose MMA M side is the
  weight rows; narrow weights split K across thread blocks (split-K).
  ``int8_decode_linear`` writes one or two segments that share K into one
  output (e.g. an INT8 and a BF16 projection of the same input), contiguously
  or through column maps.
- Prefill: a fused W8A16 GEMM, or dequantization to a BF16 scratch buffer
  followed by cuBLAS where that is faster.

Per-row scales multiply the fp32 accumulator; per-group scales are applied to
the weight tile before the dot product. ``w8a16_int8_linear`` is the opaque
custom op that dispatches on the row count at run time. The dispatch cutoffs
and tile configurations were tuned on an RTX 5090 (SM120) and apply to any
SM80+ GPU.
"""

import functools
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass
class Segment:
    """Weight rows and their destination columns in a concatenated projection."""

    weight: torch.Tensor
    scale: torch.Tensor | None = None
    cols: torch.Tensor | None = None
    col0: int = 0

    @property
    def n(self) -> int:
        return self.weight.shape[0]


@dataclass
class Int8Weight:
    """An INT8 weight ``[N, K]`` (row-major) with per-row or per-group scales."""

    weight: torch.Tensor  # int8 [N, K]
    scale: torch.Tensor  # fp32 [N], or fp16 / fp32 [N, K // group]

    @property
    def n(self) -> int:
        return self.weight.shape[0]

    @property
    def k(self) -> int:
        return self.weight.shape[1]

    @property
    def group(self) -> int:
        return seg_group(self.weight, self.scale)


def seg_group(weight: torch.Tensor, scale: torch.Tensor | None) -> int:
    """0: per-row scale (or none); G: one scale per G weights along K."""
    if scale is None or scale.dim() == 1:
        return 0
    return weight.shape[1] // scale.shape[1]


# Scale modes of a segment (constexpr SCALE)
_NONE, _ROW, _GROUP = 0, 1, 2


def _scale_mode(s: Segment) -> int:
    if s.scale is None:
        return _NONE
    return _ROW if s.scale.dim() == 1 else _GROUP


# -- kernels -----------------------------------------------------------------


@triton.jit
def _deq_tile(
    w,
    s_ptr,
    offs_n,
    n_ok,
    k0,
    stride_sn,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    """``w [BN, BK]`` int8 -> fp32 with its group scales (BK % G == 0)."""
    offs_g = k0 // G + tl.arange(0, BK // G)
    s = tl.load(
        s_ptr + offs_n[:, None].to(tl.int64) * stride_sn + offs_g[None, :],
        mask=n_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    w3 = tl.reshape(w.to(tl.float32), [BN, BK // G, G])
    return tl.reshape(w3 * s[:, :, None], [BN, BK])


@triton.jit
def _gemv_seg(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    c_ptr,
    pid,
    N,
    K,
    stride_wn,
    stride_sn,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """``y[cols[n]] = sum_k deq(w)[n, k] x[k]`` for BN rows (K % BK == 0)."""
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BK)
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    acc = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptrs + k0, mask=n_mask[:, None], other=0)
        x = tl.load(x_ptr + k0 + offs_k).to(tl.float32)
        if SCALE == 2:
            wf = _deq_tile(w, s_ptr, offs_n, n_mask, k0, stride_sn, BN, BK, G)
        else:
            wf = w.to(tl.float32)
        acc += wf * x[None, :]
    y = tl.sum(acc, axis=1)
    if SCALE == 1:
        y = y * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    cols = tl.load(c_ptr + offs_n, mask=n_mask, other=0) if HAS_MAP else offs_n
    tl.store(y_ptr + cols, y.to(y_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _i8_gemv_kernel(
    x_ptr,
    K,
    w1_ptr,
    s1_ptr,
    y1_ptr,
    c1_ptr,
    N1,
    stride_w1,
    stride_s1,
    NB1,
    w2_ptr,
    s2_ptr,
    y2_ptr,
    c2_ptr,
    N2,
    stride_w2,
    stride_s2,
    SCALE1: tl.constexpr,
    G1: tl.constexpr,
    HAS_MAP1: tl.constexpr,
    SEG2: tl.constexpr,
    SCALE2: tl.constexpr,
    G2: tl.constexpr,
    HAS_MAP2: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    if SEG2:
        if pid < NB1:
            _gemv_seg(
                x_ptr,
                w1_ptr,
                s1_ptr,
                y1_ptr,
                c1_ptr,
                pid,
                N1,
                K,
                stride_w1,
                stride_s1,
                SCALE1,
                G1,
                HAS_MAP1,
                BN,
                BK,
            )
        else:
            _gemv_seg(
                x_ptr,
                w2_ptr,
                s2_ptr,
                y2_ptr,
                c2_ptr,
                pid - NB1,
                N2,
                K,
                stride_w2,
                stride_s2,
                SCALE2,
                G2,
                HAS_MAP2,
                BN,
                BK,
            )
    else:
        _gemv_seg(
            x_ptr,
            w1_ptr,
            s1_ptr,
            y1_ptr,
            c1_ptr,
            pid,
            N1,
            K,
            stride_w1,
            stride_s1,
            SCALE1,
            G1,
            HAS_MAP1,
            BN,
            BK,
        )


@triton.jit
def _skinny_store(
    acc,
    s_ptr,
    y_ptr,
    c_ptr,
    offs_n,
    offs_m,
    n_ok,
    m_ok,
    stride_ym,
    SCALE: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    if SCALE == 1:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_ok, other=0.0)[:, None]
    cols = tl.load(c_ptr + offs_n, mask=n_ok, other=0) if HAS_MAP else offs_n
    y_ptrs = y_ptr + offs_m[None, :] * stride_ym + cols[:, None]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_ok[:, None] & m_ok[None, :])


@triton.jit
def _skinny_seg(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    c_ptr,
    ws_ptr,
    lock_ptr,
    tile,
    pid,
    pid_k,
    M,
    N,
    K_SPLIT,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_ym,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    HAS_MAP: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """``y[m, cols[n]] = sum_k deq(w)[n, k] x[m, k]``: the BN weight rows are
    the MMA's M side, the BM activation rows its N side. SPLIT > 1: this
    program covers K_SPLIT of K; the partial tiles meet in a workspace and the
    last program of the tile (atomic counter, reset for the next launch) sums
    them and stores."""
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    k_base = pid_k * K_SPLIT
    offs_k = k_base + tl.arange(0, BK)
    n_ok = offs_n < N
    m_ok = offs_m < M
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for kk in range(0, K_SPLIT, BK):
        w = tl.load(w_ptrs, mask=n_ok[:, None], other=0)
        x = tl.load(x_ptrs, mask=m_ok[None, :], other=0.0)
        if SCALE == 2:
            wb = _deq_tile(w, s_ptr, offs_n, n_ok, k_base + kk, stride_sn, BN, BK, G)
            wb = wb.to(tl.bfloat16)
        else:
            wb = w.to(tl.bfloat16)
        acc = tl.dot(wb, x.to(tl.bfloat16), acc)
        w_ptrs += BK
        x_ptrs += BK
    if SPLIT > 1:
        offs_t = tl.arange(0, BN)[:, None] * BM + offs_m[None, :]
        tl.store(ws_ptr + (tile * SPLIT + pid_k) * (BN * BM) + offs_t, acc)
        tl.debug_barrier()
        cnt = tl.atomic_add(lock_ptr + tile, 1, sem="acq_rel")
        if cnt == SPLIT - 1:
            acc = tl.zeros([BN, BM], dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                acc += tl.load(
                    ws_ptr + (tile * SPLIT + s) * (BN * BM) + offs_t,
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(lock_ptr + tile, 0)
            _skinny_store(
                acc,
                s_ptr,
                y_ptr,
                c_ptr,
                offs_n,
                offs_m,
                n_ok,
                m_ok,
                stride_ym,
                SCALE,
                HAS_MAP,
            )
    else:
        _skinny_store(
            acc,
            s_ptr,
            y_ptr,
            c_ptr,
            offs_n,
            offs_m,
            n_ok,
            m_ok,
            stride_ym,
            SCALE,
            HAS_MAP,
        )


@triton.jit(do_not_specialize=["M"])
def _i8_skinny_kernel(
    x_ptr,
    M,
    K_SPLIT,
    stride_xm,
    stride_ym,
    ws_ptr,
    lock_ptr,
    w1_ptr,
    s1_ptr,
    y1_ptr,
    c1_ptr,
    N1,
    stride_w1,
    stride_s1,
    NB1,
    w2_ptr,
    s2_ptr,
    y2_ptr,
    c2_ptr,
    N2,
    stride_w2,
    stride_s2,
    SCALE1: tl.constexpr,
    G1: tl.constexpr,
    HAS_MAP1: tl.constexpr,
    SEG2: tl.constexpr,
    SCALE2: tl.constexpr,
    G2: tl.constexpr,
    HAS_MAP2: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    if SEG2:
        if pid < NB1:
            _skinny_seg(
                x_ptr,
                w1_ptr,
                s1_ptr,
                y1_ptr,
                c1_ptr,
                ws_ptr,
                lock_ptr,
                pid,
                pid,
                pid_k,
                M,
                N1,
                K_SPLIT,
                stride_xm,
                stride_w1,
                stride_s1,
                stride_ym,
                SCALE1,
                G1,
                HAS_MAP1,
                SPLIT,
                BM,
                BN,
                BK,
            )
        else:
            _skinny_seg(
                x_ptr,
                w2_ptr,
                s2_ptr,
                y2_ptr,
                c2_ptr,
                ws_ptr,
                lock_ptr,
                pid,
                pid - NB1,
                pid_k,
                M,
                N2,
                K_SPLIT,
                stride_xm,
                stride_w2,
                stride_s2,
                stride_ym,
                SCALE2,
                G2,
                HAS_MAP2,
                SPLIT,
                BM,
                BN,
                BK,
            )
    else:
        _skinny_seg(
            x_ptr,
            w1_ptr,
            s1_ptr,
            y1_ptr,
            c1_ptr,
            ws_ptr,
            lock_ptr,
            pid,
            pid,
            pid_k,
            M,
            N1,
            K_SPLIT,
            stride_xm,
            stride_w1,
            stride_s1,
            stride_ym,
            SCALE1,
            G1,
            HAS_MAP1,
            SPLIT,
            BM,
            BN,
            BK,
        )


@triton.jit
def _i8_gemv1_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    N,
    K,
    stride_wn,
    stride_sn,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """One INT8 segment, contiguous output: fewer arguments than
    _i8_gemv_kernel (Python launch time counts for eager callers)."""
    _gemv_seg(
        x_ptr,
        w_ptr,
        s_ptr,
        y_ptr,
        s_ptr,
        tl.program_id(0),
        N,
        K,
        stride_wn,
        stride_sn,
        SCALE,
        G,
        False,
        BN,
        BK,
    )


@triton.jit(do_not_specialize=["M"])
def _i8_skinny1_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    ws_ptr,
    lock_ptr,
    M,
    N,
    K_SPLIT,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_ym,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    _skinny_seg(
        x_ptr,
        w_ptr,
        s_ptr,
        y_ptr,
        s_ptr,
        ws_ptr,
        lock_ptr,
        pid,
        pid,
        tl.program_id(1),
        M,
        N,
        K_SPLIT,
        stride_xm,
        stride_wn,
        stride_sn,
        stride_ym,
        SCALE,
        G,
        False,
        SPLIT,
        BM,
        BN,
        BK,
    )


# -- prefill kernels ---------------------------------------------------------


@triton.jit(do_not_specialize=["M"])
def _w8a16_gemm_kernel(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_sn,
    stride_cm,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """``C = A @ deq(B).T``: A bf16 [M, K], B int8 [N, K] upcast to bf16 per
    tile (K % BK == 0), fp32 accumulation, per-row scale on the accumulator."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % group_m
    pid_n = (pid % num_in_group) // group_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_ok = offs_m < M
    n_ok = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=m_ok[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_ok[None, :], other=0)
        if SCALE == 2:
            bb = tl.trans(
                _deq_tile(tl.trans(b), s_ptr, offs_n, n_ok, k0, stride_sn, BN, BK, G)
            )
            bb = bb.to(tl.bfloat16)
        else:
            bb = b.to(tl.bfloat16)
        acc = tl.dot(a, bb, acc)
        a_ptrs += BK
        b_ptrs += BK
    if SCALE == 1:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_ok, other=0.0)[None, :]
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def _dequant_kernel(
    w_ptr,
    s_ptr,
    o_ptr,
    N,
    K,
    stride_sn,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """``o = bf16(q * s)`` for a [BN, BK] tile."""
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    k0 = tl.program_id(1) * BK
    offs_k = k0 + tl.arange(0, BK)
    n_ok = offs_n < N
    p = offs_n[:, None].to(tl.int64) * K + offs_k[None, :]
    mask = n_ok[:, None] & (offs_k < K)[None, :]
    w = tl.load(w_ptr + p, mask=mask, other=0).to(tl.float32)
    if SCALE == 2:
        offs_g = k0 // G + tl.arange(0, BK // G)
        scales = tl.load(
            s_ptr + offs_n[:, None].to(tl.int64) * stride_sn + offs_g[None, :],
            mask=n_ok[:, None] & (offs_g < K // G)[None, :],
            other=0,
        ).to(tl.float32)
        grouped = tl.reshape(w, [BN, BK // G, G])
        wf = tl.reshape(grouped * scales[:, :, None], [BN, BK])
    else:
        wf = w * tl.load(s_ptr + offs_n, mask=n_ok, other=0.0)[:, None]
    tl.store(o_ptr + p, wf.to(tl.bfloat16), mask=mask)


# -- launch ------------------------------------------------------------------

SKINNY_INT8_MAX_M = 128
_WS_BYTES = 8 << 20
_split_bufs: dict = {}


def _split_buffers(device) -> tuple[torch.Tensor, torch.Tensor]:
    # Counters are reset by the last block. Independent streams need separate
    # partial sums and counters when two launches overlap.
    stream = torch.cuda.current_stream(device)
    if stream not in _split_bufs:
        _split_bufs[stream] = (
            torch.empty(_WS_BYTES // 4, dtype=torch.float32, device=device),
            torch.zeros(4096, dtype=torch.int32, device=device),
        )
    return _split_bufs[stream]


def _gemv_cfg(n_total: int, K: int) -> tuple[int, int, int, int]:
    """(BN, BK, warps, stages) for one row (RTX 5090 sweep, cold weights)."""
    if n_total >= 2048 or K % 256:
        return 8, 128, 8, 1
    if K % 512 == 0 and K <= 3072:
        return 1, 512, 4, 2
    return 1, 256, 4, 3


def _skinny_cfg(M: int, n_total: int, K: int) -> tuple[int, int, int, int, int]:
    """(BN, BK, SPLIT, warps, stages) for 2..128 rows."""
    narrow = n_total <= 2048
    if narrow:
        if K % (6 * 128) == 0:
            if M <= 16 and K <= 3072:
                return 32, 128, 3, 4, 4
            if M <= 32:
                return 32, 64, 6, 4, 4
            return (32, 128, 6, 4, 3) if M <= 64 else (32, 64, 6, 4, 3)
        return 16, 128, 1, 4, 4
    if M <= 32:
        return 64, 128, 1, 4, 4
    return (64, 64, 1, 4, 3) if M <= 64 else (64, 64, 1, 8, 3)


def _bm(M: int) -> int:
    return 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128))


def _fit_stages(ns: int, bm: int, bn: int, bk: int, w_bytes: int = 1) -> int:
    """Pipeline stages that fit the ~99 KB of shared memory of sm_86 / sm_120
    (x tile bf16 + weight tile per stage)."""
    per_stage = bm * bk * 2 + bn * bk * w_bytes
    return max(1, min(ns, (96 * 1024) // per_stage))


def _fit_group(bk: int, groups: list[int]) -> int:
    """BK must be a multiple of every per-group segment's G."""
    for g in groups:
        if g and bk % g:
            bk = g * -(-bk // g)
    return bk


def _seg_args(s: Segment | None, out: torch.Tensor, dummy: torch.Tensor) -> tuple:
    if s is None:
        return (dummy, dummy, out, dummy, 0, 0, 0)
    y = out[:, s.col0 :] if s.cols is None and s.col0 else out
    scale = s.scale if s.scale is not None else dummy
    return (
        s.weight,
        scale,
        y,
        s.cols if s.cols is not None else dummy,
        s.n,
        s.weight.stride(0),
        scale.stride(0) if s.scale is not None and s.scale.dim() == 2 else 0,
    )


def int8_decode_linear(
    x: torch.Tensor,
    segs: list[Segment],
    out: torch.Tensor | None = None,
    width: int | None = None,
    cfg: tuple | None = None,
) -> torch.Tensor:
    """``out[:, cols] = x @ deq(W).T`` over one or two segments sharing K (INT8
    with per-row / per-group scales, or bf16 with ``scale`` None), for ``x: (M,
    K)`` bf16 with M <= SKINNY_INT8_MAX_M and K % 128 == 0. ``out`` (M, width)
    is allocated when not given (width: max column + 1). ``cfg``: (BN, BK,
    warps, stages) at one row, (BN, BK, SPLIT, warps, stages) beyond."""
    M, K = x.shape
    assert 1 <= len(segs) <= 2 and M <= SKINNY_INT8_MAX_M
    if out is None:
        if width is None:
            assert all(s.cols is None for s in segs), "width required with column maps"
            width = max(s.col0 + s.n for s in segs)
        out = torch.empty(M, width, dtype=x.dtype, device=x.device)
    s1 = segs[0]
    s2 = segs[1] if len(segs) == 2 else None
    n_total = sum(s.n for s in segs)
    dummy = s1.weight
    a1 = _seg_args(s1, out, dummy)
    a2 = _seg_args(s2, out, dummy)
    g1 = seg_group(s1.weight, s1.scale)
    g2 = seg_group(s2.weight, s2.scale) if s2 is not None else 0
    flags = dict(
        SCALE1=_scale_mode(s1),
        G1=g1 or 1,
        HAS_MAP1=s1.cols is not None,
        SEG2=s2 is not None,
        SCALE2=_scale_mode(s2) if s2 is not None else _NONE,
        G2=g2 or 1,
        HAS_MAP2=s2 is not None and s2.cols is not None,
    )
    if M == 1:
        bn, bk, nw, ns = cfg or _gemv_cfg(n_total, K)
        bk = _fit_group(bk, [g1, g2])
        nb1 = triton.cdiv(s1.n, bn)
        nb2 = triton.cdiv(s2.n, bn) if s2 is not None else 0
        _i8_gemv_kernel[(nb1 + nb2,)](
            x,
            K,
            *a1,
            nb1,
            *a2,
            **flags,
            BN=bn,
            BK=bk,
            num_warps=nw,
            num_stages=ns,
        )
        return out
    bn, bk, split, nw, ns = cfg or _skinny_cfg(M, n_total, K)
    bk = _fit_group(bk, [g1, g2])
    bm = _bm(M)
    ns = _fit_stages(ns, bm, bn, bk, 2 if s2 is not None and s2.scale is None else 1)
    nb1 = triton.cdiv(s1.n, bn)
    nb2 = triton.cdiv(s2.n, bn) if s2 is not None else 0
    if K % (split * bk):
        split = 1
    assert K % bk == 0
    if split > 1:
        ws, locks = _split_buffers(x.device)
        assert (
            nb1 + nb2
        ) * split * bn * bm <= ws.numel() and nb1 + nb2 <= locks.numel()
    else:
        ws = locks = x
    _i8_skinny_kernel[(nb1 + nb2, split)](
        x,
        M,
        K // split,
        x.stride(0),
        out.stride(0),
        ws,
        locks,
        *a1,
        nb1,
        *a2,
        **flags,
        SPLIT=split,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=nw,
        num_stages=ns,
    )
    return out


def int8_linear(x: torch.Tensor, w: Int8Weight, cfg=None, out=None) -> torch.Tensor:
    """``x @ deq(W).T`` for M <= SKINNY_INT8_MAX_M rows of bf16 ``x`` (one
    segment: the lean kernels)."""
    M, K = x.shape
    N = w.n
    y = torch.empty(M, N, dtype=x.dtype, device=x.device) if out is None else out
    g = w.group
    mode = _GROUP if g else _ROW
    ss = w.scale.stride(0) if g else 0
    if M == 1:
        bn, bk, nw, ns = cfg or _gemv_cfg(N, K)
        bk = _fit_group(bk, [g])
        _i8_gemv1_kernel[(triton.cdiv(N, bn),)](
            x,
            w.weight,
            w.scale,
            y,
            N,
            K,
            w.weight.stride(0),
            ss,
            SCALE=mode,
            G=g or 1,
            BN=bn,
            BK=bk,
            num_warps=nw,
            num_stages=ns,
        )
        return y
    bn, bk, split, nw, ns = cfg or _skinny_cfg(M, N, K)
    bk = _fit_group(bk, [g])
    bm = _bm(M)
    ns = _fit_stages(ns, bm, bn, bk)
    nb = triton.cdiv(N, bn)
    if K % (split * bk):
        split = 1
    if split > 1:
        ws, locks = _split_buffers(x.device)
        assert nb * split * bn * bm <= ws.numel() and nb <= locks.numel()
    else:
        ws = locks = x
    _i8_skinny1_kernel[(nb, split)](
        x,
        w.weight,
        w.scale,
        y,
        ws,
        locks,
        M,
        N,
        K // split,
        x.stride(0),
        w.weight.stride(0),
        ss,
        y.stride(0),
        SCALE=mode,
        G=g or 1,
        SPLIT=split,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=nw,
        num_stages=ns,
    )
    return y


# -- prefill -----------------------------------------------------------------


def w8a16_gemm(
    x: torch.Tensor,
    w: Int8Weight,
    cfg: dict | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton W8A16 GEMM for prefill-size batches (``x`` bf16 [M, K], K % 64 == 0)."""
    M, K = x.shape
    N = w.n
    y = torch.empty(M, N, dtype=x.dtype, device=x.device) if out is None else out
    g = w.group
    c = cfg or _gemm_cfg(M, N, K)
    bk = _fit_group(c["BK"], [g])
    if K % bk:
        # E.g. K=192 with the 128-wide tile. A narrower tile must still
        # cover an integer number of quantization groups.
        if K % 64 or (g and 64 % g):
            return torch.mm(x, int8_dequant(w).t(), out=y)
        bk = 64
    grid = (triton.cdiv(M, c["BM"]) * triton.cdiv(N, c["BN"]),)
    _w8a16_gemm_kernel[grid](
        x,
        w.weight,
        w.scale,
        y,
        M,
        N,
        K,
        x.stride(0),
        w.weight.stride(0),
        w.scale.stride(0) if g else 0,
        y.stride(0),
        SCALE=_GROUP if g else _ROW,
        G=g or 1,
        BM=c["BM"],
        BN=c["BN"],
        BK=bk,
        GROUP_M=c.get("GROUP_M", 8),
        num_warps=c["num_warps"],
        num_stages=c["num_stages"],
    )
    return y


def _gemm_cfg(M: int, N: int, K: int) -> dict:
    if M <= 256:
        return dict(BM=64, BN=64, BK=128, num_warps=4, num_stages=3)
    return dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3)


def int8_dequant(w: Int8Weight, out: torch.Tensor | None = None) -> torch.Tensor:
    """``bf16(q * s)`` [N, K]."""
    N, K = w.weight.shape
    o = (
        torch.empty(N, K, dtype=torch.bfloat16, device=w.weight.device)
        if out is None
        else out
    )
    g = w.group
    bk = _fit_group(128 if K % 128 == 0 else 64, [g])
    _dequant_kernel[(triton.cdiv(N, 32), triton.cdiv(K, bk))](
        w.weight,
        w.scale,
        o,
        N,
        K,
        w.scale.stride(0) if g else 0,
        SCALE=_GROUP if g else _ROW,
        G=g or 1,
        BN=32,
        BK=bk,
        num_warps=4,
    )
    return o


_scratch: dict = {}


def _deq_scratch(device, numel: int) -> torch.Tensor | None:
    """Per-stream bf16 scratch for dequantize + cuBLAS, or None for a weight
    larger than VLLM_W8A16_INT8_DEQ_SCRATCH_MB. Allocated by the first prefill
    (the profiling run, which sizes the KV cache after it)."""
    budget = envs.VLLM_W8A16_INT8_DEQ_SCRATCH_MB << 20
    if numel * 2 > budget:
        return None
    key = torch.cuda.current_stream(device)
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = torch.empty(
            budget // 2, dtype=torch.bfloat16, device=device
        )
    return buf[:numel]


def _prefill_uses_cublas(M: int, N: int, K: int) -> bool:
    """Dequantize + cuBLAS where it wins on the RTX 5090: from 1024 rows on
    (the GEMMs are compute bound, the dequantization costs 1-4 %), and for
    narrow (N <= 2048) or long-K (K >= 3072) weights at any size; the Triton
    GEMM for wide K = 1536 weights below 1024 rows."""
    return M >= 1024 or N <= 2048 or K >= 3072


def int8_prefill_mm(x: torch.Tensor, w: Int8Weight) -> torch.Tensor:
    """``x @ deq(W).T`` for any number of rows of bf16 ``x``."""
    M, K = x.shape
    N = w.n
    if x.dtype != torch.bfloat16:
        raise ValueError("W8A16 INT8 linears require BF16 activations")
    if x.stride(-1) != 1:
        x = x.contiguous()
    if _prefill_uses_cublas(M, N, K):
        buf = _deq_scratch(x.device, N * K)
        if buf is not None:
            wd = int8_dequant(w, out=buf.view(N, K))
            return torch.mm(x, wd.t())
    if K % 64 == 0:
        return w8a16_gemm(x, w)
    return torch.mm(x, int8_dequant(w).t())


# -- dispatch ----------------------------------------------------------------


@functools.cache
def int8_decode_enabled() -> bool:
    """W8A16 decode kernels for INT8 linears (any SM80+ GPU: bf16 mma.sync);
    VLLM_W8A16_INT8_DECODE=0 runs every batch size through int8_prefill_mm."""
    return (
        envs.VLLM_W8A16_INT8_DECODE
        and current_platform.is_cuda()
        and current_platform.has_device_capability(80)
    )


def int8_decode_max_m(n: int, k: int, n_bf16: int = 0) -> int:
    """Largest row count the decode kernels take for an ``[n, k]`` INT8 weight
    (plus ``n_bf16`` bf16 rows in the same launch); int8_prefill_mm beyond.
    RTX 5090, cold weights: vocabulary-size weights (n >= 65536) up to 32 rows
    (151936 x 1536: 147 us; 166 at 64 rows in the prefill GEMM vs 199), narrow
    weights (n <= 2048) and wide K = 1536 ones (n >= 6144) up to 128 (7344 x
    1536: 21.5 vs 23.5 us at 128 rows), the others up to 64 (4608 x 1536: 21.2
    vs 14.3 us at 128 rows)."""
    if not int8_decode_enabled() or k % 128:
        return 0
    if n >= 65536:
        return 32
    if n <= 2048 or n_bf16 >= 512 or (k == 1536 and n >= 6144):
        return SKINNY_INT8_MAX_M
    return 64


def int8_mm(x: torch.Tensor, w: Int8Weight) -> torch.Tensor:
    """``x @ deq(W).T`` for a 2-D bf16 ``x``: decode kernels up to
    int8_decode_max_m rows, int8_prefill_mm beyond."""
    M = x.shape[0]
    if (
        0 < M <= int8_decode_max_m(w.n, w.k)
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
    ):
        return int8_linear(x, w)
    if M == 0:
        return x.new_empty(0, w.n)
    return int8_prefill_mm(x, w)


def _op(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    y = int8_mm(x.reshape(-1, shape[-1]), Int8Weight(weight, scale))
    return y.view(*shape[:-1], weight.shape[0])


def _op_fake(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


direct_register_custom_op(
    op_name="w8a16_int8_linear", op_func=_op, mutates_args=[], fake_impl=_op_fake
)


def w8a16_int8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``x @ (scale * weight).T`` for bf16 ``x [..., K]``, int8 ``weight [N, K]``
    and per-row fp32 ``scale [N]`` or per-group ``scale [N, K // G]`` (opaque
    op: the row count is dispatched at run time)."""
    return torch.ops.vllm.w8a16_int8_linear(x, weight, scale)

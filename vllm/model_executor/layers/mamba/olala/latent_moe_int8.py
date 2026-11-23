# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only Olala experts, shared expert and latent projections.

Small batches fuse routed and shared projections. Medium batches compact the
routed rows before grouped GEMMs. Prefill sorts by expert using vLLM's
moe_align_block_size. All paths use BF16 activations and accumulate in FP32.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.model_executor.layers.mamba.olala.pdl import pdl_kwargs
from vllm.triton_utils import tl, triton

# Weight formats (constexpr FMT)
FMT_BF16, FMT_ROW, FMT_GROUP, FMT_INT4 = 0, 1, 2, 3


@dataclass
class WqWeight:
    """A weight-only quantized ``[N, K]`` (dense) or ``[E, N, K]`` (experts)
    matrix as the kernels read it."""

    # int8 [(E,) N, K]; bf16 for FMT_BF16; uint8 [(E,) N, K // 2] for FMT_INT4
    weight: torch.Tensor
    # fp32 [(E,) N] or fp16 / fp32 [(E,) N, K // G]; FMT_INT4: int32
    # [(E,) N, K // G], bf16 scale in the low and bf16 bias in the high half
    scale: torch.Tensor | None = None

    def __post_init__(self):
        assert self.weight.is_contiguous()
        assert self.scale is None or self.scale.is_contiguous()
        assert self.weight.dtype != torch.uint8 or (
            self.scale is not None and self.scale.dtype == torch.int32
        )

    @classmethod
    def int4_affine(
        cls, packed: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
    ) -> "WqWeight":
        """INT4 affine weight, MLX q4 layout: ``w[.., n, k] = s[.., n, k // G] *
        q[.., n, k] + b[.., n, k // G]`` (fp32, rounded to bf16), q in [0, 15]
        with ``packed[.., n, j] = q[.., n, 2j] | q[.., n, 2j + 1] << 4``;
        ``scale`` / ``bias`` bf16 ``[(E,) N, K // G]``."""
        assert packed.dtype == torch.uint8
        assert scale.dtype == bias.dtype == torch.bfloat16 and scale.shape == bias.shape
        sb = torch.stack((scale, bias), dim=-1).contiguous().view(torch.int32)
        return cls(packed.contiguous(), sb.squeeze(-1).contiguous())

    @property
    def n(self) -> int:
        return self.weight.shape[-2]

    @property
    def k(self) -> int:
        return self.weight.shape[-1] * (2 if self.fmt == FMT_INT4 else 1)

    @property
    def fmt(self) -> int:
        if self.scale is None:
            return FMT_BF16
        if self.weight.dtype == torch.uint8:
            return FMT_INT4
        return FMT_ROW if self.scale.dim() == self.weight.dim() - 1 else FMT_GROUP

    @property
    def group(self) -> int:
        """G for FMT_GROUP / FMT_INT4 (1 otherwise: unused constexpr)."""
        if self.fmt in (FMT_GROUP, FMT_INT4):
            return self.k // self.scale.shape[-1]
        return 1

    def args(self) -> tuple:
        """(weight, scale) pointers (the weight stands in for a missing scale)."""
        return self.weight, self.scale if self.scale is not None else self.weight


# -- tiles --------------------------------------------------------------------


@triton.jit
def _wq_expert(
    w_ptr,
    s_ptr,
    e,
    N: tl.constexpr,
    KD: tl.constexpr,
    FMT: tl.constexpr,
    G: tl.constexpr,
):
    """Weight and scale pointers of expert ``e`` of an ``[E, N, KD]`` weight."""
    if FMT == 3:
        w = w_ptr + e * N * (KD // 2)
        s = s_ptr + e * N * (KD // G)
    elif FMT == 2:
        w = w_ptr + e * N * KD
        s = s_ptr + e * N * (KD // G)
    else:
        w = w_ptr + e * N * KD
        s = s_ptr + e * N
    return w, s


@triton.jit
def _i4_tiles(
    w_ptr,
    s_ptr,
    offs_n,
    k0,
    KD: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    """FMT_INT4: the even (``k0 + 2j``) and odd (``k0 + 2j + 1``) columns of
    ``W[offs_n, k0:k0 + BK]^T`` as two ``[BK / 2, BN]`` bf16 tiles of
    ``s * q + b`` computed in fp32 (``s * q`` is exact). The codes become fp32
    through the 2^23 exponent trick, the bf16 scale / bias by a shift."""
    BKH: tl.constexpr = BK // 2
    offs_h = k0 // 2 + tl.arange(0, BKH)
    p = tl.load(w_ptr + offs_n[None, :] * (KD // 2) + offs_h[:, None]).to(tl.int32)
    offs_g = k0 // G + tl.arange(0, BK // G)
    sb = tl.load(s_ptr + offs_n[None, :] * (KD // G) + offs_g[:, None])
    s = tl.expand_dims((sb << 16).to(tl.float32, bitcast=True), 1)
    b = tl.expand_dims((sb & -65536).to(tl.float32, bitcast=True), 1)
    lo = ((p & 15) | 0x4B000000).to(tl.float32, bitcast=True) - 8388608.0
    hi = ((p >> 4) | 0x4B000000).to(tl.float32, bitcast=True) - 8388608.0
    lo = tl.reshape(tl.reshape(lo, [BK // G, G // 2, BN]) * s + b, [BKH, BN])
    hi = tl.reshape(tl.reshape(hi, [BK // G, G // 2, BN]) * s + b, [BKH, BN])
    return lo.to(tl.bfloat16), hi.to(tl.bfloat16)


@triton.jit
def _wq_tile_t(
    w_ptr,
    s_ptr,
    offs_n,
    k0,
    KD: tl.constexpr,
    FMT: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    """Load a BF16 weight tile with K as the leading dimension.
    Group scales are applied here and per-row scales in the epilogue."""
    if FMT == 3:
        lo, hi = _i4_tiles(w_ptr, s_ptr, offs_n, k0, KD, G, BK, BN)
        return tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), [BK, BN])
    offs_k = k0 + tl.arange(0, BK)
    w = tl.load(w_ptr + offs_n[None, :] * KD + offs_k[:, None])
    if FMT == 2:
        offs_g = k0 // G + tl.arange(0, BK // G)
        s = tl.load(s_ptr + offs_n[None, :] * (KD // G) + offs_g[:, None]).to(
            tl.float32
        )
        w3 = tl.reshape(w.to(tl.float32), [BK // G, G, BN])
        wb = tl.reshape(w3 * s[:, None, :], [BK, BN]).to(tl.bfloat16)
    else:
        wb = w.to(tl.bfloat16)
    return wb


@triton.jit
def _wq_row_scale(acc, s_ptr, offs_n, FMT: tl.constexpr):
    """Epilogue of ``acc [BM, BN]``: the per-row scales of FMT_ROW."""
    if FMT == 1:
        acc = acc * tl.load(s_ptr + offs_n)[None, :]
    return acc


@triton.jit
def _wq_mm(
    acc,
    a_ptr,
    rows,
    rmask,
    stride_a,
    w_ptr,
    s_ptr,
    offs_n,
    KD: tl.constexpr,
    FMT: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    ONE_ROW: tl.constexpr = False,
):
    """``acc [BM, BN] += A[rows, :KD] @ deq(W)[offs_n, :KD].T`` (row scales not
    applied). ONE_ROW: one activation row (scalar ``rows``) broadcast over the
    BM rows of the MMA tile (only row 0 of ``acc`` is meaningful). The loop is
    pipelined with the launch's num_stages."""
    offs_k = tl.arange(0, BK)
    for k0 in tl.range(0, KD, BK):
        if ONE_ROW:
            a = tl.load(a_ptr + rows * stride_a + (k0 + offs_k)[None, :]).to(
                tl.bfloat16
            )
            a = tl.broadcast_to(a, [BM, BK])
        else:
            a = tl.load(
                a_ptr + rows[:, None] * stride_a + (k0 + offs_k)[None, :],
                mask=rmask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
        w = _wq_tile_t(w_ptr, s_ptr, offs_n, k0, KD, FMT, G, BK, BN)
        acc = tl.dot(a, w, acc)
    return acc


@triton.jit
def _relu2_bf16(acc):
    h = acc.to(tl.bfloat16).to(tl.float32)
    h = tl.maximum(h, 0.0)
    return (h * h).to(tl.bfloat16)


@triton.jit
def _topk_slot(
    logits_ptr,
    logits_lo_ptr,
    bias_ptr,
    m,
    k_slot,
    stride_lm,
    E: tl.constexpr,
    K: tl.constexpr,
    LOGITS_LO: tl.constexpr,
):
    """Expert and normalized weight of slot ``k_slot`` of token ``m``: sigmoid
    scores ranked by score + bias, weights renormalized over the k winners
    (topk_sigmoid semantics; see latent_moe_decode._latent_moe_up_kernel)."""
    offs_e = tl.arange(0, E)
    lg = tl.load(logits_ptr + m * stride_lm + offs_e)
    if LOGITS_LO:
        lg += tl.load(logits_lo_ptr + m * stride_lm + offs_e)
    sc = 1.0 / (1.0 + tl.exp(-lg))
    sb = sc + tl.load(bias_ptr + offs_e)
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
    return e, wsel / tot


# -- small path ---------------------------------------------------------------


@triton.jit(do_not_specialize=["M"])
def _i8_small_up_kernel(
    x_ptr,  # [M, D] latent (bf16, or fp32 rounded to bf16 on load)
    logits_ptr,  # [M, E] fp32
    logits_lo_ptr,  # [M, E] fp32 low-part logits added to logits (LOGITS_LO)
    bias_ptr,  # [E] fp32
    ids_ptr,  # [M*K] int32 out
    tw_ptr,  # [M*K] fp32 out (normalized routing weights)
    w1_ptr,  # [E, N, D] int8 (FMT_INT4: [E, N, D // 2] uint8)
    w1s_ptr,  # [E, N] / [E, N, D // G1] scales
    h_ptr,  # [M*K, N] bf16 out: relu^2
    hid_ptr,  # [M, H] bf16 block input (shared expert)
    su_ptr,  # [S, H] shared up
    sus_ptr,  # its scales
    s_ptr,  # [M, S] bf16 out: relu^2(shared up)
    M,
    stride_xm,
    stride_lm,
    stride_hm,
    E: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FMT1: tl.constexpr,
    G1: tl.constexpr,
    FMTS: tl.constexpr,
    GS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BNS: tl.constexpr,
    BKS: tl.constexpr,
    GEMV: tl.constexpr = False,
    LOGITS_LO: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """Programs [0, S/BNS): shared up projection, one N block for all M (<= 16)
    tokens (the longest programs, scheduled first); the rest: routed expert
    (token, slot, N block) with the router top-k recomputed from the logits."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    NB: tl.constexpr = N // BN
    NBS: tl.constexpr = S // BNS
    offs_r = tl.arange(0, 16)
    if pid >= NBS:
        pid -= NBS
        row = pid // NB
        nb = pid % NB
        m = row // K
        e, wsel = _topk_slot(
            logits_ptr, logits_lo_ptr, bias_ptr, m, row % K, stride_lm, E, K, LOGITS_LO
        )
        if nb == 0:
            tl.store(ids_ptr + row, e.to(tl.int32))
            tl.store(tw_ptr + row, wsel)
        e64 = e.to(tl.int64)
        offs_n = (nb * BN + tl.arange(0, BN)).to(tl.int64)
        w1, w1s = _wq_expert(w1_ptr, w1s_ptr, e64, N, D, FMT1, G1)
        if GEMV:
            acc1 = _wq_gemv_x(
                x_ptr, m.to(tl.int64) * stride_xm, w1, w1s, offs_n, D, FMT1, G1, BK, BN
            )
        else:
            acc = tl.zeros([16, BN], dtype=tl.float32)
            acc = _wq_mm(
                acc,
                x_ptr,
                m.to(tl.int64),
                offs_r < 1,
                stride_xm,
                w1,
                w1s,
                offs_n,
                D,
                FMT1,
                G1,
                16,
                BK,
                BN,
                ONE_ROW=True,
            )
            acc = _wq_row_scale(acc, w1s, offs_n, FMT1)
            acc1 = tl.sum(tl.where(offs_r[:, None] < 1, acc, 0.0), axis=0)
        tl.store(h_ptr + row * N + offs_n, _relu2_bf16(acc1))
    else:
        offs_ns = (pid * BNS + tl.arange(0, BNS)).to(tl.int64)
        rmask = offs_r < M
        acc_sh = tl.zeros([16, BNS], dtype=tl.float32)
        acc_sh = _wq_mm(
            acc_sh,
            hid_ptr,
            offs_r.to(tl.int64),
            rmask,
            stride_hm,
            su_ptr,
            sus_ptr,
            offs_ns,
            H,
            FMTS,
            GS,
            16,
            BKS,
            BNS,
        )
        acc_sh = _wq_row_scale(acc_sh, sus_ptr, offs_ns, FMTS)
        tl.store(
            s_ptr + offs_r[:, None] * S + offs_ns[None, :],
            _relu2_bf16(acc_sh),
            mask=rmask[:, None],
        )


@triton.jit
def _wq_gemv_x(
    a_ptr,
    a_off,
    w_ptr,
    s_ptr,
    offs_n,
    KD: tl.constexpr,
    FMT: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    """``[BN]`` fp32 = ``deq(W)[offs_n, :KD] @ a`` for the activation row at
    ``a_ptr + a_off`` (bf16 or fp32 rounded to bf16), fp32 FMAs, row scales
    applied."""
    if FMT == 3:
        # even / odd columns as two half tiles: no interleave of the codes
        offs_h = tl.arange(0, BK // 2)
        acc4 = tl.zeros([BK // 2, BN], dtype=tl.float32)
        for k0 in tl.range(0, KD, BK):
            a_lo = (
                tl.load(a_ptr + a_off + k0 + 2 * offs_h).to(tl.bfloat16).to(tl.float32)
            )
            a_hi = (
                tl.load(a_ptr + a_off + k0 + 2 * offs_h + 1)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            lo, hi = _i4_tiles(w_ptr, s_ptr, offs_n, k0, KD, G, BK, BN)
            acc4 += (
                lo.to(tl.float32) * a_lo[:, None] + hi.to(tl.float32) * a_hi[:, None]
            )
        return tl.sum(acc4, axis=0)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros([BK, BN], dtype=tl.float32)
    for k0 in tl.range(0, KD, BK):
        a = tl.load(a_ptr + a_off + k0 + offs_k).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + offs_n[None, :] * KD + (k0 + offs_k)[:, None]).to(
            tl.float32
        )
        if FMT == 2:
            offs_g = k0 // G + tl.arange(0, BK // G)
            s = tl.load(s_ptr + offs_n[None, :] * (KD // G) + offs_g[:, None]).to(
                tl.float32
            )
            w = tl.reshape(tl.reshape(w, [BK // G, G, BN]) * s[:, None, :], [BK, BN])
        acc += w * a[:, None]
    y = tl.sum(acc, axis=0)
    if FMT == 1:
        y = y * tl.load(s_ptr + offs_n)
    return y


@triton.jit
def _i8_small_down_kernel(
    h_ptr,  # [M*K, N1] bf16
    ids_ptr,  # [M*K] int32
    tw_ptr,  # [M*K] fp32
    w2_ptr,  # [E, D2, N1] int8
    w2s_ptr,  # [E, D2] / [E, D2, N1 // G2] scales
    part_ptr,  # [M*K, D2] fp32 out (routing weight applied)
    N1: tl.constexpr,
    D2: tl.constexpr,
    FMT2: tl.constexpr,
    G2: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GEMV: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    NB: tl.constexpr = D2 // BN
    row = pid // NB
    nb = pid % NB
    e = tl.load(ids_ptr + row).to(tl.int64)
    offs_n = (nb * BN + tl.arange(0, BN)).to(tl.int64)
    w2, w2s = _wq_expert(w2_ptr, w2s_ptr, e, D2, N1, FMT2, G2)
    if GEMV:
        part = _wq_gemv_x(
            h_ptr, row.to(tl.int64) * N1, w2, w2s, offs_n, N1, FMT2, G2, BK, BN
        )
    else:
        offs_r = tl.arange(0, 16)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        acc = _wq_mm(
            acc,
            h_ptr,
            row.to(tl.int64),
            offs_r < 1,
            N1,
            w2,
            w2s,
            offs_n,
            N1,
            FMT2,
            G2,
            16,
            BK,
            BN,
            ONE_ROW=True,
        )
        acc = _wq_row_scale(acc, w2s, offs_n, FMT2)
        part = tl.sum(tl.where(offs_r[:, None] < 1, acc, 0.0), axis=0)
    tl.store(part_ptr + row * D2 + offs_n, part * tl.load(tw_ptr + row))


@triton.jit(do_not_specialize=["M"])
def _i8_exit_kernel(
    r_ptr,  # SUM_K: [M*K, D2] fp32 partials; else [M, D2] bf16 routed (scaled)
    fc2_ptr,  # [H, D2]
    fc2s_ptr,  # its scales
    s_ptr,  # [M, S] bf16 relu^2(shared up)
    sd_ptr,  # [H, S] shared down
    sds_ptr,  # its scales
    out_ptr,  # [M, H] bf16
    M,
    scale,
    stride_r,
    H: tl.constexpr,
    D2: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FMTF: tl.constexpr,
    GF: tl.constexpr,
    FMTD: tl.constexpr,
    GD: tl.constexpr,
    SUM_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BKL: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """``out = fc2_latent(routed) + shared_down(relu^2(s))``; with SUM_K the
    routed latent ``bf16(scale * sum_k partials)`` is formed on the fly."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    rmask = rows < M
    offs_n = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
    offs_l = tl.arange(0, BKL)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    # not pipelined: the SUM_K loads of K partial tiles per step would not fit
    for k0 in tl.range(0, D2, BKL, num_stages=1):
        if SUM_K:
            r = tl.zeros([BM, BKL], dtype=tl.float32)
            for kk in tl.static_range(K):
                r += tl.load(
                    r_ptr + (rows * K + kk)[:, None] * D2 + (k0 + offs_l)[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
            r = (r * scale).to(tl.bfloat16)
        else:
            r = tl.load(
                r_ptr + rows[:, None] * stride_r + (k0 + offs_l)[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
        w = _wq_tile_t(fc2_ptr, fc2s_ptr, offs_n, k0, D2, FMTF, GF, BKL, BN)
        acc = tl.dot(r, w, acc)
    acc = _wq_row_scale(acc, fc2s_ptr, offs_n, FMTF)
    acc_s = tl.zeros([BM, BN], dtype=tl.float32)
    acc_s = _wq_mm(
        acc_s, s_ptr, rows, rmask, S, sd_ptr, sds_ptr, offs_n, S, FMTD, GD, BM, BK, BN
    )
    acc += _wq_row_scale(acc_s, sds_ptr, offs_n, FMTD)
    tl.store(
        out_ptr + rows[:, None] * H + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit(do_not_specialize=["M"])
def _i8_exit_gemv_kernel(
    part_ptr,  # [M*K, D2] fp32 routed partials
    fc2_ptr,  # [H, D2]
    fc2s_ptr,  # its scales
    s_ptr,  # [M, S] bf16 relu^2(shared up)
    sd_ptr,  # [H, S] shared down
    sds_ptr,  # its scales
    out_ptr,  # [M, H] bf16
    M,
    scale,
    D2: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    FMTF: tl.constexpr,
    GF: tl.constexpr,
    FMTD: tl.constexpr,
    GD: tl.constexpr,
    BN: tl.constexpr,
    BKL: tl.constexpr,
    BKS: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """_i8_exit_kernel for one or two tokens with fp32 FMAs: BN (< 16) output
    rows per program, i.e. more programs than the 16-row MMA tiles allow
    (1536 / 16 = 96 for 170 SMs). Same values: bf16 routed latent, bf16(q) or
    bf16(q * s) weights, fp32 sums."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    offs_n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    offs_l = tl.arange(0, BKL)
    offs_s = tl.arange(0, BKS)
    for m in range(M):
        acc = tl.zeros([BN, BKL], dtype=tl.float32)
        for k0 in tl.range(0, D2, BKL):
            r = tl.zeros([BKL], dtype=tl.float32)
            for kk in tl.static_range(K):
                r += tl.load(part_ptr + (m * K + kk) * D2 + k0 + offs_l)
            r = (r * scale).to(tl.bfloat16).to(tl.float32)
            w = tl.trans(
                _wq_tile_t(fc2_ptr, fc2s_ptr, offs_n, k0, D2, FMTF, GF, BKL, BN)
            )
            acc += w.to(tl.float32) * r[None, :]
        y = tl.sum(acc, axis=1)
        if FMTF == 1:
            y = y * tl.load(fc2s_ptr + offs_n)
        acc_s = tl.zeros([BN, BKS], dtype=tl.float32)
        for k0 in tl.range(0, S, BKS):
            a = tl.load(s_ptr + m * S + k0 + offs_s).to(tl.float32)
            w = tl.trans(_wq_tile_t(sd_ptr, sds_ptr, offs_n, k0, S, FMTD, GD, BKS, BN))
            acc_s += w.to(tl.float32) * a[None, :]
        ys = tl.sum(acc_s, axis=1)
        if FMTD == 1:
            ys = ys * tl.load(sds_ptr + offs_n)
        tl.store(out_ptr + m * H + offs_n, (y + ys).to(tl.bfloat16))


# -- medium path --------------------------------------------------------------


@triton.jit(do_not_specialize=["M", "RCAP"])
def _i8_moe_gemm_kernel(
    a_ptr,  # up: latent [M, Kd] (row entry // TOPK); down: h [M*TOPK, Kd]
    b_ptr,  # [E, N, Kd]
    bs_ptr,  # its scales
    c_ptr,  # up: h [M*TOPK, N] bf16; down: part [M*TOPK, N] fp32
    tw_ptr,  # [M*TOPK] fp32 (down only)
    rows_ptr,  # [E, RCAP] int32
    cnt_ptr,  # [E] int32
    # shared up (UP launches with SHARED): relu^2(hid @ su.T) for all M rows
    hid_ptr,
    su_ptr,
    sus_ptr,
    s_ptr,
    M,
    RCAP,
    stride_am,
    stride_hm,
    E: tl.constexpr,
    N: tl.constexpr,
    KD: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    TOPK: tl.constexpr,
    UP: tl.constexpr,
    SHARED: tl.constexpr,
    FMTB: tl.constexpr,
    GB: tl.constexpr,
    FMTS: tl.constexpr,
    GS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BKS: tl.constexpr = 64,
    BN_SH: tl.constexpr = 0,
    PDL: tl.constexpr = False,
):
    """Grouped W8A16 GEMM of the medium path: program (row block, N block) over
    every expert's routed rows; with SHARED the first ``cdiv(M, BLOCK_SIZE_M) *
    S / BN_SH`` programs run the shared expert's up projection."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BLOCK_SIZE_N
    BNS: tl.constexpr = BLOCK_SIZE_N if BN_SH == 0 else BN_SH
    nps: tl.constexpr = S // BNS
    nsh = tl.cdiv(M, BLOCK_SIZE_M) * nps if SHARED else 0
    if pid < nsh:
        q = pid
        rows = ((q // nps) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        offs_ns = ((q % nps) * BNS + tl.arange(0, BNS)).to(tl.int64)
        rmask = rows < M
        acc_sh = tl.zeros((BLOCK_SIZE_M, BNS), dtype=tl.float32)
        acc_sh = _wq_mm(
            acc_sh,
            hid_ptr,
            rows,
            rmask,
            stride_hm,
            su_ptr,
            sus_ptr,
            offs_ns,
            H,
            FMTS,
            GS,
            BLOCK_SIZE_M,
            BKS,
            BNS,
        )
        acc_sh = _wq_row_scale(acc_sh, sus_ptr, offs_ns, FMTS)
        tl.store(
            s_ptr + rows[:, None] * S + offs_ns[None, :],
            _relu2_bf16(acc_sh),
            mask=rmask[:, None],
        )
        return
    pid -= nsh
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_e = tl.arange(0, E)
    nblk = (tl.load(cnt_ptr + offs_e) + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cum = tl.cumsum(nblk, axis=0)
    e = tl.sum((cum <= pid_m).to(tl.int32), axis=0)
    if e >= E:
        return
    first = tl.sum(tl.where(offs_e == e, cum - nblk, 0), axis=0)
    e = e.to(tl.int64)
    cnt = tl.load(cnt_ptr + e)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    offs_r = (pid_m - first) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rmask = offs_r < cnt
    entry = tl.load(rows_ptr + e * RCAP + offs_r, mask=rmask, other=0).to(tl.int64)
    arow = entry // TOPK if UP else entry
    b, bs = _wq_expert(b_ptr, bs_ptr, e, N, KD, FMTB, GB)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc = _wq_mm(
        acc,
        a_ptr,
        arow,
        rmask,
        stride_am,
        b,
        bs,
        offs_n,
        KD,
        FMTB,
        GB,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
    )
    acc = _wq_row_scale(acc, bs, offs_n, FMTB)
    c_ptrs = c_ptr + entry[:, None] * N + offs_n[None, :]
    if UP:
        tl.store(c_ptrs, _relu2_bf16(acc), mask=rmask[:, None])
    else:
        w = tl.load(tw_ptr + entry, mask=rmask, other=0.0)
        tl.store(c_ptrs, acc * w[:, None], mask=rmask[:, None])


# -- launch -------------------------------------------------------------------

# Small path up to this many tokens, medium path beyond (RTX 5090, cold
# weights: bench_latent_moe_int8.py).
INT8_SMALL_MAX_TOKENS = 12

_SMALL_UP = dict(BN=32, BK=128, BNS=16, BKS=256, num_warps=4, num_stages=3)
_SMALL_DOWN = dict(BN=32, BK=256, num_warps=4, num_stages=3)
_EXIT_SMALL = dict(BM=16, BN=16, BK=256, BKL=128, num_warps=4, num_stages=3)
# (max tokens, up, down, exit): RTX 5090 sweep (uniform
# routing, cold weights): 16.8 us / layer at 1 token, 31.3 at 8.
_SMALL = [
    # 1-2 tokens: the GEMV exit (GEMV_BN rows per program): 15.1 / 21.5 us per layer at
    # 1 / 2 tokens vs 17.9 / 22.4 with the MMA exit
    # (and the GEMV routed tiles, GEMV in the up config: 14.8 / 21.3 us)
    (
        1,
        dict(_SMALL_UP, BKS=512, GEMV=True),
        dict(_SMALL_DOWN, BN=8),
        dict(GEMV_BN=4, BKL=128, BKS=256, num_warps=4, num_stages=1),
    ),
    (
        2,
        dict(_SMALL_UP, BKS=512, GEMV=True),
        dict(_SMALL_DOWN, BN=8, BK=128),
        dict(GEMV_BN=2, BKL=128, BKS=512, num_warps=2, num_stages=2),
    ),
    (
        16,
        dict(_SMALL_UP, BN=64),
        dict(_SMALL_DOWN, BN=16, BK=128, num_warps=2),
        dict(_EXIT_SMALL, BN=32, BK=512),
    ),
]


# INT4 affine routed experts (FMT_INT4; dense INT8 / bf16 as above, the exits
# unchanged): RTX 5090 sweep (uniform routing, cold weights): 14.9 / 18.9 / 27.7 /
# 31.4 us per layer at 1 / 2 / 8 / 12 tokens (INT8 14.8 / 21.3 / 31.2 / 37.7);
# bench_latent_moe_int8.py --format int4-affine times these configs.
_SMALL_I4 = [
    (
        1,
        dict(_SMALL_UP, BN=16, BKS=512, num_warps=2, num_stages=2, GEMV=True),
        dict(_SMALL_DOWN, BN=8, num_stages=2),
        _SMALL[0][3],
    ),
    (
        2,
        dict(_SMALL_UP, BN=16, BKS=512, num_warps=2, num_stages=2, GEMV=True),
        dict(_SMALL_DOWN, BN=8),
        _SMALL[1][3],
    ),
    (
        16,
        dict(_SMALL_UP, BN=64, BKS=256, num_warps=2, num_stages=2),
        dict(_SMALL_DOWN, BN=16, BK=128, num_warps=2, num_stages=2),
        _SMALL[2][3],
    ),
]


def _pick_small(M: int, fmt: int = FMT_ROW):
    table = _SMALL_I4 if fmt == FMT_INT4 else _SMALL
    for mx, *cfgs in table:
        if mx >= M:
            return cfgs
    return table[-1][1:]


def _sm80_stages(cfg: dict, gemv: bool, fmt: int, dev: torch.device) -> None:
    """Triton 3.8 fails to compile the pipelined MMA form of the small-path up
    and down kernels for sm_80 / 86 / 89 when the routed experts are bf16 or
    per-row INT8 (TritonGPUPrefetch: "PassManager::run failed"); grouped INT8
    and INT4 experts compile. The tables above were tuned on sm_120: one stage
    on the SM80 family. The stage count does not change the result."""
    if (
        not gemv
        and fmt in (FMT_BF16, FMT_ROW)
        and cfg.get("num_stages", 1) > 1
        and gpu_family(dev) == GpuFamily.SM80
    ):
        cfg["num_stages"] = 1


_UP_MEDIUM = dict(
    BLOCK_SIZE_M=16,
    BLOCK_SIZE_N=64,
    BLOCK_SIZE_K=64,
    BKS=128,
    BN_SH=32,
    num_warps=4,
    num_stages=4,
)
_DOWN_MEDIUM = dict(
    BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, num_warps=4, num_stages=2
)
# (max tokens, up, down, exit, top-k sum in the exit): RTX 5090 sweep (uniform
# routing, cold weights): 51.0 us / layer at 32 tokens, 76.5 at 128.
_MEDIUM = [
    (
        64,
        _UP_MEDIUM,
        _DOWN_MEDIUM,
        dict(BM=16, BN=32, BK=256, BKL=128, num_warps=4, num_stages=3),
        False,
    ),
    (
        512,
        dict(_UP_MEDIUM, BN_SH=64, num_stages=2),
        dict(_DOWN_MEDIUM, num_stages=4),
        dict(BM=16, BN=64, BK=128, BKL=128, num_warps=4, num_stages=3),
        False,
    ),
]


# INT4 affine routed experts: 35.3 / 41.9 / 50.5 / 61.1 us per layer at 16 / 32 /
# 64 / 128 tokens (INT8 42.3 / 50.6 / 63.2 / 75.9), same sweep.
_MEDIUM_I4 = [
    (
        64,
        dict(_UP_MEDIUM, BLOCK_SIZE_K=128, num_stages=3),
        dict(_DOWN_MEDIUM, num_warps=2, num_stages=2),
        _MEDIUM[0][3],
        False,
    ),
    (
        512,
        dict(_UP_MEDIUM, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, BN_SH=64, num_stages=2),
        dict(_DOWN_MEDIUM, num_stages=3),
        _MEDIUM[1][3],
        False,
    ),
]


def _pick_medium(M: int, fmt: int = FMT_ROW):
    table = _MEDIUM_I4 if fmt == FMT_INT4 else _MEDIUM
    for mx, *cfgs in table:
        if mx >= M:
            return cfgs
    return table[-1][1:]


def _fit_bk(bk: int, w: WqWeight) -> int:
    g = w.group if w.fmt in (FMT_GROUP, FMT_INT4) else 1
    return max(bk, g) if g > 1 else bk


def olala_moe_small_int8(
    hidden: torch.Tensor,  # [M, H] bf16 (shared expert input)
    x_in: torch.Tensor,  # [M, >= D1] latent (bf16 or fp32), any row stride
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w13: WqWeight,  # [E, N1, D1]
    w2: WqWeight,  # [E, D2, N1]
    shared_up: WqWeight,  # [S, H]
    shared_down: WqWeight,  # [H, S]
    fc2: WqWeight,  # [H, D2]
    top_k: int,
    routed_scale: float,
    configs=None,
    router_logits_lo: torch.Tensor | None = None,  # added to router_logits
    gemv: bool = False,
) -> torch.Tensor:
    """Small-batch (M <= 16) latent MoE with weight-only quantized (or bf16)
    routed experts, shared expert and fc2_latent."""
    M, H = hidden.shape
    E, N1, D1 = w13.weight.shape[0], w13.n, w13.k
    D2 = w2.n
    S = shared_up.n
    assert M <= 16 and w2.k == N1 and shared_down.k == S and (fc2.n, fc2.k) == (H, D2)
    dev = hidden.device
    MK = M * top_k
    ids = torch.empty(MK, dtype=torch.int32, device=dev)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    h = torch.empty(MK, N1, dtype=torch.bfloat16, device=dev)
    s = torch.empty(M, S, dtype=torch.bfloat16, device=dev)
    up_cfg, down_cfg, exit_cfg = configs or _pick_small(M, w13.fmt)
    cfg = dict(up_cfg)
    gemv = cfg.pop("GEMV", False) or gemv
    _sm80_stages(cfg, gemv, w13.fmt, dev)
    cfg["BK"] = _fit_bk(cfg["BK"], w13)
    cfg["BKS"] = _fit_bk(cfg["BKS"], shared_up)
    _i8_small_up_kernel[(MK * (N1 // cfg["BN"]) + S // cfg["BNS"],)](
        x_in,
        router_logits,
        router_logits if router_logits_lo is None else router_logits_lo,
        e_score_bias,
        ids,
        tw,
        *w13.args(),
        h,
        hidden,
        *shared_up.args(),
        s,
        M,
        x_in.stride(0),
        router_logits.stride(0),
        hidden.stride(0),
        E=E,
        D=D1,
        N=N1,
        H=H,
        S=S,
        K=top_k,
        FMT1=w13.fmt,
        G1=w13.group,
        FMTS=shared_up.fmt,
        GS=shared_up.group,
        GEMV=gemv,
        LOGITS_LO=router_logits_lo is not None,
        **cfg,
        **pdl_kwargs(),
    )
    assert router_logits_lo is None or router_logits_lo.stride(
        0
    ) == router_logits.stride(0)
    part = torch.empty(MK, D2, dtype=torch.float32, device=dev)
    cfg = dict(down_cfg)
    _sm80_stages(cfg, gemv, w2.fmt, dev)
    cfg["BK"] = _fit_bk(cfg["BK"], w2)
    _i8_small_down_kernel[(MK * (D2 // cfg["BN"]),)](
        h,
        ids,
        tw,
        *w2.args(),
        part,
        N1=N1,
        D2=D2,
        FMT2=w2.fmt,
        G2=w2.group,
        GEMV=gemv,
        **cfg,
        **pdl_kwargs(),
    )
    out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    cfg = dict(exit_cfg)
    if "GEMV_BN" in cfg:
        bn = cfg.pop("GEMV_BN")
        cfg["BKL"] = _fit_bk(cfg["BKL"], fc2)
        cfg["BKS"] = _fit_bk(cfg["BKS"], shared_down)
        _i8_exit_gemv_kernel[(H // bn,)](
            part,
            *fc2.args(),
            s,
            *shared_down.args(),
            out,
            M,
            routed_scale,
            D2=D2,
            S=S,
            K=top_k,
            H=H,
            FMTF=fc2.fmt,
            GF=fc2.group,
            FMTD=shared_down.fmt,
            GD=shared_down.group,
            BN=bn,
            **cfg,
            **pdl_kwargs(),
        )
        return out
    cfg["BK"] = _fit_bk(cfg["BK"], shared_down)
    cfg["BKL"] = _fit_bk(cfg["BKL"], fc2)
    _i8_exit_kernel[(H // cfg["BN"], triton.cdiv(M, cfg["BM"]))](
        part,
        *fc2.args(),
        s,
        *shared_down.args(),
        out,
        M,
        routed_scale,
        D2,
        H=H,
        D2=D2,
        S=S,
        K=top_k,
        FMTF=fc2.fmt,
        GF=fc2.group,
        FMTD=shared_down.fmt,
        GD=shared_down.group,
        SUM_K=True,
        **cfg,
        **pdl_kwargs(),
    )
    return out


def olala_moe_medium_int8(
    hidden: torch.Tensor,  # [M, H] bf16
    x_in: torch.Tensor,  # [M, >= D1] latent (bf16 or fp32), any row stride
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w13: WqWeight,
    w2: WqWeight,
    shared_up: WqWeight,
    shared_down: WqWeight,
    fc2: WqWeight,
    top_k: int,
    routed_scale: float,
    configs=None,
    router_logits_lo: torch.Tensor | None = None,
) -> torch.Tensor:
    """Medium-batch latent MoE: routing, compaction, grouped up GEMM (+ shared
    up), grouped down GEMM, exit GEMM (top-k sum + fc2_latent + shared down)."""
    from vllm.model_executor.layers.mamba.olala.latent_moe_decode import (
        _latent_moe_epilogue_kernel,
    )
    from vllm.model_executor.layers.mamba.olala.latent_moe_medium import (
        _compact_kernel,
        _route_topk_kernel,
    )

    M, H = hidden.shape
    E, N1, D1 = w13.weight.shape[0], w13.n, w13.k
    D2 = w2.n
    S = shared_up.n
    up_cfg, down_cfg, exit_cfg, sum_k = configs or _pick_medium(M, w13.fmt)
    up_cfg, down_cfg, exit_cfg = dict(up_cfg), dict(down_cfg), dict(exit_cfg)
    up_cfg["BLOCK_SIZE_K"] = _fit_bk(up_cfg["BLOCK_SIZE_K"], w13)
    up_cfg["BKS"] = _fit_bk(up_cfg["BKS"], shared_up)
    down_cfg["BLOCK_SIZE_K"] = _fit_bk(down_cfg["BLOCK_SIZE_K"], w2)
    exit_cfg["BK"] = _fit_bk(exit_cfg["BK"], shared_down)
    exit_cfg["BKL"] = _fit_bk(exit_cfg["BKL"], fc2)
    bn_sh = up_cfg.get("BN_SH", 0) or up_cfg["BLOCK_SIZE_N"]
    dev = hidden.device
    MK = M * top_k
    ids = torch.empty(MK, dtype=torch.int32, device=dev)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    rows = torch.empty(E, M, dtype=torch.int32, device=dev)
    cnt = torch.empty(E, dtype=torch.int32, device=dev)
    RB = 8
    # Two rows per CTA shorten the 256-expert split-logit route on RTX 5090.
    if (
        router_logits_lo is not None
        and E == 256
        and top_k == 6
        and gpu_family(dev) == GpuFamily.SM120
    ):
        RB = 2
    _route_topk_kernel[(triton.cdiv(M, RB),)](
        router_logits,
        e_score_bias,
        ids,
        tw,
        M,
        router_logits.stride(0),
        E=E,
        K=top_k,
        KP=triton.next_power_of_2(top_k),
        RB=RB,
        num_warps=4,
        logits_lo_ptr=router_logits_lo,
        stride_lo=router_logits_lo.stride(0) if router_logits_lo is not None else 0,
        **pdl_kwargs(),
    )
    _compact_kernel[(E,)](
        ids,
        rows,
        cnt,
        MK,
        M,
        QP=max(triton.next_power_of_2(MK), 16),
        num_warps=4,
        **pdl_kwargs(),
    )
    bm, bn = up_cfg["BLOCK_SIZE_M"], up_cfg["BLOCK_SIZE_N"]
    nbm = E + triton.cdiv(MK, bm)
    h = torch.empty(MK, N1, dtype=torch.bfloat16, device=dev)
    s = torch.empty(M, S, dtype=torch.bfloat16, device=dev)
    _i8_moe_gemm_kernel[(nbm * (N1 // bn) + triton.cdiv(M, bm) * (S // bn_sh),)](
        x_in,
        *w13.args(),
        h,
        tw,
        rows,
        cnt,
        hidden,
        *shared_up.args(),
        s,
        M,
        M,
        x_in.stride(0),
        hidden.stride(0),
        E=E,
        N=N1,
        KD=D1,
        H=H,
        S=S,
        TOPK=top_k,
        UP=True,
        SHARED=True,
        FMTB=w13.fmt,
        GB=w13.group,
        FMTS=shared_up.fmt,
        GS=shared_up.group,
        **up_cfg,
        **pdl_kwargs(),
    )
    part = torch.empty(MK, D2, dtype=torch.float32, device=dev)
    bm, bn = down_cfg["BLOCK_SIZE_M"], down_cfg["BLOCK_SIZE_N"]
    nbm = E + triton.cdiv(MK, bm)
    _i8_moe_gemm_kernel[(nbm * (D2 // bn),)](
        h,
        *w2.args(),
        part,
        tw,
        rows,
        cnt,
        h,
        *w2.args(),
        part,
        M,
        M,
        N1,
        N1,
        E=E,
        N=D2,
        KD=N1,
        H=H,
        S=S,
        TOPK=top_k,
        UP=False,
        SHARED=False,
        FMTB=w2.fmt,
        GB=w2.group,
        FMTS=w2.fmt,
        GS=w2.group,
        **down_cfg,
        **pdl_kwargs(),
    )
    if sum_k:
        routed, stride_r = part, D2
    else:
        routed = torch.empty(M, D2, dtype=torch.bfloat16, device=dev)
        _latent_moe_epilogue_kernel[(M, D2 // 128)](
            part,
            part,
            routed,
            routed_scale,
            0,
            D2=D2,
            S=0,
            BS=128,
            K=top_k,
            num_warps=1,
            **pdl_kwargs(),
        )
        stride_r = D2
    out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    _i8_exit_kernel[(H // exit_cfg["BN"], triton.cdiv(M, exit_cfg["BM"]))](
        routed,
        *fc2.args(),
        s,
        *shared_down.args(),
        out,
        M,
        routed_scale,
        stride_r,
        H=H,
        D2=D2,
        S=S,
        K=top_k,
        FMTF=fc2.fmt,
        GF=fc2.group,
        FMTD=shared_down.fmt,
        GD=shared_down.group,
        SUM_K=sum_k,
        **exit_cfg,
        **pdl_kwargs(),
    )
    return out


# -- prefill (any number of tokens) -------------------------------------------


@triton.jit(do_not_specialize=["num_valid", "EM"])
def _wq_moe_sorted_kernel(
    a_ptr,  # up: latent [M, KD] (row = entry // TOPK); down: h [M*TOPK, KD]
    b_ptr,  # [E, N, KD]
    bs_ptr,  # its scales
    c_ptr,  # up: BF16 relu^2; down: FP32 weighted expert outputs
    tw_ptr,  # [M*TOPK] fp32 routing weights (down)
    sorted_ptr,  # [EM] int32 entries sorted by expert (padding: >= num_valid)
    eid_ptr,  # [EM / BLOCK_SIZE_M] int32 expert of each row block
    ntp_ptr,  # [1] int32 rows after per-expert padding
    num_valid,
    EM,
    stride_am,
    N: tl.constexpr,
    KD: tl.constexpr,
    TOPK: tl.constexpr,
    UP: tl.constexpr,
    FMTB: tl.constexpr,
    GB: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Grouped W8A16 GEMM over moe_align_block_size's sorted rows (fused_moe's
    layout, the weight tiles through _wq_tile_t): prefill-size batches."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n: tl.constexpr = N // BLOCK_SIZE_N
    num_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_SIZE_M
    group_m = min(num_pid_m - first_m, GROUP_SIZE_M)
    pid_m = first_m + (pid % num_in_group) % group_m
    pid_n = (pid % num_in_group) // group_m
    if pid_m * BLOCK_SIZE_M >= tl.load(ntp_ptr):
        return
    offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    entry = tl.load(sorted_ptr + offs).to(tl.int64)
    rmask = entry < num_valid
    e = tl.load(eid_ptr + pid_m).to(tl.int64)
    arow = tl.where(rmask, entry // TOPK if UP else entry, 0)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    b, bs = _wq_expert(b_ptr, bs_ptr, e, N, KD, FMTB, GB)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc = _wq_mm(
        acc,
        a_ptr,
        arow,
        rmask,
        stride_am,
        b,
        bs,
        offs_n,
        KD,
        FMTB,
        GB,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
    )
    acc = _wq_row_scale(acc, bs, offs_n, FMTB)
    c_ptrs = c_ptr + entry[:, None] * N + offs_n[None, :]
    if UP:
        tl.store(c_ptrs, _relu2_bf16(acc), mask=rmask[:, None])
    else:
        w = tl.load(tw_ptr + entry, mask=rmask, other=0.0)
        tl.store(c_ptrs, acc * w[:, None], mask=rmask[:, None])


@triton.jit(do_not_specialize=["M"])
def _topk_sum_kernel(
    part_ptr,
    out_ptr,
    scale,
    M,
    D: tl.constexpr,
    K: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    """``out[m] = bf16(scale * sum_k part[m * K + k])`` for BR rows x BD columns per
    program."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    offs = tl.program_id(1) * BD + tl.arange(0, BD)
    rmask = rows < M
    acc = tl.zeros([BR, BD], dtype=tl.float32)
    for k in tl.static_range(K):
        acc += tl.load(
            part_ptr + (rows * K + k)[:, None].to(tl.int64) * D + offs[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
    tl.store(
        out_ptr + rows[:, None].to(tl.int64) * D + offs[None, :],
        (acc * scale).to(tl.bfloat16),
        mask=rmask[:, None],
    )


_PREFILL_UP = dict(
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8,
    num_warps=4,
    num_stages=3,
)
_PREFILL_DOWN = dict(
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8,
    num_warps=4,
    num_stages=3,
)
# INT4 affine experts, up and down: 102 / 417 us per layer for 1024 / 8192 tokens
# (INT8 99 / 397; RTX 5090).
_PREFILL_I4 = dict(_PREFILL_UP, BLOCK_SIZE_K=128, num_stages=2)


def wq_routed_experts(
    x: torch.Tensor,  # [M, D1] bf16 latent
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w13: WqWeight,  # [E, N1, D1]
    w2: WqWeight,  # [E, D2, N1]
    top_k: int,
    routed_scale: float,
    configs=None,
) -> torch.Tensor:
    """Routed experts for any number of tokens (prefill): sigmoid top-k routing
    (latent_moe_medium._route_topk_kernel), moe_align_block_size, grouped up
    GEMM + relu^2, grouped down GEMM weighted by the routing weights, top-k sum
    times the routed scale. Returns the routed latent ``[M, D2]`` bf16."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.mamba.olala.latent_moe_medium import (
        _route_topk_kernel,
    )

    M, D1 = x.shape
    E, N1 = w13.weight.shape[0], w13.n
    D2 = w2.n
    dev = x.device
    MK = M * top_k
    up_cfg, down_cfg = configs or (
        (_PREFILL_I4, _PREFILL_I4)
        if w13.fmt == FMT_INT4
        else (_PREFILL_UP, _PREFILL_DOWN)
    )
    up_cfg, down_cfg = dict(up_cfg), dict(down_cfg)
    up_cfg["BLOCK_SIZE_K"] = _fit_bk(up_cfg["BLOCK_SIZE_K"], w13)
    down_cfg["BLOCK_SIZE_K"] = _fit_bk(down_cfg["BLOCK_SIZE_K"], w2)
    ids = torch.empty(M, top_k, dtype=torch.int32, device=dev)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    RB = 8
    _route_topk_kernel[(triton.cdiv(M, RB),)](
        router_logits,
        e_score_bias,
        ids,
        tw,
        M,
        router_logits.stride(0),
        E=E,
        K=top_k,
        KP=triton.next_power_of_2(top_k),
        RB=RB,
        num_warps=4,
    )
    bm = up_cfg["BLOCK_SIZE_M"]
    assert down_cfg["BLOCK_SIZE_M"] == bm
    sorted_ids, expert_ids, ntp = moe_align_block_size(ids, bm, E)
    EM = sorted_ids.numel()
    h = torch.empty(MK, N1, dtype=torch.bfloat16, device=dev)
    _wq_moe_sorted_kernel[(triton.cdiv(EM, bm) * (N1 // up_cfg["BLOCK_SIZE_N"]),)](
        x,
        *w13.args(),
        h,
        tw,
        sorted_ids,
        expert_ids,
        ntp,
        MK,
        EM,
        x.stride(0),
        N=N1,
        KD=D1,
        TOPK=top_k,
        UP=True,
        FMTB=w13.fmt,
        GB=w13.group,
        **up_cfg,
    )
    part = torch.empty(MK, D2, dtype=torch.float32, device=dev)
    _wq_moe_sorted_kernel[(triton.cdiv(EM, bm) * (D2 // down_cfg["BLOCK_SIZE_N"]),)](
        h,
        *w2.args(),
        part,
        tw,
        sorted_ids,
        expert_ids,
        ntp,
        MK,
        EM,
        N1,
        N=D2,
        KD=N1,
        TOPK=top_k,
        UP=False,
        FMTB=w2.fmt,
        GB=w2.group,
        **down_cfg,
    )
    out = torch.empty(M, D2, dtype=torch.bfloat16, device=dev)
    BR = 8
    _topk_sum_kernel[(triton.cdiv(M, BR), D2 // 128)](
        part, out, routed_scale, M, D=D2, K=top_k, BR=BR, BD=128, num_warps=4
    )
    return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode preamble / epilogue for Olala's Differential-TPA layers.

For a pure-decode batch the eager path between the input projections and the
attention call is ~20 small kernels per layer: rank-``R`` K/V reconstruction
(bmm + div), the token-shift blend against the one-token pool, q/k RMSNorm,
and the scalable-softmax scaling of q. One program per (token, kv-head) does
all of it — the kv head's K/V rows and its ``snr + 1`` query heads — keeping
the eager bf16 rounding points. A second kernel folds the differential
recombination ``sig - sigmoid(lam) * noise`` after attention.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tpa_decode_qkv_kernel(
    q_ptr, sq_n, sq_h,                 # q_all (N, Hq*D) view: row stride, head stride
    ak_ptr, av_ptr, sa_n,              # A_k / A_v (N, Hkv*R): row stride
    bk_ptr, bv_ptr, sb_n,              # B_k / B_v (N, R*D): row stride
    alk_ptr, alv_ptr, sal_n,           # shift logits (N, Hkv): row stride
    pos_ptr, slots_ptr,
    kpool_ptr, vpool_ptr, sp_n, sp_h, P,
    qw_ptr, kw_ptr, eps,               # norm gains (D,), zero-centered
    scaler_ptr, wsize,                 # softmax scaler (Hq,), window clamp (0 = none)
    qo_ptr, ko_ptr, vo_ptr,            # outputs (N, Hq*D) / (N, Hkv*D) contiguous
    HKV: tl.constexpr,
    GQ: tl.constexpr,                  # query heads per kv head (= snr + 1)
    R: tl.constexpr,
    D: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)

    # ---- rank-R K/V reconstruction: bf16(sum_r A[r] * B[r]) / R ----------
    k = tl.zeros([D], dtype=tl.float32)
    v = tl.zeros([D], dtype=tl.float32)
    for r in range(R):
        ak_r = tl.load(ak_ptr + n * sa_n + h * R + r).to(tl.float32)
        av_r = tl.load(av_ptr + n * sa_n + h * R + r).to(tl.float32)
        bk_r = tl.load(bk_ptr + n * sb_n + r * D + offs).to(tl.float32)
        bv_r = tl.load(bv_ptr + n * sb_n + r * D + offs).to(tl.float32)
        k += ak_r * bk_r
        v += av_r * bv_r
    k = (k.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)
    v = (v.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)

    # ---- token shift against the one-token pool ---------------------------
    slot = tl.load(slots_ptr + n)
    srow = tl.minimum(tl.maximum(slot, 0), P - 1).to(tl.int64)
    kp = tl.load(kpool_ptr + srow * sp_n + h * sp_h + offs).to(tl.float32)
    vp = tl.load(vpool_ptr + srow * sp_n + h * sp_h + offs).to(tl.float32)
    a_k = tl.sigmoid(tl.load(alk_ptr + n * sal_n + h).to(tl.float32))
    a_v = tl.sigmoid(tl.load(alv_ptr + n * sal_n + h).to(tl.float32))
    pos = tl.load(pos_ptr + n)
    a_k = tl.where(pos == 0, 0.0, a_k)
    a_v = tl.where(pos == 0, 0.0, a_v)
    ks = (a_k * kp + (1.0 - a_k) * k).to(tl.bfloat16).to(tl.float32)
    vs = (a_v * vp + (1.0 - a_v) * v).to(tl.bfloat16)
    tl.store(kpool_ptr + srow * sp_n + h * sp_h + offs, k.to(kpool_ptr.dtype.element_ty))
    tl.store(vpool_ptr + srow * sp_n + h * sp_h + offs, v.to(vpool_ptr.dtype.element_ty))
    tl.store(vo_ptr + n * (HKV * D) + h * D + offs, vs)

    # ---- k norm (zero-centered gain), same rounding as OlalaRMSNorm -------
    kw = (1.0 + tl.load(kw_ptr + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(ks * ks, axis=0) / D + eps)
    kn = (ks * inv).to(tl.bfloat16).to(tl.float32)
    tl.store(ko_ptr + n * (HKV * D) + h * D + offs, (kn * kw).to(tl.bfloat16))

    # ---- q heads of this kv group: norm + scalable-softmax scale -----------
    qw = (1.0 + tl.load(qw_ptr + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    p1 = tl.maximum(pos.to(tl.float32) + 1.0, 1.0)
    if wsize > 0:
        p1 = tl.minimum(p1, wsize)
    log_pos = tl.log(p1).to(tl.bfloat16).to(tl.float32)
    for j in range(GQ):
        hq = h * GQ + j
        q = tl.load(q_ptr + n * sq_n + hq * sq_h + offs).to(tl.float32)
        inv_q = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / D + eps)
        qn = (q * inv_q).to(tl.bfloat16).to(tl.float32)
        qn = (qn * qw).to(tl.bfloat16).to(tl.float32)
        sc = tl.load(scaler_ptr + hq).to(tl.bfloat16).to(tl.float32)
        scale = (sc * log_pos).to(tl.bfloat16).to(tl.float32)
        tl.store(qo_ptr + n * (HKV * GQ * D) + hq * D + offs, (qn * scale).to(tl.bfloat16))


@triton.jit
def _tpa_qkv_row(
    n, h, k, v, kprev, vprev, use_prev,
    q_ptr, sq_n, sq_h, alk_ptr, alv_ptr, sal_n, pos_ptr,
    qw_ptr, kw_ptr, eps, scaler_ptr, wsize, qo_ptr, ko_ptr, vo_ptr,
    HKV: tl.constexpr, GQ: tl.constexpr, D: tl.constexpr,
):
    """Shift blend, k norm and the GQ query heads of one (token, kv head), as
    _tpa_decode_qkv_kernel."""
    offs = tl.arange(0, D)
    a_k = tl.sigmoid(tl.load(alk_ptr + n * sal_n + h).to(tl.float32))
    a_v = tl.sigmoid(tl.load(alv_ptr + n * sal_n + h).to(tl.float32))
    pos = tl.load(pos_ptr + n)
    a_k = tl.where((pos == 0) | (use_prev == 0), 0.0, a_k)
    a_v = tl.where((pos == 0) | (use_prev == 0), 0.0, a_v)
    ks = (a_k * kprev + (1.0 - a_k) * k).to(tl.bfloat16).to(tl.float32)
    vs = (a_v * vprev + (1.0 - a_v) * v).to(tl.bfloat16)
    tl.store(vo_ptr + n * (HKV * D) + h * D + offs, vs)
    kw = (1.0 + tl.load(kw_ptr + offs).to(tl.float32)).to(tl.bfloat16)
    kw = kw.to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(ks * ks, axis=0) / D + eps)
    kn = (ks * inv).to(tl.bfloat16).to(tl.float32)
    tl.store(ko_ptr + n * (HKV * D) + h * D + offs, (kn * kw).to(tl.bfloat16))
    qw = (1.0 + tl.load(qw_ptr + offs).to(tl.float32)).to(tl.bfloat16)
    qw = qw.to(tl.float32)
    p1 = tl.maximum(pos.to(tl.float32) + 1.0, 1.0)
    if wsize > 0:
        p1 = tl.minimum(p1, wsize)
    log_pos = tl.log(p1).to(tl.bfloat16).to(tl.float32)
    for j in range(GQ):
        hq = h * GQ + j
        q = tl.load(q_ptr + n * sq_n + hq * sq_h + offs).to(tl.float32)
        inv_q = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / D + eps)
        qn = (q * inv_q).to(tl.bfloat16).to(tl.float32)
        qn = (qn * qw).to(tl.bfloat16).to(tl.float32)
        sc = tl.load(scaler_ptr + hq).to(tl.bfloat16).to(tl.float32)
        scale = (sc * log_pos).to(tl.bfloat16).to(tl.float32)
        tl.store(
            qo_ptr + n * (HKV * GQ * D) + hq * D + offs,
            (qn * scale).to(tl.bfloat16),
        )


@triton.jit
def _tpa_raw_kv(
    n, h, ak_ptr, av_ptr, sa_n, bk_ptr, bv_ptr, sb_n,
    R: tl.constexpr, D: tl.constexpr,
):
    offs = tl.arange(0, D)
    k = tl.zeros([D], dtype=tl.float32)
    v = tl.zeros([D], dtype=tl.float32)
    for r in range(R):
        ak_r = tl.load(ak_ptr + n * sa_n + h * R + r).to(tl.float32)
        av_r = tl.load(av_ptr + n * sa_n + h * R + r).to(tl.float32)
        k += ak_r * tl.load(bk_ptr + n * sb_n + r * D + offs).to(tl.float32)
        v += av_r * tl.load(bv_ptr + n * sb_n + r * D + offs).to(tl.float32)
    k = (k.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)
    v = (v.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)
    return k, v


@triton.jit
def _tpa_decode_qkv_spec_kernel(
    q_ptr, sq_n, sq_h,
    ak_ptr, av_ptr, sa_n,
    bk_ptr, bv_ptr, sb_n,
    alk_ptr, alv_ptr, sal_n,
    pos_ptr,
    qsl_ptr, cols_ptr, scol, ncols, acc_ptr, lanes, N,
    kpool_ptr, vpool_ptr, sp_n, sp_h, sp_c, P,
    qw_ptr, kw_ptr, eps,
    scaler_ptr, wsize,
    qo_ptr, ko_ptr, vo_ptr,
    HKV: tl.constexpr,
    GQ: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    LIGHT: tl.constexpr,
):
    """Spec-decode verify rows (column-slot protocol, see olala_spec_utils):
    one program per (request, kv head) walks the request's positions in order.
    Position 0 shifts against the pool column of the accepted state, position
    t > 0 against the raw K/V of position t - 1; position t's raw K/V goes to
    column t. Program ``lanes`` passes the CUDA-graph padding rows past the
    last request through unshifted, without touching the pool."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)
    if r < lanes:
        start = tl.load(qsl_ptr + r)
        qlen = tl.minimum(tl.load(qsl_ptr + r + 1) - start, ncols)
        c0 = tl.maximum(tl.load(acc_ptr + r) - 1, 0)
        if LIGHT:
            # light columns (state.py): one slot, column c0 inside its page
            c0 = tl.minimum(c0, ncols - 1)
            init = tl.load(cols_ptr + r * scol)
            irow = tl.minimum(tl.maximum(init, 0), P - 1).to(tl.int64)
            ioff = irow * sp_n + c0.to(tl.int64) * sp_c
        else:
            init = tl.load(cols_ptr + r * scol + c0)
            irow = tl.minimum(tl.maximum(init, 0), P - 1).to(tl.int64)
            ioff = irow * sp_n
        kprev = tl.load(kpool_ptr + ioff + h * sp_h + offs).to(tl.float32)
        vprev = tl.load(vpool_ptr + ioff + h * sp_h + offs).to(tl.float32)
        for t in range(qlen):
            n = start + t
            k, v = _tpa_raw_kv(
                n, h, ak_ptr, av_ptr, sa_n, bk_ptr, bv_ptr, sb_n, R, D
            )
            _tpa_qkv_row(
                n, h, k, v, kprev, vprev, 1,
                q_ptr, sq_n, sq_h, alk_ptr, alv_ptr, sal_n, pos_ptr,
                qw_ptr, kw_ptr, eps, scaler_ptr, wsize, qo_ptr, ko_ptr, vo_ptr,
                HKV, GQ, D,
            )
            if LIGHT:
                doff = irow * sp_n + t * sp_c
            else:
                dst = tl.load(cols_ptr + r * scol + t)
                doff = tl.minimum(tl.maximum(dst, 0), P - 1).to(tl.int64) * sp_n
            dpos = doff + h * sp_h + offs
            tl.store(kpool_ptr + dpos, k.to(kpool_ptr.dtype.element_ty))
            tl.store(vpool_ptr + dpos, v.to(vpool_ptr.dtype.element_ty))
            kprev = k
            vprev = v
    else:
        for n in range(tl.load(qsl_ptr + lanes), N):
            k, v = _tpa_raw_kv(
                n, h, ak_ptr, av_ptr, sa_n, bk_ptr, bv_ptr, sb_n, R, D
            )
            _tpa_qkv_row(
                n, h, k, v, k, v, 0,
                q_ptr, sq_n, sq_h, alk_ptr, alv_ptr, sal_n, pos_ptr,
                qw_ptr, kw_ptr, eps, scaler_ptr, wsize, qo_ptr, ko_ptr, vo_ptr,
                HKV, GQ, D,
            )


@triton.jit
def _tpa_diff_combine_kernel(
    attn_ptr,          # (N, Hkv*(SNR+1)*D) contiguous
    lam_ptr, sl_n,     # lambda logits (N, Hkv): row stride
    out_ptr, so_n,     # (N, Hkv*SNR*D): row stride
    HKV: tl.constexpr,
    HP: tl.constexpr,
    SNR: tl.constexpr,
    D: tl.constexpr,
):
    """One token per program, all kv heads as one [HP, D] tile per signal head."""
    n = tl.program_id(0).to(tl.int64)
    hh = tl.arange(0, HP)
    hm = hh[:, None] < HKV
    offs = tl.arange(0, D)
    base = attn_ptr + n * HKV * (SNR + 1) * D + hh[:, None] * (SNR + 1) * D + offs[None, :]
    noi = tl.load(base + SNR * D, mask=hm, other=0.0).to(tl.float32)
    lam = tl.load(lam_ptr + n * sl_n + hh, mask=hh < HKV, other=0.0).to(tl.float32)
    s = tl.sigmoid(lam).to(tl.bfloat16).to(tl.float32)
    t = (s[:, None] * noi).to(tl.bfloat16).to(tl.float32)
    obase = out_ptr + n * so_n + hh[:, None] * SNR * D + offs[None, :]
    for j in tl.static_range(SNR):
        sig = tl.load(base + j * D, mask=hm, other=0.0).to(tl.float32)
        tl.store(obase + j * D, (sig - t).to(out_ptr.dtype.element_ty), mask=hm)


def tpa_decode_qkv(q_all, A_k, A_v, B_k, B_v, alpha_k, alpha_v, positions, slots,
                   k_pool, v_pool, q_w, k_w, eps, scaler, wsize, hkv, gq, rank, d):
    n = q_all.shape[0]
    dt = q_all.dtype
    q_out = torch.empty(n, hkv * gq * d, dtype=dt, device=q_all.device)
    k_out = torch.empty(n, hkv * d, dtype=dt, device=q_all.device)
    v_out = torch.empty(n, hkv * d, dtype=dt, device=q_all.device)
    _tpa_decode_qkv_kernel[(n, hkv)](
        q_all, q_all.stride(0), d,
        A_k, A_v, A_k.stride(0),
        B_k, B_v, B_k.stride(0),
        alpha_k, alpha_v, alpha_k.stride(0),
        positions, slots,
        k_pool, v_pool, k_pool.stride(0), k_pool.stride(1), k_pool.shape[0],
        q_w, k_w, eps, scaler, float(wsize),
        q_out, k_out, v_out,
        HKV=hkv, GQ=gq, R=rank, D=d, num_warps=2,
    )
    return q_out, k_out, v_out


def tpa_decode_qkv_spec(q_all, A_k, A_v, B_k, B_v, alpha_k, alpha_v, positions,
                        qsl, cols, num_accepted, k_pool, v_pool, q_w, k_w, eps,
                        scaler, wsize, hkv, gq, rank, d):
    """tpa_decode_qkv for a verify batch (decode rows only, ragged per request):
    ``qsl`` (lanes + 1,) decode-token offsets, ``cols`` (lanes, 1 + k) pool
    columns, ``num_accepted`` (lanes,) previous acceptance. With light columns
    (4-D pools (slots, 1 + k, kv, D)) ``cols`` holds the slot in column 0."""
    n = q_all.shape[0]
    dt = q_all.dtype
    q_out = torch.empty(n, hkv * gq * d, dtype=dt, device=q_all.device)
    k_out = torch.empty(n, hkv * d, dtype=dt, device=q_all.device)
    v_out = torch.empty(n, hkv * d, dtype=dt, device=q_all.device)
    lanes = cols.shape[0]
    # light columns (state.py): pools (slots, 1 + k, kv, D), cols holds the slot in column 0
    light = k_pool.dim() == 4
    ncols = k_pool.shape[1] if light else cols.shape[1]
    sp_h = k_pool.stride(2) if light else k_pool.stride(1)
    sp_c = k_pool.stride(1) if light else 0
    assert v_pool.stride() == k_pool.stride()
    _tpa_decode_qkv_spec_kernel[(lanes + 1, hkv)](
        q_all, q_all.stride(0), d,
        A_k, A_v, A_k.stride(0),
        B_k, B_v, B_k.stride(0),
        alpha_k, alpha_v, alpha_k.stride(0),
        positions,
        qsl, cols, cols.stride(0), ncols, num_accepted, lanes, n,
        k_pool, v_pool, k_pool.stride(0), sp_h, sp_c, k_pool.shape[0],
        q_w, k_w, eps, scaler, float(wsize),
        q_out, k_out, v_out,
        HKV=hkv, GQ=gq, R=rank, D=d, LIGHT=light, num_warps=2,
    )
    return q_out, k_out, v_out


def tpa_diff_combine(attn_out, lam_all, out, hkv, snr, d):
    n = attn_out.shape[0]
    if n == 0:
        return
    _tpa_diff_combine_kernel[(n,)](
        attn_out, lam_all, lam_all.stride(0), out, out.stride(0),
        HKV=hkv, HP=triton.next_power_of_2(hkv), SNR=snr, D=d, num_warps=8,
    )

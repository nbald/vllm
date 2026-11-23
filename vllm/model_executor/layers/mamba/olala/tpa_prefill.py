# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused prefill glue of the TPA-factorized DiffTPA layer.

Replaces the per-token torch chain of ``OlalaDiffTPAAttention._factor_prefill`` (k/v rank
contractions, previous-token seeds from the batch or the shift pools, shift gates, k RMS
norm, factor rows, row scatter, pool updates) and of ``_q_for_attention`` (q RMS norm and
scalable-softmax scale) by three kernels, with the same bf16 roundings as the torch code.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from vllm.model_executor.layers.mamba.olala import fa4_softcap
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gencode, gpu_family, sm_arch
from vllm.triton_utils import tl, triton


@triton.jit
def _tpa_rows_kernel(
    Ak_ptr, Av_ptr, Bk_ptr, Bv_ptr, ak_ptr, av_ptr, rs,
    pos_ptr, slots_ptr, starts_ptr, ends_ptr, pslots_ptr, init_ptr,
    kpool_ptr, vpool_ptr, akpool_ptr, avpool_ptr, s_kp, s_vp, s_akp, s_avp,
    kw_ptr, eps,
    kplane_ptr, vplane_ptr, bstride, block_size,
    H: tl.constexpr, HP: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
):
    """(request, token): K and V factor rows [B' | C | D] of the token, written to the paged
    planes at its slot (slots < 0 skipped). The C / D coefficients go out as contiguous [H, R]
    tiles."""
    req = tl.program_id(0)
    t0 = tl.load(starts_ptr + req)
    t_end = tl.load(ends_ptr + req)
    t = t0 + tl.program_id(1)
    if t >= t_end:
        return
    inb = t > t0
    pslot = tl.load(pslots_ptr + req).to(tl.int64)
    hinit = tl.load(init_ptr + req) != 0
    seed = (t == t0) & hinit
    hh = tl.arange(0, HP)
    hm = hh < H
    d = tl.arange(0, D)
    rr = tl.arange(0, R)
    t64 = t.to(tl.int64)
    p64 = t64 - 1
    keep = tl.load(pos_ptr + t) != 0
    a_k = tl.where(keep, tl.sigmoid(tl.load(ak_ptr + t64 * rs + hh, mask=hm, other=0.0).to(tl.float32)), 0.0)
    a_v = tl.where(keep, tl.sigmoid(tl.load(av_ptr + t64 * rs + hh, mask=hm, other=0.0).to(tl.float32)), 0.0)
    k = tl.zeros([HP, D], dtype=tl.float32)
    kprev = tl.zeros([HP, D], dtype=tl.float32)
    for r in tl.static_range(R):
        a_r = tl.load(Ak_ptr + t64 * rs + hh * R + r, mask=hm, other=0.0).to(tl.float32)
        b_r = tl.load(Bk_ptr + t64 * rs + r * D + d).to(tl.float32)
        k += a_r[:, None] * b_r[None, :]
        ap_r = tl.load(Ak_ptr + p64 * rs + hh * R + r, mask=hm & inb, other=0.0).to(tl.float32)
        bp_r = tl.load(Bk_ptr + p64 * rs + r * D + d, mask=inb & (d < D), other=0.0).to(tl.float32)
        kprev += ap_r[:, None] * bp_r[None, :]
    k = k.to(tl.bfloat16).to(tl.float32) / R
    kprev = kprev.to(tl.bfloat16).to(tl.float32) / R
    kseed = tl.load(kpool_ptr + pslot * s_kp + hh[:, None] * D + d[None, :], mask=hm[:, None] & hinit, other=0.0).to(tl.bfloat16).to(tl.float32)
    kprev = tl.where(seed, kseed, kprev)
    a2 = a_k[:, None]
    ks = (a2 * kprev + (1.0 - a2) * k).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(ks * ks, axis=1) / D + eps)  # [HP]
    slot = tl.load(slots_ptr + t).to(tl.int64)
    if slot < 0:
        return
    kw = (tl.load(kw_ptr + d).to(tl.float32) + 1.0).to(tl.bfloat16).to(tl.float32)
    kbase = kplane_ptr + (slot // block_size) * bstride + (slot % block_size) * (R * D + 2 * H * R)
    vbase = vplane_ptr + (slot // block_size) * bstride + (slot % block_size) * (R * D + 2 * H * R)
    # B rows: [R, D] contiguous
    rd = rr[:, None] * D + d[None, :]
    bk = tl.load(Bk_ptr + t64 * rs + rd).to(tl.float32)
    tl.store(kbase + rd, (bk * kw[None, :]).to(kplane_ptr.dtype.element_ty))
    tl.store(vbase + rd, tl.load(Bv_ptr + t64 * rs + rd).to(vplane_ptr.dtype.element_ty))
    # A coefficients: [H, R] tiles, contiguous h * R + r
    hr = hh[:, None] * R + rr[None, :]
    m2 = hm[:, None]
    A = tl.load(Ak_ptr + t64 * rs + hr, mask=m2, other=0.0).to(tl.float32)
    Ap = tl.load(Ak_ptr + p64 * rs + hr, mask=m2 & inb, other=0.0).to(tl.float32)
    Ap = tl.where(seed, tl.load(akpool_ptr + pslot * s_akp + hr, mask=m2 & hinit, other=0.0).to(tl.bfloat16).to(tl.float32), Ap)
    scale = ((1.0 - a_k) / R * inv)[:, None]
    dscale = (a_k / R * inv)[:, None]
    tl.store(kbase + R * D + hr, (A * scale).to(kplane_ptr.dtype.element_ty), mask=m2)
    tl.store(kbase + R * D + H * R + hr, (Ap * dscale).to(kplane_ptr.dtype.element_ty), mask=m2)
    V = tl.load(Av_ptr + t64 * rs + hr, mask=m2, other=0.0).to(tl.float32)
    Vp = tl.load(Av_ptr + p64 * rs + hr, mask=m2 & inb, other=0.0).to(tl.float32)
    Vp = tl.where(seed, tl.load(avpool_ptr + pslot * s_avp + hr, mask=m2 & hinit, other=0.0).to(tl.bfloat16).to(tl.float32), Vp)
    tl.store(vbase + R * D + hr, (V * ((1.0 - a_v) / R)[:, None]).to(vplane_ptr.dtype.element_ty), mask=m2)
    tl.store(vbase + R * D + H * R + hr, (Vp * (a_v / R)[:, None]).to(vplane_ptr.dtype.element_ty), mask=m2)


@triton.jit
def _tpa_pools_kernel(
    Ak_ptr, Av_ptr, Bk_ptr, Bv_ptr, rs, ends_ptr, pslots_ptr,
    kpool_ptr, vpool_ptr, akpool_ptr, avpool_ptr, s_kp, s_vp, s_akp, s_avp,
    H: tl.constexpr, HP: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
):
    """Shift pools of one request <- its last token's k, v, A_k, A_v."""
    req = tl.program_id(0)
    l = (tl.load(ends_ptr + req) - 1).to(tl.int64)
    pslot = tl.load(pslots_ptr + req).to(tl.int64)
    hh = tl.arange(0, HP)
    hm = hh < H
    d = tl.arange(0, D)
    k = tl.zeros([HP, D], dtype=tl.float32)
    v = tl.zeros([HP, D], dtype=tl.float32)
    for r in tl.static_range(R):
        ak = tl.load(Ak_ptr + l * rs + hh * R + r, mask=hm, other=0.0)
        av = tl.load(Av_ptr + l * rs + hh * R + r, mask=hm, other=0.0)
        k += ak.to(tl.float32)[:, None] * tl.load(Bk_ptr + l * rs + r * D + d).to(tl.float32)[None, :]
        v += av.to(tl.float32)[:, None] * tl.load(Bv_ptr + l * rs + r * D + d).to(tl.float32)[None, :]
        tl.store(akpool_ptr + pslot * s_akp + hh * R + r, ak.to(akpool_ptr.dtype.element_ty), mask=hm)
        tl.store(avpool_ptr + pslot * s_avp + hh * R + r, av.to(avpool_ptr.dtype.element_ty), mask=hm)
    k = k.to(tl.bfloat16).to(tl.float32) / R
    v = v.to(tl.bfloat16).to(tl.float32) / R
    tl.store(kpool_ptr + pslot * s_kp + hh[:, None] * D + d[None, :], k.to(kpool_ptr.dtype.element_ty), mask=hm[:, None])
    tl.store(vpool_ptr + pslot * s_vp + hh[:, None] * D + d[None, :], v.to(vpool_ptr.dtype.element_ty), mask=hm[:, None])


@triton.jit
def _tpa_q_kernel(
    q_ptr, rs, pos_ptr, w_ptr, scaler_ptr, out_ptr, eps, wsize, out_scale,
    HQ: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, HAS_WS: tl.constexpr,
    HAS_OUT_SCALE: tl.constexpr,
):
    """q <- (rms_norm(q) * (1 + w)) * (scaler[h] * log(clamp(pos + 1, 1, wsize))), torch roundings.
    HAS_OUT_SCALE: also times ``out_scale`` in fp32 before the final rounding."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    hm = h < HQ
    d = tl.arange(0, D)
    x = tl.load(q_ptr + t * rs + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    y = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    w = (tl.load(w_ptr + d).to(tl.float32) + 1.0).to(tl.bfloat16).to(tl.float32)
    y = (y * w[None, :]).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(pos_ptr + t).to(tl.float32) + 1.0
    pos = tl.maximum(pos, 1.0)
    if HAS_WS:
        pos = tl.minimum(pos, wsize)
    lp = tl.log(pos).to(tl.bfloat16).to(tl.float32)
    sc = (tl.load(scaler_ptr + h, mask=hm, other=0.0).to(tl.bfloat16).to(tl.float32) * lp).to(tl.bfloat16).to(tl.float32)
    o = y * sc[:, None]
    if HAS_OUT_SCALE:
        o = o * out_scale
    tl.store(out_ptr + t * (HQ * D) + h[:, None] * D + d[None, :], o.to(out_ptr.dtype.element_ty), mask=hm[:, None])


def tpa_prefill_rows(A_k, A_v, B_k, B_v, alpha_k, alpha_v, positions, slots, starts, ends, pslots, init,
                     pools, k_norm_w, eps, kplane, vplane, H, R, D, max_len=None):
    """Factor rows of every prefill token into the paged planes, then the shift pools."""
    n = A_k.shape[0]
    nk = starts.shape[0]
    if n == 0 or nk == 0:
        return
    rs = A_k.stride(0)
    for x in (A_v, B_k, B_v, alpha_k, alpha_v):
        assert x.stride(0) == rs and x.stride(-1) == 1
    k_pool, v_pool, ak_pool, av_pool = pools
    assert kplane.stride(1) == R * D + 2 * H * R and kplane.stride(2) == 1 and vplane.stride(0) == kplane.stride(0)
    HP = triton.next_power_of_2(H)
    # programs past a request's end return at once; the host knows the longest request.
    # One token per program and one warp: 2x faster than 2 warps on B300.
    max_len = n if max_len is None else max_len
    _tpa_rows_kernel[(nk, max_len)](
        A_k, A_v, B_k, B_v, alpha_k, alpha_v, rs, positions, slots, starts, ends, pslots, init,
        k_pool, v_pool, ak_pool, av_pool, k_pool.stride(0), v_pool.stride(0), ak_pool.stride(0), av_pool.stride(0),
        k_norm_w, eps, kplane, vplane, kplane.stride(0), kplane.shape[1],
        H=H, HP=HP, R=R, D=D, num_warps=1,
    )
    _tpa_pools_kernel[(nk,)](
        A_k, A_v, B_k, B_v, rs, ends, pslots,
        k_pool, v_pool, ak_pool, av_pool, k_pool.stride(0), v_pool.stride(0), ak_pool.stride(0), av_pool.stride(0),
        H=H, HP=HP, R=R, D=D, num_warps=4,
    )


def tpa_prefill_q(q_all, positions, q_norm_w, scaler, eps, wsize, HQ, D, out_scale=None):
    n = q_all.shape[0]
    out = torch.empty(n, HQ * D, device=q_all.device, dtype=q_all.dtype)
    if n == 0:
        return out.view(n, HQ, D)
    HB = 8
    _tpa_q_kernel[(n, triton.cdiv(HQ, HB))](
        q_all, q_all.stride(0), positions, q_norm_w, scaler, out, eps, float(wsize),
        float(out_scale or 1.0),
        HQ=HQ, HB=HB, D=D, HAS_WS=wsize > 0, HAS_OUT_SCALE=out_scale is not None, num_warps=4,
    )
    return out.view(n, HQ, D)


@lru_cache(maxsize=8)
def _fa4_softcap_mod(softcap: float):
    """FA4 score_mod for ``softcap * tanh(s / softcap)``.

    FA4's own ``softcap=`` mod divides by the cap (an IEEE ``div.rn`` per
    score); multiplying by the reciprocal lowers to the same ``tanh.approx``
    and makes the causal hd128 forward 3.6x faster on B300.
    """
    import cutlass.cute as cute

    inv = 1.0 / softcap

    @cute.jit
    def softcap_mod(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return softcap * cute.math.tanh(acc_S_SSA * inv, fastmath=True)

    return softcap_mod


def prefill_q_prescale(softmax_scale: float, softcap: float) -> float | None:
    """Factor to fold into q for ``tpa_varlen_attention(..., prescaled=True)``, or None.

    The native softcap kernel multiplies every score by ``scale / softcap`` before the tanh;
    with q carrying it (one fp32 multiply before q's bf16 rounding) the kernel skips that
    multiply: ~4% faster causal hd128 prefill attention on B300. ``OLALA_FA4_PRESCALE_Q=0``
    disables it.
    """
    if softcap <= 0.0 or os.environ.get("OLALA_FA4_PRESCALE_Q", "1") == "0":
        return None
    if os.environ.get("OLALA_FA4_SOFTCAP", "native") != "native":
        return None
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    if get_flash_attn_version() != 4 or not fa4_softcap.native_softcap_available():
        return None
    return softmax_scale / softcap


@lru_cache(maxsize=1)
def _volta_attn_ext():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(f"olala_tpa_prefill_attn_volta_sm{sm_arch()}", "tpa_prefill_attn_volta.cu",
                          ["-O3", "-std=c++20", "--use_fast_math", gencode()])


def _volta_attn_ok(q, k, prescaled) -> bool:
    """SM70 family (V100 through volta-forever, which emulates FA2's bf16 mma.sync / ldmatrix / cp.async with
    scalar code): the FMA edition csrc/tpa_prefill_attn_volta.cu, which returns FA2's values (head dim 128, 4 query
    heads per KV head). OLALA_VOLTA_PREFILL=0 keeps FA2."""
    return (not prescaled and os.environ.get("OLALA_VOLTA_PREFILL", "1") == "1"
            and gpu_family() is GpuFamily.SM70 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and q.shape[-1] == 128 and q.shape[1] == 4 * k.shape[1])


def _vec_ok(t: torch.Tensor) -> torch.Tensor:
    # 16-byte rows for the kernel's vector loads and stores
    if t.stride(-1) != 1 or t.stride(0) % 8 or t.stride(1) % 8 or t.data_ptr() % 16:
        return t.contiguous()
    return t


def tpa_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                         softmax_scale, softcap, prescaled=False):
    """Causal varlen attention of the dense prefill K/V.

    With FA4 and a softcap, ``OLALA_FA4_SOFTCAP`` picks: ``native`` (default on sm_10x, see
    fa4_softcap.py), ``mod`` (the fast score_mod above) or ``fa4`` (FA4's own softcap).
    ``prescaled``: q already times ``prefill_q_prescale(...)`` (native path only).
    On the SM70 family: the FMA kernel csrc/tpa_prefill_attn_volta.cu (FA2's values).
    """
    if _volta_attn_ok(q, k, prescaled):
        q, k, v = _vec_ok(q), _vec_ok(k), _vec_ok(v)
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        _volta_attn_ext().attn_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, out, int(max_seqlen_q),
                                   float(softmax_scale), float(softcap))
        return out
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version

    fa_version = get_flash_attn_version()
    mode = os.environ.get("OLALA_FA4_SOFTCAP", "native") if softcap > 0.0 and fa_version == 4 else "fa4"
    if mode == "native":
        if fa4_softcap.native_softcap_available():
            return fa4_softcap.fa4_softcap_varlen_fwd(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                softmax_scale, softcap, prescaled=prescaled,
            )
        mode = "mod"
    assert not prescaled, "prescaled q needs the native softcap path"
    if mode == "mod":
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

        return _flash_attn_fwd(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
            causal=True, score_mod=_fa4_softcap_mod(float(softcap)),
        )[0]
    return flash_attn_varlen_func(
        q=q, k=k, v=v, max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k, softmax_scale=softmax_scale,
        causal=True, softcap=softcap, fa_version=fa_version,
    )

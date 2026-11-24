# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-fused Triton kernels of the DSpine draft step (OLALA_DSPINE_FUSED=3,
qwen3_dspine_fast.py).

The captured draft step is about 180 small kernels, and on GPUs with a high
per-kernel cost (a V100 through volta-forever: 1.6 us of graph launch per
kernel plus a latency-bound body) their count, not their work, sets the draft
time. Each function here replaces a chain of them with one launch. The kernels
use plain loads, reductions and FMAs (no tl.dot), so the same code runs on
SM70, the SM80 family and SM120. They are not bit-identical to the chains they
replace (fp32 math, one rounding at the end); the acceptance gate is offline
tau.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr, stride_qkv, pos_ptr, cs_ptr, stride_cs, qw_ptr, kw_ptr, q_ptr, k_ptr, eps,
    NQ: tl.constexpr, NK: tl.constexpr, HD: tl.constexpr,
):
    """One (row, head) of q (heads 0..NQ-1) or k (heads NQ..): RMSNorm over the
    head (fp32 statistics, bf16 rounding before the weight, as vLLM's rms_norm
    kernel), then neox rotary over the full head with cos / sin of the row's
    position."""
    row = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    offs = tl.arange(0, HALF)
    base = qkv_ptr + row * stride_qkv + h * HD   # k heads follow the q heads in qkv
    x1 = tl.load(base + offs).to(tl.float32)
    x2 = tl.load(base + HALF + offs).to(tl.float32)
    r = 1.0 / tl.sqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / HD + eps)
    is_q = h < NQ
    w1 = tl.load(qw_ptr + offs, mask=is_q & (offs >= 0), other=0.0).to(tl.float32) + tl.load(
        kw_ptr + offs, mask=(~is_q) & (offs >= 0), other=0.0
    ).to(tl.float32)
    w2 = tl.load(qw_ptr + HALF + offs, mask=is_q & (offs >= 0), other=0.0).to(tl.float32) + tl.load(
        kw_ptr + HALF + offs, mask=(~is_q) & (offs >= 0), other=0.0
    ).to(tl.float32)
    n1 = (x1 * r).to(tl.bfloat16).to(tl.float32) * w1
    n2 = (x2 * r).to(tl.bfloat16).to(tl.float32) * w2
    pos = tl.load(pos_ptr + row)
    # cos / sin rounded to bf16, as the reference rotary (cache cast to the query dtype)
    c = tl.load(cs_ptr + pos * stride_cs + offs).to(tl.bfloat16).to(tl.float32)
    s = tl.load(cs_ptr + pos * stride_cs + HALF + offs).to(tl.bfloat16).to(tl.float32)
    o1 = (n1 * c - n2 * s).to(tl.bfloat16)
    o2 = (n2 * c + n1 * s).to(tl.bfloat16)
    mq = is_q & (offs >= 0)
    mk = (~is_q) & (offs >= 0)
    qo = q_ptr + row * (NQ * HD) + h * HD
    tl.store(qo + offs, o1, mask=mq)
    tl.store(qo + HALF + offs, o2, mask=mq)
    ko = k_ptr + row * (NK * HD) + (h - NQ) * HD
    tl.store(ko + offs, o1, mask=mk)
    tl.store(ko + HALF + offs, o2, mask=mk)


def _qk_norm_rope(
    qkv: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float, num_q_heads: int,
    num_kv_heads: int, head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = qkv.shape[0]
    q = torch.empty(n, num_q_heads * head_dim, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(n, num_kv_heads * head_dim, dtype=qkv.dtype, device=qkv.device)
    if n:
        _qk_norm_rope_kernel[(n, num_q_heads + num_kv_heads)](
            qkv, qkv.stride(0), positions, cos_sin_cache, cos_sin_cache.stride(0), q_weight,
            k_weight, q, k, eps, NQ=num_q_heads, NK=num_kv_heads, HD=head_dim, num_warps=1,
        )
    return q, k


def _qk_norm_rope_fake(qkv, positions, cos_sin_cache, q_weight, k_weight, eps, num_q_heads,
                       num_kv_heads, head_dim):
    n = qkv.shape[0]
    return (qkv.new_empty(n, num_q_heads * head_dim), qkv.new_empty(n, num_kv_heads * head_dim))


direct_register_custom_op(
    op_name="dspine_qk_norm_rope",
    op_func=_qk_norm_rope,
    mutates_args=[],
    fake_impl=_qk_norm_rope_fake,
)


def qk_norm_rope(qkv, positions, cos_sin_cache, q_weight, k_weight, eps, num_q_heads, num_kv_heads, head_dim):
    """q and k of a qkv projection ``[N, (num_q_heads + 2 num_kv_heads) head_dim]``:
    per-head RMSNorm (``q_weight`` / ``k_weight``, ``eps``) and neox rotary over
    the full head with ``cos_sin_cache[positions]`` (cos | sin halves), as
    contiguous ``[N, heads * head_dim]`` tensors. One launch instead of the two
    copies, two RMSNorms and two rotary kernels of the eager layer."""
    return torch.ops.vllm.dspine_qk_norm_rope(
        qkv, positions, cos_sin_cache, q_weight, k_weight, eps, num_q_heads, num_kv_heads, head_dim
    )


# -- Adjacent injection (qwen3_dspine_fast.inject_fn) in three launches ---------------------------------------
# A: pre = hidden (+ residual), its row RMS, projected = (pre / (rms + eps)) @ w_read.T
# B: features = sqrt(T) projected / (|projected| + eps); predecessor = anchor (block position 1) or the previous
#    row's features; gated = sigmoid(features @ w_recv.T + predecessor @ w_pred.T + bias) * (predecessor @ w_msg.T)
# C: out = pre (block position 0) or pre + rms * (gated @ w_write.T)
# Each launch tiles rows (BM) x output columns (BN) and walks K with fp32 tl.dot in IEEE precision: FMA code on
# every GPU (SM70 included), for the draft step's few rows. Same casts as inject_fn at the linear boundaries.

_TINY = tl.constexpr(1.1754943508222875e-38)  # torch.finfo(torch.float32).tiny
INJECT_MAX_ROWS = 64


@triton.jit
def _inject_read_kernel(
    h_ptr, r_ptr, w_ptr, pre_ptr, rms_ptr, proj_ptr, n_rows, eps,
    HAS_RES: tl.constexpr, H: tl.constexpr, T: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    sq = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        off = rows[:, None] * H + k0 + ks[None, :]
        x = tl.load(h_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        if HAS_RES:
            x += tl.load(r_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        x = x.to(tl.bfloat16).to(tl.float32)   # pre is a bf16 tensor
        sq += tl.sum(x * x, 1)
        if pn == 0:
            tl.store(pre_ptr + off, x.to(tl.bfloat16), mask=rmask[:, None])
    rms = tl.sqrt(tl.maximum(sq / H, _TINY))
    if pn == 0:
        tl.store(rms_ptr + rows, rms, mask=rmask)
    inv = 1.0 / (rms + eps)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        off = rows[:, None] * H + k0 + ks[None, :]
        x = tl.load(h_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        if HAS_RES:
            x += tl.load(r_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        x = x.to(tl.bfloat16).to(tl.float32)
        xn = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + cols[None, :] * H + k0 + ks[:, None]).to(tl.float32)   # [BK, BN]
        acc += tl.dot(xn, w, input_precision="ieee")
    tl.store(proj_ptr + rows[:, None] * T + cols[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _inject_gate_kernel(
    proj_ptr, anchor_ptr, wr_ptr, wpm_ptr, b_ptr, gated_ptr, n_rows, eps, scale,
    T: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    prev = tl.where(rows == 0, n_rows - 1, rows - 1)   # features.roll(1)
    pos = rows % BLOCK
    use_anchor = pos == 1
    arow = rows // BLOCK
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    sq = tl.zeros((BM,), dtype=tl.float32)
    sqp = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, T, BK):
        f = tl.load(proj_ptr + rows[:, None] * T + k0 + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        fp = tl.load(proj_ptr + prev[:, None] * T + k0 + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        sq += tl.sum(f * f, 1)
        sqp += tl.sum(fp * fp, 1)
    s = scale / (tl.sqrt(sq) + eps)
    sp = scale / (tl.sqrt(sqp) + eps)
    acc_r = tl.zeros((BM, BN), dtype=tl.float32)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_m = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, T, BK):
        f = tl.load(proj_ptr + rows[:, None] * T + k0 + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        f = (f * s[:, None]).to(tl.bfloat16).to(tl.float32)
        fp = tl.load(proj_ptr + prev[:, None] * T + k0 + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        fp = (fp * sp[:, None]).to(tl.bfloat16).to(tl.float32)
        an = tl.load(anchor_ptr + arow[:, None] * T + k0 + ks[None, :], mask=(rmask & use_anchor)[:, None],
                     other=0.0).to(tl.float32)
        pred = tl.where(use_anchor[:, None], an, fp)
        wr = tl.load(wr_ptr + cols[None, :] * T + k0 + ks[:, None]).to(tl.float32)
        wp = tl.load(wpm_ptr + cols[None, :] * T + k0 + ks[:, None]).to(tl.float32)
        wm = tl.load(wpm_ptr + (G + cols[None, :]) * T + k0 + ks[:, None]).to(tl.float32)
        acc_r += tl.dot(f, wr, input_precision="ieee")
        acc_p += tl.dot(pred, wp, input_precision="ieee")
        acc_m += tl.dot(pred, wm, input_precision="ieee")
    bias = tl.load(b_ptr + cols).to(tl.float32)
    z = acc_r.to(tl.bfloat16).to(tl.float32) + acc_p.to(tl.bfloat16).to(tl.float32) + bias[None, :]
    gate = 1.0 / (1.0 + tl.exp(-z))
    gated = (gate * acc_m.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(gated_ptr + rows[:, None] * G + cols[None, :], gated, mask=rmask[:, None])


@triton.jit
def _inject_write_kernel(
    g_ptr, w_ptr, pre_ptr, rms_ptr, out_ptr, n_rows,
    H: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, G, BK):
        g = tl.load(g_ptr + rows[:, None] * G + k0 + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols[None, :] * G + k0 + ks[:, None]).to(tl.float32)
        acc += tl.dot(g, w, input_precision="ieee")
    update = acc.to(tl.bfloat16).to(tl.float32)
    off = rows[:, None] * H + cols[None, :]
    pre = tl.load(pre_ptr + off, mask=rmask[:, None], other=0.0)
    rms = tl.load(rms_ptr + rows, mask=rmask, other=0.0)
    written = (pre.to(tl.float32) + rms[:, None] * update).to(tl.bfloat16)
    out = tl.where(((rows % BLOCK) == 0)[:, None], pre, written)
    tl.store(out_ptr + off, out, mask=rmask[:, None])


def _inject(
    hidden: torch.Tensor, residual: torch.Tensor | None, anchor: torch.Tensor, w_read: torch.Tensor,
    w_recv: torch.Tensor, w_pred_msg: torch.Tensor, w_write: torch.Tensor, gate_bias: torch.Tensor,
    block_size: int, transfer_dim: int, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, h = hidden.shape
    t, g = w_read.shape[0], w_recv.shape[0]
    pre = torch.empty_like(hidden)
    out = torch.empty_like(hidden)
    if n == 0:
        return out, pre
    rms = torch.empty(n, dtype=torch.float32, device=hidden.device)
    proj = torch.empty(n, t, dtype=hidden.dtype, device=hidden.device)
    gated = torch.empty(n, g, dtype=hidden.dtype, device=hidden.device)
    bm = 16 if n <= 16 else (32 if n <= 32 else 64)
    rm = triton.cdiv(n, bm)
    res = residual if residual is not None else hidden
    _inject_read_kernel[(rm, triton.cdiv(t, 32))](
        hidden, res, w_read, pre, rms, proj, n, eps, HAS_RES=residual is not None, H=h, T=t,
        BM=bm, BN=32, BK=32, num_warps=4,
    )
    _inject_gate_kernel[(rm, triton.cdiv(g, 16))](
        proj, anchor, w_recv, w_pred_msg, gate_bias, gated, n, eps, float(transfer_dim) ** 0.5,
        T=t, G=g, BLOCK=block_size, BM=bm, BN=16, BK=32, num_warps=4,
    )
    _inject_write_kernel[(rm, triton.cdiv(h, 32))](
        gated, w_write, pre, rms, out, n, H=h, G=g, BLOCK=block_size, BM=bm, BN=32, BK=32, num_warps=4,
    )
    return out, pre


def _inject_fake(hidden, residual, anchor, w_read, w_recv, w_pred_msg, w_write, gate_bias, block_size,
                 transfer_dim, eps):
    return torch.empty_like(hidden), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="dspine_inject",
    op_func=_inject,
    mutates_args=[],
    fake_impl=_inject_fake,
)


def inject(hidden, residual, anchor, w_read, w_recv, w_pred_msg, w_write, gate_bias, block_size, transfer_dim,
           eps):
    """AdjacentCausalInjection for one layer, from the decoder layer's output
    ``hidden`` (+ ``residual``): ``(injected, pre_injection)``. Three launches
    instead of the residual add and inject_fn's nine kernels. Rows up to
    INJECT_MAX_ROWS (the draft step of small batches); ``w_pred_msg`` =
    cat(predecessor_gate, message), ``anchor`` one row per request."""
    return torch.ops.vllm.dspine_inject(
        hidden, residual, anchor, w_read, w_recv, w_pred_msg, w_write, gate_bias, block_size, transfer_dim, eps
    )


# -- Context K/V of every draft layer into its cache (DFlash precompute_and_store_context_kv) -----------------
# After the fused K/V projection ``all_kv [num_ctx, L * 2 * nkv * hd]`` (layer-major, then K | V, heads, dims):
# per-layer K RMSNorm, neox rotary at the context positions, and the cache writes of every layer, in one launch
# instead of the layer-major copy, the grouped RMSNorm, the positions repeat, the rotary and one cache-update
# kernel per layer. TRITON_ATTN cache layout: ``kv_cache [blocks, nkv, block_size, 2 * hd]`` (K | V).

CTX_KV_MAX_LAYERS = 8


@triton.jit
def _ctx_kv_store_kernel(
    kv_ptr, stride_kv, pos_ptr, cs_ptr, stride_cs, knw_ptr, eps,
    sm0, sm1, sm2, sm3, sm4, sm5, sm6, sm7,
    c0, c1, c2, c3, c4, c5, c6, c7,
    s_blk, s_head, s_tok, block_size,
    NKV: tl.constexpr, HD: tl.constexpr,
):
    t = tl.program_id(0)
    lay = tl.program_id(1)
    h = tl.program_id(2)
    HALF: tl.constexpr = HD // 2
    offs = tl.arange(0, HALF)
    if lay == 0:
        sm, cache = sm0, c0
    elif lay == 1:
        sm, cache = sm1, c1
    elif lay == 2:
        sm, cache = sm2, c2
    elif lay == 3:
        sm, cache = sm3, c3
    elif lay == 4:
        sm, cache = sm4, c4
    elif lay == 5:
        sm, cache = sm5, c5
    elif lay == 6:
        sm, cache = sm6, c6
    else:
        sm, cache = sm7, c7
    slot = tl.load(sm + t)
    if slot >= 0:
        base = kv_ptr + t * stride_kv + lay * (2 * NKV * HD) + h * HD
        k1 = tl.load(base + offs).to(tl.float32)
        k2 = tl.load(base + HALF + offs).to(tl.float32)
        v1 = tl.load(base + NKV * HD + offs)
        v2 = tl.load(base + NKV * HD + HALF + offs)
        r = 1.0 / tl.sqrt((tl.sum(k1 * k1, 0) + tl.sum(k2 * k2, 0)) / HD + eps)
        w1 = tl.load(knw_ptr + lay * HD + offs).to(tl.float32)
        w2 = tl.load(knw_ptr + lay * HD + HALF + offs).to(tl.float32)
        n1 = (k1 * r).to(tl.bfloat16).to(tl.float32) * w1
        n2 = (k2 * r).to(tl.bfloat16).to(tl.float32) * w2
        pos = tl.load(pos_ptr + t)
        cs = tl.load(cs_ptr + pos * stride_cs + offs).to(tl.bfloat16).to(tl.float32)   # as ops.rotary_embedding
        sn = tl.load(cs_ptr + pos * stride_cs + HALF + offs).to(tl.bfloat16).to(tl.float32)
        o1 = (n1 * cs - n2 * sn).to(tl.bfloat16)
        o2 = (n2 * cs + n1 * sn).to(tl.bfloat16)
        dst = cache + (slot // block_size) * s_blk + h * s_head + (slot % block_size) * s_tok
        tl.store(dst + offs, o1)
        tl.store(dst + HALF + offs, o2)
        tl.store(dst + HD + offs, v1)
        tl.store(dst + HD + HALF + offs, v2)


def ctx_kv_store_ok(kv_caches: list[torch.Tensor], head_dim: int) -> bool:
    """The layout ctx_kv_store writes: bf16 TRITON_ATTN caches of one shape and
    stride, ``[blocks, nkv, block_size, 2 * head_dim]``, at most CTX_KV_MAX_LAYERS."""
    if not kv_caches or len(kv_caches) > CTX_KV_MAX_LAYERS:
        return False
    c0 = kv_caches[0]
    return all(
        c.dtype == torch.bfloat16 and c.dim() == 4 and c.shape == c0.shape and c.stride() == c0.stride()
        and c.shape[-1] == 2 * head_dim and c.stride(-1) == 1
        for c in kv_caches
    )


def ctx_kv_store(all_kv_flat, positions, cos_sin_cache, k_norm_weights, eps, slot_mappings, kv_caches,
                 num_kv_heads, head_dim):
    """K (per-layer RMSNorm + neox rotary) and V of every context token into
    every layer's cache at its slot (negative slot: skipped)."""
    n = all_kv_flat.shape[0]
    L = len(kv_caches)
    if n == 0:
        return
    sms = list(slot_mappings) + [slot_mappings[0]] * (CTX_KV_MAX_LAYERS - L)
    cs = list(kv_caches) + [kv_caches[0]] * (CTX_KV_MAX_LAYERS - L)
    c0 = kv_caches[0]
    _ctx_kv_store_kernel[(n, L, num_kv_heads)](
        all_kv_flat, all_kv_flat.stride(0), positions, cos_sin_cache, cos_sin_cache.stride(0), k_norm_weights, eps,
        *sms, *cs, c0.stride(0), c0.stride(1), c0.stride(2), c0.shape[2],
        NKV=num_kv_heads, HD=head_dim, num_warps=1,
    )


# -- Draft inputs: token embeddings (mask slots take the mask embedding) and anchor features ---------------
@triton.jit
def _prep_kernel(
    ids_ptr, emb_ptr, mask_emb_ptr, cb_ptr, out_ptr, anchor_ptr, n_rows, mask_id, scale,
    H: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr, HAS_MASK: tl.constexpr, BH: tl.constexpr,
    BT: tl.constexpr,
):
    p = tl.program_id(0)
    if p < n_rows:
        tok = tl.load(ids_ptr + p)
        offs_h = tl.arange(0, BH)
        mh = offs_h < H
        row = tl.load(emb_ptr + tok.to(tl.int64) * H + offs_h, mask=mh, other=0.0)
        if HAS_MASK:
            if tok == mask_id:
                row = tl.load(mask_emb_ptr + offs_h, mask=mh, other=0.0).to(row.dtype)
        tl.store(out_ptr + p.to(tl.int64) * H + offs_h, row, mask=mh)
    else:
        r = p - n_rows
        atok = tl.load(ids_ptr + r * BLOCK)
        offs_t = tl.arange(0, BT)
        mt = offs_t < T
        f = tl.load(cb_ptr + atok.to(tl.int64) * T + offs_t, mask=mt, other=0.0).to(tl.float32) * scale
        tl.store(anchor_ptr + r * T + offs_t, f.to(anchor_ptr.dtype.element_ty), mask=mt)


def prep_inputs(input_ids, embed_weight, mask_embedding, mask_token_id, codebook, block_size, transfer_dim):
    """``(input_embeds, anchor_features)`` of a draft step: embed_input_ids (mask
    slots take ``mask_embedding`` when it is given) and the injection's
    token_features of every block's first token, in one launch."""
    n = input_ids.shape[0]
    h = embed_weight.shape[1]
    t = codebook.shape[1]
    r = (n + block_size - 1) // block_size
    out = torch.empty(n, h, dtype=embed_weight.dtype, device=input_ids.device)
    anchor = torch.empty(r, t, dtype=codebook.dtype, device=input_ids.device)
    if n:
        has_mask = mask_embedding is not None and mask_token_id is not None
        _prep_kernel[(n + r,)](
            input_ids, embed_weight, mask_embedding if has_mask else embed_weight, codebook, out, anchor, n,
            int(mask_token_id) if has_mask else -1, float(transfer_dim) ** 0.5,
            H=h, T=t, BLOCK=block_size, HAS_MASK=has_mask, BH=triton.next_power_of_2(h),
            BT=triton.next_power_of_2(t), num_warps=4,
        )
    return out, anchor


# -- Adjacent injection around the existing GEMVs: one row-wise kernel between each pair -----------------------
# The four linears stay on the drafter's GEMV path (olala_linear: skinny / Volta FMA kernels); the residual add,
# RMS, feature normalization, predecessor selection, gating and update run as four row-wise launches instead of
# six (inject_fn's compiled pieces plus the add).

@triton.jit
def _inj_pre_kernel(h_ptr, r_ptr, pre_ptr, xn_ptr, rms_ptr, eps, H: tl.constexpr, BH: tl.constexpr,
                    HAS_RES: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BH)
    m = offs < H
    x = tl.load(h_ptr + i * H + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        x += tl.load(r_ptr + i * H + offs, mask=m, other=0.0).to(tl.float32)
    x = x.to(tl.bfloat16).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, _TINY))
    tl.store(pre_ptr + i * H + offs, x.to(tl.bfloat16), mask=m)
    tl.store(xn_ptr + i * H + offs, (x / (rms + eps)).to(tl.bfloat16), mask=m)
    tl.store(rms_ptr + i, rms)


@triton.jit
def _inj_feat_kernel(p_ptr, anchor_ptr, f_ptr, pred_ptr, n_rows, eps, scale, T: tl.constexpr, BT: tl.constexpr,
                     BLOCK: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BT)
    m = offs < T
    p = tl.load(p_ptr + i * T + offs, mask=m, other=0.0).to(tl.float32)
    f = (scale * p / (tl.sqrt(tl.sum(p * p, 0)) + eps)).to(tl.bfloat16)
    tl.store(f_ptr + i * T + offs, f, mask=m)
    j = tl.where(i == 0, n_rows - 1, i - 1)   # features.roll(1)
    q = tl.load(p_ptr + j * T + offs, mask=m, other=0.0).to(tl.float32)
    fq = (scale * q / (tl.sqrt(tl.sum(q * q, 0)) + eps)).to(tl.bfloat16)
    if i % BLOCK == 1:
        fq = tl.load(anchor_ptr + (i // BLOCK) * T + offs, mask=m, other=0.0).to(tl.bfloat16)
    tl.store(pred_ptr + i * T + offs, fq, mask=m)


@triton.jit
def _inj_gate_kernel(r_ptr, pm_ptr, b_ptr, g_ptr, rms_ptr, G: tl.constexpr, BG: tl.constexpr,
                     BLOCK: tl.constexpr, AS_DELTA: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BG)
    m = offs < G
    z = (tl.load(r_ptr + i * G + offs, mask=m, other=0.0).to(tl.float32)
         + tl.load(pm_ptr + i * 2 * G + offs, mask=m, other=0.0).to(tl.float32)
         + tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32))
    gate = 1.0 / (1.0 + tl.exp(-z))
    msg = tl.load(pm_ptr + i * 2 * G + G + offs, mask=m, other=0.0).to(tl.float32)
    gm = gate * msg
    if AS_DELTA:   # rms * (gated @ w_write.T) = (rms * gated) @ w_write.T; block position 0 keeps pre
        gm = gm * tl.load(rms_ptr + i) * tl.where(i % BLOCK == 0, 0.0, 1.0)
    tl.store(g_ptr + i * G + offs, gm.to(tl.bfloat16), mask=m)


@triton.jit
def _inj_out_kernel(pre_ptr, upd_ptr, rms_ptr, out_ptr, H: tl.constexpr, BH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BH)
    m = offs < H
    pre = tl.load(pre_ptr + i * H + offs, mask=m, other=0.0)
    if i % BLOCK == 0:
        tl.store(out_ptr + i * H + offs, pre, mask=m)
    else:
        upd = tl.load(upd_ptr + i * H + offs, mask=m, other=0.0).to(tl.float32)
        rms = tl.load(rms_ptr + i)
        tl.store(out_ptr + i * H + offs, (pre.to(tl.float32) + rms * upd).to(tl.bfloat16), mask=m)


def inject_rows(hidden, residual, anchor, w_read, w_recv, w_pred_msg, w_write, gate_bias, block_size,
                transfer_dim, eps, linear, as_delta=False):
    """AdjacentCausalInjection for one layer from the decoder layer's output:
    ``(injected, pre_injection)``. ``linear(x, w)`` is the drafter's GEMV
    (qwen3_dspine_fast._linear); the rest is four row-wise launches. With
    ``as_delta`` it returns ``(delta, pre_injection)`` with ``injected = pre +
    delta`` instead (three row-wise launches): the next RMSNorm adds it to its
    residual (fused_add_rms_norm), as a decoder layer does."""
    n, h = hidden.shape
    t, g = w_read.shape[0], w_recv.shape[0]
    pre = torch.empty_like(hidden)
    xn = torch.empty_like(hidden)
    rms = torch.empty(n, dtype=torch.float32, device=hidden.device)
    if n == 0:
        return hidden.clone(), pre
    bh, bt, bg = triton.next_power_of_2(h), triton.next_power_of_2(t), triton.next_power_of_2(g)
    _inj_pre_kernel[(n,)](hidden, residual if residual is not None else hidden, pre, xn, rms, eps, H=h, BH=bh,
                          HAS_RES=residual is not None, num_warps=4)
    proj = linear(xn, w_read)
    feats = torch.empty(n, t, dtype=hidden.dtype, device=hidden.device)
    pred = torch.empty_like(feats)
    _inj_feat_kernel[(n,)](proj, anchor, feats, pred, n, eps, float(transfer_dim) ** 0.5, T=t, BT=bt,
                           BLOCK=block_size, num_warps=4)
    recv = linear(feats, w_recv)
    pm = linear(pred, w_pred_msg)
    gated = torch.empty(n, g, dtype=hidden.dtype, device=hidden.device)
    _inj_gate_kernel[(n,)](recv, pm, gate_bias, gated, rms, G=g, BG=bg, BLOCK=block_size, AS_DELTA=as_delta,
                           num_warps=2)
    upd = linear(gated, w_write)
    if as_delta:
        return upd, pre
    out = torch.empty_like(hidden)
    _inj_out_kernel[(n,)](pre, upd, rms, out, H=h, BH=bh, BLOCK=block_size, num_warps=4)
    return out, pre


# -- Transition scoring (qwen3_dspine._score_transitions, tp 1) --------------------------------------------------
# Rows q = (request, position) carry the receiver state (pre-injection state of the last layer); candidate rows
# m = (q, a) the top-k successor candidates. In fp32 as the reference: per candidate row the exact predecessor
# code (codebook row of the anchor at position 0, of the previous position's candidate a elsewhere) gives
# gate = sigmoid(recv_gate[q] + code @ w_pred.T + bias) and message = code @ w_msg.T (K_A); the refined state
# pre[q] + rms[q] * ((gate * message) @ w_write.T) (K_B, with its row sum of squares); scores = RMSNorm(refined)
# . successor head rows (K_C). Weight tiles are loaded [BN, BK] (contiguous along K) and transposed.

TRANSITION_MAX_ROWS = 2048


@triton.jit
def _tr_gate_kernel(
    cand_ptr, anchor_ptr, cb_ptr, recv_ptr, wp_ptr, wm_ptr, b_ptr, gm_ptr, n_rows, n_pos, scale,
    TOPK: tl.constexpr, T: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    MMA: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    q = rows // TOPK
    a = rows % TOPK
    req = q // n_pos
    pos = q % n_pos
    prev_cand = tl.load(cand_ptr + (q - 1) * TOPK + a, mask=rmask & (pos > 0), other=0)
    anc = tl.load(anchor_ptr + req, mask=rmask & (pos == 0), other=0)
    pid = tl.where(pos == 0, anc, prev_cand).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_m = tl.zeros((BM, BN), dtype=tl.float32)
    # (scale * code) @ w.T = scale * (code @ w.T): bf16 operands, exact products, fp32 sums
    for k0 in range(0, T, BK):
        code = tl.load(cb_ptr + pid[:, None] * T + k0 + ks[None, :], mask=rmask[:, None], other=0.0)
        wp = tl.load(wp_ptr + cols[:, None] * T + k0 + ks[None, :])   # [BN, BK]
        wm = tl.load(wm_ptr + cols[:, None] * T + k0 + ks[None, :])
        if MMA:
            acc_p += tl.dot(code, tl.trans(wp))
            acc_m += tl.dot(code, tl.trans(wm))
        else:   # SM70: FMA code (no bf16 mma)
            code_f = code.to(tl.float32)
            acc_p += tl.dot(code_f, tl.trans(wp.to(tl.float32)), input_precision="ieee")
            acc_m += tl.dot(code_f, tl.trans(wm.to(tl.float32)), input_precision="ieee")
    acc_p = acc_p * scale
    acc_m = acc_m * scale
    r = tl.load(recv_ptr + q[:, None] * G + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + cols).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-(r + acc_p + bias[None, :])))
    tl.store(gm_ptr + rows[:, None] * G + cols[None, :], gate * acc_m, mask=rmask[:, None])


@triton.jit
def _tr_write_kernel(
    gm_ptr, ww_ptr, pre_ptr, rms_ptr, ref_ptr, ss_ptr, n_rows,
    TOPK: tl.constexpr, H: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    q = rows // TOPK
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, G, BK):
        g = tl.load(gm_ptr + rows[:, None] * G + k0 + ks[None, :], mask=rmask[:, None], other=0.0)
        w = tl.load(ww_ptr + cols[:, None] * G + k0 + ks[None, :]).to(tl.float32)   # [BN, BK]
        acc += tl.dot(g, tl.trans(w), input_precision="ieee")
    pre = tl.load(pre_ptr + q[:, None] * H + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    rms = tl.load(rms_ptr + q, mask=rmask, other=0.0)
    refined = pre + rms[:, None] * acc
    tl.store(ref_ptr + rows[:, None] * H + cols[None, :], refined, mask=rmask[:, None])
    tl.atomic_add(ss_ptr + rows, tl.sum(refined * refined, 1), mask=rmask)


@triton.jit
def _tr_score_kernel(
    ref_ptr, ss_ptr, nw_ptr, cand_ptr, head_ptr, hs_ptr, out_ptr, eps,
    TOPK: tl.constexpr, H: tl.constexpr, BK: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    q = tl.program_id(0)
    k0 = tl.program_id(1) * BK   # one slice of H per program, summed into the zeroed output
    a = tl.arange(0, TOPK)
    rows = q * TOPK + a
    ks = tl.arange(0, BK)
    inv = 1.0 / tl.sqrt(tl.load(ss_ptr + rows) / H + eps)
    cid = tl.load(cand_ptr + rows).to(tl.int64)   # the successors j of row q are its own candidates
    nrm = tl.load(ref_ptr + rows[:, None] * H + k0 + ks[None, :]) * inv[:, None]
    nrm = nrm * tl.load(nw_ptr + k0 + ks).to(tl.float32)[None, :]
    succ = tl.load(head_ptr + cid[:, None] * H + k0 + ks[None, :]).to(tl.float32)   # [j, BK]
    if HAS_SCALE:
        succ = succ * tl.load(hs_ptr + cid).to(tl.float32)[:, None]
    acc = tl.dot(nrm, tl.trans(succ), input_precision="ieee")
    tl.atomic_add(out_ptr + q * TOPK * TOPK + a[:, None] * TOPK + a[None, :], acc)


def score_transitions(final_pre, candidate_ids, anchor_ids, w_read, w_recv, w_pred, w_msg, w_write, gate_bias,
                      norm_w, head_w, head_scale, codebook, transfer_dim, eps, norm_eps, linear):
    """Transition scores ``[R, n, k, k]`` (the reference's einsum layout) of
    ``final_pre [R, n, H]`` and its ``candidate_ids [R, n, k]``; ``anchor_ids [R]``
    are the predecessors of position 0. ``linear`` is the drafter's GEMV for the
    receiver-side bf16 linears; the candidate side runs in fp32 as the reference."""
    R, n, h = final_pre.shape
    k = candidate_ids.shape[-1]
    t, g = w_read.shape[0], w_recv.shape[0]
    qn = R * n
    m = qn * k
    dev = final_pre.device
    pre2 = final_pre.reshape(qn, h)
    pre_c = torch.empty_like(pre2)
    xn = torch.empty_like(pre2)
    rms = torch.empty(qn, dtype=torch.float32, device=dev)
    bh, bt = triton.next_power_of_2(h), triton.next_power_of_2(t)
    _inj_pre_kernel[(qn,)](pre2, pre2, pre_c, xn, rms, eps, H=h, BH=bh, HAS_RES=False, num_warps=4)
    proj = linear(xn, w_read)
    feats = torch.empty(qn, t, dtype=final_pre.dtype, device=dev)
    _tr_feat_kernel[(qn,)](proj, feats, eps, float(transfer_dim) ** 0.5, T=t, BT=bt, num_warps=4)
    recv = linear(feats, w_recv)
    cand = candidate_ids.reshape(m).contiguous()
    anchor_ids = anchor_ids.contiguous()   # the speculator passes a strided view of the input ids
    gm = torch.empty(m, g, dtype=torch.float32, device=dev)
    bm = 16
    _tr_gate_kernel[(triton.cdiv(m, bm), g // 16)](
        cand, anchor_ids, codebook, recv, w_pred, w_msg, gate_bias, gm, m, n, float(transfer_dim) ** 0.5,
        TOPK=k, T=t, G=g, BM=bm, BN=16, BK=64, MMA=_has_bf16_mma(), num_warps=4,
    )
    refined = torch.empty(m, h, dtype=torch.float32, device=dev)
    zeros = torch.zeros(m + qn * k * k, dtype=torch.float32, device=dev)   # row sums of squares | scores
    ss = zeros[:m]
    _tr_write_kernel[(triton.cdiv(m, bm), h // 64)](
        gm, w_write, pre2, rms, refined, ss, m, TOPK=k, H=h, G=g, BM=bm, BN=64, BK=32, num_warps=4,
    )
    out = zeros[m:].view(R, n, k, k)
    _tr_score_kernel[(qn, h // 64)](
        refined, ss, norm_w, cand, head_w, head_scale if head_scale is not None else norm_w, out, norm_eps,
        TOPK=k, H=h, BK=64, HAS_SCALE=head_scale is not None, num_warps=4,
    )
    return out


def _has_bf16_mma() -> bool:
    """bf16 tensor-core dots: not on the SM70 family (also through volta-forever,
    which tells the process sm_80 and would emulate them)."""
    from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family

    sm70 = getattr(GpuFamily, "SM70", None)   # trees with the V100 support only
    return sm70 is None or gpu_family() is not sm70


@triton.jit
def _tr_feat_kernel(p_ptr, f_ptr, eps, scale, T: tl.constexpr, BT: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BT)
    msk = offs < T
    p = tl.load(p_ptr + i * T + offs, mask=msk, other=0.0).to(tl.float32)
    tl.store(f_ptr + i * T + offs, (scale * p / (tl.sqrt(tl.sum(p * p, 0)) + eps)).to(tl.bfloat16), mask=msk)

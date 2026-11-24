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
    c = tl.load(cs_ptr + pos * stride_cs + offs).to(tl.float32)
    s = tl.load(cs_ptr + pos * stride_cs + HALF + offs).to(tl.float32)
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
        cs = tl.load(cs_ptr + pos * stride_cs + offs).to(tl.float32)
        sn = tl.load(cs_ptr + pos * stride_cs + HALF + offs).to(tl.float32)
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

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

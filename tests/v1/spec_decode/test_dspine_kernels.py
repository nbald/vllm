# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hand-fused DSpine draft kernels (qwen3_dspine_kernels.py) against the
eager chains they replace, on the drafter's shapes (12 q heads, 4 KV heads,
head dim 128) at the draft step's row counts. Not bit-identical by design
(fp32 math, one rounding); the bound is a few bf16 ulps."""

import pytest
import torch

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _rope_cache(max_pos: int, head_dim: int, base: float = 1e6) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


@requires_gpu
@pytest.mark.parametrize("n", [1, 4, 6, 48, 300])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
def test_qk_norm_rope_matches_eager(n, cache_dtype):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.model_executor.models.qwen3_dspine_kernels import qk_norm_rope

    torch.manual_seed(n)
    nq, nk, hd, eps = 12, 4, 128, 1e-6
    qkv = (torch.randn(n, (nq + 2 * nk) * hd, device="cuda") * 3).to(torch.bfloat16)
    qw = (torch.rand(hd, device="cuda") + 0.5).to(torch.bfloat16)
    kw = (torch.rand(hd, device="cuda") + 0.5).to(torch.bfloat16)
    pos = torch.randint(0, 8192, (n,), device="cuda")
    cache = _rope_cache(8192, hd).to("cuda", cache_dtype)

    def ref_norm(x, w):  # vLLM's rms_norm: fp32 statistics, bf16 before the weight
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        return (y.to(torch.bfloat16).float() * w.float()).to(torch.bfloat16)

    q = ref_norm(qkv[:, : nq * hd].view(n, nq, hd), qw).view(n, -1)
    k = ref_norm(qkv[:, nq * hd : (nq + nk) * hd].view(n, nk, hd), kw).view(n, -1)
    q_ref, k_ref = RotaryEmbedding.forward_static(pos, q.float(), k.float(), hd, hd, cache.float(), True)
    q_out, k_out = qk_norm_rope(qkv, pos, cache, qw, kw, eps, nq, nk, hd)
    assert q_out.shape == (n, nq * hd) and k_out.shape == (n, nk * hd)
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=2e-2, atol=3e-2)

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


@requires_gpu
@pytest.mark.parametrize("n", [1, 6, 30, 200])
def test_ctx_kv_store_matches_reference(n):
    from vllm.model_executor.models.qwen3_dspine_kernels import ctx_kv_store, ctx_kv_store_ok

    torch.manual_seed(n)
    L, nkv, hd, eps, blocks, bs = 5, 4, 128, 1e-6, 64, 16
    bf = torch.bfloat16
    all_kv = (torch.randn(n, L * 2 * nkv * hd, device="cuda") * 3).to(bf)
    knw = (torch.rand(L, hd, device="cuda") + 0.5).to(bf)
    pos = torch.randint(0, 4096, (n,), device="cuda")
    cache_cs = _rope_cache(4096, hd).to("cuda", bf)
    caches = [torch.zeros(blocks, nkv, bs, 2 * hd, device="cuda", dtype=bf) for _ in range(L)]
    slots = [torch.randperm(blocks * bs, device="cuda")[:n] for _ in range(L)]
    slots[2][0] = -1   # a padded row: nothing written for it
    assert ctx_kv_store_ok(caches, hd)
    ctx_kv_store(all_kv, pos, cache_cs, knw, eps, slots, caches, nkv, hd)

    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    kv = all_kv.view(n, L, 2, nkv, hd)
    for lay in range(L):
        k = kv[:, lay, 0].float()
        kn = (k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps)).to(bf).float() * knw[lay].float()
        k_ref, _ = RotaryEmbedding.forward_static(pos, kn.to(bf).float().view(n, -1), None, hd, hd,
                                                  cache_cs.float(), True)
        k_ref = k_ref.view(n, nkv, hd)
        for i in range(n):
            slot = int(slots[lay][i])
            if slot < 0:
                continue
            got = caches[lay][slot // bs, :, slot % bs]
            torch.testing.assert_close(got[:, :hd].float(), k_ref[i].float(), rtol=2e-2, atol=3e-2)
            assert torch.equal(got[:, hd:], kv[i, lay, 1])
    written = sum(int((s >= 0).sum()) for s in slots)
    assert sum(int(c.abs().sum(-1).ne(0).sum()) for c in caches) == written * nkv


@requires_gpu
@pytest.mark.parametrize("n,block", [(4, 4), (6, 6), (48, 6), (13, 4)])
@pytest.mark.parametrize("with_mask", [True, False])
def test_prep_inputs_matches_eager(n, block, with_mask):
    from vllm.model_executor.models.qwen3_dspine_kernels import prep_inputs

    torch.manual_seed(n + block)
    v, h, t, mask_id = 1000, 1536, 512, 999
    emb = torch.randn(v, h, device="cuda").to(torch.bfloat16)
    mask_emb = torch.randn(h, device="cuda")  # fp32 parameter, cast like embed_input_ids
    cb = torch.randn(v, t, device="cuda").to(torch.bfloat16)
    ids = torch.randint(0, v - 1, (n,), device="cuda")
    ids[1::block] = mask_id
    out, anchor = prep_inputs(ids, emb, mask_emb if with_mask else None, mask_id if with_mask else None, cb,
                              block, t)
    ref = emb[ids]
    if with_mask:
        ref = torch.where((ids == mask_id)[:, None], mask_emb.to(ref.dtype), ref)
    assert torch.equal(out, ref)
    anchor_ref = (cb[ids[::block]].float() * t**0.5).to(cb.dtype)
    assert torch.equal(anchor, anchor_ref)

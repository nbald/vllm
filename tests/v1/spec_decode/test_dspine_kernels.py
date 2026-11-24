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
    q_ref, k_ref = RotaryEmbedding.forward_static(pos, q.float(), k.float(), hd, hd, cache.to(torch.bfloat16).float(), True)
    q_out, k_out = qk_norm_rope(qkv, pos, cache, qw, kw, eps, nq, nk, hd)
    assert q_out.shape == (n, nq * hd) and k_out.shape == (n, nk * hd)
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=2e-2, atol=3e-2)


@requires_gpu
@pytest.mark.parametrize("n,block", [(4, 4), (6, 6), (12, 6), (48, 6), (64, 8), (33, 3)])
@pytest.mark.parametrize("with_residual", [True, False])
def test_inject_matches_inject_fn(n, block, with_residual):
    from vllm.model_executor.models.qwen3_dspine_fast import inject_fn
    from vllm.model_executor.models.qwen3_dspine_kernels import inject

    torch.manual_seed(n * 10 + block)
    h, t, g, eps = 1536, 512, 256, 1e-6
    bf = torch.bfloat16
    hidden = (torch.randn(n, h, device="cuda") * 2).to(bf)
    residual = (torch.randn(n, h, device="cuda") * 2).to(bf) if with_residual else None
    anchor = (torch.randn((n + block - 1) // block, t, device="cuda") * 4).to(bf)
    w_read = (torch.randn(t, h, device="cuda") / h**0.5).to(bf)
    w_recv = (torch.randn(g, t, device="cuda") / t**0.5).to(bf)
    w_pred_msg = (torch.randn(2 * g, t, device="cuda") / t**0.5).to(bf)
    w_write = (torch.randn(h, g, device="cuda") / g**0.5).to(bf)
    bias = (torch.randn(g, device="cuda") * 0.1).to(bf)
    pre_ref = hidden if residual is None else hidden + residual
    out_ref = inject_fn(pre_ref, anchor, w_read, w_recv, w_pred_msg, w_write, bias, block, t, eps)
    out, pre = inject(hidden, residual, anchor, w_read, w_recv, w_pred_msg, w_write, bias, block, t, eps)
    assert torch.equal(pre, pre_ref)
    # block position 0 keeps the pre-injection state exactly
    first = torch.arange(n, device="cuda") % block == 0
    assert torch.equal(out[first], pre_ref[first])
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-2, atol=5e-2)


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


@requires_gpu
@pytest.mark.parametrize("n,block", [(4, 4), (6, 6), (48, 6), (64, 8), (33, 3), (768, 6)])
@pytest.mark.parametrize("with_residual", [True, False])
def test_inject_rows_matches_inject_fn(n, block, with_residual):
    import torch.nn.functional as F

    from vllm.model_executor.models.qwen3_dspine_fast import inject_fn
    from vllm.model_executor.models.qwen3_dspine_kernels import inject_rows

    torch.manual_seed(n * 7 + block)
    h, t, g, eps = 1536, 512, 256, 1e-6
    bf = torch.bfloat16
    hidden = (torch.randn(n, h, device="cuda") * 2).to(bf)
    residual = (torch.randn(n, h, device="cuda") * 2).to(bf) if with_residual else None
    anchor = (torch.randn((n + block - 1) // block, t, device="cuda") * 4).to(bf)
    w_read = (torch.randn(t, h, device="cuda") / h**0.5).to(bf)
    w_recv = (torch.randn(g, t, device="cuda") / t**0.5).to(bf)
    w_pred_msg = (torch.randn(2 * g, t, device="cuda") / t**0.5).to(bf)
    w_write = (torch.randn(h, g, device="cuda") / g**0.5).to(bf)
    bias = (torch.randn(g, device="cuda") * 0.1).to(bf)
    pre_ref = hidden if residual is None else hidden + residual
    out_ref = inject_fn(pre_ref, anchor, w_read, w_recv, w_pred_msg, w_write, bias, block, t, eps)
    out, pre = inject_rows(hidden, residual, anchor, w_read, w_recv, w_pred_msg, w_write, bias, block, t, eps,
                           F.linear)
    assert torch.equal(pre, pre_ref)
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-2, atol=5e-2)


@requires_gpu
@pytest.mark.parametrize("kernel", ["rows", "rows_delta", "tile"])
@pytest.mark.parametrize("fill", [float("nan"), float("inf"), 1e4])
def test_inject_ignores_padded_rows(kernel, fill):
    """A CUDA graph pads the draft batch with whole blocks, whose rows can hold NaN (an empty attention) or
    anything else. The real rows must not see them: bit for bit those of the same batch padded with zeros, and
    those of the unpadded batch up to the GEMMs' row-count dependent rounding. Regression: block position 0 took
    the previous row (the padded block's last row for the first request) and the delta form zeroed it with a
    multiply, so a NaN reached the first request; served greedy tau at 5 requests padded to 6, K5: 2.22
    against 3.26 unpadded."""
    import torch.nn.functional as F

    from vllm.model_executor.models import qwen3_dspine_kernels as k

    torch.manual_seed(11)
    h, t, g, eps, block, reqs = 1536, 512, 256, 1e-6, 6, 5
    bf = torch.bfloat16
    n = reqs * block
    hidden = (torch.randn(n, h, device="cuda") * 2).to(bf)
    residual = (torch.randn(n, h, device="cuda") * 2).to(bf)
    anchor = (torch.randn(reqs, t, device="cuda") * 4).to(bf)
    ws = [(torch.randn(o, i, device="cuda") / i**0.5).to(bf) for o, i in ((t, h), (g, t), (2 * g, t), (h, g))]
    bias = (torch.randn(g, device="cuda") * 0.1).to(bf)

    def run(pad_value: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        hs, rs, an = hidden, residual, anchor
        if pad_value is not None:
            pad = torch.full((block, h), pad_value, device="cuda", dtype=bf)
            hs, rs = torch.cat([hidden, pad]), torch.cat([residual, pad])
            an = torch.cat([anchor, torch.zeros(1, t, device="cuda", dtype=bf)])
        if kernel == "tile":
            out, pre = k.inject(hs, rs, an, *ws, bias, block, t, eps)
        else:
            out, pre = k.inject_rows(hs, rs, an, *ws, bias, block, t, eps, F.linear, as_delta=kernel == "rows_delta")
        return out[:n], pre[:n]

    out, pre = run(fill)
    out_zero, pre_zero = run(0.0)
    assert torch.isfinite(out).all()
    assert torch.equal(out, out_zero) and torch.equal(pre, pre_zero)
    out_exact, _ = run(None)
    torch.testing.assert_close(out.float(), out_exact.float(), rtol=2e-2, atol=5e-2)


@requires_gpu
@pytest.mark.parametrize("n,block", [(6, 6), (48, 6), (33, 3)])
def test_inject_rows_delta_adds_up(n, block):
    import torch.nn.functional as F

    from vllm.model_executor.models.qwen3_dspine_fast import inject_fn
    from vllm.model_executor.models.qwen3_dspine_kernels import inject_rows

    torch.manual_seed(n + 3 * block)
    h, t, g, eps = 1536, 512, 256, 1e-6
    bf = torch.bfloat16
    hidden = (torch.randn(n, h, device="cuda") * 2).to(bf)
    residual = (torch.randn(n, h, device="cuda") * 2).to(bf)
    anchor = (torch.randn((n + block - 1) // block, t, device="cuda") * 4).to(bf)
    ws = [(torch.randn(o, i, device="cuda") / i**0.5).to(bf) for o, i in ((t, h), (g, t), (2 * g, t), (h, g))]
    bias = (torch.randn(g, device="cuda") * 0.1).to(bf)
    out_ref = inject_fn(hidden + residual, anchor, *ws, bias, block, t, eps)
    delta, pre = inject_rows(hidden, residual, anchor, *ws, bias, block, t, eps, F.linear, as_delta=True)
    first = torch.arange(n, device="cuda") % block == 0
    assert torch.equal(delta[first], torch.zeros_like(delta[first]))
    # rms * gated is rounded before the GEMV instead of after: a few elements move by a bf16 step or two
    torch.testing.assert_close((pre.float() + delta.float()).to(bf).float(), out_ref.float(), rtol=5e-2, atol=1e-1)


@requires_gpu
@pytest.mark.parametrize("R,n", [(1, 5), (1, 3), (8, 5), (3, 7)])
@pytest.mark.parametrize("int8_head", [True, False])
def test_score_transitions_matches_transitions_fn(R, n, int8_head):
    import torch.nn.functional as F

    from vllm.model_executor.models.qwen3_dspine_fast import transitions_fn
    from vllm.model_executor.models.qwen3_dspine_kernels import score_transitions

    torch.manual_seed(R * 10 + n)
    h, t, g, k, v, eps, neps = 1536, 512, 256, 16, 5000, 1e-6, 1e-6
    bf = torch.bfloat16
    pre = (torch.randn(R, n, h, device="cuda") * 2).to(bf)
    cand = torch.randint(0, v, (R, n, k), device="cuda")
    # as DSpineSpeculator passes it: every request's first query token, a strided view of the input ids
    anchor = torch.randint(0, v, (R * (n + 1),), device="cuda")[:: n + 1]
    pred_ids = torch.cat((anchor[:, None, None].expand(-1, 1, k), cand[:, :-1]), dim=1)
    cb = torch.randn(v, t, device="cuda").to(bf)
    w_read = (torch.randn(t, h, device="cuda") / h**0.5).to(bf)
    w_recv = (torch.randn(g, t, device="cuda") / t**0.5).to(bf)
    w_pred = (torch.randn(g, t, device="cuda") / t**0.5).to(bf)
    w_msg = (torch.randn(g, t, device="cuda") / t**0.5).to(bf)
    w_write = (torch.randn(h, g, device="cuda") / g**0.5).to(bf)
    bias = (torch.randn(g, device="cuda") * 0.1).to(bf)
    norm_w = (torch.rand(h, device="cuda") + 0.5).to(bf)
    if int8_head:
        head = torch.randint(-127, 128, (v, h), device="cuda", dtype=torch.int8)
        scale = torch.rand(v, device="cuda") / 100
    else:
        head, scale = (torch.randn(v, h, device="cuda") * 0.02).to(bf), None
    ref = transitions_fn(pre, cand, pred_ids, head, scale, cb, w_read, w_recv, w_pred, w_msg, w_write, bias,
                         norm_w, t, eps, neps)
    out = score_transitions(pre, cand, anchor, w_read, w_recv, w_pred, w_msg, w_write, bias, norm_w, head, scale,
                            cb, t, eps, neps, F.linear)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref.float(), rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


def test_row_cap_default():
    """CPU: level 3 serves draft steps of at most OLALA_DSPINE_FUSED_MAX_ROWS rows (64)."""
    from vllm.model_executor.models import qwen3_dspine_fast as fp

    assert fp.HAND_MAX_ROWS == int(__import__("os").environ.get("OLALA_DSPINE_FUSED_MAX_ROWS", "64"))
    assert fp._rows_ok(fp.HAND_MAX_ROWS) and not fp._rows_ok(fp.HAND_MAX_ROWS + 1)


def test_unknown_fused_mode_warns(monkeypatch):
    """CPU: an OLALA_DSPINE_FUSED value the engine does not know is reported, not
    silently run as level 0."""
    from vllm.model_executor.models import qwen3_dspine_fast as fp

    seen = []
    monkeypatch.setattr(fp.logger, "warning", lambda msg, *args: seen.append(msg % args))
    for mode in ("0", "1", "2", "3"):
        assert fp.check_fused_mode(mode)
    assert not seen
    assert not fp.check_fused_mode("4")
    assert len(seen) == 1 and "'4'" in seen[0] and "level 0" in seen[0]

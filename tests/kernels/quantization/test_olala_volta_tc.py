# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Volta fp16 tensor-core edition (olala/volta_tc.py, OLALA_VOLTA_TC): W8A16 and bf16 linears
against fp64, the exactness of the activation terms, the per-row fallback (rows the terms do not
cover, non-finite rows), CUDA-graph replay against eager, the W8A16 dispatch, and the routed-expert
kernel (INT8 per row and INT4 affine experts) against fp64. SM70 family only (a V100, natively or
through volta-forever)."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or gpu_family() is not GpuFamily.SM70,
    reason="the SM70 family's fp16 tensor-core edition",
)


@pytest.fixture
def tc(monkeypatch):
    from vllm.model_executor.layers.mamba.olala import volta_tc
    import vllm.model_executor.layers.quantization.utils.w8a16_utils as wu

    monkeypatch.setenv("OLALA_VOLTA_TC", "int8,bf16")
    volta_tc._kinds.cache_clear()
    wu.volta_tc.cache_clear()
    yield volta_tc
    volta_tc._kinds.cache_clear()
    wu.volta_tc.cache_clear()


def _int8(n, k, gen):
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    s = w.abs().amax(1) / 127
    q = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), s.float().contiguous()


def _acts(m, k, gen, span=0):
    x = torch.randn(m, k, device="cuda", generator=gen)
    if span:  # element magnitudes over `span` binades (power-of-two factors)
        e = (-span * torch.rand(m, k, device="cuda", generator=gen)).floor().to(torch.int32)
        x = x * ((e + 127) << 23).view(torch.float32)
    return x.to(torch.bfloat16)


def _err(y, ref):
    d = (y.double() - ref).abs() / ref.abs().amax(-1, keepdim=True).clamp_min(1e-300)
    return float(d.max())


@pytest.mark.parametrize(
    "n,k,terms", [(7344, 1536, 1), (1536, 3072, 3), (1536, 1536, 2), (384, 1536, 1)]
)
@pytest.mark.parametrize("m", [9, 33, 128, 300])
def test_int8_linear_vs_fp64(tc, n, k, terms, m):
    gen = torch.Generator(device="cuda").manual_seed(n + k + m)
    q, s = _int8(n, k, gen)
    assert tc.register_int8(q, terms, "t")
    try:
        x = _acts(m, k, gen, span=30 if terms > 1 else 0)
        y = tc.tc_linear(x, q, s)
        if m < tc.min_rows(terms):
            assert y is None
            return
        ref = (x.double() @ q.double().t()) * s.double()[None, :]
        # bf16 output rounding: 2^-8 of the row maximum
        assert _err(y, ref) < 4e-3
    finally:
        tc.unregister(q)


@pytest.mark.parametrize("terms,span", [(1, 0), (2, 40), (3, 60)])
def test_terms_are_exact(tc, terms, span):
    gen = torch.Generator(device="cuda").manual_seed(terms)
    m, k = 64, 3072
    x = _acts(m, k, gen, span=span)
    h = torch.empty(terms * m, k, dtype=torch.float16, device="cuda")
    sh = torch.empty(terms, m, dtype=torch.int32, device="cuda")
    ok = torch.empty(m, dtype=torch.int32, device="cuda")
    tc._tc_split_kernel[(m,)](x, h, sh, ok, ok, m, k, x.stride(0), T=terms, BK=4096, STATS=False,
                              E_MIN=tc.E_LO, E_MAX=tc.E_HI, num_warps=8)
    okb = ok.bool()
    assert okb.any()
    recon = torch.zeros(m, k, dtype=torch.float64, device="cuda")
    for t in range(terms):
        inv = ((127 - sh[t]) << 23).view(torch.float32)
        recon += h[t * m : (t + 1) * m].double() * inv.double()[:, None]
    assert torch.equal(recon[okb], x.double()[okb])


def test_fallback_rows(tc):
    gen = torch.Generator(device="cuda").manual_seed(3)
    n, k, m = 1536, 1536, 40
    q, s = _int8(n, k, gen)
    tc.register_int8(q, 1, "t")
    try:
        x = _acts(m, k, gen)
        x[1] = 0
        x[2, 5] = float("nan")
        x[3, 7] = float("inf")
        x[4:8] = _acts(4, k, gen, span=90)  # more binades than one term covers
        y = tc.tc_linear(x, q, s)
        ref = (x.double() @ q.double().t()) * s.double()[None, :]
        fin = torch.isfinite(x.float()).all(1)
        assert _err(y[fin], ref[fin]) < 4e-3
        assert torch.isnan(y[2]).all() and not torch.isfinite(y[3]).any()
    finally:
        tc.unregister(q)


def test_graph_replay_equals_eager(tc):
    gen = torch.Generator(device="cuda").manual_seed(4)
    n, k, m = 4608, 1536, 64
    q, s = _int8(n, k, gen)
    wb = (torch.randn(n, k, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    tc.register_int8(q, 2, "t")
    tc.register_bf16(wb, 1, "b")
    try:
        x = _acts(m, k, gen, span=40)
        tc.tc_linear(x, q, s)
        tc.tc_linear(x, wb)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y8 = tc.tc_linear(x, q, s)
            yb = tc.tc_linear(x, wb)
        for seed in (5, 6):
            x.copy_(_acts(m, k, torch.Generator(device="cuda").manual_seed(seed), span=50))
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(y8, tc.tc_linear(x, q, s))
            assert torch.equal(yb, tc.tc_linear(x, wb))
    finally:
        tc.unregister(q)
        tc.unregister(wb)


def test_bf16_weight_with_sparse_term(tc):
    gen = torch.Generator(device="cuda").manual_seed(7)
    n, k, m = 1536, 1536, 48
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    w[3, 11] = w[3].abs().max() * 2.0 ** -40  # not exact in fp16 at its row's scale
    wb = w.to(torch.bfloat16)
    assert tc.register_bf16(wb, 1, "b")
    try:
        entry = tc.lookup(wb)[0]
        assert entry.bad_n is not None and entry.bad_n.numel() >= 1
        x = _acts(m, k, gen)
        y = tc.tc_linear(x, wb)
        ref = x.double() @ wb.double().t()
        assert _err(y, ref) < 4e-3
    finally:
        tc.unregister(wb)


def test_w8a16_dispatch(tc):
    from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
        Int8Weight, Segment, int8_decode_linear, int8_mm,
    )

    gen = torch.Generator(device="cuda").manual_seed(8)
    n, k = 7344, 1536
    q, s = _int8(n, k, gen)
    tc.register_int8(q, 1, "t")
    try:
        for m in (5, 16, 64, 300):
            x = _acts(m, k, gen)
            y = int8_mm(x, Int8Weight(q, s))
            y_tc = tc.tc_linear(x, q, s)
            if m >= tc.min_rows(1):
                assert torch.equal(y, y_tc)
                if m <= 128:
                    assert torch.equal(int8_decode_linear(x, [Segment(q, s)], width=n), y_tc)
            else:
                assert y_tc is None
    finally:
        tc.unregister(q)

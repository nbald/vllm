# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W8A16 INT8 kernels (quantization/utils/w8a16_utils.py) against the
dequantized weights in fp32: one row (GEMV), 2..128 rows (skinny, with and
without split-K), two segments (INT8 + bf16) written contiguously or through
column maps, split-K counters surviving CUDA graph replays, per-row and
per-group (32) scales, and the prefill GEMMs (Triton W8A16, dequantize +
cuBLAS)."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
    Int8Weight,
    Segment,
    int8_decode_linear,
    int8_dequant,
    int8_linear,
    int8_mm,
    int8_prefill_mm,
    w8a16_gemm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="bf16 mma.sync needs sm_80+",
)


def _int8(n, k, gen, group=0):
    """Symmetric INT8 (round half away from zero), per-row fp32 scales or
    per-group fp16 scales (ggml Q8_0); returns (q, scale, fp32 reference)."""
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    if group:
        wg = w.view(n, k // group, group)
        d = (wg.abs().amax(-1) / 127).half()
        q = torch.round(wg / d.float()[..., None]).clamp(-127, 127).to(torch.int8)
        ref = (q.float() * d.float()[..., None]).view(n, k)
        # the per-group kernels dequantize to bf16 before the MMA (except the GEMV)
        return q.view(n, k), d, ref
    scale = w.abs().amax(1) / 127
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale, q.float() * scale[:, None]


def _close(y, ref, tol=4e-3):
    # bf16 output rounding: 2^-8 of the largest magnitude
    err = (y.float() - ref).abs().max() / ref.abs().max()
    assert err < tol, err


@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize(
    "n,k", [(6144, 1536), (1536, 3072), (1536, 4608), (384, 1536), (1536, 384)]
)
@pytest.mark.parametrize("m", [1, 2, 5, 16, 31, 64, 100, 128])
def test_int8_linear(n, k, m, group):
    gen = torch.Generator(device="cuda").manual_seed(n + k + m + group)
    q, scale, ref_w = _int8(n, k, gen, group)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    _close(int8_linear(x, Int8Weight(q, scale)), x.float() @ ref_w.T)


@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize(
    "n,k", [(6144, 1536), (1536, 3072), (1536, 4608), (384, 1536), (1536, 384)]
)
@pytest.mark.parametrize("m", [1, 2, 3, 8, 9, 16, 32])
def test_volta_gemv_int8(n, k, m, group):
    """The FMA GEMVs that Volta takes for W8A16 decode batches (plain Triton, any GPU): per-row
    scales on the fp32 row sums, per-group scales on the weights, written into a column slice
    of a wider output."""
    from vllm.model_executor.layers.mamba.olala.decode_gemv import volta_gemv

    gen = torch.Generator(device="cuda").manual_seed(n + k + m + group + 1)
    q, scale, ref_w = _int8(n, k, gen, group)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    out = torch.zeros(m, n + 64, device="cuda", dtype=torch.float32)
    volta_gemv(x, q, torch.float32, scale, out=out[:, 64:])
    ref = x.float() @ ref_w.T
    err = (out[:, 64:] - ref).abs().max() / ref.abs().max()
    assert err < 1e-5, err
    assert out[:, :64].abs().max() == 0
    _close(volta_gemv(x, q, torch.bfloat16, scale), ref)


@pytest.mark.parametrize("split", [2, 3, 6])
@pytest.mark.parametrize("m", [3, 32])
def test_split_k(split, m):
    gen = torch.Generator(device="cuda").manual_seed(split * 100 + m)
    q, scale, ref_w = _int8(1536, 4608, gen)
    x = torch.randn(m, 4608, device="cuda", dtype=torch.bfloat16, generator=gen)
    for _ in range(3):  # the tile counters reset themselves
        y = int8_linear(x, Int8Weight(q, scale), cfg=(32, 128, split, 4, 3))
        _close(y, x.float() @ ref_w.T)


@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize("m", [1, 4, 32, 128])
def test_two_segments_graph(use_map, m, group):
    """INT8 rows + bf16 rows of one output in one launch, replayed from a CUDA
    graph with split-K."""
    gen = torch.Generator(device="cuda").manual_seed(m + 7 * use_map + group)
    n1, n2, k = 6144, 1200, 1536
    q, scale, ref1 = _int8(n1, k, gen, group)
    w2 = (torch.randn(n2, k, device="cuda", generator=gen) * 0.02).bfloat16()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    out = torch.empty(m, n1 + n2, device="cuda", dtype=torch.bfloat16)
    if use_map:
        perm = torch.randperm(n1 + n2, device="cuda", generator=gen).int()
        segs = [Segment(q, scale, cols=perm[:n1]), Segment(w2, cols=perm[n1:])]
        cols = perm.long()
    else:
        segs = [Segment(q, scale), Segment(w2, col0=n1)]
        cols = torch.arange(n1 + n2, device="cuda")
    cfg = (32, 128, 3, 4, 3) if 1 < m <= 32 else None
    int8_decode_linear(x, segs, out=out, cfg=cfg)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        int8_decode_linear(x, segs, out=out, cfg=cfg)
    out.zero_()
    for _ in range(3):
        graph.replay()
    ref = torch.empty(m, n1 + n2, device="cuda")
    ref[:, cols] = torch.cat([x.float() @ ref1.T, x.float() @ w2.float().T], 1)
    _close(out, ref)


def test_fp32_output_two_segments():
    """INT8 rows + bf16 rows written to an fp32 output (e.g. a projection fused
    with fp32 router logits)."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    q, scale, ref1 = _int8(384, 1536, gen)
    r = (torch.randn(512, 1536, device="cuda", generator=gen) * 0.02).bfloat16()
    for m in (1, 3, 16):
        x = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16, generator=gen)
        out = torch.empty(m, 384 + 512, device="cuda", dtype=torch.float32)
        int8_decode_linear(x, [Segment(q, scale), Segment(r, col0=384)], out=out)
        ref = torch.cat([x.float() @ ref1.T, x.float() @ r.float().T], 1)
        err = (out - ref).abs().max() / ref.abs().max()
        assert err < 1e-4, err


@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize("n,k", [(7344, 1536), (1536, 3072), (4608, 1536), (1536, 384)])
@pytest.mark.parametrize("m", [129, 1000, 4096])
def test_prefill(n, k, m, group):
    gen = torch.Generator(device="cuda").manual_seed(n + k + m + group)
    q, scale, ref_w = _int8(n, k, gen, group)
    w = Int8Weight(q, scale)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    ref = x.float() @ ref_w.T
    _close(w8a16_gemm(x, w), ref)
    # The cuBLAS path uses BF16-rounded weights and
    # cuBLAS's bf16 reductions, as the bf16 model's own GEMMs
    _close(int8_prefill_mm(x, w), ref, tol=8e-3)
    _close(int8_mm(x, w), ref, tol=8e-3)
    torch.testing.assert_close(int8_dequant(w), ref_w.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("k", [96, 192])
@pytest.mark.parametrize("group", [0, 32])
def test_prefill_partial_k_tile(k, group):
    """K can end inside either the dequantization or GEMM tile."""
    gen = torch.Generator(device="cuda").manual_seed(k + group)
    q, scale, ref_w = _int8(79, k, gen, group)
    w = Int8Weight(q, scale)
    x = torch.randn(129, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    torch.testing.assert_close(int8_dequant(w), ref_w.bfloat16(), rtol=0, atol=0)
    _close(w8a16_gemm(x, w), x.float() @ ref_w.T, tol=8e-3)

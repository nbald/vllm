# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala decode MoE with weight-only INT8 (W8A16) or INT4 affine (W4A16) routed
experts, shared expert and fc2_latent (latent_moe_int8.py) must match a torch
reference over the dequantized weights in fp32: per-row and per-group (32)
INT8 scales, INT4 affine groups of 32 (MLX q4 layout) for the routed experts,
bf16 shared / fc2 mixed in, small and medium paths, MMA and GEMV variants of
the small path, the FMA-only small path of Volta, the prefill grouped GEMMs."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.has_device_capability(80):
    pytest.skip("bf16 mma.sync needs sm_80+", allow_module_level=True)

from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import (  # noqa: E402
    FMT_INT4,
    WqWeight,
    _i4_tiles,
    _wq_tile_t,
    olala_moe_medium_int8,
    olala_moe_small_int8,
    olala_moe_small_int8_volta,
)
from vllm.triton_utils import tl, triton  # noqa: E402

E, TOPK, H, D1, N1, S = 256, 6, 1536, 384, 768, 1536
INT4 = "int4"  # routed experts INT4 affine g32, the rest INT8 per row (Q4A2)


def _roundf(x):
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def _make(w: torch.Tensor, group: int):
    """build_q8.py recipe. w [..., N, K] fp32 -> (WqWeight, dequantized fp32)."""
    if group == 0:
        s = (w.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-30)
        q = _roundf(w / s).clamp(-127, 127)
        return WqWeight(
            q.to(torch.int8).contiguous(), s.squeeze(-1).contiguous()
        ), q * s
    k = w.shape[-1]
    wg = w.reshape(*w.shape[:-1], k // group, group)
    d = wg.abs().amax(-1, keepdim=True) / 127
    q = _roundf(wg / d).clamp(-127, 127)
    dh = d.half()
    deq = (q * dh.float()).reshape(w.shape)
    # the MMA paths dequantize the group tiles to bf16
    return (
        WqWeight(
            q.to(torch.int8).reshape(w.shape).contiguous(), dh.squeeze(-1).contiguous()
        ),
        deq.bfloat16().float(),
    )


def _mlx_affine_quantize(w: torch.Tensor, group: int = 32, bits: int = 4):
    """mlx.core.quantize (affine mode, ops.cpp affine_quantize) from its
    definition: per group of ``group`` along K, the larger-magnitude extreme is
    the edge (exactly representable), q = clip(round((w - b) / s), 0, 15).
    Returns (packed uint8 [..., N, K / 2] = MLX's uint32 words as little-endian
    bytes), BF16 scales and biases, and the reference dequantized weights."""
    n_bins = (1 << bits) - 1
    wg = w.reshape(*w.shape[:-1], w.shape[-1] // group, group).float()
    w_max, w_min = wg.amax(-1, keepdim=True), wg.amin(-1, keepdim=True)
    mask = w_min.abs() > w_max.abs()
    scale = ((w_max - w_min) / n_bins).clamp(min=1e-7)
    scale = torch.where(mask, scale, -scale)
    edge = torch.where(mask, w_min, w_max)
    q0 = torch.round(edge / scale)
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    sb, bb = scale.bfloat16(), bias.bfloat16()
    q = torch.clamp(torch.round((wg - bb.float()) / sb.float()), 0, n_bins).to(
        torch.int64
    )
    q = q.reshape(w.shape)
    # MLX packs 32 // bits codes per uint32 (least significant code first)
    words = q.reshape(*w.shape[:-1], w.shape[-1] // 8, 8)
    words = (words << (bits * torch.arange(8, device=w.device))).sum(-1).to(torch.int32)
    packed = words.contiguous().view(torch.uint8)
    deq = sb.float() * q.reshape(wg.shape).float() + bb.float()
    return (
        packed,
        sb.squeeze(-1).contiguous(),
        bb.squeeze(-1).contiguous(),
        deq.reshape(w.shape).bfloat16().float(),
    )


def _make_int4(w: torch.Tensor):
    packed, sb, bb, deq = _mlx_affine_quantize(w)
    return WqWeight.int4_affine(packed, sb, bb), deq


def _make_expert(w: torch.Tensor, group):
    return _make_int4(w) if group == INT4 else _make(w, group)


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x.bfloat16().float()).square().bfloat16()


def _reference(hidden, latent, logits, bias, w13, w2, su, sd, fc2):
    M = hidden.shape[0]
    sc = torch.sigmoid(logits)
    ids = torch.topk(sc + bias, TOPK, dim=-1).indices
    tw = sc.gather(1, ids)
    tw = tw / tw.sum(-1, keepdim=True)
    lat = latent.bfloat16().float()
    routed = torch.zeros(M, D1, device=hidden.device)
    for m in range(M):
        for k in range(TOPK):
            e = ids[m, k]
            h = _relu2(lat[m] @ w13[e].T)
            routed[m] += tw[m, k] * (h.float() @ w2[e].T)
    routed = (routed * 2.45).bfloat16()
    s = _relu2(hidden.float() @ su.T)
    return (routed.float() @ fc2.T + s.float() @ sd.T).bfloat16()


def _setup(m: int, group, dense_bf16: bool):
    torch.manual_seed(m + (1000 if group == INT4 else group))
    dev = "cuda"
    hidden = torch.randn(m, H, device=dev).bfloat16()
    latent = torch.randn(m, D1, device=dev)  # fp32, as the entry launch writes it
    logits = torch.randn(m, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    w13, dq13 = _make_expert(torch.randn(E, N1, D1, device=dev) * D1**-0.5, group)
    w2, dq2 = _make_expert(torch.randn(E, D1, N1, device=dev) * N1**-0.5, group)
    dense_group = 0 if group == INT4 else group
    if dense_bf16:
        su_b = (torch.randn(S, H, device=dev) * H**-0.5).bfloat16()
        sd_b = (torch.randn(H, S, device=dev) * S**-0.5).bfloat16()
        f2_b = (torch.randn(H, D1, device=dev) * D1**-0.5).bfloat16()
        su, sd, fc2 = WqWeight(su_b), WqWeight(sd_b), WqWeight(f2_b)
        dqsu, dqsd, dqf2 = su_b.float(), sd_b.float(), f2_b.float()
    else:
        su, dqsu = _make(torch.randn(S, H, device=dev) * H**-0.5, dense_group)
        sd, dqsd = _make(torch.randn(H, S, device=dev) * S**-0.5, dense_group)
        fc2, dqf2 = _make(torch.randn(H, D1, device=dev) * D1**-0.5, dense_group)
    ref = _reference(hidden, latent, logits, bias, dq13, dq2, dqsu, dqsd, dqf2)
    return (hidden, latent, logits, bias, w13, w2, su, sd, fc2), ref


@pytest.mark.parametrize("m", [1, 2, 3, 5, 12])
@pytest.mark.parametrize("group", [0, 32, INT4])
@pytest.mark.parametrize("gemv", [False, True])
@pytest.mark.parametrize("dense_bf16", [False, True])
def test_small_path_matches_reference(m, group, gemv, dense_bf16):
    args, ref = _setup(m, group, dense_bf16)
    out = olala_moe_small_int8(*args, TOPK, 2.45, gemv=gemv)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 12, 17, 32])
@pytest.mark.parametrize("group", [0, 32, INT4])
@pytest.mark.parametrize("dense_bf16", [False, True])
def test_volta_small_path_matches_reference(m, group, dense_bf16):
    """The Volta small path (FMA kernels only; plain Triton, so it runs on any GPU): routed
    group and INT4 weights stay fp32 after dequantization (as in the GEMV variant), within bf16
    rounding of the reference."""
    args, ref = _setup(m, group, dense_bf16)
    out = olala_moe_small_int8_volta(*args, TOPK, 2.45)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m", [6, 33, 128, 512])
@pytest.mark.parametrize("group", [0, 32, INT4])
def test_medium_path_matches_reference(m, group):
    args, ref = _setup(m, group, False)
    out = olala_moe_medium_int8(*args, TOPK, 2.45)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m", [1, 100, 1000, 3000])
@pytest.mark.parametrize("group", [0, 32, INT4])
def test_prefill_routed_experts(m, group):
    """wq_routed_experts (sorted grouped GEMMs, any batch) = the routed part of
    the reference."""
    from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import wq_routed_experts

    torch.manual_seed(m + (1000 if group == INT4 else group))
    dev = "cuda"
    latent = torch.randn(m, D1, device=dev).bfloat16()
    logits = torch.randn(m, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    w13, dq13 = _make_expert(torch.randn(E, N1, D1, device=dev) * D1**-0.5, group)
    w2, dq2 = _make_expert(torch.randn(E, D1, N1, device=dev) * N1**-0.5, group)
    out = wq_routed_experts(latent, logits, bias, w13, w2, TOPK, 2.45)
    sc = torch.sigmoid(logits)
    ids = torch.topk(sc + bias, TOPK, dim=-1).indices
    tw = sc.gather(1, ids)
    tw = tw / tw.sum(-1, keepdim=True)
    ref = torch.zeros(m, D1, device=dev)
    for e in torch.unique(ids).tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        h = _relu2(latent[rows].float() @ dq13[e].T)
        ref.index_add_(0, rows, tw[rows, slots][:, None] * (h.float() @ dq2[e].T))
    torch.testing.assert_close(
        out.float(), (ref * 2.45).bfloat16().float(), atol=3e-2, rtol=3e-2
    )


# -- INT4 affine: the kernels' dequantized tiles, bit for bit -------------------


@triton.jit
def _deq_tiles_kernel(
    w_ptr,
    s_ptr,
    out_ptr,
    out2_ptr,
    N: tl.constexpr,
    KD: tl.constexpr,
    FMT: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    """Out = the MMA tiles (_wq_tile_t), out2 = the GEMV half tiles (_i4_tiles)
    re-interleaved, both bf16 [N, KD]."""
    offs_n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_h = tl.arange(0, BK // 2)
    for k0 in tl.range(0, KD, BK):
        w = _wq_tile_t(w_ptr, s_ptr, offs_n, k0, KD, FMT, G, BK, BN)
        tl.store(out_ptr + offs_n[None, :] * KD + (k0 + offs_k)[:, None], w)
        lo, hi = _i4_tiles(w_ptr, s_ptr, offs_n, k0, KD, G, BK, BN)
        tl.store(out2_ptr + offs_n[None, :] * KD + (k0 + 2 * offs_h)[:, None], lo)
        tl.store(out2_ptr + offs_n[None, :] * KD + (k0 + 2 * offs_h + 1)[:, None], hi)


@pytest.mark.parametrize("shape", [(N1, D1), (D1, N1)])
@pytest.mark.parametrize("bk", [32, 64, 128])
def test_int4_tiles_bit_exact(shape, bk):
    """The kernels' bf16 weights of an MLX-layout INT4 matrix (codes packed by
    mlx.quantize's uint32 rule, bf16 scale and bias) equal bf16(s * q + b)."""
    torch.manual_seed(bk)
    n, k = shape
    w = torch.randn(n, k, device="cuda") * k**-0.5
    w[:, :32] = (
        w[:, :32].abs() + 0.01
    )  # a group without 0 (bias = min, not an exact zero point)
    packed, sb, bb, ref = _mlx_affine_quantize(w)
    wq = WqWeight.int4_affine(packed, sb, bb)
    assert wq.fmt == FMT_INT4 and (wq.n, wq.k, wq.group) == (n, k, 32)
    out = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    out2 = torch.empty_like(out)
    _deq_tiles_kernel[(n // 32,)](
        *wq.args(), out, out2, N=n, KD=k, FMT=wq.fmt, G=wq.group, BK=bk, BN=32
    )
    assert torch.equal(out.float(), ref)
    assert torch.equal(out2.float(), ref)


@pytest.mark.parametrize(
    "m,run", [(4, olala_moe_small_int8), (33, olala_moe_medium_int8)]
)
def test_split_router_logits(m, run):
    """Router logits passed as two strided slices of the entry projection
    (``router_logits + router_logits_lo``) give the same output as their sum."""
    args, _ = _setup(m, 0, False)
    hidden, latent, logits, *rest = args
    hi, lo = torch.empty((m, 2 * E), device=logits.device).split(E, dim=1)
    hi.copy_(logits.bfloat16())
    lo.copy_(logits - hi)
    ref = run(hidden, latent, hi + lo, *rest, TOPK, 2.45)
    out = run(hidden, latent, hi, *rest, TOPK, 2.45, router_logits_lo=lo)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# -- Volta multi-row MoE (volta_cuda.moe_rows: up to 32 tokens in three launches, dispatched from 3 to 8) ---------


def _rows_setup(m: int, group, seed: int):
    """A layer as the rows path reads it: INT8 per-row fc1_latent, shared expert and fc2_latent, the router as bf16
    hi + lo rows (bf16x2), routed experts INT8 per row (group 0) or INT4 affine; bf16 hidden rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, device="cuda", generator=g)

    w13, dq13 = _make_expert(rn(E, N1, D1) * D1**-0.5, group)
    w2, dq2 = _make_expert(rn(E, D1, N1) * N1**-0.5, group)
    su, dqsu = _make(rn(S, H) * H**-0.5, 0)
    sd, dqsd = _make(rn(H, S) * S**-0.5, 0)
    fc2, dqf2 = _make(rn(H, D1) * D1**-0.5, 0)
    fc1, dqf1 = _make(rn(D1, H) * H**-0.5, 0)
    router = rn(E, H) * 0.03
    hi = router.bfloat16()
    rhl = torch.cat([hi, (router - hi.float()).bfloat16()]).contiguous()
    bias = rn(E) * 0.05
    hidden = rn(m, H).bfloat16()
    dq = (dqf1, dq13, dq2, dqsu, dqsd, dqf2, hi.float() + rhl[E:].float())
    return (hidden, rhl, fc1, bias, w13, w2, su, sd, fc2), dq


def _rows_reference(hidden, bias, dq, scale=2.45):
    """fp32 over the dequantized weights with torch's routing of the fp32 logits; bf16 roundings where the kernels
    round (latent, relu^2 outputs, routed latent, output)."""
    f1, w13, w2, su, sd, f2, router = dq
    h = hidden.float()
    sc = torch.sigmoid(h @ router.T)
    ids = torch.topk(sc + bias, TOPK, dim=-1).indices
    tw = sc.gather(1, ids)
    tw = tw / tw.sum(-1, keepdim=True)
    lat = (h @ f1.T).bfloat16().float()
    routed = torch.zeros(h.shape[0], D1, device=h.device)
    for i in range(h.shape[0]):
        for k in range(TOPK):
            e = int(ids[i, k])
            routed[i] += tw[i, k] * (_relu2(lat[i] @ w13[e].T).float() @ w2[e].T)
    routed = (routed * scale).bfloat16().float()
    s = _relu2(h @ su.T).float()
    return (routed @ f2.T + s @ sd.T).bfloat16()


def _rows_base(args, scale=2.45):
    """The current small path on the same layer: the entry launch (INT8 fc1 + bf16 router rows) and
    olala_moe_small_int8_volta."""
    from vllm.model_executor.layers.quantization.utils.w8a16_utils import Segment, int8_decode_linear

    hidden, rhl, fc1, bias, w13, w2, su, sd, fc2 = args
    m = hidden.shape[0]
    y = torch.empty(m, D1 + 2 * E, dtype=torch.float32, device=hidden.device)
    int8_decode_linear(hidden, [Segment(fc1.weight, fc1.scale), Segment(rhl, col0=D1)], out=y)
    return olala_moe_small_int8_volta(hidden, y[:, :D1], y[:, D1:D1 + E], bias, w13, w2, su, sd, fc2, TOPK, scale,
                                      router_logits_lo=y[:, D1 + E:])


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("m", [2, 3, 4, 5, 8, 12, 16, 17, 24, 32])
@pytest.mark.parametrize("group", [0, INT4])
def test_volta_rows_path_matches_reference(m, group):
    """The rows path (Volta multi-row MoE: entry with routing, experts by slice, exit) within the current small path's
    error of the fp32 reference, deterministic, its counters back to zero; a 2 % routed-scale error must not pass
    the same check (control)."""
    from vllm.model_executor.layers.mamba.olala import volta_cuda

    args, dq = _rows_setup(m, group, seed=100 + m)
    hidden, rhl, fc1, bias, w13, w2, su, sd, fc2 = args
    ref = _rows_reference(hidden, bias, dq)
    base = _rows_base(args)
    out = volta_cuda.moe_rows(*args, 2.45)
    again = volta_cuda.moe_rows(*args, 2.45)
    rel, rel_base = _rel(out, ref), _rel(base, ref)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    assert rel <= 1.5 * rel_base + 1e-3, (rel, rel_base)
    assert torch.equal(out, again)
    _, ctr = volta_cuda.moe_rows_buffers(hidden.device)
    assert int(ctr.abs().sum()) == 0
    wrong = volta_cuda.moe_rows(*args, 2.45 * 1.02)
    assert _rel(wrong, ref) > 1.5 * rel_base + 1e-3


@pytest.mark.parametrize("m", [1, 2, 3, 8, 9, 16, 32])
def test_volta_rows_path_gate(m, monkeypatch):
    """moe_rows_ok: 3..8 tokens of the model's layouts with INT8 routed experts, and only with OLALA_VOLTA_MOE_ROWS=1
    on the SM70 family; INT4 affine experts stay on the small path."""
    from vllm.model_executor.layers.mamba.olala import volta_cuda
    from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family

    args, _ = _rows_setup(m, 0, seed=7)
    args4, _ = _rows_setup(m, INT4, seed=7)
    monkeypatch.delenv("OLALA_VOLTA_MOE_ROWS", raising=False)
    assert not volta_cuda.moe_rows_ok(m, *args, TOPK)
    monkeypatch.setenv("OLALA_VOLTA_MOE_ROWS", "1")
    volta = gpu_family() is GpuFamily.SM70
    assert volta_cuda.moe_rows_ok(m, *args, TOPK) == (volta and 3 <= m <= 8)
    assert not volta_cuda.moe_rows_ok(m, *args4, TOPK)
    hidden, rhl, *rest = args
    assert not volta_cuda.moe_rows_ok(m, hidden.float(), rhl, *rest, TOPK)  # fp32 rows

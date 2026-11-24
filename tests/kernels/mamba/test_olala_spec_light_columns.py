# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Light speculative columns (OLALA_SPEC_LIGHT, layers/mamba/olala/state.py)
against the column-slot protocol.

Light layout: one state page per request, the fp32 SSM state once and 1 + k
columns of the small pieces; the verify kernel commits last step's accepted
drafts before verifying. Column-slot layout: 1 + k state pages per request,
acceptance selects one. Over several consecutive verify steps with random
ragged query lengths and random acceptances, both must give bit-identical
outputs, and the committed state (the light page's column 0 after a final
one-token step against the column-slot page written by that step) must be
bit-identical too. The light pools are strided views (padded slot stride),
like the pages of the hybrid allocator.
"""

import pytest
import torch
from vllm.model_executor.layers.mamba.olala.diff_tpa_decode import tpa_decode_qkv_spec
from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import mamba3_verify_cuda

requires_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80+ GPU",
)

H, D, S, R, NA = 48, 64, 128, 4, 32


def _strided(shape, dtype, pad_cols=1, fill=None):
    """(P, C, ...) view of a (P, C + pad_cols, ...) buffer: slot stride larger
    than C columns."""
    P, C, *rest = shape
    buf = torch.zeros(P, C + pad_cols, *rest, device="cuda", dtype=dtype)
    v = buf[:, :C]
    if fill is not None:
        v.copy_(fill)
    return v


def _step_inputs(qlens, k):
    bf = torch.bfloat16
    lanes = len(qlens)
    qsl = torch.zeros(lanes + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(qlens), 0)
    N = int(qsl[-1]) + 2  # CUDA-graph padding rows past the last request
    return (
        qsl,
        N,
        dict(
            A=-torch.rand(N, H, device="cuda") * 2 - 0.05,
            dt=torch.rand(N, H, device="cuda") * 0.3 + 0.01,
            trap=torch.rand(N, H, device="cuda"),
            BC=torch.randn(N, 2, R, S, device="cuda").to(bf),
            x=torch.randn(N, H, D, device="cuda").to(bf),
            z=torch.randn(N, H, D, device="cuda").to(bf),
            angp=torch.randn(N, NA + 8, device="cuda").to(bf)[:, 8:],
        ),
    )


@requires_sm80
@pytest.mark.parametrize("lanes,k,steps", [(1, 3, 5), (6, 3, 6), (37, 7, 4), (3, 5, 6)])
def test_mamba3_light_matches_column_slots(lanes, k, steps):
    torch.manual_seed(lanes * 100 + k)
    bf = torch.bfloat16
    C = 1 + k
    P = lanes * C + 4
    cols = (
        (torch.randperm(P - 1)[: lanes * C] + 1).to(torch.int32).view(lanes, C).cuda()
    )
    slots = torch.arange(1, lanes + 1, dtype=torch.int32, device="cuda").view(lanes, 1)
    PL = lanes + 3
    xpj = torch.rand(R, H, D, device="cuda") * 0.5
    zpj = torch.rand(R, H, D, device="cuda")
    opj = torch.rand(R, H, D, device="cuda") * 0.5
    bq = torch.randn(R, H, S, device="cuda") * 0.1 + 1
    bk = torch.randn(R, H, S, device="cuda") * 0.1 + 1
    Dp = torch.randn(H, device="cuda")

    # column-slot pools (reference) and light pools, same initial committed state
    ang_c = torch.rand(P, H, NA, device="cuda") * 6.28
    ssm_c = torch.randn(P, H, D, S, device="cuda") * 0.5
    k_c = torch.randn(P, R, H, S, device="cuda").to(bf)
    v_c = torch.randn(P, H, D, device="cuda").to(bf)
    ang_l = _strided((PL, C, H, NA), torch.float32)
    ssm_l = torch.zeros(PL, H, D, S, device="cuda")
    k_l = _strided((PL, C, R, H, S), bf, pad_cols=2)
    v_l = _strided((PL, C, H, D), bf)
    agb_l = _strided((PL, C, H, 4), torch.float32)
    c0 = cols[:, 0].long()
    s0 = slots[:, 0].long()
    ang_l[s0, 0], ssm_l[s0], k_l[s0, 0], v_l[s0, 0] = (
        ang_c[c0],
        ssm_c[c0],
        k_c[c0],
        v_c[c0],
    )
    num_acc = torch.ones(lanes, dtype=torch.int32, device="cuda")

    for step in range(steps + 1):
        final = step == steps
        if final:
            qlens = (
                [1] * lanes
            )  # one more position: both layouts then hold the committed state + 1 token
        else:
            qlens = torch.randint(1, C + 1, (lanes,)).tolist()
            qlens[0] = C
        qsl, N, inp = _step_inputs(qlens, k)
        qsl = qsl.cuda()
        y_c = torch.full((N, H, D), float("nan"), device="cuda", dtype=bf)
        y_l = torch.full((N, H, D), float("nan"), device="cuda", dtype=bf)
        common = (
            inp["A"],
            inp["BC"][:, 0],
            inp["BC"][:, 1],
            Dp,
            inp["x"],
            inp["dt"],
            inp["trap"],
            xpj,
            zpj,
            opj,
            inp["z"],
            bq,
            bk,
            inp["angp"],
            qsl,
        )
        mamba3_verify_cuda(ssm_c, k_c, v_c, ang_c, *common, cols, num_acc, y_c)
        mamba3_verify_cuda(
            ssm_l, k_l, v_l, ang_l, *common, slots, num_acc, y_l, agb=agb_l
        )
        torch.accelerator.synchronize()
        assert torch.equal(y_l, y_c), f"step {step}: outputs differ"
        # random acceptance in [1, qlen]: the next step starts from column num_acc - 1
        num_acc = torch.tensor(
            [int(torch.randint(1, q + 1, ())) for q in qlens],
            dtype=torch.int32,
            device="cuda",
        )

    # after the final one-token step: column 0 of both layouts holds the same state
    assert torch.equal(ssm_l[s0], ssm_c[c0])
    assert torch.equal(ang_l[s0, 0], ang_c[c0])
    assert torch.equal(k_l[s0, 0], k_c[c0])
    assert torch.equal(v_l[s0, 0], v_c[c0])


@requires_sm80
def test_mamba3_light_invalid_slot_and_padding_lane():
    """A lane without a valid slot gets zero outputs and writes nothing; a
    zero-length lane is skipped."""
    torch.manual_seed(1)
    bf = torch.bfloat16
    k = 3
    C = 1 + k
    PL = 6
    qlens = [C, 0, 2]
    qsl, N, inp = _step_inputs(qlens, k)
    slots = torch.tensor([[1], [2], [-1]], dtype=torch.int32, device="cuda")
    ang = _strided(
        (PL, C, H, NA), torch.float32, fill=torch.rand(PL, C, H, NA, device="cuda")
    )
    ssm = torch.randn(PL, H, D, S, device="cuda")
    kp = _strided(
        (PL, C, R, H, S), bf, fill=torch.randn(PL, C, R, H, S, device="cuda").to(bf)
    )
    vp = _strided(
        (PL, C, H, D), bf, fill=torch.randn(PL, C, H, D, device="cuda").to(bf)
    )
    agb = _strided(
        (PL, C, H, 4), torch.float32, fill=torch.rand(PL, C, H, 4, device="cuda")
    )
    before = [t.clone() for t in (ang, ssm, kp, vp, agb)]
    y = torch.full((N, H, D), float("nan"), device="cuda", dtype=bf)
    mamba3_verify_cuda(
        ssm,
        kp,
        vp,
        ang,
        inp["A"],
        inp["BC"][:, 0],
        inp["BC"][:, 1],
        torch.randn(H, device="cuda"),
        inp["x"],
        inp["dt"],
        inp["trap"],
        torch.rand(R, H, D, device="cuda"),
        torch.rand(R, H, D, device="cuda"),
        torch.rand(R, H, D, device="cuda"),
        inp["z"],
        torch.ones(R, H, S, device="cuda"),
        torch.ones(R, H, S, device="cuda"),
        inp["angp"],
        qsl.cuda(),
        slots,
        torch.tensor([2, 1, 1], dtype=torch.int32, device="cuda"),
        y,
        agb=agb,
    )
    torch.accelerator.synchronize()
    n0, n2 = int(qsl[2]), int(qsl[3])
    assert torch.isfinite(y[:n0].float()).all()
    assert torch.equal(y[n0:n2], torch.zeros_like(y[n0:n2]))  # invalid slot -> zeros
    assert torch.equal(y[n2:], torch.zeros_like(y[n2:]))  # padding rows -> zeros
    for got, orig in zip((ang, ssm, kp, vp, agb), before):
        assert torch.equal(
            got[2:], orig[2:]
        )  # slot 2 (zero-length lane) and beyond untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("lanes,k,steps", [(1, 3, 4), (5, 3, 5), (9, 7, 3)])
def test_tpa_light_matches_column_slots(lanes, k, steps):
    torch.manual_seed(7 + lanes)
    bf = torch.bfloat16
    HKV, GQ, RK, DD = 12, 4, 4, 128
    C = 1 + k
    P = lanes * C + 4
    dev = "cuda"
    cols = (
        (torch.randperm(P - 1)[: lanes * C] + 1).to(torch.int32).view(lanes, C).to(dev)
    )
    slots = torch.arange(1, lanes + 1, dtype=torch.int32, device=dev).view(lanes, 1)
    PL = lanes + 2
    kp_c = torch.randn(P, HKV, DD, device=dev).to(bf)
    vp_c = torch.randn(P, HKV, DD, device=dev).to(bf)
    kp_l = _strided((PL, C, HKV, DD), bf)
    vp_l = kp_l.new_zeros(PL, C + 1, HKV, DD)[:, :C]  # same strides as kp_l
    s0, c0 = slots[:, 0].long(), cols[:, 0].long()
    kp_l[s0, 0], vp_l[s0, 0] = kp_c[c0], vp_c[c0]
    q_w = torch.randn(DD, device=dev).to(bf) * 0.1
    k_w = torch.randn(DD, device=dev).to(bf) * 0.1
    scaler = torch.rand(HKV * GQ, device=dev).to(bf) + 0.5
    args = (q_w, k_w, 1e-6, scaler, 65536.0, HKV, GQ, RK, DD)
    num_acc = torch.ones(lanes, dtype=torch.int32, device=dev)
    pos0 = torch.randint(0, 50, (lanes,))
    for step in range(steps + 1):
        qlens = (
            [1] * lanes if step == steps else torch.randint(1, C + 1, (lanes,)).tolist()
        )
        qsl = torch.zeros(lanes + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(torch.tensor(qlens), 0)
        N = int(qsl[-1]) + 2
        positions = torch.zeros(N, dtype=torch.long)
        for r in range(lanes):
            positions[int(qsl[r]) : int(qsl[r + 1])] = torch.arange(
                int(pos0[r]), int(pos0[r]) + qlens[r]
            )
        positions = positions.to(dev)
        ins = [
            torch.randn(N, n, device=dev).to(bf)
            for n in (HKV * GQ * DD, HKV * RK, HKV * RK, RK * DD, RK * DD, HKV, HKV)
        ]
        out_c = tpa_decode_qkv_spec(
            *ins, positions, qsl.to(dev), cols, num_acc, kp_c, vp_c, *args
        )
        out_l = tpa_decode_qkv_spec(
            *ins, positions, qsl.to(dev), slots, num_acc, kp_l, vp_l, *args
        )
        n_tok = int(qsl[-1])
        for a, b in zip(out_l, out_c):
            assert torch.equal(a[:n_tok], b[:n_tok]), f"step {step}"
        num_acc = torch.tensor(
            [int(torch.randint(1, q + 1, ())) for q in qlens],
            dtype=torch.int32,
            device=dev,
        )
        pos0 = pos0 + num_acc.cpu()
    assert torch.equal(kp_l[s0, 0], kp_c[c0]) and torch.equal(vp_l[s0, 0], vp_c[c0])

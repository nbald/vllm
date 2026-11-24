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
from vllm.model_executor.layers.mamba.olala import state as st
from vllm.model_executor.layers.mamba.olala.diff_tpa_decode import tpa_decode_qkv_spec
from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import mamba3_verify_cuda

# The Mamba-3 light-column commit: the Volta edition of the verify kernel (SM70, natively or through
# volta-forever, which reports sm_80) implements the column-slot protocol only and refuses light pools.
requires_light_verify = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0) or st._sm70(),
    reason="light-column Mamba-3 verify: tensor-core edition only (column slots on SM70)",
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


@requires_light_verify
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


@requires_light_verify
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


# ---------------------------------------------------------------------------
# TPA factor cache in verify (OLALA_TPA_FACTOR_SPEC, F1 phase (a))
# ---------------------------------------------------------------------------


def _factor_spec_inputs(lanes, k, N, P, dev):
    from vllm.model_executor.layers.mamba.olala.tpa_factor import HEAD_DIM, HKV, HQ, RANK

    bf = torch.bfloat16
    C = 1 + k
    ins = [torch.randn(N, n, device=dev).to(bf) for n in
           (HQ * HEAD_DIM, HKV * RANK, HKV * RANK, RANK * HEAD_DIM, RANK * HEAD_DIM, HKV, HKV)]
    pools0 = [torch.randn(P, HKV, HEAD_DIM, device=dev).to(bf), torch.randn(P, HKV, HEAD_DIM, device=dev).to(bf),
              torch.randn(P, HKV, RANK, device=dev).to(bf), torch.randn(P, HKV, RANK, device=dev).to(bf)]
    q_w = torch.randn(HEAD_DIM, device=dev).to(bf) * 0.1
    k_w = torch.randn(HEAD_DIM, device=dev).to(bf) * 0.1
    scaler = torch.rand(HQ, device=dev).to(bf) + 0.5
    cols = (torch.randperm(P - 1)[: lanes * C] + 1).to(torch.int32).view(lanes, C).to(dev)
    num_acc = torch.randint(1, C + 1, (lanes,), dtype=torch.int32).to(dev)
    return ins, pools0, (q_w, k_w, 1e-6, scaler, 65536.0), cols, num_acc


def _chained_columns(cols, num_acc, qlens, qsl, t):
    """Rows of position t and the (source, destination) pool columns of the column-slot chain."""
    init = cols.gather(1, (num_acc.long() - 1).clamp(min=0)[:, None]).squeeze(1)
    rows, src, dst, reqs = [], [], [], []
    for r in range(cols.shape[0]):
        if t < int(qlens[r]):
            rows.append(int(qsl[r]) + t)
            src.append(int(init[r]) if t == 0 else int(cols[r, t - 1]))
            dst.append(int(cols[r, t]))
            reqs.append(r)
    return rows, src, dst, reqs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("lanes,k,light", [(1, 3, False), (5, 3, False), (4, 3, True), (7, 7, True)])
def test_factor_write_spec_matches_chained_decode_write(lanes, k, light):
    """factor_decode_write_spec (verify rows, OLALA_TPA_FACTOR_SPEC) against factor_decode_write chained over
    the positions with the column-slot protocol: q, factor rows and token-shift buffers bit-identical. With
    ``light`` the spec side uses in-page light columns and is compared to the column-slot chain."""
    from vllm.model_executor.layers.mamba.olala.tpa_factor import (
        HEAD_DIM, HKV, HQ, ROWP, factor_decode_write, factor_decode_write_spec)

    torch.manual_seed(11 + lanes + k)
    bf = torch.bfloat16
    dev = "cuda"
    C = 1 + k
    P = lanes * C + 4
    GQ = HQ // HKV
    qlens = torch.randint(1, C + 1, (lanes,))
    qlens[0] = C
    if lanes > 2:
        qlens[1] = 0
    qsl = torch.zeros(lanes + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(qlens, 0)
    n_tok = int(qsl[-1])
    N = n_tok + 2
    ins, pools0, tail, cols, num_acc = _factor_spec_inputs(lanes, k, N, P, dev)
    positions = torch.zeros(N, dtype=torch.long)
    for r in range(lanes):
        p0 = int(torch.randint(0, 40, ()))
        positions[int(qsl[r]): int(qsl[r + 1])] = torch.arange(p0, p0 + int(qlens[r]))
    positions = positions.to(dev)
    BS, NB = 48, 8
    cache_slots = torch.full((N,), -1, dtype=torch.int64)
    cache_slots[:n_tok] = torch.randperm(NB * BS)[:n_tok]
    cache_slots = cache_slots.to(dev)

    def planes():
        return (torch.zeros(NB, BS, ROWP, device=dev, dtype=bf), torch.zeros(NB, BS, ROWP, device=dev, dtype=bf))

    # reference: factor_decode_write per position (in place on its pool slot: copy the source column first)
    ref = [p.clone() for p in pools0]
    kp_r, vp_r = planes()
    q_ref = torch.zeros(N, HQ * HEAD_DIM, device=dev, dtype=bf)
    for t in range(C):
        rows, src, dst, _ = _chained_columns(cols, num_acc, qlens, qsl, t)
        if not rows:
            continue
        ri = torch.tensor(rows, device=dev)
        s_t, d_t = torch.tensor(src, device=dev).long(), torch.tensor(dst, device=dev).long()
        for p in ref:
            p[d_t] = p[s_t].clone()
        q_ref[ri] = factor_decode_write(*(x[ri] for x in ins), positions[ri], d_t, cache_slots[ri], *ref,
                                        *tail, kp_r, vp_r, GQ)

    kp_s, vp_s = planes()
    if light:
        # light columns: (slots, C, ...) views; slot r + 1 holds request r, column c0 its current buffer
        PL = lanes + 2
        lp = [p.new_zeros(PL, C + 1, *p.shape[1:])[:, :C] for p in pools0]
        slots = torch.arange(1, lanes + 1, dtype=torch.int32, device=dev).view(lanes, 1)
        c0 = (num_acc.long() - 1).clamp(min=0)
        init = cols.gather(1, c0[:, None]).squeeze(1)
        for p, p0_ in zip(lp, pools0):
            p[slots[:, 0].long(), c0] = p0_[init.long()]
        q_s = factor_decode_write_spec(*ins, positions, qsl.to(dev), slots, num_acc, cache_slots, *lp, *tail,
                                       kp_s, vp_s, GQ)
    else:
        sp = [p.clone() for p in pools0]
        q_s = factor_decode_write_spec(*ins, positions, qsl.to(dev), cols, num_acc, cache_slots, *sp, *tail,
                                       kp_s, vp_s, GQ)
    assert torch.equal(q_s[:n_tok], q_ref[:n_tok])
    assert torch.isfinite(q_s[n_tok:].float()).all()
    assert torch.equal(kp_s, kp_r) and torch.equal(vp_s, vp_r)
    for r in range(lanes):
        for t in range(int(qlens[r])):
            for i in range(4):
                got = lp[i][r + 1, t] if light else sp[i][int(cols[r, t])]
                assert torch.equal(got, ref[i][int(cols[r, t])]), (r, t, i)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_factor_virtual_decode_rows():
    """One single-token row per verify token: request r's block-table row, seq_len = seq_lens[r] - qlen[r] + t + 1;
    padding rows past the last request point at request 0 with seq_len 1."""
    from vllm.model_executor.layers.mamba.olala.tpa_factor import OlalaTPAFactorMetadata

    dev = "cuda"
    qsl = torch.tensor([0, 4, 5, 8, 9], dtype=torch.int32, device=dev)  # 3 decode reqs (4, 1, 3), then a prefill
    seq_lens = torch.tensor([100, 7, 30, 50], dtype=torch.int32, device=dev)
    bt = torch.arange(4 * 6, dtype=torch.int32, device=dev).view(4, 6)
    md = OlalaTPAFactorMetadata(num_decodes=3, num_prefills=1, num_decode_tokens=10, num_prefill_tokens=1,
                                num_actual_tokens=9, block_table=bt, seq_lens=seq_lens, slot_mapping=None,
                                max_seq_len=100, max_query_len=4, query_start_loc=qsl)
    rows, sl = md.virtual_decode_rows()
    want_req = [0, 0, 0, 0, 1, 2, 2, 2, 0, 0]
    want_sl = [97, 98, 99, 100, 7, 28, 29, 30, 1, 1]
    assert sl.tolist() == want_sl
    assert torch.equal(rows, bt[torch.tensor(want_req, device=dev)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("lanes,k,nsplit", [(1, 3, 1), (4, 3, 4), (6, 7, 8)])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
def test_factor_verify_matches_sequential_decode(lanes, k, nsplit, softcap):
    """A verify step through the factor cache (factor_decode_write_spec, then factor_decode_attention on the
    virtual decode rows) against the same tokens decoded one position at a time through the non-spec decode
    path (factor_decode_write on the column-slot chain, factor_decode_attention on the request's row with
    seq_len = context + t + 1). Same KV splits, so attention outputs and factor planes are bit-identical.
    Contexts and splits cross block boundaries; the blocks are tpa_factor.block_align()'s 16 tokens on
    sm_80 / sm_120 and 26 elsewhere."""
    from vllm.model_executor.layers.mamba.olala import tpa_factor as tf
    from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family

    if not tf.factor_cache_enabled():
        pytest.skip("factor kernel unavailable (JIT build failed)")
    BS = tf.SPEC_BLOCK_ALIGN if gpu_family() in (GpuFamily.SM80, GpuFamily.SM120) else 2 * tf.TILE
    torch.manual_seed(100 * lanes + k + int(softcap))
    bf = torch.bfloat16
    dev = "cuda"
    C = 1 + k
    GQ = tf.HQ // tf.HKV
    P = lanes * C + 4
    qlens = torch.randint(1, C + 1, (lanes,))
    qlens[0] = C
    if lanes > 2:
        qlens[1] = 0
    ctx = torch.randint(1, 6 * BS, (lanes,))  # committed tokens before the verify step
    qsl = torch.zeros(lanes + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(qlens, 0)
    n_tok = int(qsl[-1])
    N = n_tok + 2  # FULL-graph padding rows
    seq_lens = (ctx + qlens).to(torch.int32)
    nblk = [-(-(int(c) + C) // BS) for c in ctx]
    NB = sum(nblk) + 2
    perm = torch.randperm(NB).tolist()
    bt = torch.zeros(lanes, max(nblk), dtype=torch.int32)
    i = 0
    for r in range(lanes):
        bt[r, : nblk[r]] = torch.tensor(perm[i : i + nblk[r]], dtype=torch.int32)
        i += nblk[r]
    positions = torch.zeros(N, dtype=torch.long)
    cache_slots = torch.full((N,), -1, dtype=torch.int64)
    for r in range(lanes):
        for t in range(int(qlens[r])):
            p = int(ctx[r]) + t
            positions[int(qsl[r]) + t] = p
            cache_slots[int(qsl[r]) + t] = int(bt[r, p // BS]) * BS + p % BS
    positions, cache_slots, bt_d = positions.to(dev), cache_slots.to(dev), bt.to(dev)
    ins, pools0, tail, cols, num_acc = _factor_spec_inputs(lanes, k, N, P, dev)
    # the committed context: random factor rows (the attention only reads them)
    kp0 = (torch.randn(NB, BS, tf.ROWP, device=dev) * 0.5).to(bf)
    vp0 = (torch.randn(NB, BS, tf.ROWP, device=dev) * 0.5).to(bf)
    md = tf.OlalaTPAFactorMetadata(
        num_decodes=lanes, num_prefills=0, num_decode_tokens=N, num_prefill_tokens=0, num_actual_tokens=N,
        block_table=bt_d, seq_lens=seq_lens.to(dev), slot_mapping=cache_slots, max_seq_len=int(seq_lens.max()),
        max_query_len=C, query_start_loc=qsl.to(dev),
    )
    split = md.split_size(nsplit)
    scale = tf.HEAD_DIM**-0.5

    ref = [p.clone() for p in pools0]
    kp_r, vp_r = kp0.clone(), vp0.clone()
    q_r = torch.zeros(N, tf.HQ * tf.HEAD_DIM, device=dev, dtype=bf)
    out_r = torch.zeros(N, tf.HQ * tf.HEAD_DIM, device=dev, dtype=bf)
    for t in range(C):
        rows, src, dst, reqs = _chained_columns(cols, num_acc, qlens, qsl, t)
        if not rows:
            continue
        ri = torch.tensor(rows, device=dev)
        s_t, d_t = torch.tensor(src, device=dev).long(), torch.tensor(dst, device=dev).long()
        for p in ref:
            p[d_t] = p[s_t].clone()
        q_t = tf.factor_decode_write(*(x[ri] for x in ins), positions[ri], d_t, cache_slots[ri], *ref, *tail,
                                     kp_r, vp_r, GQ)
        q_r[ri] = q_t
        sl_t = (ctx[reqs] + t + 1).to(torch.int32).to(dev)
        out_r[ri] = tf.factor_decode_attention(q_t, kp_r, vp_r, bt_d[torch.tensor(reqs, device=dev)], sl_t,
                                               split, nsplit, scale, softcap, BS)

    sp = [p.clone() for p in pools0]
    kp_s, vp_s = kp0.clone(), vp0.clone()
    q_s = tf.factor_decode_write_spec(*ins, positions, qsl.to(dev), cols, num_acc, cache_slots, *sp, *tail,
                                      kp_s, vp_s, GQ)
    bt_v, sl_v = md.virtual_decode_rows()
    out_s = tf.factor_decode_attention(q_s, kp_s, vp_s, bt_v, sl_v, split, nsplit, scale, softcap, BS)
    assert torch.equal(q_s[:n_tok], q_r[:n_tok])
    assert torch.equal(kp_s, kp_r) and torch.equal(vp_s, vp_r)
    assert torch.equal(out_s[:n_tok], out_r[:n_tok])
    assert torch.isfinite(out_s[n_tok:].float()).all()

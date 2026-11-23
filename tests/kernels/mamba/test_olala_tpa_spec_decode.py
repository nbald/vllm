# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala Diff-TPA fused verify preamble (tpa_decode_qkv_spec) against the
pure-decode kernel chained over the draft positions with the column-slot
protocol: position t shifts against the raw K/V of position t - 1 (the pool
column of the accepted state for t = 0) and stores its raw K/V to column t.
Ragged query lengths, a zero-length padding lane and CUDA-graph padding rows
past the last request are covered."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.diff_tpa_decode import (
    tpa_decode_qkv,
    tpa_decode_qkv_spec,
)

HKV, GQ, R, D = 4, 4, 8, 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("lanes,k", [(1, 7), (5, 3)])
def test_tpa_spec_matches_chained_decode(lanes, k):
    torch.manual_seed(0)
    bf = torch.bfloat16
    ncols = 1 + k
    P = lanes * ncols + 4
    qlens = torch.randint(1, ncols + 1, (lanes,))
    qlens[0] = ncols
    if lanes > 2:
        qlens[1] = 0
    qsl = torch.zeros(lanes + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(qlens, 0)
    N = int(qsl[-1]) + 2
    cols = (
        (torch.randperm(P - 1)[: lanes * ncols] + 1).to(torch.int32).view(lanes, ncols)
    )
    num_acc = torch.randint(1, ncols + 1, (lanes,), dtype=torch.int32)
    positions = torch.zeros(N, dtype=torch.long)
    for r in range(lanes):
        p0 = int(torch.randint(0, 50, ()))  # p0 == 0 exercises the sequence-start rule
        positions[int(qsl[r]) : int(qsl[r + 1])] = torch.arange(p0, p0 + int(qlens[r]))
    dev = "cuda"
    q_all = torch.randn(N, HKV * GQ * D, device=dev).to(bf)
    A_k = torch.randn(N, HKV * R, device=dev).to(bf)
    A_v = torch.randn(N, HKV * R, device=dev).to(bf)
    B_k = torch.randn(N, R * D, device=dev).to(bf)
    B_v = torch.randn(N, R * D, device=dev).to(bf)
    al_k = torch.randn(N, HKV, device=dev).to(bf)
    al_v = torch.randn(N, HKV, device=dev).to(bf)
    k_pool0 = torch.randn(P, HKV, D, device=dev).to(bf)
    v_pool0 = torch.randn(P, HKV, D, device=dev).to(bf)
    q_w = torch.randn(D, device=dev).to(bf) * 0.1
    k_w = torch.randn(D, device=dev).to(bf) * 0.1
    scaler = torch.rand(HKV * GQ, device=dev).to(bf) + 0.5
    args = (q_w, k_w, 1e-6, scaler, 4096.0, HKV, GQ, R, D)
    positions = positions.to(dev)

    # reference: the decode kernel, one launch per position, column-slot protocol
    kp, vp = k_pool0.clone(), v_pool0.clone()
    q_ref = torch.zeros(N, HKV * GQ * D, device=dev, dtype=bf)
    k_ref = torch.zeros(N, HKV * D, device=dev, dtype=bf)
    v_ref = torch.zeros(N, HKV * D, device=dev, dtype=bf)
    init = cols.gather(1, (num_acc.long() - 1).clamp(min=0)[:, None]).squeeze(1)
    for t in range(ncols):
        rows, src, dst = [], [], []
        for r in range(lanes):
            if t < int(qlens[r]):
                rows.append(int(qsl[r]) + t)
                src.append(int(init[r]) if t == 0 else int(cols[r, t - 1]))
                dst.append(int(cols[r, t]))
        if not rows:
            continue
        ri = torch.tensor(rows, device=dev)
        s_t, d_t = (
            torch.tensor(src, device=dev).long(),
            torch.tensor(dst, device=dev).long(),
        )
        kp[d_t], vp[d_t] = kp[s_t].clone(), vp[s_t].clone()
        qo, ko, vo = tpa_decode_qkv(
            q_all[ri],
            A_k[ri],
            A_v[ri],
            B_k[ri],
            B_v[ri],
            al_k[ri],
            al_v[ri],
            positions[ri],
            d_t.to(torch.int32),
            kp,
            vp,
            *args,
        )
        q_ref[ri], k_ref[ri], v_ref[ri] = qo, ko, vo

    kp2, vp2 = k_pool0.clone(), v_pool0.clone()
    qo, ko, vo = tpa_decode_qkv_spec(
        q_all,
        A_k,
        A_v,
        B_k,
        B_v,
        al_k,
        al_v,
        positions,
        qsl.to(dev),
        cols.to(dev),
        num_acc.to(dev),
        kp2,
        vp2,
        *args,
    )
    n_tok = int(qsl[-1])
    assert torch.equal(qo[:n_tok], q_ref[:n_tok])
    assert torch.equal(ko[:n_tok], k_ref[:n_tok])
    assert torch.equal(vo[:n_tok], v_ref[:n_tok])
    written = sorted(
        {int(cols[r, t]) for r in range(lanes) for t in range(int(qlens[r]))}
    )
    w = torch.tensor(written, device=dev).long()
    assert torch.equal(kp2[w], kp[w]) and torch.equal(vp2[w], vp[w])
    # padding rows: finite, unshifted, and the pool is left alone elsewhere
    assert (
        torch.isfinite(qo[n_tok:].float()).all()
        and torch.isfinite(ko[n_tok:].float()).all()
    )
    keep = torch.ones(P, dtype=torch.bool, device=dev)
    keep[w] = False
    assert torch.equal(kp2[keep], k_pool0[keep])

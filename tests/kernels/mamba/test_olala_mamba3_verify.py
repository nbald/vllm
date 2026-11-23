# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala fused speculative verify (csrc/mamba3_verify.cu) against the CUDA
decode step chained over the draft positions, which is what the column-slot
protocol means: position t starts from the state written by position t - 1
(the accepted column for t = 0) and writes column t.

Ragged query lengths, zero-length CUDA-graph padding lanes, rows past the last
request and an invalid start slot are covered; the large batch gives each
persistent CTA several (request, head) items."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
    mamba3_step_cuda,
    mamba3_verify_cuda,
)

requires_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80+ GPU",
)

H, D, S, R, NA = 48, 64, 128, 4, 32


def _inputs(N):
    bf = torch.bfloat16
    return dict(
        A=-torch.rand(N, H, device="cuda") * 2 - 0.05,
        dt=torch.rand(N, H, device="cuda") * 0.3 + 0.01,
        trap=torch.rand(N, H, device="cuda"),
        BC=torch.randn(N, 2, R, S, device="cuda").to(bf),
        x=torch.randn(N, H, D, device="cuda").to(bf),
        z=torch.randn(N, H, D, device="cuda").to(bf),
        # a row view with a padded stride, like the in_proj_dyn output slice
        angp=torch.randn(N, NA + 8, device="cuda").to(bf)[:, 8:],
    )


@requires_sm80
@pytest.mark.parametrize("lanes,k", [(1, 7), (3, 3), (40, 7)])
def test_verify_matches_chained_step(lanes, k):
    torch.manual_seed(0)
    bf = torch.bfloat16
    ncols = 1 + k
    P = lanes * ncols + 8
    # ragged query lengths; lane 1 (when present) is a zero-length padding lane
    qlens = torch.randint(1, ncols + 1, (lanes,))
    qlens[0] = ncols
    if lanes > 2:
        qlens[1] = 0
    qsl = torch.zeros(lanes + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(qlens, 0)
    n_tok = int(qsl[-1])
    N = n_tok + 3  # rows past the last request
    perm = (torch.randperm(P - 1)[: lanes * ncols] + 1).to(torch.int32)
    cols = perm.view(lanes, ncols).cuda()
    num_acc = torch.randint(1, ncols + 1, (lanes,), dtype=torch.int32).cuda()
    if lanes > 3:
        # no valid start state: outputs zero, nothing written
        cols[3, num_acc[3] - 1] = -1

    ang0 = torch.rand(P, H, NA, device="cuda") * 6.28
    ssm0 = torch.randn(P, H, D, S, device="cuda") * 0.5
    k0 = torch.randn(P, R, H, S, device="cuda").to(bf)
    v0 = torch.randn(P, H, D, device="cuda").to(bf)
    xpj = torch.rand(R, H, D, device="cuda") * 0.5
    zpj = torch.rand(R, H, D, device="cuda")
    opj = torch.rand(R, H, D, device="cuda") * 0.5
    bq = torch.randn(R, H, S, device="cuda") * 0.1 + 1
    bk = torch.randn(R, H, S, device="cuda") * 0.1 + 1
    Dp = torch.randn(H, device="cuda")
    inp = _inputs(N)

    ref = [t.clone() for t in (ang0, ssm0, k0, v0)]
    y_ref = torch.zeros(N, H, D, device="cuda", dtype=bf)
    in_slots = cols.gather(1, (num_acc.long() - 1).clamp(min=0)[:, None]).squeeze(1)
    for t in range(ncols):
        # one chained step: lanes still active read column t - 1 (start column at
        # t = 0) and write column t
        rows, src, dst = [], [], []
        for r in range(lanes):
            if t < int(qlens[r]) and int(in_slots[r]) >= 0:
                rows.append(int(qsl[r]) + t)
                src.append(int(in_slots[r]) if t == 0 else int(cols[r, t - 1]))
                dst.append(int(cols[r, t]))
        if not rows:
            continue
        rows_t = torch.tensor(rows, device="cuda")
        # the step kernel updates its slot in place: copy the source state into
        # the destination column first
        src_t = torch.tensor(src, device="cuda").long()
        dst_t = torch.tensor(dst, device="cuda").long()
        for pool in ref:
            pool[dst_t] = pool[src_t]
        y_t = torch.empty(len(rows), H, D, device="cuda", dtype=bf)
        BC = inp["BC"][rows_t]
        mamba3_step_cuda(
            ref[1],
            ref[2],
            ref[3],
            ref[0],
            inp["A"][rows_t],
            BC[:, 0],
            BC[:, 1],
            Dp,
            inp["x"][rows_t].contiguous(),
            inp["dt"][rows_t].contiguous(),
            inp["trap"][rows_t].contiguous(),
            xpj,
            zpj,
            opj,
            inp["z"][rows_t].contiguous(),
            bq,
            bk,
            inp["angp"][rows_t].contiguous(),
            dst_t.to(torch.int32),
            y_t,
        )
        y_ref[rows_t] = y_t

    pools = [t.clone() for t in (ang0, ssm0, k0, v0)]
    y = torch.full((N, H, D), float("nan"), device="cuda", dtype=bf)
    mamba3_verify_cuda(
        pools[1],
        pools[2],
        pools[3],
        pools[0],
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
        qsl.cuda(),
        cols,
        num_acc,
        y,
    )
    torch.accelerator.synchronize()
    # bit-identical outputs, including zero rows for padding / invalid lanes
    assert torch.equal(y, y_ref)
    written = torch.tensor(
        sorted(
            {
                int(cols[r, t])
                for r in range(lanes)
                for t in range(int(qlens[r]))
                if int(in_slots[r]) >= 0
            }
        ),
        device="cuda",
    ).long()
    for got, want in zip(pools, ref):
        assert torch.equal(got[written], want[written])
    # every other slot (start states included unless overwritten) is untouched
    untouched = torch.ones(P, dtype=torch.bool, device="cuda")
    untouched[written] = False
    for got, orig in zip(pools, (ang0, ssm0, k0, v0)):
        assert torch.equal(got[untouched], orig[untouched])

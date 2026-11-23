# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala fused speculative verify (csrc/mamba3_verify.cu) against the CUDA
decode step chained over the draft positions, which is what the column-slot
protocol means: position t starts from the state written by position t - 1
(the accepted column for t = 0) and writes column t.

Ragged query lengths, zero-length CUDA-graph padding lanes, rows past the last
request and an invalid start slot are covered; the large batch gives each
persistent CTA several (request, head) items.

On the SM70 family (V100, natively or through volta-forever) both extensions
are Volta builds: the verify kernel's Volta edition (VERIFY_VOLTA) against the
STEP_VOLTA decode step that direct decoding uses. The control there runs the
tensor-core verify kernel (the build of the other GPUs) on the same inputs: its
fp32 split arithmetic rounds differently from the Volta FMA step, so it must
not be bit-identical to the chain (the equality checks can see a difference)
while it stays close (it computes the same recurrence)."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
    _load,
    _load_verify,
    mamba3_step_cuda,
    mamba3_verify_cuda,
)

requires_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80+ GPU",
)
requires_volta = pytest.mark.skipif(
    not torch.cuda.is_available() or gpu_family() is not GpuFamily.SM70,
    reason="needs a V100 (Volta family, natively or through volta-forever)",
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


def _case(lanes, k, seed=0):
    """Inputs of one verify launch and its reference: the CUDA decode step
    chained over the positions with the column-slot protocol."""
    torch.manual_seed(seed)
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

    c = dict(
        lanes=lanes, qlens=qlens, qsl=qsl, cols=cols, num_acc=num_acc, N=N, P=P,
        pools0=(
            torch.rand(P, H, NA, device="cuda") * 6.28,  # angle
            torch.randn(P, H, D, S, device="cuda") * 0.5,  # ssm
            torch.randn(P, R, H, S, device="cuda").to(bf),  # k (rotated B)
            torch.randn(P, H, D, device="cuda").to(bf),  # v (x)
        ),
        xpj=torch.rand(R, H, D, device="cuda") * 0.5,
        zpj=torch.rand(R, H, D, device="cuda"),
        opj=torch.rand(R, H, D, device="cuda") * 0.5,
        bq=torch.randn(R, H, S, device="cuda") * 0.1 + 1,
        bk=torch.randn(R, H, S, device="cuda") * 0.1 + 1,
        Dp=torch.randn(H, device="cuda"),
        inp=_inputs(N),
    )
    inp = c["inp"]
    ref = [t.clone() for t in c["pools0"]]
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
            ref[1], ref[2], ref[3], ref[0],
            inp["A"][rows_t], BC[:, 0], BC[:, 1], c["Dp"],
            inp["x"][rows_t].contiguous(), inp["dt"][rows_t].contiguous(),
            inp["trap"][rows_t].contiguous(), c["xpj"], c["zpj"], c["opj"],
            inp["z"][rows_t].contiguous(), c["bq"], c["bk"],
            inp["angp"][rows_t].contiguous(), dst_t.to(torch.int32), y_t,
        )  # fmt: skip
        y_ref[rows_t] = y_t
    c["ref"], c["y_ref"] = ref, y_ref
    written = sorted(
        {
            int(cols[r, t])
            for r in range(lanes)
            for t in range(int(qlens[r]))
            if int(in_slots[r]) >= 0
        }
    )
    c["written"] = torch.tensor(written, device="cuda").long()
    untouched = torch.ones(P, dtype=torch.bool, device="cuda")
    untouched[c["written"]] = False
    c["untouched"] = untouched
    return c


def _run(c, verify=None):
    """One verify launch on fresh copies of the pools: mamba3_verify_cuda, or
    the ``verify`` entry point of another build of csrc/mamba3_verify.cu."""
    inp = c["inp"]
    pools = [t.clone() for t in c["pools0"]]
    y = torch.full((c["N"], H, D), float("nan"), device="cuda", dtype=torch.bfloat16)
    args = (
        pools[1], pools[2], pools[3], pools[0], inp["A"], inp["BC"][:, 0],
        inp["BC"][:, 1], c["Dp"], inp["x"], inp["dt"], inp["trap"], c["xpj"],
        c["zpj"], c["opj"], inp["z"], c["bq"], c["bk"], inp["angp"],
        c["qsl"].cuda(), c["cols"], c["num_acc"], y,
    )  # fmt: skip
    if verify is None:
        mamba3_verify_cuda(*args)
    else:
        verify(*args, 4, False)
    torch.accelerator.synchronize()
    return y, pools


def _assert_matches(c, y, pools):
    # bit-identical outputs, including zero rows for padding / invalid lanes
    assert torch.equal(y, c["y_ref"])
    for got, want in zip(pools, c["ref"]):
        assert torch.equal(got[c["written"]], want[c["written"]])
    # every other slot (start states included unless overwritten) is untouched
    for got, orig in zip(pools, c["pools0"]):
        assert torch.equal(got[c["untouched"]], orig[c["untouched"]])


@requires_sm80
@pytest.mark.parametrize("lanes,k", [(1, 7), (3, 3), (40, 7)])
def test_verify_matches_chained_step(lanes, k):
    c = _case(lanes, k)
    _assert_matches(c, *_run(c))


@requires_sm80
@requires_volta
@pytest.mark.parametrize("lanes,k", [(1, 3), (1, 5), (6, 5), (130, 5)])
def test_volta_verify_matches_chained_volta_step(lanes, k):
    """Volta: the verify kernel's Volta edition against the STEP_VOLTA step that
    direct decoding chains, bit for bit; 130 lanes give every CTA of the grid
    several requests of its head."""
    assert _load_verify().__name__.endswith("_volta")
    assert _load().__name__.endswith("_volta")
    c = _case(lanes, k, seed=lanes + k)
    _assert_matches(c, *_run(c))


@requires_sm80
@requires_volta
def test_volta_control_tensor_core_verify_is_not_bit_identical():
    """Control: the tensor-core verify kernel (the build of the other GPUs,
    emulated by volta-forever on a V100) on the same inputs differs from the
    STEP_VOLTA chain in the last bits, and only there."""
    from vllm.model_executor.layers.mamba.olala.arch import gencode, sm_arch
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    tc = load_extension(
        f"olala_mamba3_verify_sm{sm_arch()}", "mamba3_verify.cu",
        ["-O3", "-std=c++20", gencode()],
    )  # fmt: skip
    c = _case(6, 5, seed=11)
    y_tc, pools_tc = _run(c, tc.verify)
    ssm_tc, ssm_ref = pools_tc[1][c["written"]], c["ref"][1][c["written"]]
    assert not torch.equal(ssm_tc, ssm_ref)
    torch.testing.assert_close(ssm_tc, ssm_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        y_tc.float(), c["y_ref"].float(), rtol=2e-2, atol=2e-2
    )
    # the Volta edition on the same inputs: bit for bit
    _assert_matches(c, *_run(c))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70-family (V100) kernels that must return today's bits: the router chain
above 32 tokens (router_volta.py) and the Volta Mamba-3 decode step
(mamba3_step.cu, mamba3_step_volta_kernel)."""

import pytest
import torch

from vllm.platforms import current_platform


def _sm70() -> bool:
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family

    return gpu_family() is GpuFamily.SM70


requires_sm70 = pytest.mark.skipif(not _sm70(), reason="SM70-family (V100) kernels")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


@requires_sm70
@pytest.mark.parametrize("m", [33, 48, 64, 100, 128, 255, 384, 512, 700, 1024, 1500, 2048])
@torch.inference_mode()
def test_volta_router_chain_matches_cublas(m: int):
    """Above 32 tokens the router logits are mm(x, [hi ; lo].T, fp32) + the
    slice add; on a V100 cuBLAS runs that bf16 GEMM as sequential fp32 chains
    (magma), which the chain kernel reproduces bit for bit."""
    from vllm.model_executor.layers.mamba.olala.router_volta import (
        router_chain,
        router_chain_ok,
    )

    E, K = 256, 1536
    g = torch.Generator(device="cuda").manual_seed(m)
    w32 = torch.randn(E, K, device="cuda", generator=g) * 0.03
    hi = w32.to(torch.bfloat16)
    w = torch.cat([hi, (w32 - hi.float()).to(torch.bfloat16)]).contiguous()
    x = torch.randn(m, K + 64, device="cuda", generator=g)[:, :K]  # row-strided
    x[:, :8] *= 100.0  # residual-stream outliers
    x = x.to(torch.bfloat16)
    assert router_chain_ok(x, w, E)
    out = torch.mm(x, w.t(), out_dtype=torch.float32)
    ref = out[:, :E] + out[:, E:]
    assert torch.equal(_bits(router_chain(x, w, E)), _bits(ref))


@requires_sm70
@pytest.mark.parametrize("batch", [1, 3, 8, 31, 32, 48, 64])
@torch.inference_mode()
def test_volta_step_matches_one_stage(batch: int):
    """The Volta step (state in registers, round-robin items) returns the
    1-stage FMA step's outputs and pools bit for bit, over chained steps and
    with padded (invalid) slots."""
    from vllm.model_executor.layers.mamba.olala.arch import gencode
    from vllm.model_executor.layers.mamba.olala.jit import load_extension
    from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
        _load,
        cuda_step_enabled,
    )

    if not cuda_step_enabled():
        pytest.skip("CUDA step unavailable")
    one_stage = load_extension(
        "olala_mamba3_step_volta_1stage_test", "mamba3_step.cu",
        ["-O3", "-std=c++20", gencode(), "-DSTEP_VOLTA", "-DSTEP_VOLTA_1STAGE"],
    )  # fmt: skip
    H, D, S, R, NA, pool = 48, 64, 128, 4, 32, 80
    g = torch.Generator(device="cuda").manual_seed(batch)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    pools = [
        rn(pool, H, D, S, scale=0.5),
        rn(pool, R, H, S).bfloat16(),
        rn(pool, H, D).bfloat16(),
        torch.rand(pool, H, NA, device="cuda", generator=g) * 6.0,
    ]
    xpj, zpj, opj = rn(R, H, D, scale=0.3), rn(R, H, D, scale=0.3), rn(R, H, D, scale=0.3)
    bq, bk, dp = rn(R, H, S, scale=0.2), rn(R, H, S, scale=0.2), rn(H)
    slots = torch.randperm(pool, device="cuda", generator=g)[:batch].to(torch.int32)
    if batch >= 3:
        slots[1], slots[-1] = -1, pool + 3  # padding entries: no writes, zero output
    got, ref = [p.clone() for p in pools], [p.clone() for p in pools]
    for _ in range(2):
        bc = rn(batch, 2 * R * S + NA + 8).bfloat16()
        b_m = bc[:, : R * S].unflatten(1, (R, S))
        c_m = bc[:, R * S : 2 * R * S].unflatten(1, (R, S))
        angp = bc[:, 2 * R * S : 2 * R * S + NA]
        a = -torch.rand(batch, H, device="cuda", generator=g) * 4 - 0.05
        dt = torch.nn.functional.softplus(rn(batch, H) - 3)
        trap = torch.sigmoid(rn(batch, H))
        x, z = rn(batch, H, D).bfloat16(), rn(batch, H, D).bfloat16()
        y, y_ref = (torch.empty(batch, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        args = (a, b_m, c_m, dp, x, dt, trap, xpj, zpj, opj, z, bq, bk, angp, slots)
        _load().step(*got, *args, y, 4)
        one_stage.step(*ref, *args, y_ref, 4)
        assert torch.equal(_bits(y), _bits(y_ref))
        for p, q in zip(got, ref):
            assert torch.equal(_bits(p), _bits(q))

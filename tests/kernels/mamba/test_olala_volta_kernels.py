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


# -- 16-bit SSM state (OLALA_VOLTA_STATE16) -------------------------------------------------------------------------


def _round_state16(h32: torch.Tensor, fmt: str):
    """The 16-bit storage of an fp32 state (..., D, S): bf16 (nearest even), or fp16 rows at 2^(14 - e), e the
    exponent of the row max (a zero row: e = -110), with their inverse scales 2^(e - 14)."""
    if fmt == "bf16":
        return h32.to(torch.bfloat16), None
    m = h32.abs().amax(dim=-1)
    _, e = torch.frexp(m)
    ex = torch.where(m > 0, e - 1, torch.full_like(e, -110)).clamp(min=-110)
    one = torch.ones_like(m)
    return (h32 * torch.ldexp(one, 14 - ex)[..., None]).to(torch.float16), torch.ldexp(one, ex - 14)


def _widen16(v: torch.Tensor, sc):
    return v.float() if sc is None else v.float() * sc[..., None]


def _rz_half(v: torch.Tensor) -> torch.Tensor:
    """fp16 of non-negative fp32 values, rounded toward zero (as fp32)."""
    r = v.half()
    down = (r.view(torch.int16) - 1).view(torch.float16)
    return torch.where(r.float() > v, down, r).float()


def _sr16_choices(h32: torch.Tensor, v16: torch.Tensor, sc):
    """A stochastically rounded 16-bit storage v16 of h32 (fp16: at the row scales sc, those of nearest rounding):
    asserts that every stored value is one of the two 16-bit values around h32's, with h32's sign; returns, for the
    values strictly between two (fp16: in its normal range), whether the larger magnitude was stored and the position
    of |h32| between the two, which should be its probability."""
    x = h32 if sc is None else h32 / sc[..., None]   # exact: powers of two
    a = x.abs()
    bits = a.view(torch.int32)
    step = 1 << 16 if v16.dtype == torch.bfloat16 else 1 << 13
    lo = (bits & -step).view(torch.float32)
    hi = ((bits & -step) + step).view(torch.float32)
    lo_s, hi_s = (lo, hi) if v16.dtype == torch.bfloat16 else (_rz_half(lo), _rz_half(hi))
    s = v16.float().abs()
    assert ((s == lo_s) | (s == hi_s)).all()
    assert torch.equal(torch.signbit(v16.float()), torch.signbit(x))
    between = (a != lo) & (lo_s != hi_s)
    if v16.dtype == torch.float16:
        between &= a >= 2.0**-14
    frac = (a.double() - lo.double()) / (hi.double() - lo.double())
    return (s == hi_s)[between].double(), frac[between]


def test_volta_state16_layout(monkeypatch):
    """OLALA_VOLTA_STATE16 picks the SSM state dtype on the SM70 family (fp16 adds a fifth tensor of per-row
    scales), refuses an explicit --mamba-ssm-cache-dtype, and is ignored elsewhere."""
    from vllm.model_executor.layers.mamba.olala import arch, state

    def fresh():
        state.volta_state16.cache_clear()
        state.replay_config.cache_clear()
        arch.sm70_family_no_context.cache_clear()

    monkeypatch.setenv("OLALA_VOLTA", "1")
    try:
        for value, ssm_dtype, n in (("bf16", torch.bfloat16, 4), ("fp16", torch.float16, 5),
                                    ("bf16sr", torch.bfloat16, 4), ("fp16sr", torch.float16, 5),
                                    ("1", torch.float16, 5), ("0", torch.float32, 4)):
            monkeypatch.setenv("OLALA_VOLTA_STATE16", value)
            fresh()
            assert state.volta_state16() == {"1": "fp16sr", "0": None}.get(value, value)
            dtypes = state.mamba3_state_dtype(torch.bfloat16, "auto", "auto")
            shapes = state.mamba3_state_shape(1, 48, 64, 128, 4, 32)
            assert dtypes[1] == ssm_dtype and len(dtypes) == len(shapes) == n
            if n == 5:
                assert dtypes[4] == torch.float32 and shapes[4] == (48, 64)
            if value != "0":
                with pytest.raises(ValueError):
                    state.mamba3_state_dtype(torch.bfloat16, "auto", "float32")
        monkeypatch.setenv("OLALA_VOLTA_STATE16", "fp16")
        monkeypatch.setenv("OLALA_VOLTA", "0")
        fresh()
        assert state.volta_state16() is None
        assert state.mamba3_state_dtype(torch.bfloat16, "auto", "auto")[1] == torch.float32
    finally:
        monkeypatch.delenv("OLALA_VOLTA_STATE16", raising=False)
        monkeypatch.delenv("OLALA_VOLTA", raising=False)
        fresh()


@requires_sm70
@pytest.mark.parametrize("fmt", ["bf16", "fp16", "bf16sr", "fp16sr"])
@pytest.mark.parametrize("batch", [1, 3, 33, 65, 128])
@torch.inference_mode()
def test_volta_step16_matches_fp32_step(fmt: str, batch: int):
    """The 16-bit Volta step computes in fp32: from a state the fp32 step also holds exactly, its stored state is
    the fp32 step's new state rounded to the format, bit for bit (nearest even), or, with stochastic rounding, one
    of the two 16-bit values around it (fp16: the same row scales), the larger magnitude with the probability of
    the value's position between them; the k / v / angle pools are the fp32 step's; y differs at most by the C
    dot's summation order (one bf16 ulp); padded slots are neither read back nor written, and their outputs are
    zero. Chained steps, rows of very different magnitudes."""
    from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
        cuda_step_enabled,
        mamba3_step_cuda,
    )
    from vllm.model_executor.layers.mamba.olala.state import STATE16_CODES

    if not cuda_step_enabled():
        pytest.skip("CUDA step unavailable")
    H, D, S, R, NA, pool = 48, 64, 128, 4, 32, 200
    g = torch.Generator(device="cuda").manual_seed(batch)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    rnd = fmt.removesuffix("sr")   # the storage format; fmt != rnd: stochastic rounding
    h0 = rn(pool, H, D, S) * torch.exp(rn(pool, H, D, 1, scale=2.0))
    v16, sc0 = _round_state16(h0, rnd)
    ssm16 = torch.empty(pool, H * D * S + 64, device="cuda", dtype=v16.dtype)[:, : H * D * S].view(pool, H, D, S)
    ssm16.copy_(v16)
    sc = None
    if sc0 is not None:
        sc = torch.empty(pool, H * D + 16, device="cuda")[:, : H * D].view(pool, H, D)
        sc.copy_(sc0)
    small = [rn(pool, R, H, S).bfloat16(), rn(pool, H, D).bfloat16(),
             torch.rand(pool, H, NA, device="cuda", generator=g) * 6.0]  # fmt: skip
    small32 = [p.clone() for p in small]
    ssm32 = torch.empty(pool, H, D, S, device="cuda")
    xpj, zpj, opj = rn(R, H, D, scale=0.3), rn(R, H, D, scale=0.3), rn(R, H, D, scale=0.3)
    bq, bk, dp = rn(R, H, S, scale=0.2), rn(R, H, S, scale=0.2), rn(H)
    slots = torch.randperm(pool, device="cuda", generator=g)[:batch].to(torch.int32)
    if batch >= 3:
        slots[1], slots[-1] = -1, pool + 3  # padding entries
    ups, fracs = [], []
    for _ in range(3):
        ssm32.copy_(_widen16(ssm16, sc))
        before = ssm16.clone()
        bc = rn(batch, 2 * R * S + NA + 8).bfloat16()
        b_m = bc[:, : R * S].unflatten(1, (R, S))
        c_m = bc[:, R * S : 2 * R * S].unflatten(1, (R, S))
        angp = bc[:, 2 * R * S : 2 * R * S + NA]
        a = -torch.rand(batch, H, device="cuda", generator=g) * 4 - 0.05
        dt = torch.nn.functional.softplus(rn(batch, H) - 3)
        trap = torch.sigmoid(rn(batch, H))
        x, z = rn(batch, H, D).bfloat16(), rn(batch, H, D).bfloat16()
        y16, y32 = (torch.full((batch, H, D), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2))
        args = (a, b_m, c_m, dp, x, dt, trap, xpj, zpj, opj, z, bq, bk, angp, slots)
        mamba3_step_cuda(ssm32, *small32[:2], small32[2], *args[:-1], slots, y32)
        mamba3_step_cuda(ssm16, *small[:2], small[2], *args[:-1], slots, y16, ssm_scale=sc,
                         state16=STATE16_CODES[fmt])
        exp_v, exp_sc = _round_state16(ssm32, rnd)
        valid = (slots >= 0) & (slots < pool)
        touched = torch.zeros(pool, dtype=torch.bool, device="cuda")
        touched[slots[valid].long()] = True
        if fmt == rnd:
            assert torch.equal(_bits(ssm16[touched]), _bits(exp_v[touched]))
        else:
            up, frac = _sr16_choices(ssm32[touched], ssm16[touched], None if sc is None else exp_sc[touched])
            ups.append(up)
            fracs.append(frac)
        assert torch.equal(_bits(ssm16[~touched]), _bits(before[~touched]))
        if sc is not None:
            assert torch.equal(_bits(sc[touched]), _bits(exp_sc[touched]))
        for p, q in zip(small, small32):
            assert torch.equal(_bits(p), _bits(q))
        assert not torch.isnan(y16).any()
        assert torch.equal(y16[~valid], torch.zeros_like(y16[~valid]))
        # one bf16 ulp, or the C dot's reassociation where the output cancels (2^-12 of the head's max)
        d = (y16.float() - y32.float()).abs()
        tol = y32.float().abs() * 2.0**-7 + y32.float().abs().amax(dim=-1, keepdim=True) * 2.0**-12
        assert (d <= tol).all()
    if fmt != rnd:
        # unbiased: rounded up as often as the positions say, also where nearest rounding never rounds up
        # (>= 3 x 0.3 M values: the means' standard deviation is below 1e-3)
        up, frac = torch.cat(ups), torch.cat(fracs)
        low = frac < 0.5
        assert abs(up.mean() - frac.mean()) < 0.01
        assert abs(up[low].mean() - frac[low].mean()) < 0.01 and up[low].mean() > 0.2


@requires_sm70
@pytest.mark.parametrize("fmt", ["bf16", "fp16"])
@torch.inference_mode()
def test_volta_state16_prefill_state_pass(fmt: str):
    """The hybrid prefill on a 16-bit pool reads the entering states widened (fp16 times the row scale) and stores
    the final states rounded to the format: with the same values in an fp32 pool, the outputs and the angle / k /
    v pools are bit-identical and the stored states are the fp32 run's final states rounded (sequences resuming
    from pool states and starting from zero, chunk tails)."""
    from vllm.model_executor.layers.mamba.ops.mamba3 import mimo_ssd

    H, N, P, R, A = 48, 128, 64, 4, 32
    lens, init = [77, 300, 161], [True, False, True]
    T, ns = sum(lens), len(lens)
    g = torch.Generator(device="cuda").manual_seed(3)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    dt = torch.rand(H, T, device="cuda", generator=g) * 0.4
    adt = -dt * torch.rand(H, 1, device="cuda", generator=g) * 3
    args = [rn(T, R, N).bfloat16(), rn(T, R, N).bfloat16(), rn(T, H, P, scale=0.1).bfloat16(),
            rn(T, H, P, scale=0.1).bfloat16(), rn(T, A, scale=0.3), dt, adt, rn(H, T).bfloat16(),
            rn(H, R, N), rn(H, R, N), rn(H, R, P, scale=0.5), rn(H, R, P, scale=0.5), rn(H, R, P),
            5 * rn(H)]  # fmt: skip
    v16, sc0 = _round_state16(rn(ns, H, P, N, scale=0.1) * torch.exp(rn(ns, H, P, 1)), fmt)
    small = [rn(ns, H, A), rn(ns, R, H, N).bfloat16(), rn(ns, H, P).bfloat16()]
    p16 = [small[0].clone(), v16.clone(), small[1].clone(), small[2].clone()]
    p32 = [small[0].clone(), _widen16(v16, sc0), small[1].clone(), small[2].clone()]
    sc = None if sc0 is None else sc0.clone()
    cu = [0] + torch.tensor(lens).cumsum(0).tolist()
    slots = torch.arange(ns, device="cuda", dtype=torch.int32)
    has_init = torch.tensor(init, device="cuda")
    o16, o32 = (torch.empty(T, H, P, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    mimo_ssd.mamba3_mimo_hybrid_pools(*args, cu, p32, slots, has_init, o32)
    mimo_ssd.mamba3_mimo_hybrid_pools(*args, cu, p16, slots, has_init, o16, ssm_scale=sc)
    assert torch.equal(_bits(o16), _bits(o32))
    for i in (0, 2, 3):
        assert torch.equal(_bits(p16[i]), _bits(p32[i]))
    exp_v, exp_sc = _round_state16(p32[1], fmt)
    assert torch.equal(_bits(p16[1]), _bits(exp_v))
    if sc is not None:
        assert torch.equal(_bits(sc), _bits(exp_sc))

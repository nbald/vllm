# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba-3 MIMO inference kernels (ops/mamba3) at the Olala shapes."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="Mamba-3 MIMO kernels need CUDA sm90+",
)

H, N, P, R, A = 48, 128, 64, 4, 32
CHUNK = 64 // R


def _inputs(lens: list[int], seed: int = 0) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    def ru(*s, scale=1.0):
        return torch.rand(*s, device="cuda", generator=g) * scale

    T = sum(lens)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    return dict(
        Q=rn(1, T, R, 1, N).bfloat16(),
        K=rn(1, T, R, 1, N).bfloat16(),
        V=rn(1, T, H, P).bfloat16(),
        ADT=-ru(1, H, T, scale=0.1),
        DT=ru(1, H, T, scale=0.1),
        Trap=rn(1, H, T).bfloat16(),
        Q_bias=rn(H, R, N),
        K_bias=rn(H, R, N),
        MIMO_V=rn(H, R, P),
        MIMO_Z=rn(H, R, P),
        MIMO_Out=rn(H, R, P),
        Angles=rn(1, T, H, A),
        D=rn(H),
        Z=rn(1, T, H, P).bfloat16(),
        chunk_size=CHUNK,
        rotary_dim_divisor=4,
        dtype=torch.bfloat16,
        cu_seqlens=cu.cuda(),
    )


def _slice(kw: dict, a: int, b: int) -> dict:
    """Tokens [a, b) of a single packed sequence."""
    out = dict(kw)
    for k in ("Q", "K", "V", "Angles", "Z"):
        out[k] = kw[k][:, a:b]
    for k in ("ADT", "DT", "Trap"):
        out[k] = kw[k][:, :, a:b]
    out["cu_seqlens"] = torch.tensor([0, b - a], dtype=torch.int32, device="cuda")
    return out


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float) -> None:
    err = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)
    assert err < rtol, f"relative max error {err:.3e} >= {rtol}"


@torch.inference_mode()
def test_input_states_continuation():
    """Prefill in two chunks (second resumes from the first's final state)
    matches the one-shot prefill: the chunked-prefill / prefix-cache contract."""
    from vllm.model_executor.layers.mamba.ops.mamba3.mimo import mamba3_mimo

    L, split = 700, 333
    kw = _inputs([L])
    full = mamba3_mimo(**kw, return_state=True)
    first = mamba3_mimo(**_slice(kw, 0, split), return_state=True)
    second = mamba3_mimo(
        **_slice(kw, split, L), return_state=True, Input_States=tuple(first[1:])
    )
    _close(torch.cat([first[0], second[0]], dim=1), full[0], 2e-2)
    for got, ref in zip(second[1:], full[1:]):
        _close(got, ref, 2e-2)


@torch.inference_mode()
def test_grouped_prefill_matches_single_pass():
    """Group-parallel prefill (long sequences split into virtual groups) has
    the exact contract of the single-pass kernel."""
    from vllm.model_executor.layers.mamba.ops.mamba3.mimo import (
        mamba3_mimo,
        mamba3_mimo_varlen_grouped,
    )

    kw = _inputs([5000, 200, 4200], seed=1)  # forced split (auto_split=False)
    ref = mamba3_mimo(**kw, return_state=True)
    got = mamba3_mimo_varlen_grouped(
        **kw, cu_seqlens_cpu=kw["cu_seqlens"].cpu(), auto_split=False
    )
    for g, r in zip(got, ref):
        _close(g, r, 2e-2)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cuda_step_matches_cute_step(state_dtype: torch.dtype):
    """The persistent CUDA decode step agrees with the CuteDSL reference step
    (fused bias + rotary, in-place pool update, scattered slots)."""
    from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
        mamba3_step_cuda,
    )
    from vllm.model_executor.layers.mamba.ops.mamba3.step_cute import mamba3_step_fn

    g = torch.Generator(device="cuda").manual_seed(2)
    Bs, pool = 16, 20

    def rn(*s):
        return torch.randn(*s, device="cuda", generator=g)

    pools = dict(
        ssm=rn(pool, H, P, N).to(state_dtype),
        kp=rn(pool, R, H, N).bfloat16(),
        vp=rn(pool, H, P).bfloat16(),
        ang=rn(pool, H, A),
    )
    a = -torch.rand(Bs, H, device="cuda", generator=g) - 0.1
    b, c = rn(Bs, R, N).bfloat16(), rn(Bs, R, N).bfloat16()
    d, x = rn(H), rn(Bs, H, P).bfloat16()
    dt = torch.rand(Bs, H, device="cuda", generator=g)
    trap = torch.rand(Bs, H, device="cuda", generator=g)
    xpj, opj, zpj, z = rn(R, H, P), rn(R, H, P), rn(R, H, P), rn(Bs, H, P).bfloat16()
    bq, bk, angp = rn(R, H, N), rn(R, H, N), rn(Bs, A).bfloat16()
    slots = torch.randperm(pool, device="cuda", generator=g)[:Bs].to(torch.int32)

    ref = {k: v.clone() for k, v in pools.items()}
    y_ref = torch.empty(Bs, H, P, device="cuda", dtype=torch.bfloat16)
    mamba3_step_fn(
        ref["ssm"], ref["kp"], ref["vp"], a,
        b.unsqueeze(2).expand(Bs, R, H, N), c.unsqueeze(2).expand(Bs, R, H, N),
        d, x, dt, trap, xpj, opj, None, y_ref,
        z=z, zproj=zpj, state_batch_indices=slots, update_kv_state=True,
        tile_D=64, num_warps=4, rotary_dim=2 * A, rotary_bias_q=bq,
        rotary_bias_k=bk, rotary_angle_proj=angp.unsqueeze(1).expand(Bs, H, A),
        rotary_angle_state=ref["ang"],
    )  # fmt: skip
    got = {k: v.clone() for k, v in pools.items()}
    y = torch.empty_like(y_ref)
    mamba3_step_cuda(
        got["ssm"], got["kp"], got["vp"], got["ang"], a, b, c, d, x, dt, trap,
        xpj, zpj, opj, z, bq, bk, angp, slots, y,
    )  # fmt: skip
    # An fp32 state is never rounded, so y and the state agree tightly; the
    # bf16 pools (and a bf16 state) carry bf16 rounding of the rotated B.
    tight = 1e-3 if state_dtype == torch.float32 else 2e-2
    _close(y, y_ref, tight)
    _close(got["ssm"], ref["ssm"], tight)
    _close(got["ang"], ref["ang"], 1e-5)
    for k in ("kp", "vp"):
        _close(got[k], ref[k], 2e-2)


@pytest.mark.parametrize(
    "lens,expected",
    [
        ([4096], False),
        ([8192], True),
        ([24576], True),
        ([24576, 128, 300], True),  # short neighbours do not block the split
        ([8192, 8192], False),
        ([12288, 12288], True),
        ([8192, 8192, 8192], False),
    ],
)
def test_split_policy(lens: list[int], expected: bool, monkeypatch):
    """Grouped prefill only splits when the single pass under-fills the GPU
    (decisions measured on GH200: 132 SMs, 48 heads)."""
    from vllm.model_executor.layers.mamba.ops.mamba3 import mimo

    monkeypatch.setattr(mimo, "_sm_count", lambda device: 132)
    assert mimo._split_pays(lens, H, torch.device("cuda")) is expected


def _wrapped_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    d = torch.remainder(x.double() - ref, 2 * torch.pi)
    return torch.minimum(d, 2 * torch.pi - d).max().item()


@torch.inference_mode()
def test_decode_phase_matches_prefill_long_context():
    """The rotary phase is a running sum over the whole context: after 16k
    tokens the decode steps (CUDA, CuteDSL, Triton rotary) and the prefill
    angle_dt must all sit within 1e-4 rad of the fp64 phase, so prefill and
    decode do not drift apart with the context length."""
    from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
        mamba3_step_cuda,
    )
    from vllm.model_executor.layers.mamba.ops.mamba3.angle_dt import angle_dt_fwd
    from vllm.model_executor.layers.mamba.ops.mamba3.rotary_step import (
        apply_rotary_qk_inference_fwd,
    )
    from vllm.model_executor.layers.mamba.ops.mamba3.step_cute import mamba3_step_fn

    g = torch.Generator(device="cuda").manual_seed(3)
    L = 16384

    def rn(*s):
        return torch.randn(*s, device="cuda", generator=g)

    angp = rn(L, A).bfloat16()  # decode projections are bf16
    dt = torch.nn.functional.softplus(-3.0 + rn(L, H))
    inc = torch.tanh(angp.double())[:, None, :] * torch.pi * dt.double()[..., None]
    ref = inc.sum(0)  # (H, A), unwrapped fp64 phase after L tokens

    prefill = angle_dt_fwd(
        angp.float()[None, :, None, :].expand(1, L, H, A).contiguous(),
        dt.T[None].contiguous(), chunk_size=CHUNK, return_output_state=True,
    )[1]  # fmt: skip
    assert _wrapped_err(prefill[0], ref) < 1e-4

    # decode steps from a zero phase; only the angle state matters here
    a = -torch.full((1, H), 0.5, device="cuda")
    b, c = rn(1, R, N).bfloat16(), rn(1, R, N).bfloat16()
    d, x, z = rn(H), rn(1, H, P).bfloat16(), rn(1, H, P).bfloat16()
    trap = torch.rand(1, H, device="cuda", generator=g)
    xpj, opj, zpj = rn(R, H, P), rn(R, H, P), rn(R, H, P)
    bq, bk = rn(R, H, N), rn(R, H, N)
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    pools = {
        k: dict(ssm=torch.zeros(1, H, P, N, device="cuda"),
                kp=torch.zeros(1, R, H, N, device="cuda").bfloat16(),
                vp=torch.zeros(1, H, P, device="cuda").bfloat16(),
                ang=torch.zeros(1, H, A, device="cuda"))
        for k in ("cuda", "cute")
    }  # fmt: skip
    ang_tri = torch.zeros(1, H, A, device="cuda")
    q = torch.zeros(1, R, H, N, device="cuda").bfloat16()
    y = torch.empty(1, H, P, device="cuda", dtype=torch.bfloat16)
    for t in range(L):
        dtt, apt = dt[t : t + 1].contiguous(), angp[t : t + 1]
        p = pools["cuda"]
        mamba3_step_cuda(
            p["ssm"], p["kp"], p["vp"], p["ang"], a, b, c, d, x, dtt, trap,
            xpj, zpj, opj, z, bq, bk, apt, slots, y,
        )  # fmt: skip
        p = pools["cute"]
        mamba3_step_fn(
            p["ssm"], p["kp"], p["vp"], a,
            b.unsqueeze(2).expand(1, R, H, N), c.unsqueeze(2).expand(1, R, H, N),
            d, x, dtt, trap, xpj, opj, None, y,
            z=z, zproj=zpj, state_batch_indices=slots, update_kv_state=True,
            tile_D=64, num_warps=4, rotary_dim=2 * A, rotary_bias_q=bq,
            rotary_bias_k=bk, rotary_angle_proj=apt.unsqueeze(1).expand(1, H, A),
            rotary_angle_state=p["ang"],
        )  # fmt: skip
        _, _, ang_tri = apply_rotary_qk_inference_fwd(
            q, q, ang_tri, apt.float().unsqueeze(1).expand(1, H, A).contiguous(), dtt
        )
    for name, got in (("cuda", pools["cuda"]["ang"]), ("cute", pools["cute"]["ang"]),
                      ("triton", ang_tri)):  # fmt: skip
        assert got.abs().max() < 2 * torch.pi + 1e-3, f"{name}: phase not wrapped"
        assert _wrapped_err(got[0], ref) < 1e-4, name
        assert _wrapped_err(got[0], prefill[0].double()) < 1e-4, name


def _hybrid_inputs(T: int, seed: int = 0) -> list:
    """Arguments 0..13 of mamba3_mimo_hybrid_pools at the scales seen in Olala."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    dt = torch.rand(H, T, device="cuda", generator=g) * 0.4
    adt = -dt * torch.rand(H, 1, device="cuda", generator=g) * 3
    q, k = rn(T, R, N).bfloat16(), rn(T, R, N).bfloat16()
    v, z = rn(T, H, P, scale=0.1).bfloat16(), rn(T, H, P, scale=0.1).bfloat16()
    angles, trap = rn(T, A, scale=0.3), rn(H, T).bfloat16()
    q_bias, k_bias = rn(H, R, N), rn(H, R, N)
    psi, phi, zeta, d = (
        rn(H, R, P, scale=0.5),
        rn(H, R, P, scale=0.5),
        rn(H, R, P),
        rn(H),
    )
    return [q, k, v, z, angles, dt, adt, trap, q_bias, k_bias, psi, phi, zeta, 5 * d]


def _hybrid_ref(args: list, a: int, b: int):
    """fp64 token-by-token recurrence of one sequence [a, b) from a zero state:
    outputs (b - a, H, P) and the final state (H, P, N)."""
    import math

    Q, K, V, Z, ang, DT, ADT, Trap, qb, kb, psi, phi, zeta, D = (
        x.double() for x in args
    )
    Q, K, V, Z, ang = Q[a:b], K[a:b], V[a:b], Z[a:b], ang[a:b]
    DT, ADT, Trap = DT[:, a:b], ADT[:, a:b], Trap[:, a:b]
    nq = N // 4
    th = (torch.tanh(ang)[:, None, :] * math.pi * DT.T[:, :, None]).cumsum(0)
    c, s = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]

    def rot(X, bias):
        x = X[:, None] + bias[None]
        y = x.clone()
        x0, x2 = x[..., :nq], x[..., 2 * nq : 3 * nq]
        y[..., :nq], y[..., 2 * nq : 3 * nq] = c * x0 - s * x2, s * x0 + c * x2
        return y

    qh, kh, pv = rot(Q, qb), rot(K, kb), V[:, :, None, :] * psi[None]
    gam = DT * torch.sigmoid(Trap)
    ts = gam.clone()
    ts[:, :-1] += DT[:, 1:] * torch.sigmoid(-Trap[:, 1:])
    gate = phi[None] * torch.nn.functional.silu(Z[:, :, None, :] * zeta[None])
    st = torch.zeros(H, N, P, dtype=torch.float64, device="cuda")
    out = torch.empty(b - a, H, P, dtype=torch.float64, device="cuda")
    for t in range(b - a):
        kv = torch.einsum("hrn,hrp->hnp", kh[t], pv[t])
        dec = torch.exp(ADT[:, t])[:, None, None]
        y = torch.einsum("hrn,hnp->hrp", qh[t], dec * st + gam[:, t, None, None] * kv)
        st = dec * st + ts[:, t, None, None] * kv
        out[t] = ((y + D[:, None, None] * pv[t]) * gate[t]).sum(1)
    return out, st.transpose(1, 2)


def _hybrid_run(
    args: list, lens: list[int], init: list[bool], tc: bool, monkeypatch, sm80=None
):
    from vllm.model_executor.layers.mamba.ops.mamba3 import mimo_ssd

    monkeypatch.setattr(mimo_ssd, "_SCAN_TC", tc)
    if sm80 is not None:
        monkeypatch.setattr(mimo_ssd, "_CS_CUDA", sm80)
        monkeypatch.setattr(mimo_ssd, "_SCAN_SM80", sm80)
    T, ns = sum(lens), len(lens)
    g = torch.Generator(device="cuda").manual_seed(7)
    pools = [
        torch.randn(ns, H, A, device="cuda", generator=g),
        torch.randn(ns, H, P, N, device="cuda", generator=g) * 0.1,
        torch.randn(ns, R, H, N, device="cuda", generator=g).bfloat16(),
        torch.randn(ns, H, P, device="cuda", generator=g).bfloat16(),
    ]
    has_init = torch.tensor(init, device="cuda") if any(init) else None
    out = torch.empty(T, H, P, device="cuda", dtype=torch.bfloat16)
    cu = [0] + torch.tensor(lens).cumsum(0).tolist()
    slots = torch.arange(ns, device="cuda", dtype=torch.int32)
    mimo_ssd.mamba3_mimo_hybrid_pools(*args, cu, pools, slots, has_init, out)
    return out.float(), pools


def _rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.double() - ref).norm() / ref.norm()).item()


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability().major != 10,
    reason="tcgen05 prefill kernels are sm_100 / sm_103 only",
)
@torch.inference_mode()
def test_hybrid_prefill_tcgen05_vs_fp64(monkeypatch):
    """The tcgen05 output pass and chunk-local states (B300) are at least as
    close to the fp64 recurrence as the TileLang / Triton path they replace,
    for several packed sequences of uneven length (sub-block and chunk tails),
    and agree with it when sequences resume from pool states."""
    lens = [77, 300, 161]
    args = _hybrid_inputs(sum(lens))
    errs = {}
    for tc in (False, True):
        out, pools = _hybrid_run(args, lens, [False] * 3, tc, monkeypatch)
        eo = es = 0.0
        for i, (a, b) in enumerate(zip([0, 77, 377], [77, 377, 538])):
            ro, rs = _hybrid_ref(args, a, b)
            eo, es = max(eo, _rel(out[a:b], ro)), max(es, _rel(pools[1][i], rs))
        errs[tc] = (eo, es)
    assert errs[True][0] < 1e-2 and errs[True][1] < 1e-2, errs
    assert errs[True][0] < 1.25 * errs[False][0] + 1e-3, errs
    assert errs[True][1] < 1.25 * errs[False][1] + 1e-3, errs

    init = [True, False, True]
    o_tl, p_tl = _hybrid_run(args, lens, init, False, monkeypatch)
    o_tc, p_tc = _hybrid_run(args, lens, init, True, monkeypatch)
    assert _rel(o_tc, o_tl.double()) < 2e-2
    for x, y in zip(p_tc, p_tl):
        assert _rel(x.float(), y.double()) < 2e-2


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability().major not in (8, 12),
    reason="mma.sync prefill kernels (sm80 path) target sm_8x / sm_12x",
)
@torch.inference_mode()
def test_hybrid_prefill_mma_sync_vs_fp64(monkeypatch):
    """The mma.sync chunk-local states and output pass (Ampere / Ada / consumer
    Blackwell) are at least as close to the fp64 recurrence as the TileLang /
    Triton path they replace, and agree with it when sequences resume from
    pool states."""
    lens = [77, 300, 161]
    args = _hybrid_inputs(sum(lens))
    errs = {}
    for sm80 in (False, True):
        out, pools = _hybrid_run(args, lens, [False] * 3, False, monkeypatch, sm80)
        eo = es = 0.0
        for i, (a, b) in enumerate(zip([0, 77, 377], [77, 377, 538])):
            ro, rs = _hybrid_ref(args, a, b)
            eo, es = max(eo, _rel(out[a:b], ro)), max(es, _rel(pools[1][i], rs))
        errs[sm80] = (eo, es)
    assert errs[True][0] < 1e-2 and errs[True][1] < 1e-2, errs
    assert errs[True][0] < 1.25 * errs[False][0] + 1e-3, errs
    assert errs[True][1] < 1.25 * errs[False][1] + 1e-3, errs

    init = [True, False, True]
    o_tl, p_tl = _hybrid_run(args, lens, init, False, monkeypatch, False)
    o_mma, p_mma = _hybrid_run(args, lens, init, False, monkeypatch, True)
    assert _rel(o_mma, o_tl.double()) < 2e-2
    for x, y in zip(p_mma, p_tl):
        assert _rel(x.float(), y.double()) < 2e-2

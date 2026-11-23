# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_TPA_MIXED_FUSED: q / k / v of a mixed (verify + prefill) DiffTPA batch.

The verify rows must equal what a pure verify batch gives them (the fused spec
kernel alone), and the prefill rows what a pure prefill batch gives them (the
op-by-op path on those rows alone), pools included."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.mamba.olala.diff_tpa import OlalaDiffTPAAttention as A
from vllm.model_executor.layers.mamba.olala.diff_tpa_decode import tpa_decode_qkv_spec

HKV, GQ, R, D, K = 4, 4, 4, 128, 3


def _norm(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class _Norm:
    """Stand-in for the q / k norms: the fused kernel reads .norm.weight / .eps,
    the op-by-op path calls the module (any norm works: both sides of each
    comparison use the same one)."""

    def __init__(self, dev):
        self.norm = SimpleNamespace(
            weight=torch.randn(D, device=dev).bfloat16() * 0.1, eps=1e-6
        )

    def __call__(self, x):
        return _norm(x)


def _layer(k_pool, v_pool, params):
    """A stand-in DiffTPA layer over the given pools; ``params`` (norms and the
    softmax scaler) are shared by every layer of one test."""
    self = SimpleNamespace(
        num_kv_heads_local=HKV,
        rank=R,
        head_dim=D,
        num_attention_heads_local=HKV * GQ,
        q_size_local=HKV * GQ * D,
        kv_size_local=HKV * D,
        token_shift=True,
        qk_norm=True,
        rotary_emb=None,
        scalable_softmax=True,
        config=SimpleNamespace(sliding_window_size=0),
        snr=GQ - 1,
        **params,
    )
    self._ts_kv = lambda spec=False: (k_pool, v_pool)
    self._common_fused_args = lambda kp, vp: A._common_fused_args(self, kp, vp)
    self._alloc_shift = A._alloc_shift
    self._blend = A._blend
    self._seed_prefill_rows_dev = A._seed_prefill_rows_dev
    self._shift_prefill_rows = lambda *a: A._shift_prefill_rows(self, *a)
    return self


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("nosync", [False, True])
def test_mixed_fused_rows_match_pure_batches(nosync):
    torch.manual_seed(0)
    dev = "cuda"
    nd, lens_p, pad = 3, [9, 5], 2
    ndt, npt = nd * (1 + K), sum(lens_p)
    n = ndt + npt + pad  # piecewise padding rows after the prefill rows
    P = 64
    k_pool = torch.randn(P, HKV, D, device=dev).bfloat16()
    v_pool = torch.randn(P, HKV, D, device=dev).bfloat16()
    q_all = torch.randn(n, HKV * GQ * D, device=dev).bfloat16()
    ak, av = (torch.randn(n, HKV * R, device=dev).bfloat16() for _ in range(2))
    bk, bv = (torch.randn(n, R * D, device=dev).bfloat16() for _ in range(2))
    al = (
        torch.randn(n, HKV, device=dev).bfloat16(),
        torch.randn(n, HKV, device=dev).bfloat16(),
    )
    positions = torch.cat(
        [
            torch.arange(10, 10 + ndt),
            torch.arange(0, 9),
            torch.arange(40, 45),
            torch.zeros(pad, dtype=torch.long),
        ]
    ).to(dev)
    cols = torch.arange(1, 1 + nd * (1 + K), dtype=torch.int32, device=dev).view(
        nd, 1 + K
    )
    qsl_d = torch.arange(0, ndt + 1, 1 + K, dtype=torch.int32, device=dev)
    spec = SimpleNamespace(
        query_start_loc_d=qsl_d,
        state_cols=cols,
        num_accepted_tokens=torch.tensor([1, 2, 4], dtype=torch.int32, device=dev),
    )
    qsl_p = torch.tensor([0, 9, 14], dtype=torch.int32)
    slots = torch.cat(
        [cols[:, 0], torch.tensor([40, 41], dtype=torch.int32, device=dev)]
    )
    md = SimpleNamespace(
        spec=spec,
        num_decodes=nd,
        num_prefills=2,
        num_decode_tokens=ndt,
        num_prefill_tokens=npt,
        state_indices_tensor=slots,
        query_start_loc_p=qsl_p.to(dev),
        query_start_loc_p_cpu=qsl_p,
        has_initial_state=torch.tensor([False, True], device=dev),
        has_initial_state_cpu=torch.tensor([False, True]),
        state_indices_p_cpu=None if nosync else [40, 41],
    )

    kp0, vp0 = k_pool.clone(), v_pool.clone()
    params = dict(
        softmax_scaler=torch.rand(HKV * GQ, device=dev).bfloat16(),
        q_norm=_Norm(dev),
        k_norm=_Norm(dev),
    )
    lay = _layer(k_pool, v_pool, params)
    q, k, v = A._mixed_fused_qkv(lay, md, positions, q_all, ak, av, bk, bv, al)

    # verify rows: the fused kernel alone, as in a pure verify step
    kp1, vp1 = kp0.clone(), vp0.clone()
    lay1 = _layer(kp1, vp1, params)
    qd, kd, vd = tpa_decode_qkv_spec(
        q_all[:ndt],
        ak[:ndt],
        av[:ndt],
        bk[:ndt],
        bv[:ndt],
        al[0][:ndt],
        al[1][:ndt],
        positions[:ndt],
        qsl_d,
        cols,
        spec.num_accepted_tokens,
        *lay1._common_fused_args(kp1, vp1),
    )
    assert (
        torch.equal(q[:ndt], qd)
        and torch.equal(k[:ndt], kd)
        and torch.equal(v[:ndt], vd)
    )

    # prefill rows: the same batch without its verify rows
    md_p = SimpleNamespace(
        **{
            **vars(md),
            "spec": None,
            "num_decodes": 0,
            "num_decode_tokens": 0,
            "state_indices_tensor": slots[nd:],
        }
    )
    kp2, vp2 = kp0.clone(), vp0.clone()
    lay2 = _layer(kp2, vp2, params)
    r = slice(ndt, n)
    m = n - ndt
    qq = q_all[r].view(m, HKV * GQ, D)
    kk = torch.bmm(ak[r].view(m, HKV, R), bk[r].view(m, R, D)).div_(R)
    vv = torch.bmm(av[r].view(m, HKV, R), bv[r].view(m, R, D)).div_(R)
    ks, vs = A._alloc_shift(kk, vv, npt)
    A._shift_prefill_rows(
        lay2,
        md_p,
        kk,
        vv,
        slots[nd:],
        kp2,
        vp2,
        al[0][r],
        al[1][r],
        positions[r],
        0,
        2,
        0,
        npt,
        ks,
        vs,
    )
    assert torch.equal(k[ndt:], _norm(ks).reshape(m, -1)) and torch.equal(
        v[ndt:], vs.reshape(m, -1)
    )
    p1 = (positions[r].float() + 1.0).clamp_min(1.0).log()
    scale = lay2.softmax_scaler.to(torch.bfloat16) * p1.to(torch.bfloat16).unsqueeze(-1)
    q_ref = (_norm(qq) * scale.unsqueeze(-1)).reshape(m, -1)
    assert torch.equal(q[ndt:], q_ref)
    # pools: verify columns from the kernel, prefill slots from the op-by-op path
    for c in cols.flatten().tolist():
        assert torch.equal(k_pool[c], kp1[c]) and torch.equal(v_pool[c], vp1[c])
    for s in (40, 41):
        assert torch.equal(k_pool[s], kp2[s]) and torch.equal(v_pool[s], vp2[s])

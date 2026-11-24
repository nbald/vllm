# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_PREFILL_NOSYNC: the Olala metadata builders build a prefill step without
reading device tensors on the host. CPU only.

The proof uses the meta device: the runner's device tensors (query_start_loc,
seq_lens, block table) live on "meta", where any host read (.to("cpu"),
.tolist(), .item()) raises. The flag-off build must raise there, the flag-on
build must not, and both must agree on CPU tensors."""

from types import SimpleNamespace

import pytest
import torch
from tests.v1.core.utils import create_scheduler
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import olala_spec_utils
from vllm.v1.attention.backends.mamba3_attn import Mamba3AttentionMetadataBuilder
from vllm.v1.attention.backends.olala_diff_tpa_attn import OlalaDiffTPAMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test

K = 3


@pytest.fixture
def nosync(monkeypatch):
    def set_(on: bool, check: bool = False):
        monkeypatch.setenv("OLALA_PREFILL_NOSYNC", "1" if on else "0")
        if check:
            monkeypatch.setenv("OLALA_PREFILL_NOSYNC_CHECK", "1")
        else:
            monkeypatch.delenv("OLALA_PREFILL_NOSYNC_CHECK", raising=False)
        olala_spec_utils.prefill_nosync.cache_clear()

    yield set_
    olala_spec_utils.prefill_nosync.cache_clear()


def _vllm_config():
    return create_scheduler(num_speculative_tokens=K).vllm_config


def _builders(device):
    cfg = _vllm_config()
    spec = MambaSpec(
        block_size=16, shapes=((4,),), dtypes=(torch.float32,), num_speculative_blocks=K
    )
    return (
        Mamba3AttentionMetadataBuilder(spec, ["m"], cfg, device),
        OlalaDiffTPAMetadataBuilder(spec, ["v"], cfg, device),
    )


def _batch(
    device, computed=(57, 31, 0, 100), qlens=(4, 4, 30, 50), ub_extra=(K, K, 0, 0)
):
    """Two verify rows (1 + K tokens) then two prefill rows: a fresh prompt and a
    continuation chunk. The CPU upper bound is optimistic for the verify rows
    (async spec) and exact for the prefill rows."""
    n = len(qlens)
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(qlens, dtype=torch.int32), 0)
    seq = torch.tensor(computed, dtype=torch.int32) + torch.tensor(
        qlens, dtype=torch.int32
    )
    ub = seq + torch.tensor(ub_extra, dtype=torch.int32)
    bt = torch.arange(n * (1 + K), dtype=torch.int32).view(n, 1 + K) + 1
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=seq.to(device),
        num_reqs=n,
        num_actual_tokens=int(qsl[-1]),
        max_query_len=max(qlens),
        max_seq_len=int(seq.max()),
        block_table_tensor=bt.to(device),
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64, device=device),
        seq_lens_cpu_upper_bound=ub,
        is_prefilling=torch.tensor([False, False, True, True]),
    )


def _build(builder, m, device):
    acc = torch.ones(m.num_reqs, dtype=torch.int32, device=device)
    return builder.build(0, m, num_accepted_tokens=acc)


def test_prefill_computed_cpu_exact(nosync):
    m = _batch("cpu")
    got = olala_spec_utils.prefill_computed_cpu(m, 2, 2)
    assert got.tolist() == [0, 100]
    assert torch.equal(got, m.compute_num_computed_tokens()[2:].to(torch.int64))
    # capture dummies carry no CPU bound: the builders fall back
    m.seq_lens_cpu_upper_bound = None
    assert olala_spec_utils.prefill_computed_cpu(m, 2, 2) is None


def test_nosync_check_catches_a_wrong_bound(nosync):
    nosync(True, check=True)
    m = _batch("cpu", ub_extra=(K, K, 1, 0))  # wrong (non-exact) bound on a prefill row
    with pytest.raises(AssertionError, match="CPU computed tokens"):
        olala_spec_utils.prefill_computed_cpu(m, 2, 2)


@pytest.mark.parametrize("which", [0, 1])
def test_nosync_matches_sync_metadata(nosync, which):
    nosync(False)
    ref = _build(_builders("cpu")[which], _batch("cpu"), "cpu")
    nosync(True, check=True)
    got = _build(_builders("cpu")[which], _batch("cpu"), "cpu")
    assert (
        (got.num_decodes, got.num_prefills)
        == (ref.num_decodes, ref.num_prefills)
        == (2, 2)
    )
    assert torch.equal(got.has_initial_state_cpu, ref.has_initial_state_cpu)
    assert got.has_initial_state_cpu.tolist() == [False, True]
    assert torch.equal(got.query_start_loc_p_cpu, ref.query_start_loc_p_cpu)
    assert torch.equal(got.has_initial_state, ref.has_initial_state)
    if which == 0:
        assert got.has_initial_any is ref.has_initial_any is True
    else:
        # the TPA builder no longer copies the slots back; the V layers use
        # the device slots
        assert ref.state_indices_p_cpu == [9, 13]
        assert got.state_indices_p_cpu is None


@pytest.mark.parametrize("which", [0, 1])
def test_nosync_build_reads_no_device_tensor(nosync, which):
    """On the meta device every host read raises: the flag-off build does one,
    the flag-on build none."""
    meta = torch.device("meta")
    nosync(False)
    with pytest.raises((NotImplementedError, RuntimeError)):
        _build(_builders(meta)[which], _batch(meta), meta)
    nosync(True)
    md = _build(_builders(meta)[which], _batch(meta), meta)
    assert md.has_initial_state_cpu.tolist() == [False, True]
    assert md.has_initial_state.device.type == "meta"


# --------------------------------------------------------------- DiffTPA prefill seeds


def _seed_case(np_, lens, init, device="cpu"):
    from vllm.model_executor.layers.mamba.olala.diff_tpa import OlalaDiffTPAAttention

    torch.manual_seed(np_)
    kv, d = 4, 16
    npt = sum(lens)
    qsl = torch.zeros(np_ + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    nd = 3
    slots = torch.tensor([0] * nd + [7, 2, 11, 5][:np_], dtype=torch.int32)
    md = SimpleNamespace(
        query_start_loc_p_cpu=qsl,
        query_start_loc_p=qsl.clone(),
        has_initial_state=torch.tensor(init),
        has_initial_state_cpu=torch.tensor(init),
        state_indices_p_cpu=slots[nd:].tolist(),
    )
    k = torch.randn(npt, kv, d).bfloat16()
    v = torch.randn(npt, kv, d).bfloat16()
    pool_k = torch.randn(16, kv, d).bfloat16()
    pool_v = torch.randn(16, kv, d).bfloat16()
    return OlalaDiffTPAAttention, md, k, v, slots, pool_k, pool_v, nd


@pytest.mark.parametrize(
    "lens,init",
    [
        ([5], [False]),
        ([5], [True]),
        ([3, 1, 7], [True, False, True]),
        ([4, 0, 2, 6], [True, True, False, True]),
    ],
)
def test_seed_prefill_rows_dev_matches_loop(lens, init):
    cls, md, k, v, slots, pk, pv, nd = _seed_case(len(lens), lens, init)
    np_ = len(lens)

    def loop():
        kp, vp = k.clone(), v.clone()
        pool_k, pool_v = pk.clone(), pv.clone()
        k_prev, v_prev = torch.empty_like(kp), torch.empty_like(vp)
        qsl = md.query_start_loc_p_cpu.tolist()
        for r in range(np_):
            s, e = qsl[r], qsl[r + 1]
            if e == s:
                continue
            slot = md.state_indices_p_cpu[r]
            if init[r]:
                k_prev[s] = pool_k[slot]
                v_prev[s] = pool_v[slot]
            else:
                k_prev[s] = 0
                v_prev[s] = 0
            k_prev[s + 1 : e] = kp[s : e - 1]
            v_prev[s + 1 : e] = vp[s : e - 1]
            pool_k[slot] = kp[e - 1]
            pool_v[slot] = vp[e - 1]
        return k_prev, v_prev, pool_k, pool_v

    ref = loop()
    pool_k, pool_v = pk.clone(), pv.clone()
    k_prev, v_prev = torch.empty_like(k), torch.empty_like(v)
    cls._seed_prefill_rows_dev(md, k, v, k_prev, v_prev, slots, pool_k, pool_v, nd, np_)
    for a, b in zip(ref, (k_prev, v_prev, pool_k, pool_v)):
        assert torch.equal(a, b)

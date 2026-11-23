# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec layout switches (state.py), CPU only: the token-shift fold (OLALA_TS_FOLD,
F0a) and light speculative columns (OLALA_SPEC_LIGHT, F2a) are opt-in, take
effect only where their kernels can run, and shrink the per-request state the
way the capacity design says (one Mamba-3 page instead of 1 + K; the token-shift
buffers inside the Mamba-3 pages)."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.olala import state as st

pytestmark = pytest.mark.cpu_test

# Olala-v1.0-mini-7A1B
HF = SimpleNamespace(
    hidden_size=1536,
    mamba_headdim=64,
    mamba_d_state=128,
    mamba_mimo_dim=4,
    num_attention_heads=48,
    num_signal_heads_diff=36,
    head_dim=128,
    tpa_rank=4,
    token_shift_attn=True,
    layers_config="MMMMV" * 7 + "M",
)


def _cfg(k=3, ssm="float32", tp=1):
    return SimpleNamespace(
        num_speculative_tokens=k,
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype=ssm),
        model_config=SimpleNamespace(hf_config=HF),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


@pytest.fixture
def sm80(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda c: c <= 120, raising=False
    )
    monkeypatch.delenv("OLALA_REPLAY_SSM", raising=False)
    monkeypatch.delenv("OLALA_MAMBA3_VERIFY", raising=False)
    monkeypatch.setenv("OLALA_VOLTA", "0")  # not SM70, also under volta-forever


def _bytes(shapes, dtypes):
    return sum(
        math.prod(s) * torch.empty((), dtype=d).element_size()
        for s, d in zip(shapes, dtypes)
    )


def test_layout_switches_are_opt_in(monkeypatch, sm80):
    monkeypatch.delenv("OLALA_SPEC_LIGHT", raising=False)
    monkeypatch.delenv("OLALA_TS_FOLD", raising=False)
    assert st.spec_light_cols(_cfg()) == 0
    assert not st.ts_fold_active(HF)
    monkeypatch.setenv("OLALA_SPEC_LIGHT", "1")
    monkeypatch.setenv("OLALA_TS_FOLD", "1")
    assert st.spec_light_cols(_cfg(k=3)) == 4
    assert st.spec_light_cols(_cfg(k=5)) == 6
    assert st.ts_fold_active(HF)


def test_light_columns_need_their_kernel(monkeypatch, sm80):
    monkeypatch.setenv("OLALA_SPEC_LIGHT", "1")
    assert st.spec_light_cols(_cfg(k=0)) == 0  # no speculation
    assert (
        st.spec_light_cols(_cfg(ssm="bfloat16")) == 0
    )  # the verify kernel is fp32-only
    assert st.spec_light_cols(_cfg(tp=2)) == 0  # other local head count
    monkeypatch.setenv("OLALA_MAMBA3_VERIFY", "chain")
    assert st.spec_light_cols(_cfg()) == 0  # only the fused verify commits
    monkeypatch.delenv("OLALA_MAMBA3_VERIFY")
    monkeypatch.setenv("OLALA_REPLAY_SSM", "1")
    if hasattr(st.replay_config, "cache_clear"):
        st.replay_config.cache_clear()
    try:
        assert st.spec_light_cols(_cfg()) == 0  # ReplaySSM state
    finally:
        monkeypatch.delenv("OLALA_REPLAY_SSM")
        if hasattr(st.replay_config, "cache_clear"):
            st.replay_config.cache_clear()


def test_fold_needs_a_mamba3_host_before_every_v(monkeypatch):
    monkeypatch.setenv("OLALA_TS_FOLD", "1")
    assert st.ts_fold_active(HF, pp_size=1)
    assert not st.ts_fold_active(HF, pp_size=2)
    assert not st.ts_fold_active(
        SimpleNamespace(**{**vars(HF), "layers_config": "VMMMM"})
    )
    assert not st.ts_fold_active(
        SimpleNamespace(**{**vars(HF), "token_shift_attn": False})
    )


@pytest.mark.parametrize("k", [3, 5, 7])
def test_light_page_and_fold_sizes(monkeypatch, sm80, k):
    """Per request and Mamba-3 layer: 1 + K column-slot pages against one light page;
    the folded token-shift buffers are a few KB, far below the page padding."""
    dims = dict(
        tp_world_size=1,
        num_heads=48,
        head_dim=64,
        d_state=128,
        mimo_dim=4,
        num_rope_angles=32,
    )
    base = _bytes(
        st.mamba3_state_shape(**dims),
        st.mamba3_state_dtype(torch.bfloat16, "auto", "float32"),
    )
    light_shapes = st.mamba3_state_shape(**dims, light_cols=1 + k)
    light_dtypes = st.mamba3_state_dtype(torch.bfloat16, "auto", "float32", 1 + k)
    light = _bytes(light_shapes, light_dtypes)
    assert light_shapes[1] == (48, 64, 128)  # the fp32 SSM state once
    assert light < 1.4 * base  # one page, < (1 + K) * base
    saved_mb = ((1 + k) * base - light) / 2**20
    assert saved_mb > 0.9 * k * base / 2**20
    ts_shapes, ts_dtypes = st.folded_token_shift_layout(
        HF, 1, torch.bfloat16, "auto", 1 + k
    )
    ts = _bytes(ts_shapes, ts_dtypes)
    assert ts_shapes[0] == (1 + k, 12, 128)  # one buffer per column, 12 kv heads
    assert ts < 0.03 * light


def test_sm70_keeps_column_slots(monkeypatch, sm80):
    """V100 (natively or through volta-forever): the Volta verify kernel implements the column-slot protocol
    only, so the light layout stays off there."""
    monkeypatch.setenv("OLALA_SPEC_LIGHT", "1")
    monkeypatch.setenv("OLALA_VOLTA", "1")
    assert st.spec_light_cols(_cfg(k=5)) == 0
    monkeypatch.setenv("OLALA_VOLTA", "0")
    assert st.spec_light_cols(_cfg(k=5)) == 6

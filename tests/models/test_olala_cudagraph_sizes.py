# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the Olala CUDA graph size lists: no speculation, static K and dynamic K tiers."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import (
    _olala_decode_cudagraph_sizes,
    _olala_prefill_cudagraph_sizes,
    _olala_spec_decode_sizes,
)

LADDER_2048 = sorted(
    set([1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16)) + [640, 768, 1024, 1280, 1536, 1792, 2048])
)


class _Spec(SimpleNamespace):
    def uses_dynamic_speculative_decoding(self) -> bool:
        return self.num_speculative_tokens_per_batch_size is not None


def _config(num_spec=0, schedule=None, max_num_seqs=128, max_num_batched_tokens=2048, sizes=None, max_size=None):
    spec = None
    if num_spec:
        spec = _Spec(num_speculative_tokens=num_spec, num_speculative_tokens_per_batch_size=schedule)
    return SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max_size),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OLALA_PREFILL_GRAPHS_MAX", "4096")
    monkeypatch.delenv("OLALA_PREFILL_GRAPHS", raising=False)
    monkeypatch.delenv("OLALA_SPEC_GRAPHS_MAX_REQS", raising=False)
    monkeypatch.delenv("OLALA_DECODE_GRAPHS_MAX", raising=False)


def _prefill(cfg):
    _olala_prefill_cudagraph_sizes(cfg)
    cc = cfg.compilation_config
    assert cc.cudagraph_capture_sizes == sorted(set(cc.cudagraph_capture_sizes))
    assert cc.max_cudagraph_capture_size == cc.cudagraph_capture_sizes[-1]
    return cc.cudagraph_capture_sizes


def test_no_spec_keeps_the_ladder():
    cfg = _config()
    assert _olala_spec_decode_sizes(cfg, 2048) == []
    assert _prefill(cfg) == LADDER_2048


def test_static_k2_adds_every_request_count():
    sizes = _prefill(_config(num_spec=2, max_num_seqs=16))
    assert set(LADDER_2048) <= set(sizes)
    assert all(3 * n in sizes for n in range(1, 17))  # 12 (4 requests) was missing: padded to 18
    assert sizes[-1] == 2048


def test_static_k5_caps_the_request_count(monkeypatch):
    sizes = _prefill(_config(num_spec=5, max_num_seqs=128))
    assert all(6 * n in sizes for n in range(1, 33))  # 30 (5 requests) was missing: padded to 36
    assert 6 * 33 not in sizes  # above the default cap of 32 requests, the ladder alone
    monkeypatch.setenv("OLALA_SPEC_GRAPHS_MAX_REQS", "128")
    sizes = _prefill(_config(num_spec=5, max_num_seqs=128))
    assert all(6 * n in sizes for n in range(1, 129))


def test_spec_sizes_stay_under_the_budget(monkeypatch):
    monkeypatch.setenv("OLALA_SPEC_GRAPHS_MAX_REQS", "512")
    sizes = _prefill(_config(num_spec=5, max_num_seqs=512, max_num_batched_tokens=600))
    assert sizes[-1] == 600
    assert 600 in sizes and 594 in sizes and all(s <= 600 for s in sizes)


def test_dynamic_tiers():
    # K2 at 1-2 and 7-16 requests, K0 at 3-6: K0 steps are one token per request.
    sizes = _prefill(_config(num_spec=2, schedule=[(1, 2, 2), (3, 6, 0), (7, 16, 2)], max_num_seqs=16))
    assert all(3 * n in sizes for n in range(1, 17))
    assert all(n in sizes for n in range(1, 7))
    assert 7 not in sizes  # the K0 tier stops at 6 requests
    # K3 up to 4 requests, then K1: the drafter still drafts K3 blocks at every batch size.
    spec = _olala_spec_decode_sizes(_config(num_spec=3, schedule=[(1, 4, 3), (5, 32, 1)], max_num_seqs=32), 2048)
    assert spec == sorted({4 * n for n in range(1, 33)} | {2 * n for n in range(1, 33)})


def test_user_sizes_and_small_budgets_are_left_alone():
    cfg = _config(num_spec=2, sizes=[1, 2, 4, 8])
    _olala_prefill_cudagraph_sizes(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]
    cfg = _config(num_spec=2, max_num_batched_tokens=512)  # vLLM's own list, which adds its decode sizes
    _olala_prefill_cudagraph_sizes(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes is None


def test_decode_hook():
    cfg = _config(max_num_seqs=1024)
    _olala_decode_cudagraph_sizes(cfg)
    assert cfg.compilation_config.max_cudagraph_capture_size == 1024
    assert cfg.compilation_config.cudagraph_capture_sizes is None  # no spec: vLLM's grid up to the ceiling
    cfg = _config(num_spec=2, max_num_seqs=1024)
    _olala_decode_cudagraph_sizes(cfg)
    sizes = cfg.compilation_config.cudagraph_capture_sizes
    assert sizes[-1] == cfg.compilation_config.max_cudagraph_capture_size == 1024
    assert all(3 * n in sizes for n in range(1, 33))
    assert {1, 2, 4, 8, 248, 256, 1008} <= set(sizes)

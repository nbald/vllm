# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_STEP_TRACE row assembly on CPU, with a fake CUDA event clock."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm.v1.worker.gpu import olala_step_trace as st

pytestmark = pytest.mark.cpu_test


class FakeEvent:
    clock = 0.0

    def __init__(self, enable_timing=True):
        self.t = None

    def record(self):
        self.t = FakeEvent.clock

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return other.t - self.t


def test_step_trace_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(st.torch.cuda, "Event", FakeEvent)
    path = tmp_path / "steps.jsonl"
    tr = st.OlalaStepTrace(str(path))

    def step(prefill: bool, cg, draft_cg):
        FakeEvent.clock += 0.5
        tr.step_start()
        nst = np.array([600, 4, 4] if prefill else [4, 4, 4])
        pre = np.array([True, False, False] if prefill else [False] * 3)
        n = int(nst.sum())
        batch = SimpleNamespace(
            num_reqs=3,
            num_tokens=n,
            num_tokens_after_padding=n,
            num_scheduled_tokens=nst,
            is_prefilling_np=pre,
            num_draft_tokens=6,
        )
        tr.batch(batch, cg)
        FakeEvent.clock += 1.0
        tr.forward_start()
        FakeEvent.clock += 10.0 if prefill else 3.0
        tr.forward_end()
        tr.sample_start()
        FakeEvent.clock += 0.4
        tr.sample_end()
        tr.drafter_start(3)
        FakeEvent.clock += 0.9
        tr.drafter_end(draft_cg)
        tr.step_end()

    step(True, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL)
    step(False, CUDAGraphMode.FULL, CUDAGraphMode.FULL)
    tr.flush(final=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    mixed, full = rows
    assert mixed["cg_mode"] == "PIECEWISE" and mixed["draft_cg_mode"] == "FULL"
    shape = (mixed["prefill_reqs"], mixed["prefill_tokens"], mixed["decode_reqs"])
    assert shape == (1, 600, 2)
    assert full["cg_mode"] == "FULL" and full["prefill_tokens"] == 0
    assert mixed["gpu_forward_ms"] == pytest.approx(10.0)
    assert full["gpu_step_ms"] == pytest.approx(1.0 + 3.0 + 0.4 + 0.9)
    assert full["gpu_idle_before_ms"] == pytest.approx(0.5)
    assert full["num_spec_tokens"] == 3


def test_step_trace_off_by_default(monkeypatch):
    monkeypatch.delenv("OLALA_STEP_TRACE", raising=False)
    monkeypatch.setattr(st, "_INIT", False)
    monkeypatch.setattr(st, "_TRACE", None)
    assert st.get_step_trace() is None


def test_step_trace_counts_host_syncs(tmp_path, monkeypatch):
    """OLALA_STEP_TRACE_SYNCS=1: torch's sync-debug warnings are counted per step
    (here raised by hand: no CUDA on the test host) and not printed."""
    import warnings

    monkeypatch.setattr(st.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    monkeypatch.setattr(warnings, "filters", list(warnings.filters))
    monkeypatch.setenv("OLALA_STEP_TRACE_SYNCS", "1")
    path = tmp_path / "steps.jsonl"
    tr = st.OlalaStepTrace(str(path))
    printed = []
    monkeypatch.setattr(warnings, "_showwarnmsg_impl", lambda msg: printed.append(msg))

    def sync_op():
        warnings.warn(
            "called a synchronizing CUDA operation", UserWarning, stacklevel=1
        )

    for n_syncs in (2, 0, 3):
        tr.step_start()
        for _ in range(n_syncs):
            sync_op()
        tr.step_end()
        sync_op()  # between steps: not charged to the next step
    tr.flush(final=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["host_syncs"] for r in rows] == [2, 0, 3]
    site, count = rows[0]["sync_sites"][0]
    assert site.startswith("test_olala_step_trace.py:") and count == 2
    assert not printed

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-step trace of the V2 model runner (evaluation only).

``OLALA_STEP_TRACE=<path.jsonl>`` records, for every real (non-dummy) step: the
batch shape (requests, tokens, prefill requests / tokens, draft tokens, the
target's and the drafter's CUDA graph modes), host timestamps of
``execute_model`` and ``sample_tokens``, and CUDA events around the GPU phases
(input prep, forward, sampling / rejection, drafter, tail). Events are resolved
lazily with ``query()`` a few steps later, so tracing adds no host-device
synchronization. Rows are appended to the file every 256 resolved steps and at
exit.

``OLALA_STEP_TRACE_SYNCS=1`` also counts the host-device synchronizations of
every step: torch's sync debug mode is set to "warn" and its warnings are
counted (not printed) per step, with the five most frequent call sites
(``host_syncs``, ``sync_sites``). Debug only: the check itself adds host work on
every synchronizing call.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import time
import warnings
from collections import Counter, deque

import torch

_PHASES = (
    "step_start",
    "forward_start",
    "forward_end",
    "sample_end",
    "drafter_start",
    "drafter_end",
    "step_end",
)
# (row key, start event, end event) of the GPU phases of a step, in ms.
_GPU_PHASES = (
    ("gpu_prep_ms", "step_start", "forward_start"),
    ("gpu_forward_ms", "forward_start", "forward_end"),
    ("gpu_sample_ms", "forward_end", "sample_end"),
    ("gpu_mid_ms", "sample_end", "drafter_start"),
    ("gpu_draft_ms", "drafter_start", "drafter_end"),
    ("gpu_step_ms", "step_start", "step_end"),
)


_SYNC_MSG = "synchronizing CUDA operation"


class _SyncCounter:
    """Counts torch's sync-debug warnings by call site, without printing them."""

    def __init__(self):
        self.sites: Counter = Counter()
        self.total = 0
        try:
            import torch

            torch.cuda.set_sync_debug_mode("warn")
        except Exception:  # noqa: BLE001 - no CUDA: count explicit warnings only (tests)
            pass
        warnings.filterwarnings("always", message=f".*{_SYNC_MSG}.*")
        prev = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if _SYNC_MSG in str(message):
                self.total += 1
                self.sites[f"{os.path.basename(filename)}:{lineno}"] += 1
                return
            prev(message, category, filename, lineno, file, line)

        warnings.showwarning = showwarning

    def take(self) -> tuple[int, list]:
        n, top = self.total, self.sites.most_common(5)
        self.total = 0
        self.sites = Counter()
        return n, top


class _Step:
    __slots__ = ("events", "info", "host")

    def __init__(self):
        self.events: dict[str, torch.cuda.Event] = {}
        self.info: dict = {}
        self.host: dict[str, float] = {}


class OlalaStepTrace:
    def __init__(self, path: str):
        self.path = path
        self._cur: _Step | None = None
        self._pending: deque[_Step] = deque()
        self._rows: list[dict] = []
        self._prev_end: torch.cuda.Event | None = None
        self._prev_end_host: float | None = None
        self._n = 0
        self._syncs = (
            _SyncCounter() if os.environ.get("OLALA_STEP_TRACE_SYNCS") == "1" else None
        )
        atexit.register(self.flush, True)

    # -- recording -------------------------------------------------------------
    def _record(self, name: str) -> None:
        if self._cur is None:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._cur.events[name] = ev

    def step_start(self) -> None:
        if self._syncs is not None:
            self._syncs.take()  # syncs between steps are not charged to the next one
        self._cur = _Step()
        self._cur.host["exec_start"] = time.perf_counter()
        self._record("step_start")

    def batch(self, input_batch, cg_mode, num_spec_tokens: int | None = None) -> None:
        if self._cur is None:
            return
        try:
            import numpy as np

            nst = np.asarray(input_batch.num_scheduled_tokens[: input_batch.num_reqs])
            pre = np.asarray(
                input_batch.is_prefilling_np[: input_batch.num_reqs]
            ).astype(bool)
            self._cur.info.update(
                num_reqs=int(input_batch.num_reqs),
                num_tokens=int(input_batch.num_tokens),
                num_tokens_padded=int(
                    getattr(input_batch, "num_tokens_after_padding", 0) or 0
                ),
                prefill_reqs=int(pre.sum()),
                prefill_tokens=int(nst[pre].sum()),
                decode_reqs=int((~pre).sum()),
                max_qlen=int(nst.max()) if nst.size else 0,
                draft_tokens=int(getattr(input_batch, "num_draft_tokens", 0) or 0),
                cg_mode=str(getattr(cg_mode, "name", cg_mode)),
            )
        except Exception as e:  # noqa: BLE001 - tracing must never break a step
            self._cur.info["batch_error"] = repr(e)

    def forward_start(self) -> None:
        self._record("forward_start")

    def forward_end(self) -> None:
        self._record("forward_end")
        if self._cur is not None:
            self._cur.host["exec_end"] = time.perf_counter()

    def sample_start(self) -> None:
        if self._cur is not None:
            self._cur.host["sample_start"] = time.perf_counter()

    def sample_end(self) -> None:
        self._record("sample_end")

    def drafter_start(self, num_spec_tokens: int) -> None:
        if self._cur is not None:
            self._cur.info["num_spec_tokens"] = int(num_spec_tokens)
        self._record("drafter_start")

    def drafter_end(self, cg_mode=None) -> None:
        if self._cur is not None and cg_mode is not None:
            self._cur.info["draft_cg_mode"] = str(getattr(cg_mode, "name", cg_mode))
        self._record("drafter_end")

    def step_end(self) -> None:
        if self._cur is None:
            return
        self._record("step_end")
        self._cur.host["sample_end"] = time.perf_counter()
        if self._syncs is not None:
            n, top = self._syncs.take()
            self._cur.info["host_syncs"] = n
            self._cur.info["sync_sites"] = top
        step = self._cur
        step.info["prev_end"] = self._prev_end
        step.info["prev_end_host"] = self._prev_end_host
        self._prev_end = step.events["step_end"]
        self._prev_end_host = step.host["sample_end"]
        self._pending.append(step)
        self._cur = None
        self._resolve()

    # -- resolution ------------------------------------------------------------
    def _resolve(self, force: bool = False) -> None:
        while self._pending:
            step = self._pending[0]
            last = step.events["step_end"]
            if not force and not last.query():
                break
            if force:
                last.synchronize()
            self._pending.popleft()
            ev = step.events
            row = {
                k: v
                for k, v in step.info.items()
                if k not in ("prev_end", "prev_end_host")
            }
            row["i"] = self._n
            self._n += 1
            for key, a, b in _GPU_PHASES:
                row[key] = ev[a].elapsed_time(ev[b]) if a in ev and b in ev else None
            prev = step.info.get("prev_end")
            row["gpu_idle_before_ms"] = (
                prev.elapsed_time(ev["step_start"]) if prev is not None else None
            )
            h = step.host
            row["host_exec_ms"] = (
                1e3 * (h["exec_end"] - h["exec_start"]) if "exec_end" in h else None
            )
            row["host_gap_ms"] = (
                1e3 * (h["sample_start"] - h["exec_end"])
                if "sample_start" in h and "exec_end" in h
                else None
            )
            row["host_sample_ms"] = (
                1e3 * (h["sample_end"] - h["sample_start"])
                if "sample_start" in h
                else None
            )
            ph = step.info.get("prev_end_host")
            row["host_between_ms"] = (
                1e3 * (h["exec_start"] - ph) if ph is not None else None
            )
            row["t"] = h["exec_start"]
            self._rows.append(row)
        if len(self._rows) >= 256 or (force and self._rows):
            self.flush()

    def flush(self, final: bool = False) -> None:
        if final:
            with contextlib.suppress(Exception):
                self._resolve(force=True)
        if not self._rows:
            return
        with open(self.path, "a") as f:
            for r in self._rows:
                f.write(json.dumps(r) + "\n")
        self._rows = []


_TRACE: OlalaStepTrace | None = None
_INIT = False


def get_step_trace() -> OlalaStepTrace | None:
    global _TRACE, _INIT
    if not _INIT:
        _INIT = True
        path = os.environ.get("OLALA_STEP_TRACE")
        if path:
            _TRACE = OlalaStepTrace(path)
    return _TRACE

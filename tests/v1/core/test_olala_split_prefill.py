# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split prefill (OLALA_SPLIT_PREFILL=1): prefill and decode work never share a
step while both are pending, so decode / spec-verify steps stay uniform and
replay the FULL CUDA graph. CPU only."""

import random
from collections import deque

import pytest
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from vllm.v1.core.sched.olala_split import SplitPrefillPolicy, StepKind

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

TOKEN = 7  # never EOS


# --------------------------------------------------------------------------- policy


def test_policy_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OLALA_SPLIT_PREFILL", raising=False)
    assert SplitPrefillPolicy.from_env() is None
    monkeypatch.setenv("OLALA_SPLIT_PREFILL", "0")
    assert SplitPrefillPolicy.from_env() is None


def test_policy_from_env(monkeypatch):
    monkeypatch.setenv("OLALA_SPLIT_PREFILL", "1")
    monkeypatch.setenv("OLALA_SPLIT_PREFILL_DECODE_STEPS", "3")
    monkeypatch.setenv("OLALA_SPLIT_PREFILL_MAX_DECODES", "16")
    monkeypatch.setenv("OLALA_SPLIT_PREFILL_BACKOFF", "2")
    p = SplitPrefillPolicy.from_env()
    assert (p.decode_steps, p.max_decodes, p.backoff) == (3, 16, 2)
    with pytest.raises(ValueError):
        SplitPrefillPolicy(decode_steps=0)


def _drive(p: SplitPrefillPolicy, n: int, num_decodes=4, has_prefill=True):
    kinds = []
    for _ in range(n):
        k = p.plan(num_decodes, has_prefill)
        kinds.append(k)
        p.record(k, carried_prefill=k is not StepKind.DECODE, scheduled=True)
    return kinds


@pytest.mark.parametrize("decode_steps", [1, 2, 4])
def test_policy_alternates(decode_steps):
    p = SplitPrefillPolicy(decode_steps=decode_steps)
    kinds = _drive(p, 3 * (decode_steps + 1))
    # A prefill arriving while decodes run goes first, then decode_steps decodes.
    pattern = [StepKind.PREFILL] + [StepKind.DECODE] * decode_steps
    assert kinds == pattern * 3


def test_policy_unconstrained_without_both_kinds():
    p = SplitPrefillPolicy()
    assert p.plan(0, True) is StepKind.MIXED
    assert p.plan(5, False) is StepKind.MIXED
    # A MIXED step that carried prefill (e.g. the first wave) is followed by a
    # decode-only step once decodes and more prefill are both pending.
    p.record(StepKind.MIXED, carried_prefill=True, scheduled=True)
    assert p.plan(4, True) is StepKind.DECODE


def test_policy_max_decodes_cap():
    p = SplitPrefillPolicy(max_decodes=8)
    assert p.plan(8, True) is StepKind.PREFILL
    assert p.plan(9, True) is StepKind.MIXED


def test_policy_backoff_after_empty_prefill_step():
    p = SplitPrefillPolicy(backoff=3)
    assert p.plan(4, True) is StepKind.PREFILL
    p.record(StepKind.PREFILL, carried_prefill=False, scheduled=False)
    assert p.num_empty_prefill_steps == 1
    kinds = _drive(p, 4)
    assert kinds == [StepKind.DECODE] * 3 + [StepKind.PREFILL]


def test_policy_backoff_doubles_and_resets():
    p = SplitPrefillPolicy(backoff=2)
    waits = []
    for _ in range(4):
        assert p.plan(4, True) is StepKind.PREFILL
        p.record(StepKind.PREFILL, carried_prefill=False, scheduled=False)
        n = 0
        while p.plan(4, True) is StepKind.DECODE:
            p.record(StepKind.DECODE, carried_prefill=False, scheduled=True)
            n += 1
        waits.append(n)
    assert waits == [2, 4, 8, 16]
    # A prefill that goes through restores the base backoff.
    p.record(StepKind.PREFILL, carried_prefill=True, scheduled=True)
    p.record(StepKind.DECODE, carried_prefill=False, scheduled=True)
    p.record(StepKind.PREFILL, carried_prefill=False, scheduled=False)
    assert p._need == 2


def test_policy_empty_decode_step_releases_prefill():
    p = SplitPrefillPolicy(decode_steps=4)
    p.record(StepKind.MIXED, carried_prefill=True, scheduled=True)
    assert p.plan(2, True) is StepKind.DECODE
    # Nothing schedulable among the decodes: the prefill is not held back.
    p.record(StepKind.DECODE, carried_prefill=False, scheduled=False)
    assert p.plan(2, True) is StepKind.PREFILL


# --------------------------------------------------------------------------- scheduler


class Sim:
    """Drives a scheduler like the engine core: sync (ngram drafts fed back with
    update_draft_token_ids) or async (one step in flight, GPU-side drafts as
    placeholders). Records, per step, the prefill and decode rows."""

    def __init__(self, scheduler, num_spec: int, is_async: bool, seed: int = 0):
        self.s = scheduler
        self.k = num_spec
        self.is_async = is_async
        self.rng = random.Random(seed)
        self.steps: list[tuple[set[str], set[str]]] = []
        self.inflight: deque = deque()
        self.reqs: dict = {}

    @property
    def max_tokens(self) -> dict[str, int]:
        return {rid: r.max_tokens for rid, r in self.reqs.items()}

    @property
    def finished(self) -> dict[str, int]:
        return {
            rid: r.num_output_tokens for rid, r in self.reqs.items() if r.is_finished()
        }

    def add(self, requests):
        for r in requests:
            self.reqs[r.request_id] = r
            self.s.add_request(r)

    def _sampled(self, out, prefill_rows):
        sampled = []
        for rid in out.num_scheduled_tokens:
            req = self.s.requests[rid]
            if rid in prefill_rows:
                # Partial chunk: nothing sampled; last chunk: one token.
                done = req.num_computed_tokens >= req.num_prompt_tokens
                sampled.append([TOKEN] if done else [])
            else:
                n_spec = len(out.scheduled_spec_decode_tokens.get(rid, ()))
                sampled.append([TOKEN] * (1 + self.rng.randint(0, n_spec)))
        return sampled

    def _finish(self, out, sampled):
        req_ids = list(out.num_scheduled_tokens)
        self.s.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if not self.is_async and self.k:
            ids = [
                rid
                for rid in req_ids
                if rid in self.s.requests
                and self.s.requests[rid].status == RequestStatus.RUNNING
                and not self.s.requests[rid].is_prefill_chunk
            ]
            if ids:
                self.s.update_draft_token_ids(
                    DraftTokenIds(ids, [[TOKEN] * self.k for _ in ids])
                )

    def step(self) -> bool:
        pending = {
            rid: r.num_computed_tokens < r.num_prompt_tokens
            for rid, r in self.s.requests.items()
        }
        out = self.s.schedule()
        rows = set(out.num_scheduled_tokens)
        prefill_rows = {rid for rid in rows if pending.get(rid, True)}
        decode_rows = rows - prefill_rows
        self.steps.append((prefill_rows, decode_rows))
        sampled = self._sampled(out, prefill_rows)
        if self.is_async:
            # One step in flight; an empty schedule() waits for it to land.
            if rows:
                self.inflight.append((out, sampled))
            if len(self.inflight) == 2 or (not rows and self.inflight):
                self._finish(*self.inflight.popleft())
        else:
            self._finish(out, sampled)
        return bool(rows)

    def run(self, arrivals: dict[int, list], max_steps: int = 2000):
        for i in range(max_steps):
            if i in arrivals:
                self.add(arrivals[i])
            if not self.s.has_requests() and not self.inflight and i > max(arrivals):
                return
            self.step()
        raise AssertionError("did not finish: possible scheduling deadlock")

    def mixed_steps(self):
        return [i for i, (p, d) in enumerate(self.steps) if p and d]


def _scheduler(monkeypatch, enabled: bool, is_async: bool, num_spec: int = 3, **kw):
    if enabled:
        monkeypatch.setenv("OLALA_SPLIT_PREFILL", "1")
    else:
        monkeypatch.delenv("OLALA_SPLIT_PREFILL", raising=False)
    for key in ("DECODE_STEPS", "MAX_DECODES", "BACKOFF"):
        if f"split_{key.lower()}" in kw:
            value = kw.pop(f"split_{key.lower()}")
            monkeypatch.setenv(f"OLALA_SPLIT_PREFILL_{key}", str(value))
        else:
            monkeypatch.delenv(f"OLALA_SPLIT_PREFILL_{key}", raising=False)
    return create_scheduler(
        num_speculative_tokens=num_spec or None,
        speculative_method="ngram_gpu" if is_async and num_spec else None,
        async_scheduling=is_async,
        use_v2_model_runner=True,
        **kw,
    )


def _arrivals(n: int, every: int, prompt: int, max_tokens: int, start: int = 0):
    reqs = create_requests(
        num_requests=n, num_tokens=prompt, max_tokens=max_tokens, ignore_eos=True
    )
    return {start + i * every: [r] for i, r in enumerate(reqs)}


@pytest.mark.parametrize("is_async", [False, True])
def test_baseline_mixes_prefill_and_decode(monkeypatch, is_async):
    """Without the flag, an arrival joins the running decodes in one step."""
    s = _scheduler(monkeypatch, enabled=False, is_async=is_async)
    assert s.split_prefill is None
    sim = Sim(s, 3, is_async)
    sim.run(_arrivals(6, every=5, prompt=40, max_tokens=48))
    assert sim.mixed_steps()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("num_spec", [0, 3, 5])
def test_split_never_mixes_and_finishes(monkeypatch, is_async, num_spec):
    s = _scheduler(monkeypatch, enabled=True, is_async=is_async, num_spec=num_spec)
    sim = Sim(s, num_spec, is_async, seed=num_spec)
    arrivals = _arrivals(12, every=3, prompt=40, max_tokens=40)
    sim.run(arrivals)
    assert not sim.mixed_steps()
    # Every request produced its full output.
    assert sim.finished == sim.max_tokens
    p = s.split_prefill
    assert p.num_prefill_steps > 0


@pytest.mark.parametrize("is_async", [False, True])
def test_split_decode_steps_are_uniform(monkeypatch, is_async):
    """Decode-only steps schedule every running decode with 1 + K tokens."""
    k = 3
    s = _scheduler(monkeypatch, enabled=True, is_async=is_async, num_spec=k)
    sim = Sim(s, k, is_async)
    uniform = []
    orig = s.schedule

    def schedule(*a, **kw):
        out = orig(*a, **kw)
        n = out.num_scheduled_tokens
        reqs = [s.requests[r] for r in n]
        if n and all(r.num_computed_tokens > r.num_prompt_tokens for r in reqs):
            uniform.append(all(v == 1 + k for v in n.values()))
        return out

    s.schedule = schedule
    sim.run(_arrivals(8, every=4, prompt=24, max_tokens=64))
    assert uniform and all(uniform)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("decode_steps", [1, 3])
def test_split_long_prompt_chunks_alternate(monkeypatch, is_async, decode_steps):
    """A long prompt's chunks alternate with decode-only steps: the running
    decodes keep progressing while it prefills."""
    s = _scheduler(
        monkeypatch,
        enabled=True,
        is_async=is_async,
        max_num_batched_tokens=64,
        split_decode_steps=decode_steps,
    )
    sim = Sim(s, 3, is_async)
    short = create_requests(
        num_requests=2, num_tokens=16, max_tokens=200, ignore_eos=True
    )
    (long,) = create_requests(
        num_requests=1, num_tokens=600, max_tokens=8, ignore_eos=True, req_ids=["long"]
    )
    sim.run({0: short, 6: [long]})
    assert not sim.mixed_steps()
    long_steps = [i for i, (p, _) in enumerate(sim.steps) if "long" in p]
    assert len(long_steps) >= 600 // 64
    # Between two chunks of the long prompt, at least decode_steps decode-only steps.
    for a, b in zip(long_steps, long_steps[1:]):
        between = [sim.steps[i] for i in range(a + 1, b)]
        assert sum(1 for p, d in between if d and not p) >= decode_steps
    assert sim.finished == sim.max_tokens


def test_split_max_decodes_cap(monkeypatch):
    s = _scheduler(monkeypatch, enabled=True, is_async=False, split_max_decodes=2)
    sim = Sim(s, 3, False)
    first = create_requests(
        num_requests=4, num_tokens=16, max_tokens=100, ignore_eos=True
    )
    (late,) = create_requests(
        num_requests=1, num_tokens=16, max_tokens=4, ignore_eos=True, req_ids=["late"]
    )
    sim.run({0: first, 4: [late]})
    # Four decodes running when "late" arrives: above the cap, it is mixed in.
    (step,) = [i for i, (p, _) in enumerate(sim.steps) if "late" in p]
    assert sim.steps[step][1]


@pytest.mark.parametrize("is_async", [False, True])
def test_split_kv_pressure_no_deadlock(monkeypatch, is_async):
    """A prompt that cannot be allocated while the decodes hold the cache: the
    prefill-only attempt schedules nothing, decodes continue, and the prompt is
    admitted once blocks are freed."""
    s = _scheduler(
        monkeypatch,
        enabled=True,
        is_async=is_async,
        num_blocks=24,
        block_size=16,
        split_backoff=2,
    )
    sim = Sim(s, 3, is_async)
    first = create_requests(
        num_requests=2, num_tokens=64, max_tokens=96, ignore_eos=True
    )
    (big,) = create_requests(
        num_requests=1, num_tokens=200, max_tokens=8, ignore_eos=True, req_ids=["big"]
    )
    sim.run({0: first, 6: [big]})
    assert sim.finished == sim.max_tokens
    assert not sim.mixed_steps()
    # The failed prefill-only attempts back off exponentially: few empty steps.
    assert 1 <= s.split_prefill.num_empty_prefill_steps <= 6

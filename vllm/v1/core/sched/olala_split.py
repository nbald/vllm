# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in split of prefill and decode work into separate engine steps.

A step that carries prefill tokens cannot replay a FULL CUDA graph: the model
runner dispatches it to the PIECEWISE graphs, where the stateful mixers of a
hybrid model (Olala's Mamba-3 and DiffTPA layers) run eagerly for every row of
the batch, decode and speculative-verify rows included. On Olala Q8 with
DSpine on an RTX 5090, the verify rows of such a mixed step cost 13-19 ms of
eager launches, against 6-7 ms for the same rows as a FULL-graph replay
(c = 8, K = 3).

With ``OLALA_SPLIT_PREFILL=1`` the scheduler never mixes the two kinds while
both are pending: it alternates prefill-only steps (running prefill chunks and
newly admitted prompts, the whole token budget) and decode-only steps (every
running decode, a uniform batch that replays the FULL graph). Steps with only
one kind of work are unchanged, so a single stream (c = 1) is not affected.

Knobs (environment, read once at scheduler construction):

* ``OLALA_SPLIT_PREFILL``: ``1`` enables the policy (default ``0``).
* ``OLALA_SPLIT_PREFILL_DECODE_STEPS``: decode-only steps between two
  prefill-only steps while both kinds are pending (default ``1``). Larger values
  protect decode latency and let more arrivals share one prefill step, at the
  cost of TTFT.
* ``OLALA_SPLIT_PREFILL_MAX_DECODES``: split only while at most this many
  decodes are running (default ``0``: no limit). Above it, steps mix as before.
* ``OLALA_SPLIT_PREFILL_BACKOFF``: decode-only steps to wait before retrying
  after a prefill-only step that could schedule nothing, e.g. under KV-cache
  pressure (default ``4``); doubled on each consecutive failure, up to 64.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field


class StepKind(enum.Enum):
    # No constraint: only one kind of work is pending, or the policy is capped.
    MIXED = "mixed"
    # Schedule running prefill chunks and waiting prompts only.
    PREFILL = "prefill"
    # Schedule running decodes only.
    DECODE = "decode"


@dataclass
class SplitPrefillPolicy:
    decode_steps: int = 1
    max_decodes: int = 0
    backoff: int = 4
    # Decode-only steps since the last step that carried prefill tokens, and
    # the number required before the next prefill-only step.
    _decode_run: int = field(default=0, init=False)
    _need: int = field(default=0, init=False)
    _backoff_cur: int = field(default=0, init=False)
    # Counters, for logs and tests.
    num_prefill_steps: int = field(default=0, init=False)
    num_decode_steps: int = field(default=0, init=False)
    num_mixed_steps: int = field(default=0, init=False)
    num_empty_prefill_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.decode_steps < 1:
            raise ValueError("OLALA_SPLIT_PREFILL_DECODE_STEPS must be >= 1")
        if self.max_decodes < 0 or self.backoff < 1:
            raise ValueError(
                "OLALA_SPLIT_PREFILL_MAX_DECODES must be >= 0 and "
                "OLALA_SPLIT_PREFILL_BACKOFF >= 1"
            )
        # A prefill arriving while decodes run is not held back.
        self._need = 0
        self._backoff_cur = self.backoff

    @classmethod
    def from_env(cls) -> SplitPrefillPolicy | None:
        if os.environ.get("OLALA_SPLIT_PREFILL", "0") != "1":
            return None
        return cls(
            decode_steps=int(os.environ.get("OLALA_SPLIT_PREFILL_DECODE_STEPS", "1")),
            max_decodes=int(os.environ.get("OLALA_SPLIT_PREFILL_MAX_DECODES", "0")),
            backoff=int(os.environ.get("OLALA_SPLIT_PREFILL_BACKOFF", "4")),
        )

    def plan(self, num_decodes: int, has_prefill_work: bool) -> StepKind:
        """Kind of the next step, given the running decodes and whether any
        prefill work (a running chunk or an admissible waiting prompt) is pending."""
        if num_decodes == 0 or not has_prefill_work:
            return StepKind.MIXED
        if self.max_decodes and num_decodes > self.max_decodes:
            return StepKind.MIXED
        if self._decode_run >= self._need:
            return StepKind.PREFILL
        return StepKind.DECODE

    def record(self, planned: StepKind, carried_prefill: bool, scheduled: bool) -> None:
        """Account for the step ``schedule()`` actually produced."""
        if not scheduled:
            if planned is StepKind.PREFILL:
                # Nothing could be admitted (KV pressure, blocked requests):
                # decode for a while before trying again.
                self.num_empty_prefill_steps += 1
                self._decode_run = 0
                self._need = self._backoff_cur
                self._backoff_cur = min(2 * self._backoff_cur, max(self.backoff, 64))
            elif planned is StepKind.DECODE:
                # No decode was schedulable (e.g. all of them are about to
                # finish): do not hold the pending prefill back any longer.
                self._decode_run = self._need
            return
        if planned is StepKind.PREFILL:
            self.num_prefill_steps += 1
        elif planned is StepKind.DECODE:
            self.num_decode_steps += 1
        else:
            self.num_mixed_steps += 1
        if carried_prefill:
            self._decode_run = 0
            self._need = self.decode_steps
            self._backoff_cur = self.backoff
        else:
            self._decode_run += 1

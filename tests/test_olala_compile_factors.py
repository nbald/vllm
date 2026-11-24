# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""compile_factors: the OLALA_* switches key the compile cache, the locations and diagnostics do not. CPU only."""

import pytest

from vllm import envs
from vllm.envs import compile_factors

UNKEYED = sorted(envs._OLALA_UNKEYED)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("OLALA_TS_FOLD", *UNKEYED):
        monkeypatch.delenv(var, raising=False)


def test_switches_key_the_cache(monkeypatch):
    base = compile_factors()
    monkeypatch.setenv("OLALA_TS_FOLD", "1")
    assert compile_factors() != base


def test_two_step_trace_paths_give_the_same_factors(monkeypatch):
    monkeypatch.setenv("OLALA_STEP_TRACE", "/tmp/steps_k5all_greedy_r1.jsonl")
    first = compile_factors()
    monkeypatch.setenv("OLALA_STEP_TRACE", "/tmp/steps_k5all_greedy_r2.jsonl")
    assert compile_factors() == first


@pytest.mark.parametrize("var", UNKEYED)
def test_locations_and_diagnostics_are_not_keyed(monkeypatch, var):
    base = compile_factors()
    monkeypatch.setenv(var, "1")
    assert compile_factors() == base

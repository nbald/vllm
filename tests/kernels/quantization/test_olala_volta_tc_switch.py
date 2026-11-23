# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_VOLTA_TC, the switch of the Volta fp16 tensor-core edition (olala/volta_tc.py): on by
default on the SM70 family (the int8 and bf16 kinds), "0" off, explicit kind lists, never on other
GPUs, off under VF_SAFE=1 (only that exact value); the default and VF_SAFE=1 keyed into the compile
cache (vllm/envs.py compile_factors), so that graphs cached under another setting are not reused.
No GPU needed."""

import pytest

from vllm import envs
from vllm.config.utils import hash_factors
from vllm.model_executor.layers.mamba.olala import volta_tc
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily


@pytest.fixture
def family(monkeypatch):
    monkeypatch.delenv("VF_SAFE", raising=False)

    def use(fam):
        monkeypatch.setattr(volta_tc.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(volta_tc, "gpu_family", lambda *a, **k: fam)
        volta_tc._kinds.cache_clear()
        volta_tc.vf_safe.cache_clear()

    yield use
    volta_tc._kinds.cache_clear()
    volta_tc.vf_safe.cache_clear()


def _set(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("OLALA_VOLTA_TC", raising=False)
    else:
        monkeypatch.setenv("OLALA_VOLTA_TC", raw)


@pytest.mark.parametrize(
    "raw,kinds",
    [
        (None, {"int8", "bf16"}),
        ("1", {"int8", "bf16"}),
        ("0", set()),
        ("", set()),
        ("int8", {"int8"}),
        ("int8,moe", {"int8", "moe"}),
        ("int8,bf16,moe", {"int8", "bf16", "moe"}),
    ],
)
def test_kinds_on_sm70(monkeypatch, family, raw, kinds):
    _set(monkeypatch, raw)
    family(GpuFamily.SM70)
    assert set(volta_tc._kinds()) == kinds
    assert volta_tc.tc_enabled() == bool(kinds)
    for kind in volta_tc.KINDS:
        assert volta_tc.tc_enabled(kind) == (kind in kinds)


@pytest.mark.parametrize("raw", [None, "1", "int8,bf16,moe"])
@pytest.mark.parametrize("fam", [GpuFamily.SM80, GpuFamily.SM90, GpuFamily.SM120])
def test_never_on_other_gpus(monkeypatch, family, raw, fam):
    _set(monkeypatch, raw)
    family(fam)
    assert volta_tc._kinds() == frozenset()


def test_moe_not_in_default():
    assert "moe" not in volta_tc.DEFAULT_KINDS
    assert set(volta_tc.DEFAULT_KINDS) == {"int8", "bf16"}


def test_unknown_kind_raises(monkeypatch, family):
    _set(monkeypatch, "int8,fp8")
    family(GpuFamily.SM70)
    with pytest.raises(ValueError):
        volta_tc._kinds()


def test_default_is_keyed_in_compile_cache(monkeypatch):
    _set(monkeypatch, None)
    unset = envs.compile_factors()
    assert unset["OLALA_VOLTA_TC"] == "default:" + ",".join(volta_tc.DEFAULT_KINDS)
    _set(monkeypatch, "0")
    off = envs.compile_factors()
    _set(monkeypatch, "1")
    on = envs.compile_factors()
    keys = {hash_factors(f) for f in (unset, off, on)}
    assert len(keys) == 3
    # before the default changed, an unset variable added no factor (and meant off): such a
    # cache key is none of today's
    before = {k: v for k, v in unset.items() if k != "OLALA_VOLTA_TC"}
    assert hash_factors(before) not in keys


@pytest.mark.parametrize("raw", [None, "1", "int8,moe", "int8,bf16,moe"])
def test_vf_safe_turns_it_off(monkeypatch, family, raw):
    _set(monkeypatch, raw)
    monkeypatch.setenv("VF_SAFE", "1")
    said = []
    monkeypatch.setattr(volta_tc.logger, "info", lambda msg, *a: said.append(msg % a if a else msg))
    family(GpuFamily.SM70)
    assert volta_tc._kinds() == frozenset()
    assert not volta_tc.tc_enabled() and not volta_tc.tc_enabled("moe")
    volta_tc._kinds()  # read once per process: no second line
    assert said == ["VF_SAFE=1: the fp16 tensor-core edition is off"]


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "TRUE", " 1", "1 "])
def test_only_exact_1_is_safe(monkeypatch, family, value):
    _set(monkeypatch, None)
    monkeypatch.setenv("VF_SAFE", value)
    family(GpuFamily.SM70)
    assert not volta_tc.vf_safe()
    assert set(volta_tc._kinds()) == {"int8", "bf16"}


def test_vf_safe_keyed_in_compile_cache(monkeypatch):
    _set(monkeypatch, None)
    monkeypatch.delenv("VF_SAFE", raising=False)
    plain = envs.compile_factors()
    assert "VF_SAFE" not in plain
    for value in ("0", "", "true"):
        monkeypatch.setenv("VF_SAFE", value)
        assert hash_factors(envs.compile_factors()) == hash_factors(plain)
    monkeypatch.setenv("VF_SAFE", "1")
    safe = envs.compile_factors()
    assert safe["VF_SAFE"] == "1"
    assert hash_factors(safe) != hash_factors(plain)

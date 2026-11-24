# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OlalaForCausalLMConfig: the TPA factor cache under speculative decoding.

With a speculative config, F1 (the factor cache's multi-token verify path,
OLALA_TPA_FACTOR_SPEC) is on by default on the GPUs where it was measured
(SM70, the SM80 family, SM120): the hook sets OLALA_TPA_FACTOR_SPEC=1 (read by
every process) and keeps the factor cache. Elsewhere, or with
OLALA_TPA_FACTOR_SPEC=0, the hook turns the cache off (OLALA_TPA_FACTOR=0)
unless the user asked for it explicitly, which is refused at config time
instead of at KV-cache init. Without a speculative config nothing changes.
ReplaySSM (OLALA_REPLAY_SSM) has no multi-token verify: a speculative config
with it is refused at config time too. CPU only.
"""

import os
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import OlalaForCausalLMConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


def _gpu(monkeypatch, cap):
    monkeypatch.setattr(
        current_platform, "get_device_capability",
        lambda device_id=0: None if cap is None else DeviceCapability(*cap),
        raising=False,
    )


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # setenv first so that teardown restores the original state even though
    # the hook writes os.environ directly.
    for var in ("OLALA_TPA_FACTOR_SPEC", "OLALA_TPA_FACTOR", "OLALA_REPLAY_SSM"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    _gpu(monkeypatch, (9, 0))  # a GPU without the F1 default unless a test says otherwise


def _vllm_config(speculative: bool) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        num_attention_heads=48, num_signal_heads_diff=36, moe=False
    )
    model_config = SimpleNamespace(
        hf_config=hf_config, model_arch_config=SimpleNamespace()
    )
    cache_config = SimpleNamespace(
        mamba_cache_mode="none", enable_prefix_caching=False
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=None,
        speculative_config=SimpleNamespace() if speculative else None,
    )


def _run(speculative=True):
    OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(speculative))


@pytest.mark.parametrize("cap", [(7, 0), (8, 0), (8, 6), (8, 9), (12, 0)])
def test_spec_takes_f1_by_default_on_measured_gpus(monkeypatch, cap):
    _gpu(monkeypatch, cap)
    _run()
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") == "1"
    assert os.environ.get("OLALA_TPA_FACTOR") is None  # the factor cache stays on (its default)


@pytest.mark.parametrize("cap", [(9, 0), (10, 0), None])
def test_spec_turns_factor_cache_off_elsewhere(monkeypatch, cap):
    _gpu(monkeypatch, cap)
    _run()
    assert os.environ["OLALA_TPA_FACTOR"] == "0"
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") is None


def test_explicit_f1_off_forces_the_dense_path(monkeypatch):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv("OLALA_TPA_FACTOR_SPEC", "0")
    _run()
    assert os.environ["OLALA_TPA_FACTOR"] == "0"
    assert os.environ["OLALA_TPA_FACTOR_SPEC"] == "0"


def test_explicit_f1_off_refuses_an_explicit_factor_cache(monkeypatch):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv("OLALA_TPA_FACTOR_SPEC", "0")
    monkeypatch.setenv("OLALA_TPA_FACTOR", "1")
    with pytest.raises(ValueError, match="OLALA_TPA_FACTOR"):
        _run()


def test_explicit_dense_cache_on_a_measured_gpu(monkeypatch):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv("OLALA_TPA_FACTOR", "0")
    _run()
    assert os.environ["OLALA_TPA_FACTOR"] == "0"
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") is None


@pytest.mark.parametrize("cap", [(9, 0), (12, 0)])
@pytest.mark.parametrize("value", ["1", None])
def test_explicit_f1_on_keeps_factor_cache(monkeypatch, cap, value):
    _gpu(monkeypatch, cap)
    monkeypatch.setenv("OLALA_TPA_FACTOR_SPEC", "1")
    if value is not None:
        monkeypatch.setenv("OLALA_TPA_FACTOR", value)
    _run()
    assert os.environ.get("OLALA_TPA_FACTOR") == value
    assert os.environ["OLALA_TPA_FACTOR_SPEC"] == "1"


def test_explicit_factor_cache_refused_without_f1(monkeypatch):
    monkeypatch.setenv("OLALA_TPA_FACTOR", "1")
    with pytest.raises(ValueError, match="OLALA_TPA_FACTOR"):
        _run()


@pytest.mark.parametrize("cap", [(9, 0), (12, 0)])
@pytest.mark.parametrize("value", ["0", None])
def test_no_spec_leaves_switches_alone(monkeypatch, cap, value):
    _gpu(monkeypatch, cap)
    if value is not None:
        monkeypatch.setenv("OLALA_TPA_FACTOR", value)
    _run(speculative=False)
    assert os.environ.get("OLALA_TPA_FACTOR") == value
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") is None


@pytest.mark.parametrize("value", ["1", "on", "8", "16"])
def test_spec_refuses_replay_ssm_at_config_time(monkeypatch, value):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv("OLALA_REPLAY_SSM", value)
    with pytest.raises(ValueError, match="OLALA_REPLAY_SSM=0") as err:
        _run()
    assert f"OLALA_REPLAY_SSM={value})" in str(err.value)
    assert "--speculative-config" in str(err.value)
    # refused before the F1 switches are touched
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") is None
    assert os.environ.get("OLALA_TPA_FACTOR") is None


@pytest.mark.parametrize("value", ["0", ""])
def test_spec_accepts_replay_ssm_off(monkeypatch, value):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv("OLALA_REPLAY_SSM", value)
    _run()
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") == "1"


def test_no_spec_keeps_replay_ssm(monkeypatch):
    monkeypatch.setenv("OLALA_REPLAY_SSM", "1")
    _run(speculative=False)
    assert os.environ["OLALA_REPLAY_SSM"] == "1"

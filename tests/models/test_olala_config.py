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
ReplaySSM (OLALA_REPLAY_SSM) has no multi-token verify: under a speculative
config the hook turns it off (OLALA_REPLAY_SSM=0) with one warning, and
--mamba-ssm-cache-dtype auto becomes float32 (the verify's fp32 state); an
explicit non-fp32 SSM dtype is refused. CPU only.
"""

import os
from types import SimpleNamespace

import pytest

from vllm.model_executor.models import config as config_module
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
    # The hook also sets the split prefill defaults on measured GPUs.
    for var in ("OLALA_TPA_FACTOR_SPEC", "OLALA_TPA_FACTOR", "OLALA_REPLAY_SSM",
                "OLALA_VOLTA_STATE16", "OLALA_SPLIT_PREFILL",
                "OLALA_SPLIT_PREFILL_MAX_DECODES"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    _gpu(monkeypatch, (9, 0))  # a GPU without the F1 default unless a test says otherwise


@pytest.fixture
def warnings(monkeypatch):
    """The hook's logger.warning calls, formatted."""
    seen = []
    monkeypatch.setattr(
        config_module.logger, "warning", lambda msg, *args: seen.append(msg % args)
    )
    return seen


def _vllm_config(speculative: bool, ssm_dtype: str = "auto") -> SimpleNamespace:
    hf_config = SimpleNamespace(
        num_attention_heads=48, num_signal_heads_diff=36, moe=False
    )
    model_config = SimpleNamespace(
        hf_config=hf_config, model_arch_config=SimpleNamespace()
    )
    cache_config = SimpleNamespace(
        mamba_cache_mode="none", enable_prefix_caching=False,
        mamba_ssm_cache_dtype=ssm_dtype,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=None,
        speculative_config=SimpleNamespace() if speculative else None,
    )


def _run(speculative=True, ssm_dtype="auto"):
    vllm_config = _vllm_config(speculative, ssm_dtype)
    OlalaForCausalLMConfig.verify_and_update_config(vllm_config)
    return vllm_config


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


@pytest.mark.parametrize("ssm_dtype", ["auto", "float32"])
@pytest.mark.parametrize("value", ["1", "on", "8", "16"])
def test_spec_falls_back_from_replay_ssm(monkeypatch, warnings, value, ssm_dtype):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv("OLALA_REPLAY_SSM", value)
    cfg = _run(ssm_dtype=ssm_dtype)
    assert os.environ["OLALA_REPLAY_SSM"] == "0"
    assert cfg.cache_config.mamba_ssm_cache_dtype == "float32"
    assert len(warnings) == 1
    assert f"OLALA_REPLAY_SSM={value} -> 0" in warnings[0]
    assert "fp32 Mamba-3 state" in warnings[0]
    assert ("auto -> float32" in warnings[0]) == (ssm_dtype == "auto")
    # the F1 default still applies
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") == "1"


def test_spec_fallback_drops_a_cached_replay_config(monkeypatch):
    state = pytest.importorskip("vllm.model_executor.layers.mamba.olala.state")
    monkeypatch.setenv("OLALA_REPLAY_SSM", "1")
    state.replay_config.cache_clear()
    try:
        assert state.replay_config() is not None
        _run()
        assert state.replay_config() is None
    finally:
        state.replay_config.cache_clear()


@pytest.mark.parametrize("value", ["0", "", None])
def test_spec_auto_dtype_becomes_float32_without_replay(monkeypatch, warnings, value):
    _gpu(monkeypatch, (12, 0))
    if value is not None:
        monkeypatch.setenv("OLALA_REPLAY_SSM", value)
    cfg = _run()
    assert cfg.cache_config.mamba_ssm_cache_dtype == "float32"
    assert os.environ.get("OLALA_REPLAY_SSM") == value
    assert warnings == []
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") == "1"


def test_spec_keeps_float32(monkeypatch, warnings):
    cfg = _run(ssm_dtype="float32")
    assert cfg.cache_config.mamba_ssm_cache_dtype == "float32"
    assert warnings == []


@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("value", ["1", None])
def test_spec_refuses_an_explicit_non_fp32_state(monkeypatch, ssm_dtype, value):
    _gpu(monkeypatch, (12, 0))
    if value is not None:
        monkeypatch.setenv("OLALA_REPLAY_SSM", value)
    with pytest.raises(ValueError, match=f"--mamba-ssm-cache-dtype {ssm_dtype}") as err:
        _run(ssm_dtype=ssm_dtype)
    assert "float32" in str(err.value)
    # refused before any switch is touched
    assert os.environ.get("OLALA_REPLAY_SSM") == value
    assert os.environ.get("OLALA_TPA_FACTOR_SPEC") is None
    assert os.environ.get("OLALA_TPA_FACTOR") is None


def test_spec_leaves_the_volta_state16_dtype_alone(monkeypatch, warnings):
    # OLALA_VOLTA_STATE16 needs --mamba-ssm-cache-dtype auto; the Mamba-3 mixer
    # refuses it with a speculative config.
    monkeypatch.setenv("OLALA_VOLTA_STATE16", "1")
    cfg = _run()
    assert cfg.cache_config.mamba_ssm_cache_dtype == "auto"
    assert warnings == []


@pytest.mark.parametrize("ssm_dtype", ["auto", "bfloat16"])
def test_no_spec_keeps_replay_ssm(monkeypatch, warnings, ssm_dtype):
    monkeypatch.setenv("OLALA_REPLAY_SSM", "1")
    cfg = _run(speculative=False, ssm_dtype=ssm_dtype)
    assert os.environ["OLALA_REPLAY_SSM"] == "1"
    assert cfg.cache_config.mamba_ssm_cache_dtype == ssm_dtype
    assert warnings == []

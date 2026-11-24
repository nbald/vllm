# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OlalaForCausalLMConfig: the TPA factor cache under speculative decoding.

The factor cache has a multi-token verify path only under
OLALA_TPA_FACTOR_SPEC=1 (opt-in). Without it, with a speculative config, the
hook turns the cache off (OLALA_TPA_FACTOR=0, read by every process) unless the
user asked for it explicitly, which is refused at config time instead of at
KV-cache init. With it, the hook leaves the switch alone. CPU only.
"""

import os
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import OlalaForCausalLMConfig


@pytest.fixture(autouse=True)
def _no_spec_factor(monkeypatch):
    monkeypatch.delenv("OLALA_TPA_FACTOR_SPEC", raising=False)


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


def _unset(monkeypatch):
    # setenv first so that teardown restores the original state even though
    # the hook writes os.environ directly.
    monkeypatch.setenv("OLALA_TPA_FACTOR", "")
    monkeypatch.delenv("OLALA_TPA_FACTOR")


def test_spec_turns_factor_cache_off(monkeypatch):
    _unset(monkeypatch)
    OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(True))
    assert os.environ["OLALA_TPA_FACTOR"] == "0"


@pytest.mark.parametrize("value", ["0", None])
def test_no_spec_leaves_switch_alone(monkeypatch, value):
    if value is None:
        _unset(monkeypatch)
    else:
        monkeypatch.setenv("OLALA_TPA_FACTOR", value)
    OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(False))
    assert os.environ.get("OLALA_TPA_FACTOR") == value


def test_spec_keeps_explicit_off(monkeypatch):
    monkeypatch.setenv("OLALA_TPA_FACTOR", "0")
    OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(True))
    assert os.environ["OLALA_TPA_FACTOR"] == "0"


def test_spec_refuses_explicit_factor_cache(monkeypatch):
    monkeypatch.setenv("OLALA_TPA_FACTOR", "1")
    with pytest.raises(ValueError, match="OLALA_TPA_FACTOR"):
        OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(True))


@pytest.mark.parametrize("value", ["1", None])
def test_spec_factor_keeps_factor_cache(monkeypatch, value):
    # OLALA_TPA_FACTOR_SPEC=1: the factor cache has a verify path, so the hook
    # leaves the switch alone (unset = the default, on).
    monkeypatch.setenv("OLALA_TPA_FACTOR_SPEC", "1")
    if value is None:
        _unset(monkeypatch)
    else:
        monkeypatch.setenv("OLALA_TPA_FACTOR", value)
    OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(True))
    assert os.environ.get("OLALA_TPA_FACTOR") == value


def test_spec_factor_keeps_explicit_off(monkeypatch):
    monkeypatch.setenv("OLALA_TPA_FACTOR_SPEC", "1")
    monkeypatch.setenv("OLALA_TPA_FACTOR", "0")
    OlalaForCausalLMConfig.verify_and_update_config(_vllm_config(True))
    assert os.environ["OLALA_TPA_FACTOR"] == "0"

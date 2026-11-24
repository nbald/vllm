# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OlalaForCausalLMConfig: the TPA factor cache under speculative decoding.

The factor cache has no multi-token verify path. With a speculative config,
the hook turns it off (OLALA_TPA_FACTOR=0, read by every process) unless the
user asked for it explicitly, which is refused at config time instead of at
KV-cache init. CPU only.
"""

import os
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import OlalaForCausalLMConfig


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


@pytest.fixture(autouse=True)
def _split_env(monkeypatch):
    # The hook also sets the split prefill defaults on measured GPUs: keep them
    # out of the other tests' environment.
    for var in ("OLALA_SPLIT_PREFILL", "OLALA_SPLIT_PREFILL_MAX_DECODES"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OlalaForCausalLMConfig: the split prefill policy's defaults.

On SM120 and the SM80 family, unset switches default to OLALA_SPLIT_PREFILL=1
with OLALA_SPLIT_PREFILL_MAX_DECODES=0 (no cap) without a speculative config
and 16 with one. The hook writes them to the environment, which the scheduler
reads in the engine core. Explicit values win; other GPUs, and V100s through
volta-forever (VOLTA_FOREVER), are unchanged. CPU only.
"""

import os
from types import SimpleNamespace

import pytest

from vllm.model_executor.models import config as config_module
from vllm.model_executor.models.config import OlalaForCausalLMConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

SPLIT, CAP = "OLALA_SPLIT_PREFILL", "OLALA_SPLIT_PREFILL_MAX_DECODES"


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
    for var in (SPLIT, CAP, "OLALA_TPA_FACTOR", "VOLTA_FOREVER"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    _gpu(monkeypatch, (9, 0))


@pytest.fixture
def infos(monkeypatch):
    """The hook's logger.info calls, formatted."""
    seen = []
    monkeypatch.setattr(
        config_module.logger, "info", lambda msg, *args: seen.append(msg % args)
    )
    return seen


def _run(speculative):
    hf_config = SimpleNamespace(
        num_attention_heads=48, num_signal_heads_diff=36, moe=False
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config, model_arch_config=SimpleNamespace()
        ),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none", enable_prefix_caching=False
        ),
        scheduler_config=None,
        speculative_config=SimpleNamespace() if speculative else None,
    )
    OlalaForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize("cap", [(12, 0), (8, 6), (8, 0)])
@pytest.mark.parametrize("speculative, max_decodes", [(False, "0"), (True, "16")])
def test_split_on_by_default_on_measured_gpus(monkeypatch, infos, cap, speculative,
                                              max_decodes):
    _gpu(monkeypatch, cap)
    _run(speculative)
    assert os.environ[SPLIT] == "1"
    assert os.environ[CAP] == max_decodes
    assert sum("split prefill" in line for line in infos) == 1


@pytest.mark.parametrize("cap", [(9, 0), (10, 0), (7, 0), None])
@pytest.mark.parametrize("speculative", [False, True])
def test_other_gpus_unchanged(monkeypatch, infos, cap, speculative):
    _gpu(monkeypatch, cap)
    _run(speculative)
    assert SPLIT not in os.environ and CAP not in os.environ
    assert not any("split prefill" in line for line in infos)


def test_volta_forever_unchanged(monkeypatch):
    # A V100 through volta-forever reports 8.0 to CUDA programs.
    _gpu(monkeypatch, (8, 0))
    monkeypatch.setenv("VOLTA_FOREVER", "1")
    _run(False)
    assert SPLIT not in os.environ and CAP not in os.environ


@pytest.mark.parametrize("speculative", [False, True])
def test_explicit_off_wins(monkeypatch, infos, speculative):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv(SPLIT, "0")
    _run(speculative)
    assert os.environ[SPLIT] == "0"
    assert CAP not in os.environ
    assert not any("split prefill" in line for line in infos)


@pytest.mark.parametrize("speculative", [False, True])
def test_explicit_cap_wins(monkeypatch, speculative):
    _gpu(monkeypatch, (8, 6))
    monkeypatch.setenv(CAP, "8")
    _run(speculative)
    assert os.environ[SPLIT] == "1"
    assert os.environ[CAP] == "8"


@pytest.mark.parametrize("speculative, max_decodes", [(False, "0"), (True, "16")])
def test_explicit_on_takes_the_default_cap(monkeypatch, speculative, max_decodes):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv(SPLIT, "1")
    _run(speculative)
    assert os.environ[SPLIT] == "1"
    assert os.environ[CAP] == max_decodes


def test_serve_spec_setting_unchanged(monkeypatch, infos):
    # spec/serve_spec.sh's explicit setting.
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setenv(SPLIT, "1")
    monkeypatch.setenv(CAP, "16")
    _run(True)
    assert (os.environ[SPLIT], os.environ[CAP]) == ("1", "16")
    assert not any("split prefill" in line for line in infos)


def test_explicit_on_elsewhere_unchanged(monkeypatch):
    _gpu(monkeypatch, (9, 0))
    monkeypatch.setenv(SPLIT, "1")
    _run(True)
    assert os.environ[SPLIT] == "1"
    assert CAP not in os.environ

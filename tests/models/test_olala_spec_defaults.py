# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OlalaForCausalLMConfig: the defaults of a DSpine drafter.

With a DSpine drafter (speculative method dflash, draft architecture
DSpineDraftModel), each unset switch takes its value in the benchmarked
speculative configuration, on every GPU; OLALA_DSPINE_FUSED is 3 where
the engine has the hand-fused kernels (2 otherwise); the SM70 family (V100s,
through volta-forever too) also takes OLALA_VOLTA_MOE_ROWS=1; an unset drafter
attention backend becomes TRITON_ATTN. Explicit values win. Without a DSpine
drafter nothing changes. CPU only.
"""

import importlib.util
import os
from types import SimpleNamespace

import pytest

from vllm.model_executor.models import config as config_module
from vllm.model_executor.models.config import OlalaForCausalLMConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

BENCHMARKED = {
    "OLALA_SPLIT_PREFILL": "1", "OLALA_SPLIT_PREFILL_MAX_DECODES": "16", "OLALA_PREFILL_NOSYNC": "1",
    "OLALA_TS_FOLD": "1", "OLALA_SPEC_LIGHT": "1", "VLLM_KV_CACHE_GROUP_SIZE": "7", "OLALA_TPA_MIXED_FUSED": "1",
    "OLALA_DSPINE_SHARED_HEAD": "1", "OLALA_DSPINE_GEMV": "1", "OLALA_DSPINE_CUDA_NORM": "1",
    "OLALA_MAMBA3_VERIFY": "cuda",
}
DEFAULTED = (*BENCHMARKED, "OLALA_DSPINE_FUSED", "OLALA_VOLTA_MOE_ROWS")
# also cleaned: the other switches the hook writes (the factor cache, ReplaySSM) and the volta-forever marker
ALL = (*DEFAULTED, "OLALA_TPA_FACTOR", "OLALA_TPA_FACTOR_SPEC", "OLALA_REPLAY_SSM", "VOLTA_FOREVER")
FUSED = "3" if importlib.util.find_spec("vllm.model_executor.models.qwen3_dspine_kernels") else "2"


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
    for var in ALL:
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    _gpu(monkeypatch, (12, 0))


@pytest.fixture
def infos(monkeypatch):
    seen = []
    monkeypatch.setattr(
        config_module.logger, "info", lambda msg, *args: seen.append(msg % args)
    )
    return seen


def _spec(method="dflash", arch="DSpineDraftModel", backend=None):
    return SimpleNamespace(
        method=method, attention_backend=backend,
        draft_model_config=SimpleNamespace(architectures=[arch]),
    )


def _run(spec):
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
        speculative_config=spec,
    )
    OlalaForCausalLMConfig.verify_and_update_config(vllm_config)
    return vllm_config


@pytest.mark.parametrize("cap", [(12, 0), (8, 6), (8, 0), (9, 0), (10, 0)])
def test_dspine_drafter_takes_benchmarked_values(monkeypatch, infos, cap):
    _gpu(monkeypatch, cap)
    spec = _spec()
    _run(spec)
    for name, value in BENCHMARKED.items():
        assert os.environ[name] == value, name
    assert os.environ["OLALA_DSPINE_FUSED"] == FUSED
    assert "OLALA_VOLTA_MOE_ROWS" not in os.environ
    assert spec.attention_backend == AttentionBackendEnum.TRITON_ATTN
    lines = [line for line in infos if "DSpine speculative defaults" in line]
    assert len(lines) == 1 and "OLALA_SPEC_LIGHT=1" in lines[0]


@pytest.mark.parametrize("cap, volta_forever", [((7, 0), False), ((8, 0), True)])
def test_sm70_family_also_takes_moe_rows(monkeypatch, cap, volta_forever):
    _gpu(monkeypatch, cap)
    if volta_forever:
        monkeypatch.setenv("VOLTA_FOREVER", "1")
    _run(_spec())
    assert os.environ["OLALA_VOLTA_MOE_ROWS"] == "1"
    assert os.environ["OLALA_SPLIT_PREFILL"] == "1"
    assert os.environ["OLALA_SPLIT_PREFILL_MAX_DECODES"] == "16"


@pytest.mark.parametrize("name", [*BENCHMARKED, "OLALA_DSPINE_FUSED"])
def test_explicit_value_wins(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    _run(_spec())
    assert os.environ[name] == "0"
    others = {k: v for k, v in BENCHMARKED.items() if k != name}
    for other, value in others.items():
        assert os.environ[other] == value, other


def test_explicit_moe_rows_off_wins(monkeypatch):
    _gpu(monkeypatch, (7, 0))
    monkeypatch.setenv("OLALA_VOLTA_MOE_ROWS", "0")
    _run(_spec())
    assert os.environ["OLALA_VOLTA_MOE_ROWS"] == "0"


def test_explicit_attention_backend_wins():
    spec = _spec(backend=AttentionBackendEnum.FLASH_ATTN)
    _run(spec)
    assert spec.attention_backend == AttentionBackendEnum.FLASH_ATTN


def test_all_explicit_logs_nothing(monkeypatch, infos):
    for name, value in BENCHMARKED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OLALA_DSPINE_FUSED", FUSED)
    _run(_spec(backend=AttentionBackendEnum.TRITON_ATTN))
    assert not any("DSpine speculative defaults" in line for line in infos)


@pytest.mark.parametrize("cap", [(12, 0), (8, 6), (7, 0)])
def test_no_drafter_unchanged(monkeypatch, cap):
    _gpu(monkeypatch, cap)
    _run(None)
    for name in DEFAULTED:
        if name in ("OLALA_SPLIT_PREFILL", "OLALA_SPLIT_PREFILL_MAX_DECODES") and cap == (12, 0):
            continue  # the split prefill default of SM120, without a cap
        assert name not in os.environ, name
    if cap == (12, 0):
        assert os.environ["OLALA_SPLIT_PREFILL_MAX_DECODES"] == "0"


@pytest.mark.parametrize("spec", [_spec(method="eagle3"), _spec(arch="DFlashDraftModel"), SimpleNamespace()])
def test_other_drafters_unchanged(monkeypatch, spec):
    _gpu(monkeypatch, (8, 6))
    _run(spec)
    for name in DEFAULTED:
        assert name not in os.environ, name

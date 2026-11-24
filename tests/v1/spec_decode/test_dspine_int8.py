# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_DSPINE_INT8 (qwen3_dspine_fast.py): per-channel RTN of the drafter's
bf16 linears and their W8A16 apply (the target's ModelOptW8A16LinearMethod),
and the default (on for SM70 / the SM80 family / SM120).

The quantizer is checked on CPU (symmetric per-row scales, the rounding bound,
exact zeros). On a GPU the INT8 linear method matches ``x @ (scale * q).T`` on
the drafter's shapes at the draft step's row counts (the W8A16 decode kernels)
and at a prefill row count (the prefill GEMM)."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.quantization.modelopt_w8a16 import (
    ModelOptW8A16LinearMethod,
)
from vllm.model_executor.models.qwen3_dspine_fast import (
    _INT8_NAME,
    int8_enabled,
    quantize_int8_rtn,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


def _gpu(monkeypatch, cap):
    monkeypatch.setattr(
        current_platform, "get_device_capability",
        lambda device_id=0: None if cap is None else DeviceCapability(*cap),
        raising=False,
    )


@pytest.mark.parametrize(
    "cap,value,on",
    [
        ((7, 0), None, True), ((8, 0), None, True), ((8, 6), None, True), ((8, 9), None, True),
        ((12, 0), None, True), ((9, 0), None, False), ((10, 0), None, False), (None, None, False),
        ((12, 0), "0", False), ((8, 6), "0", False), ((9, 0), "1", True), ((12, 0), "1", True),
    ],
)
def test_default_on_measured_gpus(monkeypatch, cap, value, on):
    """CPU: OLALA_DSPINE_INT8 unset is on for SM70 / the SM80 family / SM120 only;
    0 and 1 override it on any GPU."""
    _gpu(monkeypatch, cap)
    if value is None:
        monkeypatch.delenv("OLALA_DSPINE_INT8", raising=False)
    else:
        monkeypatch.setenv("OLALA_DSPINE_INT8", value)
    assert int8_enabled() is on


def test_rtn_is_symmetric_per_channel():
    torch.manual_seed(0)
    w = (torch.randn(64, 256) * torch.rand(64, 1) * 3).to(torch.bfloat16)
    w[5] = 0
    q, s = quantize_int8_rtn(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (64,)
    assert q.abs().max() <= 127
    wf = w.float()
    # the largest entry of every non-zero row maps to +-127
    nz = wf.abs().amax(1) > 0
    assert torch.equal(q[nz].abs().amax(1), torch.full((int(nz.sum()),), 127, dtype=torch.int8))
    # round to nearest: at most half a step per entry; a zero row stays zero
    assert ((s[:, None] * q.float() - wf).abs() <= s[:, None] / 2 + 1e-6).all()
    assert torch.equal(q[5], torch.zeros(256, dtype=torch.int8))


def test_selected_modules():
    for name in ("model.layers.0.self_attn.qkv_proj", "model.layers.4.self_attn.o_proj",
                 "model.layers.2.mlp.gate_up_proj", "model.layers.3.mlp.down_proj", "model.fc"):
        assert _INT8_NAME.search(name), name
    for name in ("model.injection.readouts.0", "model.injection.writes.1", "lm_head",
                 "model.embed_tokens", "model.layers.0.self_attn"):
        assert not _INT8_NAME.search(name), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("n,k", [(2560, 1536), (1536, 1536), (8192, 1536), (1536, 4096), (1536, 7680)])
@pytest.mark.parametrize("m", [4, 6, 48, 300])
def test_int8_linear_matches_dequantized(n, k, m):
    torch.manual_seed(n + k + m)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = quantize_int8_rtn(w)
    layer = nn.Module()
    layer.weight = nn.Parameter(q, requires_grad=False)
    layer.weight_scale = nn.Parameter(s, requires_grad=False)
    method = ModelOptW8A16LinearMethod()
    method.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y = method.apply(layer, x)
    ref = F.linear(x.float(), s[:, None] * q.float())
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)

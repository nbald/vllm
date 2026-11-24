# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_DSPINE_INT8 (qwen3_dspine_fast.py): per-channel RTN of the drafter's
bf16 linears and their W8A16 apply (the target's ModelOptW8A16LinearMethod),
and the default (on for SM70 / the SM80 family / SM120).

The quantizer is checked on CPU (symmetric per-row scales, the rounding bound,
exact zeros). On a GPU the INT8 linear method matches ``x @ (scale * q).T`` on
the drafter's shapes at the draft step's row counts (the W8A16 decode kernels)
and at a prefill row count (the prefill GEMM).

The drafter's own row-count dispatch (draft_int8_linear: the Triton W8A16 GEMM
with tuned tiles for 33-256 rows on sm_86-class GPUs, the target's dispatch
elsewhere) is checked on CPU: which GPUs and row counts take it, that its tiles
fit, and that the op routes there."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.quantization.modelopt_w8a16 import (
    ModelOptW8A16LinearMethod,
)
from vllm.model_executor.models import qwen3_dspine_fast as fast
from vllm.model_executor.models.qwen3_dspine_fast import (
    _INT8_NAME,
    DraftW8A16LinearMethod,
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


# The drafter's INT8 shapes: qkv, o, gate_up, down, fc, the fused context K/V.
DRAFT_SHAPES = [(2560, 1536), (1536, 1536), (8192, 1536), (1536, 4096), (1536, 7680), (5120, 1536)]


def _arch(monkeypatch, cap, volta_forever=False):
    from vllm.model_executor.layers.mamba.olala import arch

    family = {7: arch.GpuFamily.UNSUPPORTED, 8: arch.GpuFamily.SM80, 9: arch.GpuFamily.SM90,
              12: arch.GpuFamily.SM120}[cap[0]]
    monkeypatch.setattr(arch, "gpu_family", lambda device=None: family)
    monkeypatch.setattr(arch, "capability", lambda device=None: cap)
    if volta_forever:
        monkeypatch.setenv("VOLTA_FOREVER", "1")
    else:
        monkeypatch.delenv("VOLTA_FOREVER", raising=False)


@pytest.mark.parametrize("cap,vf,on", [((8, 6), False, True), ((8, 9), False, True), ((8, 0), False, False),
                                       ((8, 0), True, False), ((8, 6), True, False), ((12, 0), False, False),
                                       ((9, 0), False, False), ((7, 0), False, False)])
def test_mid_rows_dispatch_gpus(monkeypatch, cap, vf, on):
    """CPU: only sm_86-class GPUs (not A100, not a V100 through volta-forever, not the RTX 5090) take the drafter's
    own tiles; elsewhere every row count keeps the target's W8A16 dispatch."""
    _arch(monkeypatch, cap, vf)
    for n, k in DRAFT_SHAPES:
        assert (fast._mid_rows_cfg(192, n, k) is not None) is on


@pytest.mark.parametrize("n,k", DRAFT_SHAPES)
def test_mid_rows_tiles(monkeypatch, n, k):
    """CPU, sm_86: the tiles cover 33-256 rows (the target's dispatch outside), tile K exactly and fit the ~96 KB of
    shared memory the W8A16 GEMM's stages may use."""
    _arch(monkeypatch, (8, 6))
    lo, hi = fast._MID_ROWS
    assert fast._mid_rows_cfg(lo - 1, n, k) is None and fast._mid_rows_cfg(hi + 1, n, k) is None
    assert fast._mid_rows_cfg(64, n, k + 64) is None  # K not a multiple of 128
    for m in range(lo, hi + 1):
        c = fast._mid_rows_cfg(m, n, k)
        assert c is not None and k % c["BK"] == 0, (m, c)
        assert c["num_stages"] * c["BK"] * (2 * c["BM"] + c["BN"]) <= 96 * 1024, (m, c)


def test_draft_int8_linear_routes(monkeypatch):
    """CPU: the op runs the Triton GEMM with the tiles where _mid_rows_cfg gives some (any leading shape), the
    target's dispatch (int8_mm) otherwise."""
    from vllm.model_executor.layers.quantization.utils import w8a16_utils as wu

    calls = []
    def gemm(x, w, cfg=None):
        calls.append(("gemm", x.shape[0], cfg))
        return x.new_zeros(x.shape[0], w.n)

    def mm(x, w):
        calls.append(("mm", x.shape[0], None))
        return x.new_zeros(x.shape[0], w.n)

    monkeypatch.setattr(wu, "w8a16_gemm", gemm)
    monkeypatch.setattr(wu, "int8_mm", mm)
    w = torch.zeros(1536, 1536, dtype=torch.int8)
    s = torch.ones(1536)
    _arch(monkeypatch, (8, 6))
    y = fast._draft_int8_linear(torch.zeros(4, 48, 1536, dtype=torch.bfloat16), w, s)
    assert y.shape == (4, 48, 1536)
    assert calls[-1] == ("gemm", 192, fast._mid_rows_cfg(192, 1536, 1536))
    fast._draft_int8_linear(torch.zeros(6, 1536, dtype=torch.bfloat16), w, s)
    assert calls[-1][:2] == ("mm", 6)
    _arch(monkeypatch, (12, 0))
    fast._draft_int8_linear(torch.zeros(192, 1536, dtype=torch.bfloat16), w, s)
    assert calls[-1][:2] == ("mm", 192)


def test_draft_method_keeps_group_scales_on_the_target_path(monkeypatch):
    """CPU: per-group scales (not produced by the drafter's RTN) go to the target's apply."""
    seen = []
    monkeypatch.setattr(ModelOptW8A16LinearMethod, "apply", lambda self, layer, x, bias=None: seen.append(1) or x)
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.zeros(64, 256, dtype=torch.int8), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.ones(64, 2), requires_grad=False)
    DraftW8A16LinearMethod().apply(layer, torch.zeros(3, 256, dtype=torch.bfloat16))
    assert seen == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("method_cls", [ModelOptW8A16LinearMethod, DraftW8A16LinearMethod])
@pytest.mark.parametrize("n,k", DRAFT_SHAPES)
@pytest.mark.parametrize("m", [4, 6, 48, 192, 300])
def test_int8_linear_matches_dequantized(n, k, m, method_cls):
    torch.manual_seed(n + k + m)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = quantize_int8_rtn(w)
    layer = nn.Module()
    layer.weight = nn.Parameter(q, requires_grad=False)
    layer.weight_scale = nn.Parameter(s, requires_grad=False)
    method = method_cls()
    method.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y = method.apply(layer, x)
    ref = F.linear(x.float(), s[:, None] * q.float())
    # The dequantize + cuBLAS path rounds the dequantized weights to bf16: an error growing as sqrt(K)
    # (fc, K = 7680, 300 rows on an RTX 3080: 0.0207 at worst).
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2 * max(1.0, (k / 4096) ** 0.5))

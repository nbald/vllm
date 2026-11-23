# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine fast draft path (qwen3_dspine_fast.py) against the eager reference.

The fused injection / transition functions are checked on CPU in fp32 (same
math as the eager modules) and on a GPU in bf16, plain and compiled, with the
skinny GEMV; the CUDA RMSNorm binding against the native norm on a GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.models import qwen3_dspine
from vllm.model_executor.models import qwen3_dspine_fast as fast

NUM_LAYERS = 3
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def single_rank_tp(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    # The eager reference: the flags are read at import, force them off.
    monkeypatch.setattr(fast, "FUSED", False)
    monkeypatch.setattr(fast, "GEMV", False)


def _build(block_size, device, dtype, hidden=256, transfer=128, message=64, vocab=512):
    with set_current_vllm_config(VllmConfig()):
        injection = qwen3_dspine.AdjacentCausalInjection(
            hidden_size=hidden,
            transfer_dim=transfer,
            message_dim=message,
            num_layers=NUM_LAYERS,
            block_size=block_size,
            params_dtype=dtype,
            quant_config=None,
            eps=1e-6,
            prefix="injection",
        ).to(device)
    torch.manual_seed(0)
    for param in injection.parameters():
        param.data.normal_(0, 0.05)
    codebook = torch.randn(vocab, transfer, device=device, dtype=dtype)
    head = SimpleNamespace(
        weight=torch.randn(vocab, hidden, device=device, dtype=dtype) * 0.05,
        tp_size=1,
    )
    norm = SimpleNamespace(
        weight=torch.full((hidden,), hidden**-0.5, device=device, dtype=dtype),
        variance_epsilon=1e-6,
    )
    return injection, codebook, head, norm


def _inputs(injection, codebook, num_reqs, k, top_k=8):
    device, dtype = codebook.device, codebook.dtype
    hidden = injection.hidden_size
    vocab = codebook.shape[0]
    block = k + 1
    pre = torch.randn(num_reqs * block, hidden, device=device, dtype=dtype) * 0.03
    anchor = injection.token_features(
        codebook, torch.randint(0, vocab, (num_reqs,), device=device)
    )
    cand = torch.randint(0, vocab, (num_reqs, k, top_k), device=device)
    anchor_ids = torch.randint(0, vocab, (num_reqs,), device=device)
    fpi = torch.randn(num_reqs, k, hidden, device=device, dtype=dtype) * 0.03
    return pre, anchor, cand, anchor_ids, fpi


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _check(injection, codebook, head, norm, num_reqs, k, inject, score, tol_inj, tol_s):
    pre, anchor, cand, anchor_ids, fpi = _inputs(injection, codebook, num_reqs, k)
    for layer in range(NUM_LAYERS):
        ref = injection(layer, pre, anchor_features=anchor)
        got = fast.inject(injection, layer, pre, anchor, inject)
        assert torch.equal(got[:: k + 1], pre[:: k + 1])  # position 0 kept
        assert _rel(got, ref) < tol_inj, layer
    ref = qwen3_dspine._score_transitions(
        injection, norm, head, codebook, cand, fpi, anchor_ids, NUM_LAYERS - 1
    )
    predecessor = torch.cat(
        (anchor_ids[:, None, None].expand(-1, 1, 8), cand[:, :-1]), 1
    )
    got = fast.score_transitions(
        injection, norm, head, codebook, cand, fpi, predecessor, NUM_LAYERS - 1, score
    )
    assert got.shape == ref.shape
    assert _rel(got, ref) < tol_s
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree > 0.99, agree


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_reqs,k", [(1, 3), (4, 7)])
def test_fused_functions_match_eager_fp32(num_reqs, k):
    injection, codebook, head, norm = _build(k + 1, "cpu", torch.float32)
    _check(
        injection, codebook, head, norm, num_reqs, k,
        fast.inject_fn, fast.transitions_fn, 1e-5, 1e-5,
    )  # fmt: skip


@pytest.mark.cpu_test
def test_fused_weight_refresh_is_in_place():
    injection, *_ = _build(4, "cpu", torch.float32)
    fast.refresh_fused_weights(injection)
    buf = injection.fused_pred_msg_weight
    injection.message.weight.data.add_(1.0)
    fast.refresh_fused_weights(injection)
    assert injection.fused_pred_msg_weight is buf
    m = injection.predecessor_gate.weight.shape[0]
    assert torch.equal(buf[m:], injection.message.weight)


@pytest.mark.cpu_test
def test_apply_fast_path_is_idempotent(monkeypatch):
    from vllm.model_executor.layers.linear import (
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )

    monkeypatch.setattr(fast, "FUSED", True)
    monkeypatch.setattr(fast, "GEMV", True)
    injection, *_ = _build(4, "cpu", torch.bfloat16)
    with set_current_vllm_config(VllmConfig()):
        proj = ReplicatedLinear(256, 256, bias=False, params_dtype=torch.bfloat16)
    inner = nn.Module()
    inner.injection = injection
    inner.proj = proj
    inner.layers = nn.ModuleList()
    model = nn.Module()
    model.model = inner
    for _ in range(2):
        fast.apply_fast_path(model)
    assert isinstance(proj.quant_method, fast.GemvLinearMethod)
    assert type(proj.quant_method.base) is UnquantizedLinearMethod
    assert injection.fused_pred_msg_weight is not None


@requires_cuda
@pytest.mark.parametrize("gemv", [False, True])
@pytest.mark.parametrize("num_reqs,k", [(1, 3), (8, 3), (1, 7), (8, 7)])
def test_fused_functions_match_eager_bf16(monkeypatch, num_reqs, k, gemv):
    monkeypatch.setattr(fast, "GEMV", gemv)
    injection, codebook, head, norm = _build(
        k + 1, "cuda", torch.bfloat16, hidden=1536, transfer=512, message=256
    )
    # bf16 outputs: a few ulps from different GEMM splits and fusion.
    _check(
        injection, codebook, head, norm, num_reqs, k,
        fast.inject_fn, fast.transitions_fn, 2e-2, 1e-3,
    )  # fmt: skip
    _check(
        injection, codebook, head, norm, num_reqs, k,
        torch.compile(fast.inject_fn, dynamic=True, fullgraph=True),
        torch.compile(fast.transitions_fn, dynamic=True, fullgraph=True),
        2e-2, 1e-3,
    )  # fmt: skip


@requires_cuda
@pytest.mark.parametrize("rows", [4, 32])
def test_cuda_rms_norm_matches_native(rows):
    """Hidden-size norm, fused residual add, and head norms on strided views
    of the qkv output (as in the draft attention)."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        for hidden in (1536, 128):
            ref = RMSNorm(hidden, eps=1e-6).cuda().to(torch.bfloat16)
            ref.weight.data.normal_(1.0, 0.1)
            cud = RMSNorm(hidden, eps=1e-6).cuda().to(torch.bfloat16)
            cud.weight.data.copy_(ref.weight.data)
            assert fast.bind_cuda_rms_norm(cud)
            if hidden == 1536:
                x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
                r = torch.randn_like(x)
                a, b = ref(x.clone()), cud(x.clone())
                ulp = a.float().abs().max().item() * 2**-7
                assert (a.float() - b.float()).abs().max().item() <= ulp
                a2, ra = ref(x.clone(), r.clone())
                b2, rb = cud(x.clone(), r.clone())
                assert (a2.float() - b2.float()).abs().max().item() <= 2 * ulp
                assert torch.equal(ra, rb)
            else:
                qkv = torch.randn(
                    rows, 12 * 128 + 2 * 4 * 128, device="cuda", dtype=torch.bfloat16
                )
                q = qkv[:, : 12 * 128].view(rows, 12, 128)
                a, b = ref(q), cud(q)
                ulp = a.float().abs().max().item() * 2**-7
                assert (a.float() - b.float()).abs().max().item() <= ulp

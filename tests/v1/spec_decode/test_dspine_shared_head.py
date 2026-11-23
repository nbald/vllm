# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLALA_DSPINE_SHARED_HEAD: the DSpine drafter on the target's LM head. CPU only.

The transition rescoring reads the head rows of the candidates: with an INT8
W8A16 head they must be the dequantized rows (q * scale), in the eager and the
fused path, and with a floating-point head nothing changes."""

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed import parallel_state
from vllm.model_executor.models import qwen3_dspine
from vllm.model_executor.models import qwen3_dspine_fast as fast

from vllm.v1.worker.gpu.spec_decode.dspine.speculator import _maybe_share_target_head

from .test_dspine_fast import NUM_LAYERS, _build, _inputs

pytestmark = pytest.mark.cpu_test

V, H = 512, 256


@pytest.fixture(autouse=True)
def single_rank_tp(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(fast, "FUSED", False)
    monkeypatch.setattr(fast, "GEMV", False)


def _int8_head(group=0):
    torch.manual_seed(1)
    q = torch.randint(-127, 128, (V, H), dtype=torch.int8)
    scale = torch.rand(V, H // group) * 1e-3 if group else torch.rand(V) * 1e-3
    head = SimpleNamespace(weight=q, weight_scale=scale, tp_size=1)
    s = scale.repeat_interleave(group, dim=1) if group else scale[:, None]
    deq = SimpleNamespace(weight=q.float() * s, tp_size=1)
    return head, deq


def test_head_rows_float_is_unchanged():
    w = torch.randn(V, H).bfloat16()
    ids = torch.randint(0, V, (3, 5, 8))
    assert torch.equal(
        qwen3_dspine.head_rows(SimpleNamespace(weight=w), ids), w[ids].float()
    )


@pytest.mark.parametrize("group", [0, 64])
def test_head_rows_int8_are_dequantized(group):
    head, deq = _int8_head(group)
    ids = torch.randint(0, V, (3, 5, 8))
    assert torch.equal(qwen3_dspine.head_rows(head, ids), deq.weight[ids])


@pytest.mark.parametrize("group", [0, 64])
@pytest.mark.parametrize("num_reqs,k", [(1, 3), (4, 5)])
def test_transition_scores_on_an_int8_head(group, num_reqs, k):
    """Eager and fused rescoring on the INT8 head equal the same rescoring on
    its fp32 dequantization."""
    injection, codebook, _, norm = _build(
        k + 1, "cpu", torch.float32, vocab=V, hidden=H
    )
    head, deq = _int8_head(group)
    _, _, cand, anchor_ids, fpi = _inputs(injection, codebook, num_reqs, k)
    args = (codebook, cand, fpi, anchor_ids, NUM_LAYERS - 1)
    ref = qwen3_dspine._score_transitions(injection, norm, deq, *args)
    got = qwen3_dspine._score_transitions(injection, norm, head, *args)
    assert torch.equal(got, ref)
    predecessor = torch.cat(
        (anchor_ids[:, None, None].expand(-1, 1, 8), cand[:, :-1]), 1
    )
    fargs = (codebook, cand, fpi, predecessor, NUM_LAYERS - 1, fast.transitions_fn)
    ref_f = fast.score_transitions(injection, norm, deq, *fargs)
    got_f = fast.score_transitions(injection, norm, head, *fargs)
    assert torch.equal(got_f, ref_f)


def test_share_flag_turns_the_own_head_off(monkeypatch):
    cfg = SimpleNamespace(has_own_lm_head=True)
    monkeypatch.delenv("OLALA_DSPINE_SHARED_HEAD", raising=False)
    assert not _maybe_share_target_head(cfg) and cfg.has_own_lm_head
    monkeypatch.setenv("OLALA_DSPINE_SHARED_HEAD", "1")
    assert _maybe_share_target_head(cfg) and not cfg.has_own_lm_head
    # a drafter without its own head already shares the target's
    assert not _maybe_share_target_head(SimpleNamespace(has_own_lm_head=False))


def test_own_head_weights_are_not_loaded():
    w = torch.zeros(1)
    names = [
        "lm_head.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight_scale",
        "model.lm_head_bias",
    ]
    kept = [n for n, _ in qwen3_dspine.drop_head_weights((n, w) for n in names)]
    assert kept == [
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "model.lm_head_bias",
    ]

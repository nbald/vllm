# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the Olala W8A16 MoE paths to dequantized FP32 references."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.has_device_capability(80):
    pytest.skip("bf16 mma.sync needs sm_80+", allow_module_level=True)

from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import (  # noqa: E402
    WqWeight,
    olala_moe_medium_int8,
    olala_moe_small_int8,
)

E, TOPK, H, D1, N1, S = 256, 6, 1536, 384, 768, 1536


def _roundf(x):
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def _make(w: torch.Tensor, group: int):
    """build_q8.py recipe. w [..., N, K] fp32 -> (WqWeight, dequantized fp32)."""
    if group == 0:
        s = (w.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-30)
        q = _roundf(w / s).clamp(-127, 127)
        return WqWeight(
            q.to(torch.int8).contiguous(), s.squeeze(-1).contiguous()
        ), q * s
    k = w.shape[-1]
    wg = w.reshape(*w.shape[:-1], k // group, group)
    d = wg.abs().amax(-1, keepdim=True) / 127
    q = _roundf(wg / d).clamp(-127, 127)
    dh = d.half()
    deq = (q * dh.float()).reshape(w.shape)
    # the MMA paths dequantize the group tiles to bf16
    return (
        WqWeight(
            q.to(torch.int8).reshape(w.shape).contiguous(), dh.squeeze(-1).contiguous()
        ),
        deq.bfloat16().float(),
    )


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x.bfloat16().float()).square().bfloat16()


def _reference(hidden, latent, logits, bias, w13, w2, su, sd, fc2):
    M = hidden.shape[0]
    sc = torch.sigmoid(logits)
    ids = torch.topk(sc + bias, TOPK, dim=-1).indices
    tw = sc.gather(1, ids)
    tw = tw / tw.sum(-1, keepdim=True)
    lat = latent.bfloat16().float()
    routed = torch.zeros(M, D1, device=hidden.device)
    for m in range(M):
        for k in range(TOPK):
            e = ids[m, k]
            h = _relu2(lat[m] @ w13[e].T)
            routed[m] += tw[m, k] * (h.float() @ w2[e].T)
    routed = (routed * 2.45).bfloat16()
    s = _relu2(hidden.float() @ su.T)
    return (routed.float() @ fc2.T + s.float() @ sd.T).bfloat16()


def _setup(m: int, group, dense_bf16: bool):
    torch.manual_seed(m + group)
    dev = "cuda"
    hidden = torch.randn(m, H, device=dev).bfloat16()
    latent = torch.randn(m, D1, device=dev)  # fp32, as the entry launch writes it
    logits = torch.randn(m, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    w13, dq13 = _make(torch.randn(E, N1, D1, device=dev) * D1**-0.5, group)
    w2, dq2 = _make(torch.randn(E, D1, N1, device=dev) * N1**-0.5, group)
    dense_group = group
    if dense_bf16:
        su_b = (torch.randn(S, H, device=dev) * H**-0.5).bfloat16()
        sd_b = (torch.randn(H, S, device=dev) * S**-0.5).bfloat16()
        f2_b = (torch.randn(H, D1, device=dev) * D1**-0.5).bfloat16()
        su, sd, fc2 = WqWeight(su_b), WqWeight(sd_b), WqWeight(f2_b)
        dqsu, dqsd, dqf2 = su_b.float(), sd_b.float(), f2_b.float()
    else:
        su, dqsu = _make(torch.randn(S, H, device=dev) * H**-0.5, dense_group)
        sd, dqsd = _make(torch.randn(H, S, device=dev) * S**-0.5, dense_group)
        fc2, dqf2 = _make(torch.randn(H, D1, device=dev) * D1**-0.5, dense_group)
    ref = _reference(hidden, latent, logits, bias, dq13, dq2, dqsu, dqsd, dqf2)
    return (hidden, latent, logits, bias, w13, w2, su, sd, fc2), ref


@pytest.mark.parametrize("m", [1, 2, 3, 5, 12])
@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize("gemv", [False, True])
@pytest.mark.parametrize("dense_bf16", [False, True])
def test_small_path_matches_reference(m, group, gemv, dense_bf16):
    args, ref = _setup(m, group, dense_bf16)
    out = olala_moe_small_int8(*args, TOPK, 2.45, gemv=gemv)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m", [6, 33, 128, 512])
@pytest.mark.parametrize("group", [0, 32])
def test_medium_path_matches_reference(m, group):
    args, ref = _setup(m, group, False)
    out = olala_moe_medium_int8(*args, TOPK, 2.45)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m", [1, 100, 1000, 3000])
@pytest.mark.parametrize("group", [0, 32])
def test_prefill_routed_experts(m, group):
    """wq_routed_experts (sorted grouped GEMMs, any batch) = the routed part of
    the reference."""
    from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import wq_routed_experts

    torch.manual_seed(m + group)
    dev = "cuda"
    latent = torch.randn(m, D1, device=dev).bfloat16()
    logits = torch.randn(m, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    w13, dq13 = _make(torch.randn(E, N1, D1, device=dev) * D1**-0.5, group)
    w2, dq2 = _make(torch.randn(E, D1, N1, device=dev) * N1**-0.5, group)
    out = wq_routed_experts(latent, logits, bias, w13, w2, TOPK, 2.45)
    sc = torch.sigmoid(logits)
    ids = torch.topk(sc + bias, TOPK, dim=-1).indices
    tw = sc.gather(1, ids)
    tw = tw / tw.sum(-1, keepdim=True)
    ref = torch.zeros(m, D1, device=dev)
    for e in torch.unique(ids).tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        h = _relu2(latent[rows].float() @ dq13[e].T)
        ref.index_add_(0, rows, tw[rows, slots][:, None] * (h.float() @ dq2[e].T))
    torch.testing.assert_close(
        out.float(), (ref * 2.45).bfloat16().float(), atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("m,run", [(4, olala_moe_small_int8)])
def test_split_router_logits(m, run):
    """Router logits passed as two strided slices of the entry projection
    (``router_logits + router_logits_lo``) give the same output as their sum."""
    args, _ = _setup(m, 0, False)
    hidden, latent, logits, *rest = args
    hi, lo = torch.empty((m, 2 * E), device=logits.device).split(E, dim=1)
    hi.copy_(logits.bfloat16())
    lo.copy_(logits - hi)
    ref = run(hidden, latent, hi + lo, *rest, TOPK, 2.45)
    out = run(hidden, latent, hi, *rest, TOPK, 2.45, router_logits_lo=lo)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

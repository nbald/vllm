# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FMA edition of the DiffTPA prefill attention (olala/csrc/tpa_prefill_attn_volta.cu, used on
the SM70 family) against FlashAttention-2, whose values it computes, and an fp64 reference."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="FlashAttention-2 (the reference kernel) needs sm_80+",
)

HQ, HKV, D = 48, 12, 128


def _reference(q, k, v, cuq, cuk, scale, softcap):
    """fp64 causal attention, the query rows aligned with the last keys of their sequence."""
    q, k, v = q.double(), k.double(), v.double()
    out = torch.zeros_like(q)
    for s in range(len(cuq) - 1):
        a, b, c, e = int(cuq[s]), int(cuq[s + 1]), int(cuk[s]), int(cuk[s + 1])
        ks = k[c:e].repeat_interleave(HQ // HKV, dim=1)
        vs = v[c:e].repeat_interleave(HQ // HKV, dim=1)
        sc = torch.einsum("thd,shd->hts", q[a:b], ks) * scale
        if softcap:
            sc = softcap * torch.tanh(sc / softcap)
        nq, nk = b - a, e - c
        rows = torch.arange(nq, device=q.device)[:, None]
        mask = torch.arange(nk, device=q.device)[None, :] > rows + nk - nq
        p = torch.softmax(sc.masked_fill(mask[None], float("-inf")), -1)
        out[a:b] = torch.einsum("hts,shd->thd", p, vs)
    return out


@pytest.mark.parametrize("softcap", [0.0, 150.0])
@torch.inference_mode()
def test_volta_prefill_attention_matches_fa2(softcap: float):
    """Bit for bit through volta-forever (which runs FA2's bf16 mma.sync as fmaf chains in
    the order of k, as the FMA kernel does), as close to fp64 as FA2 elsewhere; uneven
    sequences, one resuming after cached tokens (more keys than queries)."""
    from vllm.model_executor.layers.mamba.olala import tpa_prefill
    from vllm.model_executor.layers.mamba.olala.arch import volta_forever
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    g = torch.Generator(device="cuda").manual_seed(0)
    q_lens, k_lens = [77, 300, 161], [77, 413, 161]
    cuq = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    cuk = torch.tensor([0] + torch.tensor(k_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    q = (torch.randn(sum(q_lens), HQ, D, device="cuda", generator=g) * 3).bfloat16()
    k = (torch.randn(sum(k_lens), HKV, D, device="cuda", generator=g) * 3).bfloat16()
    v = torch.randn(sum(k_lens), HKV, D, device="cuda", generator=g).bfloat16()
    scale = D**-0.5

    fa2 = flash_attn_varlen_func(
        q=q, k=k, v=v, max_seqlen_q=max(q_lens), cu_seqlens_q=cuq, max_seqlen_k=max(k_lens),
        cu_seqlens_k=cuk, softmax_scale=scale, causal=True, softcap=softcap, fa_version=2,
    )
    out = torch.empty_like(q)
    tpa_prefill._volta_attn_ext().attn_fwd(q, k, v, cuq, cuk, out, max(q_lens), scale, softcap)
    ref = _reference(q, k, v, cuq, cuk, scale, softcap)

    def rel(x: torch.Tensor) -> float:
        return ((x.double() - ref).norm() / ref.norm()).item()

    assert rel(out) < 1e-2, rel(out)
    assert rel(out) < 1.25 * rel(fa2) + 1e-4, (rel(out), rel(fa2))
    if volta_forever():
        assert torch.equal(out, fa2)

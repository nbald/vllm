# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA editions of volta_cuda.py return their Triton kernels' bits: the geodesic norm (models/olala.py
_geodesic_norm_kernel) and the weight-only latent-MoE small path at 3-32 tokens (latent_moe_int8.py's Volta small
path: routing, routed / shared up, routed down, exit; INT8 per-row and INT4 affine routed experts), on the SM70 family
through volta-forever (told sm_80), where they are dispatched."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family, has_bf16_mma

if not (torch.cuda.is_available() and gpu_family() is GpuFamily.SM70 and has_bf16_mma()):
    pytest.skip("the Volta CUDA editions run on the SM70 family told sm_80+", allow_module_level=True)

from vllm.model_executor.layers.mamba.olala import volta_cuda  # noqa: E402
from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import (  # noqa: E402
    _volta_small_cfg,
    olala_moe_small_int8_volta,
)
from vllm.model_executor.models.olala import _geodesic_norm_kernel  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

from tests.kernels.mamba.test_olala_latent_moe_w8a16 import INT4, TOPK, _setup  # noqa: E402

D = 1536


def _geo_rows(m, kind, gen):
    x = torch.randn(m, D, device="cuda", generator=gen)
    g = torch.randn(m, D, device="cuda", generator=gen)
    if kind == "massive":
        x[:, :4] *= 300.0
        g *= 0.05
    elif kind == "parallel":
        g = 0.37 * x + 1e-6 * g
    elif kind == "zero":
        x.zero_()
    return x.bfloat16(), g.bfloat16()


@pytest.mark.parametrize("m", [1, 7, 32, 128])
@pytest.mark.parametrize("kind", ["normal", "massive", "parallel", "zero"])
def test_geodesic_cuda_matches_triton(m, kind):
    gen = torch.Generator(device="cuda").manual_seed(m)
    for layer in (0, 5, 35):
        x, g = _geo_rows(m, kind, gen)
        scale = (torch.rand(1, device="cuda", generator=gen) * 4 - 1).bfloat16()
        bias = (torch.rand(1, device="cuda", generator=gen) * 2 - 1).bfloat16()
        ref = torch.empty_like(x)
        _geodesic_norm_kernel[(m,)](
            x, g, ref, scale, bias, 1.0 / (layer + 1), torch.pi / 4, D, x.stride(0), g.stride(0), ref.stride(0),
            BLOCK=triton.next_power_of_2(D), num_warps=2,
        )
        out = torch.empty_like(x)
        assert volta_cuda.geodesic_ok(x, g, out, scale, bias)
        volta_cuda.geodesic(x, g, out, scale, bias, 1.0 / (layer + 1), torch.pi / 4)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_geodesic_cuda_graph_replay():
    gen = torch.Generator(device="cuda").manual_seed(0)
    x, g = _geo_rows(32, "normal", gen)
    scale, bias = torch.tensor([0.9], device="cuda").bfloat16(), torch.tensor([0.1], device="cuda").bfloat16()
    ref = volta_cuda.geodesic(x, g, torch.empty_like(x), scale, bias, 0.5, torch.pi / 4).clone()
    out = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        volta_cuda.geodesic(x, g, out, scale, bias, 0.5, torch.pi / 4)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def _triton_small(args, m, lo=None):
    hidden, latent, logits, bias, w13, w2, su, sd, fc2 = args
    return olala_moe_small_int8_volta(
        hidden, latent, logits, bias, w13, w2, su, sd, fc2, TOPK, 2.45, router_logits_lo=lo,
        config=_volta_small_cfg(triton.next_power_of_2(m)),
    )


@pytest.mark.parametrize("m", [3, 4, 5, 8, 9, 12, 16, 17, 24, 32])
@pytest.mark.parametrize("group", [0, INT4])
@pytest.mark.parametrize("split_logits", [False, True])
def test_moe_small_cuda_matches_triton(m, group, split_logits):
    """Q8 layout (INT8 per row everywhere) and Q4A2 layout (INT4 affine routed experts): the whole CUDA path = the
    Triton Volta small path, bit for bit."""
    args, _ = _setup(m, group, False)
    hidden, latent, logits, bias, w13, w2, su, sd, fc2 = args
    lo = None
    if split_logits:  # the entry launch's bf16x2 router: hi and lo parts, added in the routing
        hi = logits.bfloat16().float().contiguous()
        lo = (logits - hi).contiguous()
        args = (hidden, latent, hi, bias, w13, w2, su, sd, fc2)
    # dispatched from MOE_SMALL_MIN_TOKENS on; the kernels cover 3..32 tokens
    want = m >= volta_cuda.MOE_SMALL_MIN_TOKENS["int4" if group == INT4 else "int8"]
    assert volta_cuda.moe_small_ok(m, hidden, latent, args[2], lo, bias, w13, w2, su, sd, fc2, TOPK) == want
    ref = _triton_small(args, m, lo)
    out = volta_cuda.moe_small(hidden, latent, args[2], lo, bias, w13, w2, su, sd, fc2, TOPK, 2.45)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_moe_small_cuda_graph_replay_and_repeats():
    """Graph replay = eager, and repeated launches of mixed token counts give the same bits (the routing's
    compaction is deterministic)."""
    outs = {}
    for _ in range(2):
        for m in (32, 3, 9, 17, 32, 5, 16):
            args, _ = _setup(m, 0, False)
            hidden, latent, logits, bias, w13, w2, su, sd, fc2 = args
            o = volta_cuda.moe_small(hidden, latent, logits, None, bias, w13, w2, su, sd, fc2, TOPK, 2.45)
            if m in outs:
                assert torch.equal(o, outs[m])
            outs[m] = o
    args, _ = _setup(24, 0, False)
    hidden, latent, logits, bias, w13, w2, su, sd, fc2 = args
    ref = volta_cuda.moe_small(hidden, latent, logits, None, bias, w13, w2, su, sd, fc2, TOPK, 2.45)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = volta_cuda.moe_small(hidden, latent, logits, None, bias, w13, w2, su, sd, fc2, TOPK, 2.45)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)

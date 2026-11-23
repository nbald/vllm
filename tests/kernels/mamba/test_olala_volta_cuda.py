# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA editions of volta_cuda.py return their Triton kernels' bits: the geodesic norm (models/olala.py
_geodesic_norm_kernel), on the SM70 family through volta-forever (told sm_80), where they are dispatched."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family, has_bf16_mma

if not (torch.cuda.is_available() and gpu_family() is GpuFamily.SM70 and has_bf16_mma()):
    pytest.skip("the Volta CUDA editions run on the SM70 family told sm_80+", allow_module_level=True)

from vllm.model_executor.layers.mamba.olala import volta_cuda  # noqa: E402
from vllm.model_executor.models.olala import _geodesic_norm_kernel  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402


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

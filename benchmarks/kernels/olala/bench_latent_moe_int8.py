# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the complete weight-only decode MoE at the Olala 7A1B shapes.

Routed, shared and latent weights rotate above L2 in a CUDA graph.
Example: python bench_latent_moe_int8.py --batch 1 8 32 128
"""

import argparse
from functools import partial

import torch
from bench_decode_gemv import timeit_graph

from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import (
    INT8_SMALL_MAX_TOKENS,
    WqWeight,
    olala_moe_medium_int8,
    olala_moe_small_int8,
)
from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
    Segment,
    int8_decode_linear,
)

H, D, N, E, TOP_K = 1536, 384, 768, 256, 6


def weight(shape):
    q = torch.randint(-127, 128, shape, dtype=torch.int8, device="cuda")
    scales = torch.rand(shape[:-1], device="cuda") * 0.001
    return WqWeight(q, scales)


def layer():
    fc1 = weight((D, H))
    router = torch.randn((2 * E, H), device="cuda", dtype=torch.bfloat16) * 0.02
    segs = [Segment(fc1.weight, fc1.scale), Segment(router, col0=D)]
    experts = (weight((E, N, D)), weight((E, D, N)))
    projections = (weight((H, H)), weight((H, H)), weight((H, D)))
    return segs, torch.randn(E, device="cuda") * 0.1, (*experts, *projections)


def forward(hidden, params):
    segs, bias, weights = params
    m = hidden.shape[0]
    out = torch.empty((m, D + 2 * E), device="cuda", dtype=torch.float32)
    int8_decode_linear(hidden, segs, out=out)
    latent, hi, lo = out.split((D, E, E), dim=-1)
    if m <= INT8_SMALL_MAX_TOKENS:
        return olala_moe_small_int8(
            hidden, latent, hi, bias, *weights, TOP_K, 2.45, router_logits_lo=lo
        )
    return olala_moe_medium_int8(hidden, latent, hi + lo, bias, *weights, TOP_K, 2.45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--layers", type=int, default=36)
    args = parser.parse_args()
    if args.layers < 1 or any(not 1 <= b <= 128 for b in args.batch):
        parser.error("use at least one layer and batches of 1 to 128 tokens")
    torch.manual_seed(0)
    layers = [layer() for _ in range(args.layers)]
    for batch in args.batch:
        hidden = torch.randn(batch, H, dtype=torch.bfloat16, device="cuda")
        calls = [partial(forward, hidden, p) for p in layers]
        print(f"batch={batch} {1000 * timeit_graph(calls):.2f} us/layer")


if __name__ == "__main__":
    main()

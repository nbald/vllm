# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-weight W8A16 and BF16 decode projections at the Olala shapes.

Each CUDA graph rotates through enough weight matrices to exceed L2.
Use --m to select batch sizes and --group 32 to measure per-group scaling.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_decode_gemv import timeit_graph  # noqa: E402

import vllm.model_executor.layers.quantization.utils.w8a16_utils as i8  # noqa: E402
from vllm.model_executor.layers.mamba.olala import decode_gemv as dg  # noqa: E402

# (INT8 rows, bf16 rows, K, name)
SHAPES = [
    (7344, 0, 1536, "in_proj+dyn"),
    (7300, 0, 1536, "tpa q/k/v"),
    (7264, 36, 1536, "tpa 8+16"),
    (4608, 0, 1536, "gate_proj"),
    (1536, 0, 1536, "shared up/dn"),
    (1536, 0, 3072, "mixer_proj M"),
    (1536, 0, 4608, "mixer_proj V"),
    (384, 512, 1536, "entry fc1+rt"),
    (384, 0, 1536, "fc1_latent"),
    (1536, 0, 384, "fc2_latent"),
    (151936, 0, 1536, "lm_head"),
]
LLC_TARGET_BYTES = 400 * 1024**2
PEAK_GBS = 1790


def make(n8, n16, K, gen, group):
    w = torch.randn(n8, K, device="cuda", generator=gen) * 0.02
    if group:
        d = (w.view(n8, K // group, group).abs().amax(-1) / 127).half()
        q = torch.round(w.view(n8, K // group, group) / d.float()[..., None]).clamp(
            -127, 127
        )
        segs = [i8.Segment(q.to(torch.int8).view(n8, K), d)]
    else:
        sc = w.abs().amax(1) / 127
        segs = [
            i8.Segment(torch.round(w / sc[:, None]).clamp(-127, 127).to(torch.int8), sc)
        ]
    if n16:
        w16 = (torch.randn(n16, K, device="cuda", generator=gen) * 0.02).bfloat16()
        segs.append(i8.Segment(w16, col0=n8))
    return segs


def seg_bytes(segs):
    return sum(
        s.weight.numel() * s.weight.element_size()
        + (s.scale.numel() * s.scale.element_size() if s.scale is not None else 0)
        for s in segs
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--shapes", nargs="*", default=None, help="subset of shape names")
    ap.add_argument("--group", type=int, choices=(0, 32), default=0)
    ap.add_argument("--no-ref", action="store_true", help="skip the BF16 reference")
    a = ap.parse_args()
    arch = "".join(map(str, torch.cuda.get_device_capability()))
    print(f"{torch.cuda.get_device_name()} sm_{arch} group={a.group}")
    gen = torch.Generator(device="cuda").manual_seed(0)
    for n8, n16, K, name in SHAPES:
        if a.shapes and name not in a.shapes:
            continue
        N = n8 + n16
        segs0 = make(n8, n16, K, gen, a.group)
        b8 = seg_bytes(segs0)
        r8 = max(2, -(-LLC_TARGET_BYTES // b8))
        segs = [segs0] + [make(n8, n16, K, gen, a.group) for _ in range(r8 - 1)]
        rb = max(2, -(-LLC_TARGET_BYTES // (N * K * 2)))
        wb = (
            []
            if a.no_ref
            else [
                torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=gen)
                * 0.02
                for _ in range(rb)
            ]
        )
        out_dtype = torch.float32 if name.startswith("entry") else torch.bfloat16
        for M in a.m:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=gen)
            line = f"{name:13s} M={M:3d}"
            if wb and M <= dg.SKINNY_MAX_M:
                us = (
                    timeit_graph([lambda w=w, x=x: dg.decode_gemv(x, w) for w in wb])
                    * 1e3
                )
                line += f" | bf16 {us:6.1f}us"
            out = torch.empty(M, N, device="cuda", dtype=out_dtype)
            if M <= i8.SKINNY_INT8_MAX_M:
                us = (
                    timeit_graph(
                        [
                            lambda s=s, x=x, out=out: i8.int8_decode_linear(
                                x, s, out=out
                            )
                            for s in segs
                        ]
                    )
                    * 1e3
                )
                gbs = b8 / us / 1e3
                line += (
                    f" | int8 {us:6.1f}us {gbs:5.0f}GB/s ({100 * gbs / PEAK_GBS:3.0f}%)"
                )
            print(line, flush=True)
        del segs, wb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

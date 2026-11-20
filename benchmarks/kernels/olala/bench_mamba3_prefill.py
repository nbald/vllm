# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala Mamba-3 MIMO prefill (one layer, mamba3_mimo_hybrid_pools) per kernel path.

Paths: "tilelang" (Triton chunk states + TileLang output pass) and "cuda" (the
arch's CUDA chunk-state / output-pass kernels: sm90, tcgen05 or mma.sync).

    python benchmarks/kernels/olala/bench_mamba3_prefill.py --lens 512 2048 8192 32768
"""

import argparse

import torch
from triton.testing import do_bench

from vllm.model_executor.layers.mamba.ops.mamba3 import mimo_ssd

H, N, P, R, A = 48, 128, 64, 4, 32


def make_inputs(T: int, seed: int = 0) -> list:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    dt = torch.rand(H, T, device="cuda", generator=g) * 0.4
    adt = -dt * torch.rand(H, 1, device="cuda", generator=g) * 3
    q, k = rn(T, R, N).bfloat16(), rn(T, R, N).bfloat16()
    v, z = rn(T, H, P, scale=0.1).bfloat16(), rn(T, H, P, scale=0.1).bfloat16()
    angles, trap = rn(T, A, scale=0.3), rn(H, T).bfloat16()
    q_bias, k_bias = rn(H, R, N), rn(H, R, N)
    psi, phi, zeta = rn(H, R, P, scale=0.5), rn(H, R, P, scale=0.5), rn(H, R, P)
    d = 5 * rn(H)
    return [q, k, v, z, angles, dt, adt, trap, q_bias, k_bias, psi, phi, zeta, d]


def set_path(path: str) -> None:
    cuda = path == "cuda"
    mimo_ssd._CS_CUDA = cuda
    mimo_ssd._SCAN_CUDA = cuda
    mimo_ssd._SCAN_SM80 = cuda
    mimo_ssd._SCAN_TC = cuda


def bench(lens: list[int], path: str) -> float:
    set_path(path)
    args = make_inputs(sum(lens))
    ns = len(lens)
    pools = [
        torch.zeros(ns, H, A, device="cuda"),
        torch.zeros(ns, H, P, N, device="cuda"),
        torch.zeros(ns, R, H, N, device="cuda", dtype=torch.bfloat16),
        torch.zeros(ns, H, P, device="cuda", dtype=torch.bfloat16),
    ]
    out = torch.empty(sum(lens), H, P, device="cuda", dtype=torch.bfloat16)
    cu = [0] + torch.tensor(lens).cumsum(0).tolist()
    slots = torch.arange(ns, device="cuda", dtype=torch.int32)

    def fn():
        mimo_ssd.mamba3_mimo_hybrid_pools(*args, cu, pools, slots, None, out)

    return do_bench(fn, warmup=50, rep=300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16],
                    help="number of sequences of each length packed together")
    ap.add_argument("--paths", nargs="+", default=["tilelang", "cuda"])
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"{'seqs':>5} {'len':>6} " + " ".join(f"{p + ' ms':>12} {'Mtok/s':>7}" for p in a.paths))
    for b in a.batch:
        for L in a.lens:
            if b * L > 65536:
                continue
            row = []
            for p in a.paths:
                ms = bench([L] * b, p)
                row.append(f"{ms:12.3f} {b * L / ms / 1e3:7.2f}")
            print(f"{b:5d} {L:6d} " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()

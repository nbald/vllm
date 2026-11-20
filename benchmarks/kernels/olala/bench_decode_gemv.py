# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala decode projections (x @ W.T, x: (M, K)) per kernel, at the model's weight shapes.

One CUDA graph cycling over rotating weight copies (> 400 MB, above the L2): every call
streams its weights cold from DRAM, back to back as in a graphed decode step. Prints us per
call and the achieved weight-streaming bandwidth (GB/s, weights read once).

    python benchmarks/kernels/olala/bench_decode_gemv.py --m 1 2 4 8 16 32 64
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.olala import decode_gemv as dg

# (N, K, name): Mamba in_proj (+dyn), TPA concat projections, attention gate, Mamba / TPA
# mixer_proj, latent-MoE in (latent down | shared up) and out (shared down | latent up).
SHAPES = [
    (7344, 1536, "mamba in_proj"),
    (7300, 1536, "tpa q/k/v concat"),
    (4608, 1536, "attn gate_proj"),
    (1536, 3072, "mamba mixer_proj"),
    (1536, 4608, "tpa mixer_proj"),
    (1920, 1536, "moe in"),
    (1536, 1920, "moe out"),
]


def kernels(M: int):
    ks = {"cublas": lambda x, w: F.linear(x, w)}
    if M <= dg.SKINNY_MAX_M:
        ks["skinny"] = lambda x, w: dg.skinny_linear(x, w)
    if M == 1:
        ks["bw_fullk"] = dg._decode_gemv_blackwell
        ks["m1_gemv"] = dg._gemv_m1
    ks["dispatch"] = dg.decode_gemv
    return ks


LLC_TARGET_BYTES = 400 * 1024**2


def timeit_graph(calls, inner=2, replays=10, rounds=5, warmup=5) -> float:
    """ms per call: one graph of len(calls) * inner calls, median over rounds."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        g.replay()
    for st, en in events:
        st.record()
        for _ in range(replays):
            g.replay()
        en.record()
    torch.cuda.synchronize()
    return statistics.median(st.elapsed_time(en) / (replays * len(calls) * inner) for st, en in events)


def bench(fn, M: int, N: int, K: int) -> tuple[float, bool]:
    g = torch.Generator(device="cuda").manual_seed(0)
    r = max(2, -(-LLC_TARGET_BYTES // (N * K * 2)))
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 for _ in range(r)]
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    try:
        ok = torch.allclose(fn(x, ws[0]).float(), F.linear(x.float(), ws[0].float()), atol=2e-2, rtol=2e-2)
    except Exception as e:  # noqa: BLE001
        print(f"    ! {e}")
        return float("nan"), False
    calls = [lambda w=w: fn(x, w) for w in ws]
    return timeit_graph(calls) * 1e3, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    for M in a.m:
        print(f"\n== M = {M}")
        for N, K, name in SHAPES:
            res = []
            for kname, fn in kernels(M).items():
                us, ok = bench(fn, M, N, K)
                gbs = N * K * 2 / (us * 1e3)
                res.append(f"{kname} {us:6.1f}us {gbs:5.0f}GB/s{'' if ok else ' WRONG'}")
            print(f"  {name:18s} {N:5d}x{K:<5d} | " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()

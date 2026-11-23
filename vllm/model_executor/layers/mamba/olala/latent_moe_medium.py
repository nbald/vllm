# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Medium-batch path for Olala's latent MoE (decode batches, 9..512 tokens).

The generic triton path costs ~20 us of small kernels per layer on top of the
two grouped GEMMs (topk gating, block alignment, count & sort, activation,
top-k sum, shared concat). Here:

* the routing (sigmoid + bias top-k, renormalized weights, same semantics as
  the small path) is one kernel, and one program per expert lists the entries
  routed to it; each GEMM program finds its (expert, row block) by a prefix
  sum over the 256 counts, so there is no padded sort and a heavily routed
  expert still spreads over many programs;
* the up GEMM applies relu^2 in its epilogue;
* the down GEMM applies the routing weight and writes fp32 partials, which the
  small path's epilogue kernel sums over top-k (fixed order: deterministic)
  and concatenates with relu^2 of the shared-expert pre-activation.

At a 256-token decode batch every expert is touched, so the GEMMs read all the
routed weights once per step; the configs below are tuned on B300 for that.
"""

import functools

import torch

from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.model_executor.layers.mamba.olala.decode_gemv import decode_gemv
from vllm.model_executor.layers.mamba.olala.latent_moe_decode import (
    _latent_moe_epilogue_kernel,
)
from vllm.model_executor.layers.mamba.olala.pdl import pdl_kwargs
from vllm.triton_utils import tl, triton

MEDIUM_BATCH_MAX_TOKENS = 512


@triton.jit
def _route_topk_kernel(
    logits_ptr,  # [M, E] fp32 router logits
    bias_ptr,  # [E] fp32 e_score_correction_bias
    ids_ptr,  # [M*K] int32 out: expert of entry m*K+k
    tw_ptr,  # [M*K] fp32 out: renormalized routing weight
    M,
    stride_l,
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    RB: tl.constexpr,
    PDL: tl.constexpr = False,
    logits_lo_ptr=None,
    stride_lo=0,
):
    """Sigmoid scores ranked by score + bias; weights renormalized over the top k."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    offs_e = tl.arange(0, E)
    offs_k = tl.arange(0, KP)
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    rmask = rows < M
    lg = tl.load(
        logits_ptr + rows[:, None] * stride_l + offs_e[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    if logits_lo_ptr is not None:
        lg += tl.load(
            logits_lo_ptr + rows[:, None] * stride_lo + offs_e[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
    # Padded CUDA-graph rows can hold NaN logits. Scoring them 0 keeps the k
    # winners distinct (ranked by the bias alone): a repeated expert would
    # overflow its RCAP-row list in _compact_kernel and corrupt other rows.
    sc = tl.where(lg != lg, 0.0, 1.0 / (1.0 + tl.exp(-lg)))
    sb = sc + tl.load(bias_ptr + offs_e)[None, :]
    sb = tl.where(sb != sb, float("-inf"), sb)
    kid = tl.zeros([RB, KP], dtype=tl.int32)
    kw = tl.zeros([RB, KP], dtype=tl.float32)
    for kk in tl.static_range(K):
        mx = tl.max(sb, axis=1)
        idx = tl.minimum(
            tl.min(tl.where(sb == mx[:, None], offs_e[None, :], E), axis=1), E - 1
        )
        hit = offs_e[None, :] == idx[:, None]
        w = tl.sum(tl.where(hit, sc, 0.0), axis=1)
        sel = offs_k[None, :] == kk
        kid = tl.where(sel, idx[:, None], kid)
        kw = tl.where(sel, w[:, None], kw)
        sb = tl.where(hit, float("-inf"), sb)
    kw = kw / tl.maximum(tl.sum(kw, axis=1), 1e-20)[:, None]
    m2 = rmask[:, None] & (offs_k[None, :] < K)
    o = rows[:, None] * K + offs_k[None, :]
    tl.store(ids_ptr + o, kid, mask=m2)
    tl.store(tw_ptr + o, kw, mask=m2)


@triton.jit
def _compact_kernel(
    ids_ptr,  # [MK] int32 expert of every entry
    rows_ptr,  # [E, RCAP] int32 out: entries of each expert, in order
    cnt_ptr,  # [E] int32 out
    MK,
    RCAP,
    QP: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """One program per expert: the list of entries routed to it (no padding, no
    global sort; any number of rows per expert)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    e = tl.program_id(0)
    q = tl.arange(0, QP)
    hit = (tl.load(ids_ptr + q, mask=q < MK, other=-1) == e).to(tl.int32)
    pos = tl.cumsum(hit, axis=0) - 1
    tl.store(rows_ptr + e * RCAP + pos, q, mask=hit > 0)
    tl.store(cnt_ptr + e, tl.sum(hit, axis=0))


@triton.jit
def _moe_gemm_kernel(
    a_ptr,  # up: latent [M, Kd] (row entry // TOPK); down: h [M*TOPK, Kd] (row entry)
    b_ptr,  # [E, N, Kd] bf16
    c_ptr,  # up: h [M*TOPK, N] bf16; down: part [M*TOPK, N] fp32
    tw_ptr,  # [M*TOPK] fp32 (down only)
    rows_ptr,  # [E, RCAP] int32
    cnt_ptr,  # [E] int32
    N,
    Kd,
    RCAP,
    stride_am,
    stride_cm,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    UP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Program (row block, N block). Row blocks enumerate every expert's rows in
    BLOCK_SIZE_M chunks (prefix sum over the per-expert counts, done in each
    program), so a heavily routed expert spreads over several programs."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    num_pid_n = N // BLOCK_SIZE_N
    pid = tl.program_id(0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_e = tl.arange(0, E)
    nblk = (tl.load(cnt_ptr + offs_e) + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cum = tl.cumsum(nblk, axis=0)
    e = tl.sum((cum <= pid_m).to(tl.int32), axis=0)
    if e >= E:
        return
    first = tl.sum(tl.where(offs_e == e, cum - nblk, 0), axis=0)
    e = e.to(tl.int64)
    cnt = tl.load(cnt_ptr + e)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    r0 = (pid_m - first) * BLOCK_SIZE_M
    offs_r = r0 + tl.arange(0, BLOCK_SIZE_M)
    rmask = offs_r < cnt
    entry = tl.load(rows_ptr + e * RCAP + offs_r, mask=rmask, other=0).to(tl.int64)
    arow = entry // TOPK if UP else entry
    a_ptrs = a_ptr + arow[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + e * N * Kd + offs_n[None, :] * Kd + offs_k[:, None]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, Kd // BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        a = a.to(b.dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c_ptrs = c_ptr + entry[:, None] * stride_cm + offs_n[None, :]
    if UP:
        h = acc.to(tl.bfloat16).to(tl.float32)
        h = tl.maximum(h, 0.0)
        tl.store(c_ptrs, (h * h).to(tl.bfloat16), mask=rmask[:, None])
    else:
        w = tl.load(tw_ptr + entry, mask=rmask, other=0.0)
        tl.store(c_ptrs, acc * w[:, None], mask=rmask[:, None])


# Tuned on B300 with skewed routing: (max tokens, up config, down config).
_CONFIGS = [
    (
        16,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            num_warps=4,
            num_stages=3,
        ),
    ),
    (
        32,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            num_warps=4,
            num_stages=4,
        ),
    ),
    (
        64,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            num_warps=4,
            num_stages=4,
        ),
    ),
    (
        128,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=4,
        ),
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            num_warps=4,
            num_stages=4,
        ),
    ),
    (
        192,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            num_warps=4,
            num_stages=3,
        ),
    ),
    (
        256,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            num_warps=4,
            num_stages=3,
        ),
    ),
    (
        384,
        dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=32,
            num_warps=8,
            num_stages=4,
        ),
        dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
    ),
    (
        512,
        dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=32,
            num_warps=4,
            num_stages=3,
        ),
        dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            num_warps=4,
            num_stages=3,
        ),
    ),
]


@functools.lru_cache(maxsize=1)
def _max_stages() -> int:
    """The B300 configs pipeline up to 4 stages (up to 110 KB of shared memory);
    GPUs with less than ~160 KB per block (A4500 / L40S / RTX: 99 KB) get 3."""
    smem = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
    return 4 if smem >= 160 * 1024 else 3


def _pick(M: int):
    up, down = _CONFIGS[-1][1], _CONFIGS[-1][2]
    for mx, u, d in _CONFIGS:
        if mx >= M:
            up, down = u, d
            break
    ns = _max_stages()
    if max(up["num_stages"], down["num_stages"]) > ns:
        up = {**up, "num_stages": min(up["num_stages"], ns)}
        down = {**down, "num_stages": min(down["num_stages"], ns)}
    return up, down


def moe_routed_medium(
    latent: torch.Tensor,  # [M, D1] bf16 (may be a strided view)
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w13: torch.Tensor,  # [E, N1, D1] bf16
    w2: torch.Tensor,  # [E, D2, N1] bf16
    top_k: int,
    configs=None,
):
    """Routing + both grouped GEMMs. Returns the fp32 weighted partials [M*K, D2]
    (routed scale not applied) for the epilogue kernel."""
    M, D1 = latent.shape
    E, N1, _ = w13.shape
    D2 = w2.shape[1]
    up_cfg, down_cfg = configs if configs is not None else _pick(M)
    assert N1 % up_cfg["BLOCK_SIZE_N"] == 0 and D2 % down_cfg["BLOCK_SIZE_N"] == 0
    assert D1 % up_cfg["BLOCK_SIZE_K"] == 0 and N1 % down_cfg["BLOCK_SIZE_K"] == 0
    dev = latent.device
    MK = M * top_k
    ids = torch.empty(MK, dtype=torch.int32, device=dev)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    rows = torch.empty(E, M, dtype=torch.int32, device=dev)
    cnt = torch.empty(E, dtype=torch.int32, device=dev)
    RB = 8
    _route_topk_kernel[(triton.cdiv(M, RB),)](
        router_logits,
        e_score_bias,
        ids,
        tw,
        M,
        router_logits.stride(0),
        E=E,
        K=top_k,
        KP=triton.next_power_of_2(top_k),
        RB=RB,
        num_warps=4,
        **pdl_kwargs(),
    )
    _compact_kernel[(E,)](
        ids, rows, cnt, MK, M, QP=max(triton.next_power_of_2(MK), 16), num_warps=4,
        **pdl_kwargs(),
    )
    h = torch.empty(MK, N1, dtype=w13.dtype, device=dev)
    nbm = E + triton.cdiv(MK, up_cfg["BLOCK_SIZE_M"])
    _moe_gemm_kernel[(nbm * (N1 // up_cfg["BLOCK_SIZE_N"]),)](
        latent,
        w13,
        h,
        tw,
        rows,
        cnt,
        N1,
        D1,
        M,
        latent.stride(0),
        h.stride(0),
        E=E,
        TOPK=top_k,
        UP=True,
        **up_cfg,
        **pdl_kwargs(),
    )
    part = torch.empty(MK, D2, dtype=torch.float32, device=dev)
    nbm = E + triton.cdiv(MK, down_cfg["BLOCK_SIZE_M"])
    _moe_gemm_kernel[(nbm * (D2 // down_cfg["BLOCK_SIZE_N"]),)](
        h,
        w2,
        part,
        tw,
        rows,
        cnt,
        D2,
        N1,
        M,
        h.stride(0),
        part.stride(0),
        E=E,
        TOPK=top_k,
        UP=False,
        **down_cfg,
        **pdl_kwargs(),
    )
    return part


def olala_moe_medium(
    hidden: torch.Tensor,  # [M, hidden] bf16
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w_in: torch.Tensor,  # [latent + shared_inter, hidden]  (fc1 ‖ shared_up)
    w_out: torch.Tensor,  # [hidden, latent + shared_inter]  (fc2 ‖ shared_down)
    w13: torch.Tensor,  # [E, inter, latent]
    w2: torch.Tensor,  # [E, latent, inter]
    top_k: int,
    routed_scale: float,
    x_in: torch.Tensor | None = None,  # precomputed hidden @ w_in.T (fp32 or bf16)
) -> torch.Tensor:
    """Medium-batch latent MoE: shared(x) + fc2(scale * sum_k w_k * E_k(fc1(x)))."""
    M = hidden.shape[0]
    D1 = w13.shape[2]
    D2 = w2.shape[1]
    S = w_in.shape[0] - D1
    # sm_120: the skinny kernel up to 32 rows, where cuBLAS picks sm80 wmma kernels (5.5 vs 17 us
    # in, 5.2 vs 11.2 us out on an RTX 5090). Volta: cuBLAS has no Volta bf16 GEMM kernel (~250-300
    # us per projection at decode sizes); decode_gemv takes FMA / skinny kernels up to 64 rows.
    linear = (
        decode_gemv
        if gpu_family() in (GpuFamily.SM70, GpuFamily.SM120)
        else torch.nn.functional.linear
    )
    if x_in is None:
        x_in = linear(hidden, w_in)  # [M, D1 + S]
    part = moe_routed_medium(x_in[:, :D1], router_logits, e_score_bias, w13, w2, top_k)
    cat = torch.empty(M, D2 + S, dtype=hidden.dtype, device=hidden.device)
    _latent_moe_epilogue_kernel[(M, (D2 + S) // 128)](
        part,
        x_in[:, D1:],
        cat,
        routed_scale,
        x_in.stride(0),
        D2=D2,
        S=S,
        BS=128,
        K=top_k,
        num_warps=1,
        **pdl_kwargs(),
    )
    return linear(cat, w_out)

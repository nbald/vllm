# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT-built CUDA Mamba-3 MIMO decode step (see csrc/mamba3_step.cu).

Persistent CTAs (one head each, several tokens) stream the SSM state through
a 2-stage cp.async pipeline; the rank-4 state update and the C-contraction
run on tensor cores. Numerically equivalent to the CuteDSL step within bf16
rounding (the state is bf16-rounded before the C dot). GH200, B=64: 40 us
per layer vs 57 us (CuteDSL); B=16: 11.4 vs 15.1; B=1: 3.8 vs 6.1.

An fp32 SSM pool runs the update and the C dot as fp32 FMAs on the unrounded
state (matches the CuteDSL fp32 step to ~1e-4 relative). GH200, B=64: 73 us
vs 80 us (CuteDSL); B=16: 17.5 vs 18.5.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.olala.arch import (
    GpuFamily,
    gencode,
    gpu_family,
    has_bf16_mma,
    has_fp8_cvt,
    sm_arch,
)
from vllm.model_executor.layers.mamba.olala.fallback import warn_slow_path
from vllm.model_executor.layers.mamba.olala.pdl import pdl_enabled

logger = init_logger(__name__)

_ENV = "OLALA_MAMBA3_STEP"


@lru_cache(maxsize=1)
def cuda_step_enabled() -> bool:
    """OLALA_MAMBA3_STEP=cuda (default), on sm_80+ (the kernel only needs
    cp.async / ldmatrix / mma.sync; it is built for the running GPU's arch),
    and only if the JIT build succeeds; otherwise the CuteDSL step is used."""
    if os.environ.get(_ENV, "cuda") != "cuda":
        logger.info("Olala: CUDA Mamba-3 step disabled by %s.", _ENV)
        return False
    if not has_bf16_mma():
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        warn_slow_path(
            "CUDA Mamba-3 decode step disabled",
            f"the kernel needs sm_80+ (mma.sync bf16); this GPU is sm_{cap}",
            "CuteDSL decode step: ~1.4x slower Mamba-3 decode step (-10 to -15% decode throughput)",
            "run on an Ampere or newer GPU (nothing to fix elsewhere; output is correct)",
        )
        return False
    try:
        _load()
    except Exception as e:  # noqa: BLE001 - any build/load failure falls back
        warn_slow_path(
            "CUDA Mamba-3 decode step disabled",
            f"JIT build of mamba3_step.cu failed: {e}",
            "CuteDSL decode step: ~1.4x slower Mamba-3 decode step (-10 to -15% decode throughput)",
            "the JIT needs an nvcc of torch's CUDA major (torch.version.cuda); "
            "set OLALA_JIT_CUDA_HOME to such a toolkit",
        )
        return False
    return True


@lru_cache(maxsize=1)
def _load():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    sm = sm_arch()
    # One build per arch: the build directory is shared (e.g. a cache volume
    # mounted on H100 and A4500 nodes alike); sm_90 keeps its historical name.
    name = "olala_mamba3_step" if sm == "90" else f"olala_mamba3_step_sm{sm}"
    flags = ["-O3", "-std=c++20", gencode()]
    if gpu_family() is GpuFamily.SM70:
        # V100 (told sm_80 by volta-forever): the bf16 mma.sync / ldmatrix of the
        # fp32-state step are emulated with scalar code there, so build the scalar
        # FMA update (STEP_VOLTA, csrc/mamba3_step.cu). V100 through volta-forever,
        # fp32 state, graphed: 63 -> 12 us per layer at batch 1, 682 -> 87 us at 8,
        # 1296 -> 162 us at 16; its edition with the state in registers
        # (mamba3_step_volta_kernel, same bits): 1-stage 14.0 -> 12.1 us at 1,
        # 208 -> 155 us at 32, 412 -> 301 us at 64.
        name += "_volta"
        flags.append("-DSTEP_VOLTA")
    return load_extension(name, "mamba3_step.cu", flags)


def cuda_step_supported(ssm_pool, k_pool, v_pool, angle_pool, B, C, angle, num_heads, head_dim, d_state, rank, num_angles) -> bool:
    return (
        num_heads == 48 and head_dim == 64 and d_state == 128 and rank == 4 and num_angles == 32
        and ssm_pool.dtype in (torch.float32, torch.bfloat16) and k_pool.dtype == torch.bfloat16 and v_pool.dtype == torch.bfloat16
        and angle_pool.dtype == torch.float32
        and B.dim() == 3 and B.stride(1) == d_state and B.stride(2) == 1 and (B.stride(0) * 2) % 16 == 0
        and C.stride() == B.stride() and angle.stride(1) == 1 and (angle.stride(0) * 2) % 16 == 0
    )


_F32_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _fp32(w: torch.Tensor) -> torch.Tensor:
    """fp32 copy of a small constant parameter, cached by storage (D may be a bf16 parameter)."""
    if w.dtype == torch.float32:
        return w
    key = w.data_ptr()
    e = _F32_CACHE.get(key)
    if e is None:
        e = (w, w.detach().float().contiguous())
        _F32_CACHE[key] = e
    return e[1]


def refresh_f32_cache() -> None:
    """Recompute the cached fp32 copies after an in-place weight reload
    (their addresses are baked into captured CUDA graphs)."""
    with torch.inference_mode():
        for src, dst in _F32_CACHE.values():
            dst.copy_(src.detach().float())


def _as_f32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 else t.float()


def mamba3_step_cuda(ssm_pool, k_pool, v_pool, angle_pool, A, B, C, Dp, x, dt, trap, xproj, zproj, outproj, z,
                     bias_q, bias_k, angle_proj, slots, y, ctas_per_sm: int = 4) -> None:
    """B, C: (N, R, S) views (any batch stride); angle_proj: (N, 32) view; everything else contiguous."""
    Dp = _fp32(Dp)
    A, dt, trap = _as_f32(A), _as_f32(dt), _as_f32(trap)
    _load().step(ssm_pool, k_pool, v_pool, angle_pool, A, B, C, Dp, x, dt, trap, xproj, zproj, outproj, z,
                 bias_q, bias_k, angle_proj, slots, y, ctas_per_sm, pdl=pdl_enabled())


def mamba3_step_cuda_raw(ssm_pool, k_pool, v_pool, angle_pool, zxdt, bc, dt_bias,
                         a_floor, wb, wc, eps, Dp, xproj, zproj, outproj, bias_q,
                         bias_k, slots, y, ctas_per_sm: int = 4) -> None:
    """Step with the decode preamble fused (``_decode_preamble_bc_kernel``):
    ``zxdt`` is the raw head-interleaved in_proj output (N, H*(2D+3)) and
    ``bc`` the raw in_proj_dyn output (N, 2*R*S + angles), both row views.
    ``dt_bias`` and the B/C norm gains ``wb``/``wc`` (``+ 1`` already applied
    when zero-centered) must be fp32 tensors that outlive CUDA graphs."""
    R, S = 4, 128
    B = bc[:, : R * S].unflatten(1, (R, S))
    C = bc[:, R * S : 2 * R * S].unflatten(1, (R, S))
    angle_proj = bc[:, 2 * R * S :]
    Dp = _fp32(Dp)
    _load().step(ssm_pool, k_pool, v_pool, angle_pool, Dp, B, C, Dp, zxdt, Dp, Dp,
                 xproj, zproj, outproj, zxdt, bias_q, bias_k, angle_proj, slots, y,
                 ctas_per_sm, zxdt, dt_bias, float(a_floor), wb, wc, float(eps),
                 pdl_enabled())


# -- Speculative verify (csrc/mamba3_verify.cu) -------------------------------


@lru_cache(maxsize=1)
def _load_verify():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(
        f"olala_mamba3_verify_sm{sm_arch()}",
        "mamba3_verify.cu",
        ["-O3", "-std=c++20", gencode()],
    )


@lru_cache(maxsize=1)
def verify_kernel_enabled() -> bool:
    """OLALA_MAMBA3_VERIFY=cuda (default): the fused multi-position verify
    kernel; any other value keeps the per-position chain of step launches."""
    if os.environ.get("OLALA_MAMBA3_VERIFY", "cuda") != "cuda" or not has_bf16_mma():
        return False
    try:
        _load_verify()
    except Exception as e:  # noqa: BLE001 - any build/load failure falls back
        warn_slow_path(
            "Fused Mamba-3 verify kernel disabled",
            f"JIT build of mamba3_verify.cu failed: {e}",
            "per-position verify chain: ~20 launches per draft position and layer",
            "the JIT needs an nvcc of torch's CUDA major; set OLALA_JIT_CUDA_HOME",
        )
        return False
    return True


def verify_supported(
    ssm_pool, k_pool, v_pool, angle_pool, num_heads, head_dim, d_state, rank, num_angles
) -> bool:
    """The kernel's compiled shapes (Olala 7A1B at TP=1) and an fp32 state."""
    return (
        (num_heads, head_dim, d_state, rank, num_angles) == (48, 64, 128, 4, 32)
        and ssm_pool.dtype == torch.float32
        and k_pool.dtype == torch.bfloat16
        and v_pool.dtype == torch.bfloat16
        and angle_pool.dtype == torch.float32
    )


def mamba3_verify_cuda(
    ssm_pool, k_pool, v_pool, angle_pool, A, B, C, Dp, x, dt, trap, xproj, zproj,
    outproj, z, bias_q, bias_k, angle_proj, qsl, cols, num_accepted, y,
    ctas_per_sm: int = 4,
) -> None:  # fmt: skip
    """All draft positions of every request in one launch (column-slot protocol).

    ``qsl``: (lanes + 1,) int32 decode-token offsets; ``cols``: (lanes, 1 + k)
    int32 state slot columns (unit inner stride); ``num_accepted``: (lanes,)
    int32, the current state is column ``max(num_accepted - 1, 0)``. B, C:
    (N, R, S) views; angle_proj: (N, 32) view; A / dt / trap: (N, H) fp32;
    x, z, y: (N, H, D)."""
    _load_verify().verify(
        ssm_pool, k_pool, v_pool, angle_pool, _as_f32(A), B, C, _fp32(Dp), x,
        _as_f32(dt), _as_f32(trap), xproj, zproj, outproj, z, bias_q, bias_k,
        angle_proj, qsl, cols, num_accepted, y, ctas_per_sm, pdl_enabled(),
    )  # fmt: skip


# -- ReplaySSM (csrc/mamba3_replay_step.cu) ----------------------------------


@lru_cache(maxsize=1)
def _load_replay():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    flags = ["-O3", "-std=c++20", gencode()]
    if gpu_family() is GpuFamily.SM100:
        # sm_100a / sm_103a: the flush's stochastic rounding uses the hardware cvt.rs
        flags = ["-O3", "-std=c++20", gencode(arch_specific=True), "-DREPLAY_HW_SR"]
    return load_extension(f"olala_mamba3_replay_step_sm{sm_arch()}", "mamba3_replay_step.cu", flags)


# Resident CTAs per SM the persistent replay kernel sizes its grid for: 5 fit
# its 96-register / ~35 KB budget at L = 8 (csrc/mamba3_replay_step.cu).
_REPLAY_CTAS_PER_SM = 5


def replay_step_supported(num_heads_local, head_dim, d_state, rank, num_angles, ngroups) -> bool:
    return (
        head_dim == 64 and d_state == 128 and rank == 4 and num_angles == 32
        and ngroups == 1 and num_heads_local > 0
        and has_fp8_cvt()
    )


def mamba3_replay_step(pools, A, B, C, Dp, x, dt, trap, xproj, zproj, outproj, z,
                       bias_q, bias_k, angle_proj, slots, y) -> None:
    """ReplaySSM decode step. ``pools`` is the full Olala Mamba3 state tuple
    (see ``state.mamba3_state_shape``); B, C: (N, R, S) pre-bias/rotation
    views (any batch stride); angle_proj: (N, 32) view."""
    angle, anchor, kp, vp, xr, br, scr, cnt = pools[:8]
    rd, rd_scale = (pools[8], pools[9]) if len(pools) > 8 else (None, None)
    Dp = _fp32(Dp)
    A, dt, trap = _as_f32(A), _as_f32(dt), _as_f32(trap)
    _load_replay().step(angle, anchor, rd, rd_scale, kp, vp, xr, br, scr, cnt,
                        A, B, C, Dp, x, dt, trap, xproj, zproj, outproj, z,
                        bias_q, bias_k, angle_proj, slots, y, _REPLAY_CTAS_PER_SM)

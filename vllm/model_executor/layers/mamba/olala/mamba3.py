# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala Mamba3 MIMO mixer.

Native port of ``OlalaMamba3MimoFast`` from ``modeling_olala.py``. The four
temporal states (angle, ssm, k, v) live in vLLM's paged mamba state pool;
per-request slots arrive as ``Mamba3AttentionMetadata.state_indices_tensor``
from the MAMBA3 backend.

Prefill packs the batch into ``(B=1, S=total_tokens)`` and calls the TileLang
varlen kernel ``ops.mamba3.mimo.mamba3_mimo`` with ``cu_seqlens``. Decode runs
the persistent CUDA step (``mamba3_step_cuda``) or the CuteDSL
``ops.mamba3.step_cute.mamba3_step_fn`` over the whole decode-token batch,
indexing the state pools in place.
"""

from __future__ import annotations

import dataclasses
import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.olala.decode_gemv import (
    cat_rows_shared,
    decode_gemv,
)
from vllm.model_executor.layers.mamba.olala.volta_tc import dense_linear
from vllm.model_executor.layers.mamba.olala.w8a16_layers import (
    cat_int8_rows_shared,
    int8_apply,
    int8_linear_weight,
    is_int8_quant_method,
    mixer_int8_rows,
    mixer_quant_config,
)
from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
    Segment,
    int8_decode_linear,
    int8_decode_max_m,
    int8_prefill_mm,
)
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.model_executor.layers.mamba.olala.fallback import warn_slow_path
from vllm.model_executor.layers.mamba.olala.geo_linear import (
    GEO_LINEAR_MAX_TOKENS,
    geo_linear,
)
from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
    cuda_step_enabled,
    cuda_step_supported,
    mamba3_replay_step,
    mamba3_step_cuda,
    mamba3_step_cuda_raw,
    mamba3_verify_cuda,
    refresh_f32_cache,
    replay_step_supported,
    verify_kernel_enabled,
    verify_supported,
)
from vllm.model_executor.layers.mamba.olala.replay import replay_reset
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.olala.norm import OlalaNorm
from vllm.model_executor.layers.mamba.olala.state import (
    folded_token_shift_layout,
    mamba3_state_dtype,
    mamba3_state_shape,
    replay_config,
    spec_light_cols,
    ts_fold_active,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mamba3_attn import Mamba3AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# CuteDSL decode-step launch config. ``tile_D`` covers the full headdim, so
# exactly one CTA owns each (batch, head) row — the precondition for letting
# the kernel store the new B/x key/value states itself (``update_kv_state``).
_STEP_TILE_D = 64
# OLALA_RAW_STEP=1 folds the decode preamble into the CUDA step. Off by default:
# with the fp32 SSM state the in-kernel preamble lengthens each CTA's serial
# critical path more than the saved launch (-5..8% decode throughput on B300).
_RAW_STEP = os.environ.get("OLALA_RAW_STEP", "0") == "1"
_STEP_NUM_WARPS = 4

# Fusing bias+rotary into the step kernel avoids materializing the per-head
# rotated B/C, but its smem rotation stages serialize against the cp.async
# pipeline once occupancy is high. Measured on GH200 under graph capture:
# ~1 us/layer faster at batch 1, ~26 us/layer slower at batch 256. Dispatch on
# batch size; each cudagraph capture size takes one consistent branch.
_FUSE_ROTARY_MAX_BATCH = 64


def _fuse_rotary_max_batch() -> int:
    # On Blackwell the CUDA step (rotary fused, csrc/mamba3_step.cu) beats the
    # CuteDSL step + separate rotary at every batch: B300, fp32 state, batch
    # 256: 168 us vs 172 + 13.5 us per layer.
    if gpu_family() in (GpuFamily.SM100, GpuFamily.SM120) and cuda_step_enabled():
        return 1 << 30
    return _FUSE_ROTARY_MAX_BATCH

# Mamba3 kernels (vendored under ops/mamba3), imported lazily: the prefill
# kernel needs tilelang and the step kernel the CuTe DSL.
_mamba3_mimo = None
_mamba3_mimo_grouped = None
_mamba3_mimo_hybrid_pools = None
_mamba3_step_fn = None
_apply_rotary_qk_inference_fwd = None


_PREFILL_INPROJ_CAT = os.environ.get("OLALA_PREFILL_INPROJ_CAT", "1") != "0"


def _xz_strided_ok(dev) -> bool:
    from vllm.model_executor.layers.mamba.ops.mamba3.mimo_ssd import xz_strided_ok

    return xz_strided_ok(dev)


def _is_pow2(n: int) -> bool:
    return n & (n - 1) == 0


def _lazy_import_kernels() -> None:
    global _mamba3_mimo, _mamba3_mimo_grouped, _mamba3_step_fn, _mamba3_mimo_hybrid_pools
    if os.environ.get("OLALA_SYNC_DEBUG") == "1" and not getattr(torch, "_olala_sync_dbg", False):  # debug: report host syncs (warn)
        torch._olala_sync_dbg = True
        torch.cuda.set_sync_debug_mode("warn")
    global _apply_rotary_qk_inference_fwd
    if _mamba3_mimo is not None:
        return
    from vllm.model_executor.layers.mamba.ops.mamba3.mimo import (
        mamba3_mimo,
        mamba3_mimo_varlen_grouped,
    )
    from vllm.model_executor.layers.mamba.ops.mamba3.rotary_step import (
        apply_rotary_qk_inference_fwd,
    )
    from vllm.model_executor.layers.mamba.ops.mamba3.step_cute import (
        mamba3_step_fn,
    )

    _mamba3_mimo = mamba3_mimo
    _mamba3_step_fn = mamba3_step_fn
    _apply_rotary_qk_inference_fwd = apply_rotary_qk_inference_fwd
    # Group-parallel exact prefill: splits long sequences into virtual groups
    # for GPU occupancy — same math, ~1.4x on long prompts. Disable with
    # OLALA_GROUPED_PREFILL=0.
    if os.environ.get("OLALA_GROUPED_PREFILL", "1") != "0":
        _mamba3_mimo_grouped = mamba3_mimo_varlen_grouped
    # Chunk-parallel exact prefill (Triton chunk-state phases + one TileLang pass over
    # 128-token chunks): every chunk of every sequence runs in parallel. OLALA_M3_PREFILL=tilelang
    # restores the serial-per-sequence path.
    if os.environ.get("OLALA_M3_PREFILL", "hybrid") == "hybrid":
        from vllm.model_executor.layers.mamba.ops.mamba3.mimo_ssd import mamba3_mimo_hybrid_pools

        _mamba3_mimo_hybrid_pools = mamba3_mimo_hybrid_pools


@triton.jit
def _decode_preamble_kernel(
    zxdt_ptr,  # (N, H*(2D+3)) — in_proj output, head-interleaved
    x_ptr,  # out (N, H, D) contiguous, input dtype
    z_ptr,  # out (N, H, D) contiguous, input dtype
    a_ptr,  # out (N, H) fp32: clamp(-softplus(A), max=-a_floor)
    dt_ptr,  # out (N, H) fp32: softplus(dt + dt_bias[h])
    trap_ptr,  # out (N, H) fp32: sigmoid(trap)
    dt_bias_ptr,  # (H,)
    a_floor,
    H,
    stride_n,
    D: tl.constexpr,
):
    """One program per (token, head).

    Replaces the x/z ``.contiguous()`` copies and the A/dt/trap
    softplus/clamp/sigmoid chain — ~14 tiny kernels per layer per step, which
    is pure launch overhead at decode shapes.
    """
    pid = tl.program_id(0)
    h = pid % H
    base = zxdt_ptr + (pid // H) * stride_n + h * (2 * D + 3)
    offs = tl.arange(0, D)  # D is a power of 2 (guarded at the call site)
    tl.store(z_ptr + pid * D + offs, tl.load(base + offs))
    tl.store(x_ptr + pid * D + offs, tl.load(base + D + offs))
    dt = tl.load(base + 2 * D).to(tl.float32)
    a = tl.load(base + 2 * D + 1).to(tl.float32)
    trap = tl.load(base + 2 * D + 2).to(tl.float32)
    # F.softplus, including its overflow threshold (identity for x > 20).
    sp_a = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    v = dt + tl.load(dt_bias_ptr + h).to(tl.float32)
    sp_dt = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
    tl.store(a_ptr + pid, tl.minimum(-sp_a, -a_floor))
    tl.store(dt_ptr + pid, sp_dt)
    tl.store(trap_ptr + pid, 1.0 / (1.0 + tl.exp(-trap)))


@triton.jit
def _bc_norm_kernel(
    bc_ptr,  # (N, 2*R*S + n_angles) — in_proj_dyn output (ngroups == 1)
    out_ptr,  # (N, 2, R, S) contiguous: [:,0]=B_norm(B), [:,1]=C_norm(C)
    wb_ptr,  # (S,) B_norm gain
    wc_ptr,  # (S,) C_norm gain
    eps,
    off_c,  # offset of the C block inside a bc row (= R*S)
    R,
    stride_n,
    ZERO_CENTERED: tl.constexpr,
    S: tl.constexpr,
):
    """One program per (token, B/C, mimo-row): fused RMS norm + gain."""
    pid = tl.program_id(0)  # n*(2R) + which*R + r
    j = pid % (2 * R)
    base = bc_ptr + (pid // (2 * R)) * stride_n + (j % R) * S + (j // R) * off_c
    offs = tl.arange(0, S)  # S is a power of 2 (guarded at the call site)
    v = tl.load(base + offs).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(v * v) / S + eps)
    wb = tl.load(wb_ptr + offs).to(tl.float32)
    wc = tl.load(wc_ptr + offs).to(tl.float32)
    w = tl.where(j >= R, wc, wb)
    if ZERO_CENTERED:
        w = w + 1.0
    tl.store(out_ptr + pid * S + offs, (v * inv * w).to(out_ptr.dtype.element_ty))


@triton.jit
def _decode_preamble_bc_kernel(
    zxdt_ptr, x_ptr, z_ptr, a_ptr, dt_ptr, trap_ptr, dt_bias_ptr, a_floor, H,
    stride_zxdt,
    bc_ptr, bn_cn_ptr, wb_ptr, wc_ptr, eps, off_c, R, stride_bc,
    n_pre,  # number of preamble programs (= tokens * H); the rest do B/C norm
    D: tl.constexpr,
    S: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
):
    """_decode_preamble_kernel and _bc_norm_kernel in one launch: programs
    [0, n_pre) run the per-(token, head) preamble, the rest the per-(token,
    B/C, mimo-row) norm. Same math as the two kernels."""
    pid = tl.program_id(0)
    if pid < n_pre:
        h = pid % H
        base = zxdt_ptr + (pid // H) * stride_zxdt + h * (2 * D + 3)
        offs = tl.arange(0, D)
        tl.store(z_ptr + pid * D + offs, tl.load(base + offs))
        tl.store(x_ptr + pid * D + offs, tl.load(base + D + offs))
        dt = tl.load(base + 2 * D).to(tl.float32)
        a = tl.load(base + 2 * D + 1).to(tl.float32)
        trap = tl.load(base + 2 * D + 2).to(tl.float32)
        sp_a = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        v = dt + tl.load(dt_bias_ptr + h).to(tl.float32)
        sp_dt = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        tl.store(a_ptr + pid, tl.minimum(-sp_a, -a_floor))
        tl.store(dt_ptr + pid, sp_dt)
        tl.store(trap_ptr + pid, 1.0 / (1.0 + tl.exp(-trap)))
    else:
        q = pid - n_pre
        j = q % (2 * R)
        base_bc = bc_ptr + (q // (2 * R)) * stride_bc + (j % R) * S + (j // R) * off_c
        offs_s = tl.arange(0, S)
        vals = tl.load(base_bc + offs_s).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(vals * vals) / S + eps)
        wb = tl.load(wb_ptr + offs_s).to(tl.float32)
        wc = tl.load(wc_ptr + offs_s).to(tl.float32)
        wsel = tl.where(j >= R, wc, wb)
        if ZERO_CENTERED:
            wsel = wsel + 1.0
        tl.store(bn_cn_ptr + q * S + offs_s, (vals * inv * wsel).to(bn_cn_ptr.dtype.element_ty))


@triton.jit
def _prefill_prologue_kernel(
    zxdt_ptr, stride_zxdt, bc_ptr, stride_bc,
    x_ptr, z_ptr,  # out (T, H, D) contiguous
    dt_ptr, adt_ptr, trap_ptr,  # out (H, T): fp32 softplus(dt + dt_bias), fp32 A*DT, raw trap
    q_ptr, k_ptr,  # out (T, R, S): C_norm(C), B_norm(B)
    ang_ptr,  # out (T, NA) fp32
    dt_bias_ptr, a_floor, wb_ptr, wc_ptr, eps,
    T, H, R, n_pre,
    D: tl.constexpr, S: tl.constexpr, NA: tl.constexpr, BT: tl.constexpr, ZERO_CENTERED: tl.constexpr,
    R2: tl.constexpr, WRITE_XZ: tl.constexpr = True,
):
    """Prefill counterpart of _decode_preamble_bc_kernel, in one launch: programs [0, n_pre) handle a
    (BT-token tile, head) of the in_proj output (z/x copies and the dt / A / trap chain, written
    head-major), the others one token's B / C RMS norms and its rotary angles."""
    pid = tl.program_id(0)
    if pid < n_pre:
        h = pid % H
        t = (pid // H) * BT + tl.arange(0, BT)
        tm = t < T
        d = tl.arange(0, D)
        if WRITE_XZ:  # else the scan kernels read x / z straight from the in_proj output
            zrow = zxdt_ptr + t[:, None].to(tl.int64) * stride_zxdt + h * (2 * D + 3)
            tl.store(z_ptr + (t[:, None].to(tl.int64) * H + h) * D + d[None, :], tl.load(zrow + d[None, :], mask=tm[:, None]), mask=tm[:, None])
            tl.store(x_ptr + (t[:, None].to(tl.int64) * H + h) * D + d[None, :], tl.load(zrow + D + d[None, :], mask=tm[:, None]), mask=tm[:, None])
        b1 = zxdt_ptr + t.to(tl.int64) * stride_zxdt + h * (2 * D + 3) + 2 * D
        dt = tl.load(b1, mask=tm, other=0.0).to(tl.float32)
        a = tl.load(b1 + 1, mask=tm, other=0.0).to(tl.float32)
        trap = tl.load(b1 + 2, mask=tm, other=0.0)
        sp_a = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        v = dt + tl.load(dt_bias_ptr + h).to(tl.float32)
        sp_dt = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        tl.store(dt_ptr + h * T + t, sp_dt, mask=tm)
        tl.store(adt_ptr + h * T + t, tl.minimum(-sp_a, -a_floor) * sp_dt, mask=tm)
        tl.store(trap_ptr + h * T + t, trap, mask=tm)
    else:
        # one token per program: its R B rows and R C rows (the bc row is [B (R x S) | C (R x S) | angles])
        tok = (pid - n_pre).to(tl.int64)
        j = tl.arange(0, R2)
        is_c = j >= R2 // 2
        offs = tl.arange(0, S)
        vals = tl.load(bc_ptr + tok * stride_bc + j[:, None] * S + offs[None, :]).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(vals * vals, axis=1) / S + eps)
        # same roundings as OlalaRMSNorm: rms_norm output in the activation dtype, gain (1 + w)
        # in the parameter dtype, product rounded once more
        w = tl.where(is_c[:, None], tl.load(wc_ptr + offs)[None, :], tl.load(wb_ptr + offs)[None, :])
        if ZERO_CENTERED:
            w = (w.to(tl.float32) + 1.0).to(wb_ptr.dtype.element_ty)
        y = ((vals * inv[:, None]).to(q_ptr.dtype.element_ty).to(tl.float32) * w.to(tl.float32)).to(q_ptr.dtype.element_ty)
        r = j % (R2 // 2)
        tl.store(k_ptr + (tok * (R2 // 2) + r[:, None]) * S + offs[None, :], y, mask=~is_c[:, None])
        tl.store(q_ptr + (tok * (R2 // 2) + r[:, None]) * S + offs[None, :], y, mask=is_c[:, None])
        na = tl.arange(0, NA)
        tl.store(ang_ptr + tok * NA + na, tl.load(bc_ptr + tok * stride_bc + R2 * S + na).to(tl.float32))


@triton.jit
def _prefill_prologue_split_kernel(
    cat_ptr, stride_cat, bc_off,  # (T, W) prefill in_proj output: [z | x | dt | A | trap | B C angles]
    dt_ptr, adt_ptr, trap_ptr,  # out (H, T): fp32 softplus(dt + dt_bias), fp32 A*DT, raw trap
    q_ptr, k_ptr,  # out (T, R, S): C_norm(C), B_norm(B)
    ang_ptr,  # out (T, NA) fp32
    dt_bias_ptr, a_floor, wb_ptr, wc_ptr, eps,
    T, H, R, n_pre,
    D: tl.constexpr, S: tl.constexpr, NA: tl.constexpr, BT: tl.constexpr, HP: tl.constexpr,
    ZERO_CENTERED: tl.constexpr, R2: tl.constexpr,
):
    """_prefill_prologue_kernel for the de-interleaved prefill in_proj (x / z are read in place by the
    scan kernels, so there is nothing to copy): programs [0, n_pre) take a BT-token tile of every head's
    dt / A / trap, the others one token's B / C RMS norms and rotary angles."""
    pid = tl.program_id(0)
    if pid < n_pre:
        t = pid * BT + tl.arange(0, BT)
        h = tl.arange(0, HP)
        m = (t < T)[:, None] & (h < H)[None, :]
        row = cat_ptr + t[:, None].to(tl.int64) * stride_cat + 2 * H * D + h[None, :]
        dt = tl.load(row, mask=m, other=0.0).to(tl.float32)
        a = tl.load(row + H, mask=m, other=0.0).to(tl.float32)
        trap = tl.load(row + 2 * H, mask=m, other=0.0)
        sp_a = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        v = dt + tl.load(dt_bias_ptr + h, mask=h < H, other=0.0).to(tl.float32)[None, :]
        sp_dt = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        o = h[None, :] * T + t[:, None]
        tl.store(dt_ptr + o, sp_dt, mask=m)
        tl.store(adt_ptr + o, tl.minimum(-sp_a, -a_floor) * sp_dt, mask=m)
        tl.store(trap_ptr + o, trap, mask=m)
    else:
        tok = (pid - n_pre).to(tl.int64)
        bc = cat_ptr + tok * stride_cat + bc_off
        j = tl.arange(0, R2)
        is_c = j >= R2 // 2
        offs = tl.arange(0, S)
        vals = tl.load(bc + j[:, None] * S + offs[None, :]).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(vals * vals, axis=1) / S + eps)
        w = tl.where(is_c[:, None], tl.load(wc_ptr + offs)[None, :], tl.load(wb_ptr + offs)[None, :])
        if ZERO_CENTERED:
            w = (w.to(tl.float32) + 1.0).to(wb_ptr.dtype.element_ty)
        y = ((vals * inv[:, None]).to(q_ptr.dtype.element_ty).to(tl.float32) * w.to(tl.float32)).to(q_ptr.dtype.element_ty)
        r = j % (R2 // 2)
        tl.store(k_ptr + (tok * (R2 // 2) + r[:, None]) * S + offs[None, :], y, mask=~is_c[:, None])
        tl.store(q_ptr + (tok * (R2 // 2) + r[:, None]) * S + offs[None, :], y, mask=is_c[:, None])
        na = tl.arange(0, NA)
        tl.store(ang_ptr + tok * NA + na, tl.load(bc + R2 * S + na).to(tl.float32))


class OlalaMamba3Mixer(PluggableLayer, MambaBase):
    """Mamba3 MIMO mixer over vLLM's paged recurrent state.

    State layout per request slot:
        angle : (nheads, num_rope_angles)    float32
        ssm   : (nheads, headdim, d_state)   ssm cache dtype
        k     : (mimo_dim, nheads, d_state)  state dtype
        v     : (nheads, headdim)            state dtype
    """

    # Populated by the model runner via MambaSpec after init.
    kv_cache: tuple[torch.Tensor, ...]

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        tp = get_tensor_model_parallel_world_size()
        self.tp_size = tp
        # Token-shift fold (state.py): the page also carries one V layer's
        # (k_last, v_last, A_k_last, A_v_last) after the Mamba3 tensors.
        self.ts_fold = ts_fold_active(config, vllm_config.parallel_config.pipeline_parallel_size)
        # Speculative light columns (state.py: spec_light_cols): 1 + k columns
        # of small pieces in one page, 0 = one page per column.
        self.light_cols = spec_light_cols(vllm_config)

        self.d_model = config.hidden_size
        self.d_inner = 2 * self.d_model
        self.mimo_dim = config.mamba_mimo_dim
        self.d_state = config.mamba_d_state
        self.headdim = config.mamba_headdim
        self.ngroups = config.mamba_ngroups
        self.nheads = self.d_inner // self.headdim
        self.nheads_local = divide(self.nheads, tp)
        self.rope_fraction = 0.5
        self.rotary_dim_divisor = 4
        self.A_floor = 1e-4
        split = int(self.d_state * self.rope_fraction)
        if split % 2:
            split -= 1
        self.num_rope_angles = split // 2
        self.chunk_size = 64 // self.mimo_dim

        # in_proj is head-interleaved
        # ``[h0: z(headdim), x(headdim), dt, A, trap | h1: ... ]``, total width
        # ``nheads * (2*headdim + 3)``. Splitting the output axis by TP
        # therefore yields a clean per-head shard.
        int8_qc = mixer_int8_rows(vllm_config.quant_config, f"{prefix}.in_proj")
        self._int8_in_proj = int8_qc is not None
        self._int8_st: tuple | None = None
        self.in_proj = ColumnParallelLinear(
            input_size=self.d_model,
            output_size=self.d_inner * 2 + 3 * self.nheads,
            bias=False,
            gather_output=False,
            quant_config=int8_qc,
            prefix=f"{prefix}.in_proj",
        )
        # in_proj_dyn produces B/C (per group, shared across heads) and the
        # rope angles (shared across heads), so it is replicated.
        self.in_proj_dyn = ReplicatedLinear(
            input_size=self.d_model,
            output_size=(
                2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles
            ),
            bias=False,
            quant_config=mixer_quant_config(vllm_config.quant_config),
            prefix=f"{prefix}.in_proj_dyn",
        )

        self._in_proj_cat_w: torch.Tensor | None = None
        # prefill in_proj + in_proj_dyn as one GEMM with the head-interleaved in_proj rows regrouped as
        # [z | x | dt | A | trap] (16-byte aligned x / z rows the scan kernels read in place); a weight copy
        self._pf_in_w: torch.Tensor | None = None
        self._raw_consts: tuple[torch.Tensor, ...] | None = None
        self._pre_geo: nn.Module | None = None
        self._in_proj_cat_ok = (
            tp == 1 and self.in_proj.bias is None and self.in_proj_dyn.bias is None
            and not self._int8_in_proj
            and not is_int8_quant_method(self.in_proj_dyn)
        )

        head_shard = {"weight_loader": sharded_weight_loader(0)}
        self.B_bias = nn.Parameter(
            torch.ones(
                self.nheads_local, self.mimo_dim, self.d_state, dtype=torch.float32
            )
        )
        self.C_bias = nn.Parameter(
            torch.ones(
                self.nheads_local, self.mimo_dim, self.d_state, dtype=torch.float32
            )
        )
        set_weight_attrs(self.B_bias, head_shard)
        set_weight_attrs(self.C_bias, head_shard)

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        self.B_norm = OlalaNorm(self.d_state, eps=eps, zero_centered=zc)
        self.C_norm = OlalaNorm(self.d_state, eps=eps, zero_centered=zc)

        self.in_proj_mimo_x = nn.Parameter(
            torch.full(
                (self.nheads_local, self.mimo_dim, self.headdim),
                1.0 / self.mimo_dim,
                dtype=torch.float32,
            )
        )
        self.in_proj_mimo_z = nn.Parameter(
            torch.ones(
                self.nheads_local, self.mimo_dim, self.headdim, dtype=torch.float32
            )
        )
        self.out_proj_mimo = nn.Parameter(
            torch.full(
                (self.nheads_local, self.mimo_dim, self.headdim),
                1.0 / self.mimo_dim,
                dtype=torch.float32,
            )
        )
        set_weight_attrs(self.in_proj_mimo_x, head_shard)
        set_weight_attrs(self.in_proj_mimo_z, head_shard)
        set_weight_attrs(self.out_proj_mimo, head_shard)

        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(self.nheads_local) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        set_weight_attrs(self.dt_bias, head_shard)

        self.D = nn.Parameter(torch.ones(self.nheads_local))
        set_weight_attrs(self.D, head_shard)

        self._prefill_const_w: tuple[torch.Tensor, ...] | None = None
        self._decode_const_w: tuple[torch.Tensor, ...] | None = None

        self._postgate_norm = getattr(config, "mamba3_postgate_norm", False)
        pow2 = lambda n: n > 0 and (n & (n - 1)) == 0  # noqa: E731
        self._hybrid_ok = (
            self.ngroups == 1
            and self.in_proj.bias is None
            and self.in_proj_dyn.bias is None
            and pow2(self.headdim)
            and pow2(self.d_state)
            and pow2(2 * self.mimo_dim)
            and self.num_rope_angles == self.d_state // self.rotary_dim_divisor
        )
        if self._postgate_norm:
            # Per-rank local-shard norm: each rank normalizes only its own
            # d_inner slice, which differs numerically from a global RMSNorm
            # but matches how the mixer output is split.
            self.output_norm = OlalaNorm(self.d_inner // tp, eps=eps, zero_centered=zc)
            set_weight_attrs(self.output_norm.norm.weight, head_shard)

        self._replay = replay_config()
        if self._replay is not None:
            if vllm_config.speculative_config is not None:
                raise ValueError(
                    "OLALA_REPLAY_SSM does not support speculative decoding yet "
                    "(the verify step reads the full state); unset it."
                )
            if not (
                replay_step_supported(
                    self.nheads_local, self.headdim, self.d_state, self.mimo_dim,
                    self.num_rope_angles, self.ngroups,
                )
                and self._hybrid_ok
                and not self._postgate_norm
            ):
                raise ValueError(
                    "OLALA_REPLAY_SSM needs sm_89+ and the Olala 7A1B Mamba3 shapes "
                    f"(headdim 64, d_state 128, mimo 4, 32 angles, 1 group; got "
                    f"{self.headdim}/{self.d_state}/{self.mimo_dim}/"
                    f"{self.num_rope_angles}/{self.ngroups})."
                )

        # Register in the static forward-context table so the runner can find
        # this layer's metadata by prefix.
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    # -- MambaBase ----------------------------------------------------------
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA3

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = mamba3_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
            self.light_cols,
        )
        # The CuteDSL step kernel indexes the pools in place, so their dtypes
        # are a hard precondition rather than something we can cast around.
        _, ssm_dtype, k_dtype, v_dtype = dtypes[:4]
        if self.ts_fold:
            dtypes = tuple(dtypes) + self._ts_fold_layout()[1]
        if ssm_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"Olala mamba3 supports an fp32 or bf16 SSM state, got "
                f"{ssm_dtype}. Set --mamba-ssm-cache-dtype to float32 or auto."
            )
        if k_dtype != torch.bfloat16 or v_dtype != torch.bfloat16:
            raise ValueError(
                f"Olala mamba3 requires a bf16 key/value state, got "
                f"{k_dtype}/{v_dtype}. Leave --mamba-cache-dtype at auto with "
                f"a bf16 model."
            )
        return dtypes

    def _own_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return mamba3_state_shape(
            tp_world_size=self.tp_size,
            num_heads=self.nheads,
            head_dim=self.headdim,
            d_state=self.d_state,
            mimo_dim=self.mimo_dim,
            num_rope_angles=self.num_rope_angles,
            light_cols=self.light_cols,
        )

    def _ts_fold_layout(self):
        return folded_token_shift_layout(
            self.config, self.tp_size, self.model_config.dtype, self.cache_config.mamba_cache_dtype,
            self.light_cols,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        spec = super().get_kv_cache_spec(vllm_config)
        if self.light_cols and spec is not None:
            # every speculative column lives inside the one page
            spec = dataclasses.replace(spec, num_speculative_blocks=0)
        return spec

    def _base_pools(self) -> tuple[torch.Tensor, ...]:
        """(angle, ssm, k, v) of the committed state: column 0 with light columns."""
        own = tuple(self.kv_cache[: self.num_own_state_tensors])
        if self.light_cols:
            angle, ssm, k, v = own[:4]
            return (angle[:, 0], ssm, k[:, 0], v[:, 0])
        return own[:4]

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        shapes = self._own_state_shape()
        if self.ts_fold:
            shapes = tuple(shapes) + self._ts_fold_layout()[0]
        return shapes

    @property
    def num_own_state_tensors(self) -> int:
        """Mamba3 tensors at the head of ``kv_cache`` (the folded token-shift
        buffers, if any, follow)."""
        return len(self._own_state_shape())

    def folded_token_shift_pools(self) -> tuple[torch.Tensor, ...]:
        """The (k_last, v_last, A_k_last, A_v_last) views of this page (fold only)."""
        assert self.ts_fold
        return tuple(self.kv_cache[self.num_own_state_tensors :])

    # -- Forward ------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(num_tokens, hidden_size)`` in, ``(num_tokens, d_inner/tp)`` out.

        The state-dependent work runs outside the compiled graph as the
        ``vllm::olala_mamba3`` custom op, mirroring ``mamba_mixer2``.
        """
        out = torch.empty(
            hidden_states.shape[0],
            self.nheads_local * self.headdim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        torch.ops.vllm.olala_mamba3(
            hidden_states, out, _encode_layer_name(self.prefix)
        )
        return out

    def forward_geo(
        self, residual: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h = self._pre_geo(residual, g)`` (the previous block's geodesic
        update) and the mixer of ``h``; at decode sizes the update runs inside
        the in_proj GEMM (geo_linear.py). Returns ``(h, mixer output)``."""
        h = torch.empty_like(residual)
        out = torch.empty(
            residual.shape[0],
            self.nheads_local * self.headdim,
            device=residual.device,
            dtype=residual.dtype,
        )
        torch.ops.vllm.olala_mamba3(
            residual, out, _encode_layer_name(self.prefix), g, h
        )
        return h, out

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        out: torch.Tensor,
        g: torch.Tensor | None = None,
        h_out: torch.Tensor | None = None,
    ) -> None:
        _lazy_import_kernels()

        attn_metadata = get_forward_context().attn_metadata
        if g is not None:
            md = attn_metadata[self.prefix] if attn_metadata is not None else None
            n = hidden_states.shape[0]
            if (
                md is not None
                and md.num_prefills == 0
                and md.spec is None
                and self._replay is None
                and n <= GEO_LINEAR_MAX_TOKENS
                and self._in_proj_cat_ok
                and self._raw_step_ok(self._base_pools())
            ):
                cat = geo_linear(
                    hidden_states, g, self._pre_geo, self._in_proj_cat_weight(),
                    h_out, hidden_states.dtype,
                )
                n1 = self.in_proj.weight.shape[0]
                ndt = md.num_decode_tokens
                self._decode_raw_step(
                    None, out[:ndt], md.state_indices_tensor[: md.num_decodes],
                    self._base_pools(),
                    zxdt_bc=(cat[:ndt, :n1], cat[:ndt, n1:]),
                )
                return
            self._pre_geo(hidden_states, g, out=h_out)
            hidden_states = h_out

        # vLLM >= 0.33 kernel warmups (FlashInfer autotune) build metadata for
        # attention groups only: run those like the profiling dry run.
        if attn_metadata is None or self.prefix not in attn_metadata:
            self._dry_run_prefill(hidden_states, out)
            return
        md: Mamba3AttentionMetadata = attn_metadata[self.prefix]

        all_pools = tuple(self.kv_cache[: self.num_own_state_tensors])
        # (angle, ssm, k, v); ReplaySSM tensors follow. With light columns the
        # verify kernel takes the whole page, everything else column 0.
        pools = self._base_pools()
        slots = md.state_indices_tensor
        ndt, npt = md.num_decode_tokens, md.num_prefill_tokens

        if md.num_decodes > 0:
            if md.spec is not None:
                self._decode_spec(
                    hidden_states[:ndt], out[:ndt], md.spec,
                    all_pools if self.light_cols else pools,
                )
            elif self._replay is not None:
                self._decode_replay(
                    hidden_states[:ndt], out[:ndt], slots[: md.num_decodes], all_pools
                )
            else:
                self._decode(
                    hidden_states[:ndt], out[:ndt], slots[: md.num_decodes], pools
                )

        if md.num_prefills > 0:
            prefill_slots = slots[md.num_decodes : md.num_decodes + md.num_prefills]
            self._prefill(
                hidden_states[ndt : ndt + npt],
                out[ndt : ndt + npt],
                prefill_slots,
                md,
                pools,
            )
            if self._replay is not None:
                replay_reset(all_pools, prefill_slots.to(torch.int32))

    # -- Projections --------------------------------------------------------
    def _decode_preamble(self, zxdt, bc, n, H, D, R, S):
        """x/z split + A/dt/trap activations + B/C norms, one launch."""
        x = torch.empty(n, H, D, dtype=zxdt.dtype, device=zxdt.device)
        z = torch.empty_like(x)
        _A = torch.empty(n, H, dtype=torch.float32, device=zxdt.device)
        DT = torch.empty_like(_A)
        trap = torch.empty_like(_A)
        bn_cn = torch.empty(n, 2, R, S, dtype=zxdt.dtype, device=zxdt.device)
        _decode_preamble_bc_kernel[(n * H + n * 2 * R,)](
            zxdt, x, z, _A, DT, trap, self.dt_bias, self.A_floor, H, zxdt.stride(0),
            bc, bn_cn, self.B_norm.norm.weight, self.C_norm.norm.weight,
            self.B_norm.norm.eps, R * S, R, bc.stride(0), n * H,
            D=D, S=S, ZERO_CENTERED=self.B_norm.norm.zero_centered,
        )
        return x, z, _A, DT, trap, bn_cn

    def _decode_in_proj(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """in_proj / in_proj_dyn for decode rows.

        Single rank, no bias: one GEMM over the concatenated weights (cached;
        the preamble kernel takes the resulting row stride), streaming GEMV at
        batch 1. Otherwise the two parallel linears.
        """
        if self._int8_in_proj:
            segs, n1, width, n16, _ = self._int8_state()
            if 0 < u.shape[0] <= int8_decode_max_m(width, u.shape[1], n16):
                out = int8_decode_linear(u, segs, width=width)
                return out[:, :n1], out[:, n1:]
            return self._int8_parts(u)
        if self._in_proj_cat_ok:
            w = self._in_proj_cat_weight()
            out = decode_gemv(u, w)
            n1 = self.in_proj.weight.shape[0]
            return out[:, :n1], out[:, n1:]
        zxdt, _ = self.in_proj(u)
        bc, _ = self.in_proj_dyn(u)
        return zxdt, bc

    def _int8_state(self) -> tuple:
        """(decode segments, in_proj rows, width, bf16 rows, concatenated INT8
        weight or None): ``[in_proj ; in_proj_dyn]`` as one INT8 weight (shared
        storage) when both are INT8, else INT8 in_proj + bf16 in_proj_dyn rows
        in the same launch. Built on the first call (the profiling run)."""
        if self._int8_st is None:
            n1 = self.in_proj.output_size_per_partition
            n2 = self.in_proj_dyn.output_size_per_partition
            cat = None
            if self.in_proj_dyn.weight.dtype == torch.int8:
                cat = cat_int8_rows_shared([self.in_proj, self.in_proj_dyn])
                segs, n16 = ([Segment(cat.weight, cat.scale)] if cat else None), 0
            else:
                w8 = int8_linear_weight(self.in_proj)
                segs = (
                    [
                        Segment(w8.weight, w8.scale),
                        Segment(self.in_proj_dyn.weight, col0=n1),
                    ]
                    if w8
                    else None
                )
                n16 = n2
            if segs is None:
                raise RuntimeError(
                    f"{self.prefix}.in_proj: unsupported INT8 weight layout"
                )
            self._int8_st = (segs, n1, n1 + n2, n16, cat)
        return self._int8_st

    def _int8_parts(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """INT8 in_proj, prefill / large batches: ``(zxdt, bc)``, row views of one
        W8A16 GEMM over the concatenated weight (two GEMMs with a bf16
        in_proj_dyn)."""
        _, n1, width, _, cat = self._int8_state()
        if cat is not None:
            y = int8_prefill_mm(h, cat)
            return y[:, :n1], y[:, n1:]
        return int8_apply(self.in_proj, h), self.in_proj_dyn(h)[0]

    def _raw_in_proj_ok(self, n: int) -> bool:
        """The raw CUDA step needs in_proj and in_proj_dyn in the same buffer."""
        if not self._int8_in_proj:
            return self._in_proj_cat_ok
        _, _, width, n16, cat = self._int8_state()
        return cat is not None or n <= int8_decode_max_m(width, self.d_model, n16)

    def _in_proj_cat_weight(self) -> torch.Tensor:
        if self._in_proj_cat_w is None:
            self._in_proj_cat_w = cat_rows_shared(
                [self.in_proj.weight, self.in_proj_dyn.weight]
            )
        return self._in_proj_cat_w

    def _pf_split_ok(self, dev) -> bool:
        from vllm.model_executor.layers.mamba.ops.mamba3.mimo_ssd import _scan_tc_ok

        w, wd = self.in_proj.weight, self.in_proj_dyn.weight
        return (
            os.environ.get("OLALA_PF_SPLIT_INPROJ", "1") != "0"
            and type(self.in_proj.quant_method).__name__ == "UnquantizedLinearMethod"
            and type(self.in_proj_dyn.quant_method).__name__ == "UnquantizedLinearMethod"
            and w.dtype == torch.bfloat16 and wd.dtype == torch.bfloat16
            and not self._postgate_norm
            and _scan_tc_ok(dev)
        )

    def _pf_in_rows(self) -> torch.Tensor:
        H, D = self.nheads_local, self.headdim
        hd = torch.arange(H).view(H, 1) * (2 * D + 3)
        d = torch.arange(D).view(1, D)
        perm = torch.cat([(hd + d).flatten(), (hd + D + d).flatten(),
                          hd.flatten() + 2 * D, hd.flatten() + 2 * D + 1, hd.flatten() + 2 * D + 2])
        return perm.to(self.in_proj.weight.device)

    def _pf_in_weight(self) -> torch.Tensor:
        if self._pf_in_w is None:
            self._pf_in_w = torch.cat([self.in_proj.weight[self._pf_in_rows()], self.in_proj_dyn.weight]).contiguous()
        return self._pf_in_w

    def _project_in(self, h: torch.Tensor):
        """Run in_proj/in_proj_dyn on ``(N, D)`` and unpack the pieces."""
        if h.shape[0] == 1 and not self._int8_in_proj and self.in_proj.bias is None:
            # Single-token decode: streaming GEMVs beat cuBLAS's batch-1 floor.
            zxdtAtrap = decode_gemv(h, self.in_proj.weight)
        else:
            zxdtAtrap, _ = self.in_proj(h)
        per_head = zxdtAtrap.view(h.shape[0], self.nheads_local, 2 * self.headdim + 3)
        z = per_head[..., 0 : self.headdim]
        x = per_head[..., self.headdim : 2 * self.headdim]
        dt = per_head[..., 2 * self.headdim]
        A = per_head[..., 2 * self.headdim + 1]
        trap = per_head[..., 2 * self.headdim + 2]

        if (
            h.shape[0] == 1
            and not is_int8_quant_method(self.in_proj_dyn)
            and self.in_proj_dyn.bias is None
        ):
            bc = decode_gemv(h, self.in_proj_dyn.weight)
        else:
            bc, _ = self.in_proj_dyn(h)
        off = self.ngroups * self.mimo_dim * self.d_state
        B = rearrange(
            bc[..., :off], "n (G r s) -> n r G s", G=self.ngroups, r=self.mimo_dim
        )
        C = rearrange(
            bc[..., off : 2 * off],
            "n (G r s) -> n r G s",
            G=self.ngroups,
            r=self.mimo_dim,
        )
        angle = bc[..., 2 * off :]
        return z, x, dt, A, trap, B, C, angle

    # -- Prefill ------------------------------------------------------------
    def _prefill(
        self,
        h: torch.Tensor,  # (T, D)
        out: torch.Tensor,  # (T, d_inner_local)
        slots: torch.Tensor,  # (P,)
        md: Mamba3AttentionMetadata,
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        if _mamba3_mimo_hybrid_pools is not None and self._hybrid_ok:
            # pool slots read / written in place by the kernels (no host sync, no gathers)
            self._prefill_hybrid(
                h, out, slots, md.query_start_loc_p, pools,
                md.has_initial_state if md.has_initial_any else None,
                md.query_start_loc_p_cpu,
            )
            return
        # Fast path: every prompt starts from zero state, so one packed varlen
        # launch covers the batch. OlalaForCausalLMConfig disables chunked
        # prefill precisely to keep us here.
        if not md.has_initial_any:
            self._prefill_zero_start(
                h, out, slots, md.query_start_loc_p, pools,
                qsl_cpu=md.query_start_loc_p_cpu,
            )
            return

        # Continuation chunks resume a cached state: one packed varlen call
        # still covers the whole batch, zero-start sequences get zero init
        # states and continuations their pool rows.
        init = self._gather_init_states(pools, slots, md.has_initial_state, h.dtype)
        self._prefill_zero_start(
            h, out, slots, md.query_start_loc_p, pools, init_states=init,
            qsl_cpu=md.query_start_loc_p_cpu,
        )

    def _dry_run_prefill(self, h: torch.Tensor, out: torch.Tensor) -> None:
        """Profiling / dry run: the real prefill over the batch as one sequence,
        on scratch one-slot pools. The memory profiler then sees the prefill
        activation peak (sizing the KV cache from a zeroed output left too
        little headroom on 20 GB GPUs), and the TileLang / Triton JIT and
        autotuning run before the KV cache is allocated."""
        if self._int8_in_proj:
            self._int8_state()
        if self._in_proj_cat_ok:
            self._in_proj_cat_weight()  # decode-time weight cache, before KV sizing
        if self._hybrid_ok and self._pf_split_ok(h.device):
            self._pf_in_weight()  # prefill weight copy, before KV sizing
        self._decode_const_weights()
        self._raw_step_consts()
        n = h.shape[0]
        if n == 0 or torch.cuda.is_current_stream_capturing():
            out.zero_()
            return
        full = tuple(
            torch.zeros((1, *shape), dtype=dtype, device=h.device)
            for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype())
        )
        # light columns: the prefill sees column 0, as on the real pages (_base_pools)
        pools = (full[0][:, 0], full[1], full[2][:, 0], full[3][:, 0]) if self.light_cols else full[:4]
        slots = torch.zeros(1, dtype=torch.long, device=h.device)
        qsl = torch.tensor([0, n], dtype=torch.int32, device=h.device)
        self._prefill_zero_start(h, out, slots, qsl, pools, qsl_cpu=torch.tensor([0, n]))

    @staticmethod
    def _gather_init_states(
        pools: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
        has_initial: torch.Tensor,  # (P,) bool, on device
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sequence init states for the varlen kernel.

        Continuations read their pool rows; zero-start sequences get zeros
        (masked on device — no CPU sync).
        """
        angle_pool, ssm_pool, k_pool, v_pool = pools
        idx = slots.long()
        mask1 = has_initial.view(-1, *([1] * (angle_pool.dim() - 1)))
        angle = angle_pool[idx].to(torch.float32) * mask1
        mask = has_initial.view(-1, *([1] * (ssm_pool.dim() - 1)))
        ssm = ssm_pool[idx].to(torch.float32) * mask
        k = k_pool[idx].to(compute_dtype) * has_initial.view(
            -1, *([1] * (k_pool.dim() - 1))
        )
        v = v_pool[idx].to(compute_dtype) * has_initial.view(
            -1, *([1] * (v_pool.dim() - 1))
        )
        return angle, ssm, k, v

    def _prefill_zero_start(
        self,
        h: torch.Tensor,  # (T, D)
        out: torch.Tensor,  # (T, d_inner_local)
        slots: torch.Tensor,  # (P,)
        qsl: torch.Tensor,  # (P+1,) rebased to 0
        pools: tuple[torch.Tensor, ...],
        init_states: tuple[torch.Tensor, ...] | None = None,
        qsl_cpu: torch.Tensor | None = None,
    ) -> None:
        if _mamba3_mimo_hybrid_pools is not None and self._hybrid_ok and init_states is None:
            self._prefill_hybrid(h, out, slots, qsl, pools, None, qsl_cpu)
            return
        angle_pool, ssm_pool, k_pool, v_pool = pools
        z, x, dt, A, trap, B, C, angle = self._project_in(h)
        z = z.unsqueeze(0)  # (1, T, H, p)
        x = x.unsqueeze(0)  # (1, T, H, p)
        dt = dt.unsqueeze(0).to(torch.float32)  # (1, T, H)
        A = A.unsqueeze(0)  # (1, T, H)
        trap = trap.unsqueeze(0).permute(0, 2, 1).contiguous()  # (1, H, T)
        B = B.unsqueeze(0)  # (1, T, R, G, N)
        C = C.unsqueeze(0)  # (1, T, R, G, N)
        # Angles expand over heads in fp32; the kernel does the angle*dt cumsum.
        angle = (
            angle.unsqueeze(0)
            .unsqueeze(-2)
            .expand(-1, -1, self.nheads_local, -1)
            .to(torch.float32)
            .contiguous()
        )

        B = self.B_norm(B)
        C = self.C_norm(C)

        _A = torch.clamp(-F.softplus(A.to(torch.float32)), max=-self.A_floor)
        DT = F.softplus(dt + self.dt_bias)
        ADT = (_A * DT).permute(0, 2, 1).contiguous()  # (1, H, T)
        DT = DT.permute(0, 2, 1).contiguous()  # (1, H, T)

        # Constant fp32 weight casts, computed once and cached (otherwise 6
        # cast kernels + allocations per layer per prefill). Invalidated by
        # refresh_weight_caches() on weight reload.
        cw = self._prefill_const_w
        if cw is None:
            cw = (
                self.C_bias.to(torch.float32),
                self.B_bias.to(torch.float32),
                self.in_proj_mimo_x.to(torch.float32),
                self.in_proj_mimo_z.to(torch.float32),
                self.out_proj_mimo.to(torch.float32),
                self.D.to(torch.float32),
            )
            self._prefill_const_w = cw
        q_bias, k_bias, mimo_v, mimo_z, mimo_out, d_f32 = cw

        kernel_kwargs = dict(
            Q=C.contiguous().bfloat16(),
            K=B.contiguous().bfloat16(),
            V=x.contiguous().bfloat16(),
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=q_bias,
            K_bias=k_bias,
            MIMO_V=mimo_v,
            MIMO_Z=mimo_z,
            MIMO_Out=mimo_out,
            Angles=angle,
            D=d_f32,
            Z=z.contiguous(),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            cu_seqlens=qsl.to(torch.int32),
        )
        if _mamba3_mimo_grouped is not None:
            # Group-parallel exact prefill: long sequences split into virtual
            # groups (kernel occupancy), short ones take the single-pass path
            # inside the wrapper. Same numerics either way.
            Out, Final_Angle, Final_SSM, Final_K, _ = _mamba3_mimo_grouped(
                **kernel_kwargs,
                Input_States=init_states,
                cu_seqlens_cpu=qsl_cpu,
            )
        else:
            Out, Final_Angle, Final_SSM, Final_K, _ = _mamba3_mimo(
                **kernel_kwargs,
                return_state=True,
                **({"Input_States": init_states} if init_states is not None else {}),
            )

        # The kernel's Final_V is taken at the global last token, which is
        # wrong for a packed varlen batch — recompute it per sequence.
        angle_pool[slots] = Final_Angle.to(angle_pool.dtype)
        ssm_pool[slots] = Final_SSM.to(ssm_pool.dtype)
        k_pool[slots] = Final_K.to(k_pool.dtype)
        v_pool[slots] = x[0, qsl[1:].long() - 1].to(v_pool.dtype)

        y = rearrange(Out.squeeze(0), "t h p -> t (h p)")
        if self._postgate_norm:
            y = self.output_norm(y)
        out.copy_(y.to(out.dtype))

    def _prefill_hybrid(self, h, out, slots, qsl, pools, has_init, qsl_cpu):
        angle_pool, ssm_pool, k_pool, v_pool = pools
        T = h.shape[0]
        H, D, R, S = self.nheads_local, self.headdim, self.mimo_dim, self.d_state
        dev, dt_ = h.device, h.dtype
        DT = torch.empty(H, T, device=dev, dtype=torch.float32)
        ADT = torch.empty(H, T, device=dev, dtype=torch.float32)
        trap = torch.empty(H, T, device=dev, dtype=dt_)
        Q = torch.empty(T, R, S, device=dev, dtype=torch.bfloat16)
        K = torch.empty(T, R, S, device=dev, dtype=torch.bfloat16)
        ang = torch.empty(T, self.num_rope_angles, device=dev, dtype=torch.float32)
        if self._pf_split_ok(dev):
            # one GEMM, x / z stay in its output (row-strided views); OLALA_VOLTA_TC: on the
            # fp16 tensor cores (volta_tc.py)
            cat = dense_linear(h, self._pf_in_weight())
            z = cat[:, : H * D].view(T, H, D)
            x = cat[:, H * D : 2 * H * D].view(T, H, D)
            BT = 32
            n_pre = triton.cdiv(T, BT)
            _prefill_prologue_split_kernel[(n_pre + T,)](
                cat, cat.stride(0), 2 * H * D + 3 * H, DT, ADT, trap, Q, K, ang,
                self.dt_bias, self.A_floor, self.B_norm.norm.weight, self.C_norm.norm.weight, self.B_norm.norm.eps,
                T, H, R, n_pre, D=D, S=S, NA=self.num_rope_angles, BT=BT, HP=triton.next_power_of_2(H),
                ZERO_CENTERED=self.B_norm.norm.zero_centered, R2=2 * R, num_warps=4,
            )
        else:
            # Ampere / Ada / Hopper (dragon-v0.26): x / z read in place from the in_proj output
            if self._in_proj_cat_ok and _PREFILL_INPROJ_CAT:
                # one GEMM on [in_proj ; in_proj_dyn] (the decode weight cache): the narrow
                # in_proj_dyn (1056 rows) alone fills 1.3 waves on 56 SMs; the prologue reads
                # both outputs through their row strides
                n1, n2 = self.in_proj.weight.shape[0], self.in_proj_dyn.weight.shape[0]
                y = dense_linear(h, self._in_proj_cat_weight())
                zxdt, bc = y[:, :n1], y[:, n1 : n1 + n2]
            elif self._int8_in_proj:
                zxdt, bc = self._int8_parts(h)
            else:
                zxdt, _ = self.in_proj(h)
                bc, _ = self.in_proj_dyn(h)
            xz_views = _xz_strided_ok(dev) and zxdt.stride(1) == 1
            if xz_views:  # head-interleaved [z | x | dt a trap] rows of the in_proj output
                st0, off = zxdt.stride(0), zxdt.storage_offset()
                z = zxdt.as_strided((T, H, D), (st0, 2 * D + 3, 1), off)
                x = zxdt.as_strided((T, H, D), (st0, 2 * D + 3, 1), off + D)
            else:
                x = torch.empty(T, H, D, device=dev, dtype=dt_)
                z = torch.empty(T, H, D, device=dev, dtype=dt_)
            BT = 16
            n_pre = triton.cdiv(T, BT) * H
            _prefill_prologue_kernel[(n_pre + T,)](
                zxdt, zxdt.stride(0), bc, bc.stride(0), x, z, DT, ADT, trap, Q, K, ang,
                self.dt_bias, self.A_floor, self.B_norm.norm.weight, self.C_norm.norm.weight, self.B_norm.norm.eps,
                T, H, R, n_pre, D=D, S=S, NA=self.num_rope_angles, BT=BT,
                ZERO_CENTERED=self.B_norm.norm.zero_centered, R2=2 * R, WRITE_XZ=not xz_views, num_warps=2,
            )
        cw = self._prefill_const_w
        if cw is None:
            cw = (
                self.C_bias.to(torch.float32),
                self.B_bias.to(torch.float32),
                self.in_proj_mimo_x.to(torch.float32),
                self.in_proj_mimo_z.to(torch.float32),
                self.out_proj_mimo.to(torch.float32),
                self.D.to(torch.float32),
            )
            self._prefill_const_w = cw
        q_bias, k_bias, mimo_v, mimo_z, mimo_out, d_f32 = cw
        if qsl_cpu is None:
            qsl_cpu = qsl.cpu()
        cu = [int(v) for v in qsl_cpu.tolist()]
        angle_pool, ssm_pool, k_pool, v_pool = pools
        assert (ssm_pool.stride()[1:] == (D * S, S, 1) and k_pool.stride()[1:] == (H * S, S, 1)
                and v_pool.stride()[1:] == (D, 1) and angle_pool.stride()[1:] == (self.num_rope_angles, 1))
        direct = not self._postgate_norm and out.dtype == x.dtype and out.is_contiguous()
        o_buf = out.view(T, H, D) if direct else torch.empty(T, H, D, device=dev, dtype=x.dtype)
        _mamba3_mimo_hybrid_pools(
            Q, K, x, z, ang, DT, ADT, trap, q_bias, k_bias, mimo_v, mimo_out, mimo_z, d_f32, cu,
            pools, slots, has_init, o_buf,
        )
        if not direct:
            y = o_buf.reshape(T, -1)
            if self._postgate_norm:
                y = self.output_norm(y)
            out.copy_(y.to(out.dtype))

    @torch.inference_mode()
    def warmup_prefill_kernels(self, max_tokens: int) -> int:
        """Compile every Triton specialization of the hybrid prefill that a
        serving batch can hit, before serving: the prologue and the Mamba-3
        prefill kernels (mimo_ssd.py) specialize on the token count (1, a
        multiple of 16, other), on the 16-byte alignment of the chunk /
        sequence layout slices (the chunk and sequence counts mod 4) and of
        the prefill slots (a slice after the decodes: the decode count mod 4),
        on reading initial states or not, and on the long-sequence tile (32+
        chunks). One small batch per class runs through ``_prefill_hybrid`` as
        a step would, on dummy pools with the real pools' strides: the KV
        cache is not touched and no output changes. Returns the batch count."""
        _lazy_import_kernels()
        if _mamba3_mimo_hybrid_pools is None or not self._hybrid_ok:
            return 0
        from vllm.model_executor.layers.mamba.ops.mamba3.mimo_ssd import _scan_tc_ok

        real = self._base_pools()  # the committed state's views: column 0 with light columns, as _prefill sees them
        dev, dtype = real[1].device, self.model_config.dtype
        pools = tuple(
            torch.empty_strided((2, *p.shape[1:]), p.stride(), dtype=p.dtype, device=dev).zero_()
            for p in real
        )
        ls = 256 if _scan_tc_ok(dev) else 128  # chunk length of mamba3_mimo_hybrid_pools
        layouts = [[1]]
        for chunks0 in (0, 31) if max_tokens > 32 * ls else (0,):
            for ns in range(1, 5):  # sequence count mod 4
                for extra in range(4):  # chunk count mod 4
                    for div16 in (True, False):
                        t0 = ns - 1 + ls * (chunks0 + extra)
                        last = ls * (chunks0 + extra) + ((-t0) % 16 or 16) + (0 if div16 else 1)
                        layouts.append([1] * (ns - 1) + [last])
        n = 0
        for lens in layouts:
            T = sum(lens)
            if T > max_tokens:
                continue
            cu = [0]
            for length in lens:
                cu.append(cu[-1] + length)
            qsl_cpu = torch.tensor(cu, dtype=torch.int32)
            qsl = qsl_cpu.to(dev)
            h = torch.zeros(T, self.d_model, dtype=dtype, device=dev)
            out = torch.empty(T, self.nheads_local * self.headdim, dtype=dtype, device=dev)
            for off in (0, 1):  # prefill slots after 0 / 1 decodes (int32: aligned or not)
                slots = torch.zeros(off + len(lens), dtype=torch.int32, device=dev)[off:]
                for has_init in (None, torch.ones(len(lens), dtype=torch.bool, device=dev)):
                    self._prefill_hybrid(h, out, slots, qsl, pools, has_init, qsl_cpu)
                    n += 1
        torch.cuda.synchronize(dev)
        return n

    # -- Decode -------------------------------------------------------------
    def refresh_weight_caches(self) -> None:
        """Refresh the cached weight-derived tensors after a weight reload.

        Must be called after any in-place weight reload: the caches mix live
        views (the rearranged rotary biases) with snapshots (the fp32 casts
        and contiguous copies), so running on stale caches after a reload
        would silently blend old and new weights.

        The snapshots are refreshed IN PLACE rather than dropped. FULL-mode
        CUDA graphs capture ``_decode`` (inside the ``olala_mamba3`` op) and
        bake the snapshot tensors' raw device addresses into every captured
        decode graph. Dropping the cache frees those tensors; once the
        caching allocator reuses or releases the segment, every subsequent
        graph replay reads freed memory — stale weights at best, an illegal
        memory access as soon as the block is recycled. This was observed
        deterministically under RL training (verl), which reloads weights
        before every rollout; serving never resyncs weights, which is why it
        never triggered. Rewriting the same storage keeps the baked pointers
        valid and propagates the new weights into the captured graphs.

        The view entries (rearranged rotary biases) alias the params, which
        ``load_weights`` updates in place, so they need no refresh.
        ``torch.inference_mode()`` is required because the caches are
        created during forward under inference mode, and in-place writes to
        inference tensors are only legal inside it.
        """
        with torch.inference_mode():
            cw = self._decode_const_w
            if cw is not None:
                # In this tree all five entries are .contiguous() snapshots
                # (the CUDA step kernel needs contiguous fp32 biases too).
                cw[0].copy_(rearrange(self.C_bias, "h r n -> r h n"))
                cw[1].copy_(rearrange(self.B_bias, "h r n -> r h n"))
                cw[2].copy_(rearrange(self.in_proj_mimo_x, "h r p -> r h p"))
                cw[3].copy_(rearrange(self.in_proj_mimo_z, "h r p -> r h p"))
                cw[4].copy_(rearrange(self.out_proj_mimo, "h r p -> r h p"))
            pw = self._prefill_const_w
            if pw is not None:
                for dst, src in zip(
                    pw,
                    (
                        self.C_bias,
                        self.B_bias,
                        self.in_proj_mimo_x,
                        self.in_proj_mimo_z,
                        self.out_proj_mimo,
                        self.D,
                    ),
                ):
                    dst.copy_(src)
            refresh_f32_cache()
            if self._raw_consts is not None:
                for dst, src in zip(self._raw_consts, self._raw_const_values()):
                    dst.copy_(src)
            # Normally a self-copy (cat_rows_shared): only matters if a reload
            # rebound param.data instead of copying into it.
            pw2 = self._pf_in_w
            if pw2 is not None:
                n1 = self.in_proj.weight.shape[0]
                pw2[:n1].copy_(self.in_proj.weight[self._pf_in_rows()])
                pw2[n1:].copy_(self.in_proj_dyn.weight)
            w = self._in_proj_cat_w
            if w is not None:
                n1 = self.in_proj.weight.shape[0]
                w[:n1].copy_(self.in_proj.weight)
                w[n1:].copy_(self.in_proj_dyn.weight)

    def _decode_const_weights(self):
        """Rearranged constant weights for the decode-step kernels.

        The rotary biases and MIMO projections never change between weight
        loads, so their per-step ``rearrange`` + ``.contiguous()`` copies are
        pure redundant work (3 copy kernels per layer per step). Invalidated
        by refresh_weight_caches() on weight reload.
        """
        cw = self._decode_const_w
        if cw is None:
            cw = (
                rearrange(self.C_bias, "h r n -> r h n").contiguous(),
                rearrange(self.B_bias, "h r n -> r h n").contiguous(),
                rearrange(self.in_proj_mimo_x, "h r p -> r h p").contiguous(),
                rearrange(self.in_proj_mimo_z, "h r p -> r h p").contiguous(),
                rearrange(self.out_proj_mimo, "h r p -> r h p").contiguous(),
            )
            self._decode_const_w = cw
        return cw

    def _decode_spec(
        self,
        u: torch.Tensor,  # (ndt, D) — all decode tokens, ragged per request
        out: torch.Tensor,  # (ndt, d_inner_local)
        spec,  # OlalaSpecMetadata
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        """Speculative verify: chain the dual-slot step kernel over positions.

        Position t of each request reads the state written by position t-1
        (column t-1; for t = 0 the column selected by last step's acceptance)
        and writes column t. Inactive lanes (requests with fewer tokens) get
        slot -1: the kernel zeroes their output and suppresses every write.
        Always uses the fused rotary (correct at any batch size) so the angle
        pool follows the same column protocol.
        """
        angle_pool, ssm_pool, k_pool, v_pool = pools[:4]
        agb_pool = pools[4] if self.light_cols else None
        H, D = self.nheads_local, self.headdim
        R, S = self.mimo_dim, self.d_state
        if not (self.ngroups == 1 and _is_pow2(D) and _is_pow2(S)):
            raise NotImplementedError(
                "Olala spec decode requires ngroups == 1 and power-of-two "
                "headdim/d_state (the fused preamble shapes)."
            )
        assert self.headdim <= _STEP_TILE_D
        ndt = u.shape[0]

        # Preamble over every decode token at once (same kernels as _decode).
        zxdt, bc = self._decode_in_proj(u)
        x, z, _A, DT, trap, bn_cn = self._decode_preamble(zxdt, bc, ndt, H, D, R, S)
        angle = bc[..., 2 * R * S :]
        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()

        if verify_kernel_enabled() and verify_supported(
            ssm_pool, k_pool, v_pool, angle_pool, H, D, S, R, self.num_rope_angles
        ):
            # One launch for every position of every request
            # (csrc/mamba3_verify.cu), bit-identical to chaining the CUDA
            # decode step over the positions.
            direct = (
                not self._postgate_norm
                and out.dtype == x.dtype
                and out.is_contiguous()
            )
            y = out.view(ndt, H, D) if direct else torch.empty_like(x)
            mamba3_verify_cuda(
                ssm_pool, k_pool, v_pool, angle_pool, _A, bn_cn[:, 0],
                bn_cn[:, 1], self.D, x, DT, trap, xpj, zpj, outpj, z, bias_q,
                bias_k, angle, spec.query_start_loc_d, spec.state_cols,
                spec.num_accepted_tokens, y, agb=agb_pool,
            )  # fmt: skip
            if not direct:
                y = y.reshape(ndt, H * D)
                if self._postgate_norm:
                    y = self.output_norm(y)
                out.copy_(y.to(out.dtype))
            return
        if self.light_cols:
            raise NotImplementedError(
                "Light speculative columns (OLALA_SPEC_LIGHT) need the fused verify kernel "
                "(OLALA_MAMBA3_VERIFY=cuda, sm_80+ build); set OLALA_SPEC_LIGHT=0 for the chain."
            )
        self._decode_spec_chain(
            u, out, spec, pools, x, z, _A, DT, trap, bn_cn, angle
        )

    def _decode_spec_chain(
        self, u, out, spec, pools, x, z, _A, DT, trap, bn_cn, angle
    ) -> None:
        """Reference verify path (OLALA_MAMBA3_VERIFY != cuda, or shapes the
        fused kernel does not cover): the step kernel launched once per
        position."""
        angle_pool, ssm_pool, k_pool, v_pool = pools
        H, D = self.nheads_local, self.headdim
        ndt = u.shape[0]
        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()
        qsl = spec.query_start_loc_d
        starts = qsl[:-1].long()
        qlens = torch.diff(qsl)
        cols = spec.state_cols.to(torch.int32)
        in0 = spec.initial_slots().to(torch.int32)
        neg1 = torch.full_like(in0, -1)
        dummy = torch.full_like(starts, ndt)
        # Extra dummy row absorbs inactive lanes' outputs.
        y_pad = torch.zeros(ndt + 1, H, D, dtype=u.dtype, device=u.device)

        for t in range(spec.max_qlen):
            active = qlens > t
            rows = torch.where(active, starts + t, starts)
            src = in0 if t == 0 else cols[:, t - 1].contiguous()
            in_t = torch.where(active, src, neg1)
            out_t = torch.where(active, cols[:, t].contiguous(), neg1)
            bpre = bn_cn.index_select(0, rows)
            y_t = torch.empty(rows.shape[0], H, D, dtype=u.dtype, device=u.device)
            _mamba3_step_fn(
                ssm_pool,
                k_pool,
                v_pool,
                _A.index_select(0, rows),
                bpre[:, 0].unsqueeze(2).expand(-1, -1, H, -1).to(torch.bfloat16),
                bpre[:, 1].unsqueeze(2).expand(-1, -1, H, -1).to(torch.bfloat16),
                self.D,
                x.index_select(0, rows),
                DT.index_select(0, rows),
                trap.index_select(0, rows),
                xpj,
                outpj,
                None,
                y_t,
                z=z.index_select(0, rows),
                zproj=zpj,
                state_batch_indices=in_t,
                state_batch_indices_out=out_t,
                update_kv_state=True,
                tile_D=_STEP_TILE_D,
                num_warps=_STEP_NUM_WARPS,
                rotary_dim=2 * self.num_rope_angles,
                rotary_bias_q=bias_q,
                rotary_bias_k=bias_k,
                rotary_angle_proj=angle.index_select(0, rows)
                .unsqueeze(-2)
                .expand(-1, H, -1),
                rotary_angle_state=angle_pool,
            )
            y_pad.index_copy_(0, torch.where(active, starts + t, dummy), y_t)

        y = y_pad[:ndt].reshape(ndt, H * D)
        if self._postgate_norm:
            y = self.output_norm(y)
        out.copy_(y.to(out.dtype))

    def _decode(
        self,
        u: torch.Tensor,  # (N, D)
        out: torch.Tensor,  # (N, d_inner_local)
        slots: torch.Tensor,  # (N,)
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        angle_pool, ssm_pool, k_pool, v_pool = pools
        H, D = self.nheads_local, self.headdim
        R, S = self.mimo_dim, self.d_state

        # Two Triton kernels replace ~14 tiny eager kernels per call: x/z
        # head-deinterleave plus the A/dt/trap activations, and the B/C RMS
        # norm. They also keep DT/trap in fp32 where the eager chain rounded
        # to bf16 per op, so greedy output can differ within bf16 noise.
        fuse_preamble = self.ngroups == 1 and _is_pow2(D) and _is_pow2(S)
        if (
            fuse_preamble
            and self._raw_step_ok(pools)
            and self._raw_in_proj_ok(u.shape[0])
        ):
            self._decode_raw_step(u, out, slots, pools)
            return
        if fuse_preamble:
            zxdt, bc = self._decode_in_proj(u)
            n_tok = u.shape[0]
            x, z, _A, DT, trap, bn_cn = self._decode_preamble(zxdt, bc, n_tok, H, D, R, S)
            B = bn_cn[:, 0].unsqueeze(2).expand(-1, -1, H, -1)
            C = bn_cn[:, 1].unsqueeze(2).expand(-1, -1, H, -1)
            angle = bc[..., 2 * R * S :]
        else:
            z, x, dt, A, trap_raw, B, C, angle = self._project_in(u)
            # Views from _project_in are non-contiguous; the CuteDSL step
            # kernel needs strides divisible by 8.
            x = x.contiguous()
            z = z.contiguous()
            _A = torch.clamp(-F.softplus(A.to(torch.float32)), max=-self.A_floor)
            DT = F.softplus(dt.contiguous() + self.dt_bias)
            trap = torch.sigmoid(trap_raw.contiguous())
            B = self.B_norm(B).expand(-1, -1, H, -1)
            C = self.C_norm(C).expand(-1, -1, H, -1)
        angle = angle.unsqueeze(-2).expand(-1, H, -1)

        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()
        slots32 = slots.to(torch.int32).contiguous()

        # With tile_D >= headdim one CTA owns each (b, h) row, so the step
        # kernel can also store the new B/x key/value states itself.
        kernel_writes_kv = self.headdim <= _STEP_TILE_D
        # The fused rotary reads B/C head-broadcast, which only holds for a
        # single group. The kernel must also own the k_pool write, or the
        # caller-side scatter below would store the pre-rotation B.
        fuse_rotary = (
            kernel_writes_kv
            and self.ngroups == 1
            and u.shape[0] <= _fuse_rotary_max_batch()
        )

        if fuse_rotary:
            # No separate rotary launch: the step kernel applies bias+rotary to
            # the head-broadcast pre-rotation B/C and updates angle_pool itself.
            rotary_kwargs = dict(
                rotary_dim=2 * self.num_rope_angles,
                rotary_bias_q=bias_q,
                rotary_bias_k=bias_k,
                rotary_angle_proj=angle,
                rotary_angle_state=angle_pool,
            )
        else:
            rotary_kwargs = {}
            C, B, _ = _apply_rotary_qk_inference_fwd(
                q=C,
                k=B,
                angle_state=angle_pool,
                angle_proj=angle,
                dt=DT,
                bias_q=bias_q,
                bias_k=bias_k,
                conjugate=False,
                inplace=False,
                rotate_pairwise=False,
                state_batch_indices=slots32,
            )

        # Write the step kernel's output straight into ``out`` when it is a
        # contiguous (N, H*D) slice, instead of a temp plus a final copy.
        y_is_out = (
            not self._postgate_norm and out.dtype == x.dtype and out.is_contiguous()
        )
        y = out.view(out.shape[0], H, D) if y_is_out else torch.empty_like(x)

        use_cuda_step = fuse_rotary and cuda_step_enabled()
        if use_cuda_step and not (
            B.dtype == torch.bfloat16
            and cuda_step_supported(
                ssm_pool, k_pool, v_pool, angle_pool, B[:, :, 0], C[:, :, 0], angle[:, 0],
                H, D, S, R, self.num_rope_angles,
            )
        ):
            use_cuda_step = False
            warn_slow_path(
                "CUDA Mamba-3 decode step not applicable",
                f"shapes/dtypes outside the kernel (H={H} D={D} S={S} R={R} "
                f"angles={self.num_rope_angles}, B {B.dtype}, ssm pool "
                f"{ssm_pool.dtype}); it implements H=48 D=64 S=128 R=4 angles=32",
                "CuteDSL decode step: ~1.4x slower Mamba-3 decode step",
                "none needed for correctness; the kernel only covers the Olala 7A1B shapes",
            )
        if use_cuda_step:
            # Persistent CUDA step (tensor-core update + C dot, cp.async state pipeline).
            mamba3_step_cuda(
                ssm_pool, k_pool, v_pool, angle_pool, _A, B[:, :, 0], C[:, :, 0], self.D, x, DT, trap,
                xpj, zpj, outpj, z, bias_q, bias_k, angle[:, 0], slots32, y,
            )
        else:
            _mamba3_step_fn(
                ssm_pool,
                k_pool,
                v_pool,
                _A,
                B.to(torch.bfloat16),
                C.to(torch.bfloat16),
                self.D,
                x,
                DT,
                trap,
                xpj,
                outpj,
                None,  # in-place pool update
                y,
                z=z,
                zproj=zpj,
                state_batch_indices=slots32,
                update_kv_state=kernel_writes_kv,
                tile_D=_STEP_TILE_D,
                num_warps=_STEP_NUM_WARPS,
                **rotary_kwargs,
            )
        if not kernel_writes_kv:
            k_pool[slots] = B
            v_pool[slots] = x

        if not y_is_out:
            y = rearrange(y, "n h p -> n (h p)")
            if self._postgate_norm:
                y = self.output_norm(y)
            out.copy_(y.to(out.dtype))

    def _raw_step_ok(self, pools) -> bool:
        """The CUDA step can run the decode preamble itself."""
        angle_pool, ssm_pool, k_pool, v_pool = pools
        return (
            _RAW_STEP
            and (self._in_proj_cat_ok or self._int8_in_proj)
            and not self._postgate_norm
            and self.headdim <= _STEP_TILE_D
            and cuda_step_enabled()
            and self.nheads_local == 48
            and self.headdim == 64
            and self.d_state == 128
            and self.mimo_dim == 4
            and self.num_rope_angles == 32
            and ssm_pool.dtype in (torch.float32, torch.bfloat16)
            and k_pool.dtype == torch.bfloat16
            and v_pool.dtype == torch.bfloat16
            and angle_pool.dtype == torch.float32
        )

    def _raw_const_values(self) -> tuple[torch.Tensor, ...]:
        zc = 1.0 if self.B_norm.norm.zero_centered else 0.0
        return (
            self.dt_bias.detach().float().contiguous(),
            (self.B_norm.norm.weight.detach().float() + zc).contiguous(),
            (self.C_norm.norm.weight.detach().float() + zc).contiguous(),
        )

    def _raw_step_consts(self) -> tuple[torch.Tensor, ...]:
        """fp32 dt_bias and B/C gains for the raw step. Built eagerly (see
        _dry_run_prefill): a first build during CUDA graph capture would
        allocate them from a graph's private pool."""
        if self._raw_consts is None:
            self._raw_consts = self._raw_const_values()
        return self._raw_consts

    def _decode_raw_step(self, u, out, slots, pools, zxdt_bc=None) -> None:
        """in_proj GEMM then the CUDA step, which also does the x/z split,
        the A/dt/trap activations and the B/C norms (one launch less).
        ``zxdt_bc``: the in_proj output, when already computed."""
        zxdt, bc = self._decode_in_proj(u) if zxdt_bc is None else zxdt_bc
        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()
        H, D = self.nheads_local, self.headdim
        if out.is_contiguous():
            y = out.view(out.shape[0], H, D)
        else:
            y = torch.empty(out.shape[0], H, D, dtype=out.dtype, device=out.device)
        angle_pool, ssm_pool, k_pool, v_pool = pools
        dt_bias, wb, wc = self._raw_step_consts()
        mamba3_step_cuda_raw(
            ssm_pool, k_pool, v_pool, angle_pool, zxdt, bc, dt_bias, self.A_floor,
            wb, wc, self.B_norm.norm.eps, self.D, xpj, zpj, outpj, bias_q, bias_k,
            slots.to(torch.int32).contiguous(), y,
        )
        if not out.is_contiguous():
            out.copy_(y.flatten(1))

    def _decode_replay(
        self,
        u: torch.Tensor,  # (N, D)
        out: torch.Tensor,  # (N, d_inner_local)
        slots: torch.Tensor,  # (N,)
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        """ReplaySSM decode step (csrc/mamba3_replay_step.cu): reads the
        checkpoint (fp8 copy when enabled) and the ring, appends this token,
        and rebuilds + rewrites the full state only when the ring fills."""
        H, D = self.nheads_local, self.headdim
        R, S = self.mimo_dim, self.d_state
        zxdt, bc = self._decode_in_proj(u)
        x, z, _A, DT, trap, bn_cn = self._decode_preamble(zxdt, bc, u.shape[0], H, D, R, S)
        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()
        y_is_out = out.dtype == x.dtype and out.is_contiguous()
        y = out.view(out.shape[0], H, D) if y_is_out else torch.empty_like(x)
        mamba3_replay_step(
            pools, _A, bn_cn[:, 0], bn_cn[:, 1], self.D, x, DT, trap,
            xpj, zpj, outpj, z, bias_q, bias_k, bc[:, 2 * R * S :],
            slots.to(torch.int32).contiguous(), y,
        )
        if not y_is_out:
            out.copy_(y.reshape(out.shape[0], -1).to(out.dtype))


def olala_mamba3(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    g: torch.Tensor | None = None,
    h_out: torch.Tensor | None = None,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    self = get_forward_context().no_compile_layers[layer_name]
    self._forward_impl(hidden_states, output, g, h_out)


def olala_mamba3_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    g: torch.Tensor | None = None,
    h_out: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="olala_mamba3",
    op_func=olala_mamba3,
    mutates_args=["output", "h_out"],
    fake_impl=olala_mamba3_fake,
)

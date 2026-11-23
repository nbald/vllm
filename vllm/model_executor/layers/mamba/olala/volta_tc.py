# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Volta (SM70 family): dense linears on the V100's fp16 tensor cores, exactly scaled.

On a V100 (natively, or through volta-forever, which tells the process sm_80) the bf16
tensor-core instructions are emulated with scalar code and cuBLAS has no bf16 tensor-core
GEMM, so the W8A16 kernels run an emulated ``tl.dot`` or dequantize + cuBLAS's fp32-SIMT bf16
GEMM, and the bf16 model's projections that GEMM, at 3-10 TFLOPS. Volta does have fp16 tensor
cores (HMMA), and cuBLAS 12.9 (which volta-forever forwards CUDA 13's cuBLAS to) has native
Volta fp16 GEMMs: 23-66 TFLOPS on these shapes.

On the SM70 family this edition is on by default (``OLALA_VOLTA_TC`` unset or ``1``: the
DEFAULT_KINDS, W8A16 linears and the bf16 model's dense projections; ``0`` turns it off, a comma
list picks kinds; other GPUs never take it). It moves the registered linears to them from
``min_rows`` activation rows on (decode batches and prefill), with exact operands:

- weights: an fp16 copy per weight, made once. INT8 codes are exact in fp16 (W8A16, the
  per-channel scale stays on the sums). A bf16 weight row n is scaled by 2^sw[n] (sw[n] = 15 -
  exponent of the row's largest magnitude) and converted where that is exact; the rare elements
  that are not (subnormal in fp16) form a sparse second term added in the epilogue;
- activations: each row x (bf16) is split into T exact fp16 terms. Term t holds the elements not
  taken by an earlier term that convert exactly (no rounding, no underflow) at the scale 2^s_t,
  s_t = 15 - exponent of their largest magnitude. T is per linear: 1, 2 for the shared expert's
  down projection (ReLU² input), 3 for the Mamba-3 mixer_proj input (rows span up to ~100 binades);
- one cuBLAS fp16 GEMM over the T*M scaled rows with fp32 output (exact products, fp32
  tensor-core sums), then an epilogue: y[m, n] = out(scale[n] * 2^-sw[n] * sum_t acc_t[m, n] *
  2^-s_t[m]) — power-of-two factors are exact; the per-channel scale multiplies the sum as in
  the W8A16 decode kernels;
- rows that the T terms do not cover exactly (or with a non-finite element, or a row maximum
  outside 2^[-60, 60]) take an FMA fallback in the same epilogue program: fp32 sums of
  bf16(x) * w, the scale on the sum (the arithmetic of the decode kernels). Each row's path
  depends on that row only: graph-safe, independent of the batch composition.

A rounding-level change against the emulated kernels (fp32 tensor-core sums in cuBLAS's order
instead of one fp32 chain): outputs stay at the CPU-bf16 level of the checkpoints, and
``OLALA_VOLTA_TC=0`` gives the previous outputs bit for bit. The routed experts ("moe") are
opt-in only: on chat_math they moved one confident prompt position per quantized checkpoint.
``VF_SAFE=1`` (volta-forever's conservative switch) turns the whole edition off: HMMA truncates
its fp32 sums, a bias toward zero that leaves the conservative reference class.
``OLALA_VOLTA_TC_STATS=1`` counts the rows per number of terms and the fallback rows per linear
(device counters; read with ``stats()``).
"""

import functools
import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

MAX_TERMS = 3
# Row-maximum exponent window of a term: the scaled values, their fp16 grid and the epilogue's
# rescaled sums stay inside fp32's normal range.
E_LO, E_HI = -60, 60
# Sparse second weight term: at most this many non-exact elements per bf16 weight (else the
# weight keeps the stock path).
MAX_BAD_ELEMS = 256
# Stats counters per linear: rows of zeros, rows covered by 1, 2, 3 terms, fallback rows.
_NSTAT = MAX_TERMS + 2


KINDS = ("int8", "bf16", "moe")
# OLALA_VOLTA_TC unset or "1". vllm/envs.py keys this default into the compile cache when the
# variable is unset (compile_factors): keep the two in step.
DEFAULT_KINDS = ("int8", "bf16")


@functools.cache
def vf_safe() -> bool:
    """VF_SAFE, volta-forever's conservative switch: only the exact value "1" turns it on (unset,
    empty, "0", "true", ... do not). It means no path whose results leave the conservative
    reference class; read once per process."""
    return os.environ.get("VF_SAFE") == "1"


@functools.cache
def _kinds() -> frozenset[str]:
    """OLALA_VOLTA_TC on the SM70 family (V100, native or through volta-forever): unset or "1" =
    the DEFAULT_KINDS, "0" (or empty) = off, or a comma list of KINDS ("int8": W8A16 linears,
    "bf16": the bf16 model's dense projections, "moe": the weight-only routed experts above the
    small path). Never on other GPUs, and off under VF_SAFE=1 whatever OLALA_VOLTA_TC says: HMMA
    truncates the tensor-core sums (a bias toward zero), outside VF_SAFE's reference class."""
    raw = os.environ.get("OLALA_VOLTA_TC")
    raw = "1" if raw is None else raw.strip()
    if raw in ("", "0") or not torch.cuda.is_available() or gpu_family() is not GpuFamily.SM70:
        return frozenset()
    kinds = frozenset(DEFAULT_KINDS) if raw == "1" else frozenset(raw.split(","))
    unknown = kinds - set(KINDS)
    if unknown:
        raise ValueError(f"OLALA_VOLTA_TC: unknown kinds {sorted(unknown)} (known: {KINDS})")
    if vf_safe():
        logger.info("VF_SAFE=1: the fp16 tensor-core edition is off")
        return frozenset()
    return kinds


def tc_enabled(kind: str | None = None) -> bool:
    """The edition is on (for ``kind``, or for any kind)."""
    kinds = _kinds()
    return bool(kinds) if kind is None else kind in kinds


@functools.cache
def _min_rows_cfg() -> tuple[int, int]:
    """(one term, more terms): first activation row count on the tensor cores. With one term the
    fp16 GEMM beats the stock kernels from 9 rows (V100 through volta-forever, us, today -> TC
    at 16 rows: 7344 x 1536 114 -> 55, 4608 x 1536 99 -> 35, 1536 x 4608 51 -> 35, lm_head
    1282 -> 890); with 2-3 terms it only breaks even below 33 rows (1536 x 3072 three terms at
    32 rows 54 -> 52). Up to 8 rows the Volta FMA GEMVs stay. OLALA_VOLTA_TC_MIN_ROWS="one,more"
    overrides (measurements)."""
    raw = os.environ.get("OLALA_VOLTA_TC_MIN_ROWS")
    if raw:
        a, b = (int(v) for v in raw.split(","))
        return a, b
    return 9, 33


@functools.cache
def _bf16_rows_cfg() -> tuple[int, int]:
    """The same for bf16 weights. From 9 rows the fp16 GEMM beats the Volta FMA GEMV that
    decode_gemv runs up to 32 rows, also with 2-3 terms (V100 through volta-forever, 32 rows,
    us: 7344 x 1536 217 -> 56, 4608 x 1536 163 -> 39, 1536 x 3072 three terms 119 -> 52;
    lm_head at 16 rows 1476 -> 887). OLALA_VOLTA_TC_BF16_MIN_ROWS="one,more" overrides."""
    raw = os.environ.get("OLALA_VOLTA_TC_BF16_MIN_ROWS")
    if raw:
        a, b = (int(v) for v in raw.split(","))
        return a, b
    return 9, 9


def min_rows(terms: int, bf16: bool = False) -> int:
    one, more = _bf16_rows_cfg() if bf16 else _min_rows_cfg()
    return one if terms <= 1 else more


# Callers test this first (cheap): no tensor-core path below it.
MIN_ROWS = 9


def rows_ok(m: int, weight: torch.Tensor) -> bool:
    """``weight`` is registered and ``m`` activation rows take the tensor cores."""
    hit = lookup(weight)
    return hit is not None and m >= min_rows(hit[0].terms, weight.dtype != torch.int8)


@functools.cache
def _stats_on() -> bool:
    return os.environ.get("OLALA_VOLTA_TC_STATS", "0") not in ("", "0")


# -- kernels -------------------------------------------------------------------------------


# M and the activation / output row strides vary from call to call: no specializations on them
# (each would be a JIT compile while serving).
@triton.jit(do_not_specialize=["M", "stride_xm"])
def _tc_split_kernel(
    x_ptr, h_ptr, sh_ptr, ok_ptr, st_ptr, M, K, stride_xm,
    T: tl.constexpr, BK: tl.constexpr, STATS: tl.constexpr,
    E_MIN: tl.constexpr, E_MAX: tl.constexpr,
):
    """Row m of x (bf16 [M, K]) -> T exact fp16 terms h[t * M + m, :] (elements in no term or in
    a later one: 0), shifts sh[t, m] (term t = x * 2^sh), ok[m] = the terms cover the row."""
    m = tl.program_id(0)
    offs = tl.arange(0, BK)
    kmask = offs < K
    v = tl.load(x_ptr + m.to(tl.int64) * stride_xm + offs, mask=kmask, other=0.0).to(tl.float32)
    bits = v.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    good = tl.max(bits, axis=0) < 0x7F800000  # finite
    rem = bits  # |bits| of the elements no term holds yet (0: taken, or zero)
    used = 0
    zero16 = tl.zeros([BK], dtype=tl.float16)
    for t in tl.static_range(T):
        mx = tl.max(rem, axis=0)
        e = (mx >> 23) - 127
        good = good & ((mx == 0) | ((e >= E_MIN) & (e <= E_MAX)))
        s = tl.where(mx == 0, 0, 15 - e)
        s = tl.minimum(tl.maximum(s, -100), 100)
        p2 = ((s + 127) << 23).to(tl.float32, bitcast=True)
        inv = ((127 - s) << 23).to(tl.float32, bitcast=True)
        h = (v * p2).to(tl.float16)
        # exact: the fp16 value scales back to x; fp32-subnormal x never counts as exact
        ex = (rem >= 0x00800000) & (h.to(tl.float32) * inv == v)
        tl.store(h_ptr + (t * M + m).to(tl.int64) * K + offs, tl.where(ex, h, zero16), mask=kmask)
        tl.store(sh_ptr + t * M + m, s)
        used = tl.where(tl.max(ex.to(tl.int32), axis=0) > 0, t + 1, used)
        rem = tl.where(ex, 0, rem)
    ok = good & (tl.max(rem, axis=0) == 0)
    tl.store(ok_ptr + m, ok.to(tl.int32))
    if STATS:  # slots: rows of zeros, rows covered by 1..3 terms, fallback rows (slot 4)
        tl.atomic_add(st_ptr + tl.where(ok, used, 4), 1)


@triton.jit(do_not_specialize=["M", "NBAD", "stride_xm", "stride_ym"])
def _tc_epilogue_kernel(
    acc_ptr, sh_ptr, ok_ptr, sc_ptr, wi_ptr, bn_ptr, bk_ptr, bv_ptr, NBAD,
    x_ptr, w_ptr, y_ptr, M, N, K, stride_acc, stride_xm, stride_wn, stride_ym,
    T: tl.constexpr, SCALE: tl.constexpr, WINV: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """y[m, n] for BN columns: tensor-core sums rescaled (ok[m]) or the FMA fallback, then the
    per-channel scale."""
    m = tl.program_id(0)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    n_ok = offs_n < N
    a = tl.zeros([BN], dtype=tl.float32)
    if tl.load(ok_ptr + m) != 0:
        for t in tl.static_range(T):
            s = tl.load(sh_ptr + t * M + m)
            inv = ((127 - s) << 23).to(tl.float32, bitcast=True)
            row = (t * M + m).to(tl.int64) * stride_acc
            a += tl.load(acc_ptr + row + offs_n, mask=n_ok, other=0.0) * inv
        if WINV:  # bf16 weights: the fp16 copy of row n is w * 2^sw[n]
            a = a * tl.load(wi_ptr + offs_n, mask=n_ok, other=0.0)
        for i in range(0, NBAD):  # sparse second weight term (elements not exact in fp16)
            bn = tl.load(bn_ptr + i)
            xv = tl.load(x_ptr + m.to(tl.int64) * stride_xm + tl.load(bk_ptr + i)).to(tl.float32)
            a = tl.where(offs_n == bn, a + xv * tl.load(bv_ptr + i), a)
    else:
        offs_k = tl.arange(0, BK)
        for k0 in range(0, K, BK):
            k_ok = k0 + offs_k < K
            xk = tl.load(
                x_ptr + m.to(tl.int64) * stride_xm + k0 + offs_k, mask=k_ok, other=0.0
            ).to(tl.float32)
            w = tl.load(
                w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + (k0 + offs_k)[None, :],
                mask=n_ok[:, None] & k_ok[None, :], other=0,
            ).to(tl.float32)
            a += tl.sum(w * xk[None, :], axis=1)
    if SCALE:
        a = a * tl.load(sc_ptr + offs_n, mask=n_ok, other=0.0)
    tl.store(y_ptr + m.to(tl.int64) * stride_ym + offs_n, a.to(y_ptr.dtype.element_ty), mask=n_ok)


# -- weights -------------------------------------------------------------------------------


@dataclass
class _Entry:
    base: torch.Tensor  # the registered weight [N, K] (int8 or bf16; holds its storage alive)
    w16: torch.Tensor  # fp16 copy [N_pad, K] (rows padded to a multiple of 8 with zeros)
    terms: int  # exact activation terms of this linear
    name: str
    site: int  # stats row (-1: none)
    winv: torch.Tensor | None = None  # bf16: fp32 2^-sw[n]
    bad_n: torch.Tensor | None = None  # bf16: sparse second term (int32 rows, cols; fp32 values)
    bad_k: torch.Tensor | None = None
    bad_v: torch.Tensor | None = None


_REG: dict[int, list[_Entry]] = {}  # storage address -> entries on that storage
_HITS: dict[tuple, tuple[_Entry, int] | None] = {}  # view key -> (entry, first row)
_SITES: list[str] = []
_STATS: torch.Tensor | None = None


def _key(w: torch.Tensor) -> tuple:
    return (w.untyped_storage().data_ptr(), w.storage_offset(), tuple(w.shape), tuple(w.stride()),
            w.dtype)


def _new_site(name: str, device) -> int:
    global _STATS
    if not _stats_on():
        return -1
    _SITES.append(name)
    if _STATS is None or _STATS.shape[0] < len(_SITES):
        new = torch.zeros(max(256, 2 * len(_SITES)), _NSTAT, dtype=torch.int64, device=device)
        if _STATS is not None:
            new[: _STATS.shape[0]].copy_(_STATS)
        _STATS = new
    return len(_SITES) - 1


def _pad8(w16: torch.Tensor) -> torch.Tensor:
    """Rows to a multiple of 8 (cuBLAS's fp16 tensor-core kernels want aligned dimensions)."""
    n = w16.shape[0]
    if n % 8 == 0:
        return w16.contiguous()
    out = torch.zeros(-(-n // 8) * 8, w16.shape[1], dtype=w16.dtype, device=w16.device)
    out[:n].copy_(w16)
    return out


def _add(entry: _Entry) -> None:
    _REG.setdefault(entry.base.untyped_storage().data_ptr(), []).append(entry)
    _HITS.clear()


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def register_int8(weight: torch.Tensor, terms: int = 1, name: str = "") -> bool:
    """fp16 copy of an INT8 [N, K] weight (row-major, per-channel scale on the caller's side)."""
    if (
        not tc_enabled("int8") or weight.dtype != torch.int8 or weight.dim() != 2
        or weight.stride(-1) != 1
        or not weight.is_cuda or _capturing() or lookup(weight) is not None
    ):
        return False
    with torch.no_grad():
        w16 = _pad8(weight.to(torch.float16))
    _add(_Entry(weight.detach(), w16, max(1, min(MAX_TERMS, terms)), name,
                _new_site(name, weight.device)))
    return True


def register_int8_layer(layer: torch.nn.Module) -> None:
    """A W8A16 linear after loading (per-channel scales only): its fp16 copy, and the terms
    the model asks for (``layer.olala_tc_terms``, default 1)."""
    w, s = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
    if w is None or s is None or s.dim() != 1:
        return
    register_int8(w, getattr(layer, "olala_tc_terms", 1), getattr(layer, "prefix", ""))


def register_bf16(weight: torch.Tensor, terms: int = 1, name: str = "") -> bool:
    """fp16 copy of a bf16 [N, K] weight with one power-of-two scale per row (exact), plus the
    sparse second term of the elements that do not convert exactly."""
    if (
        not tc_enabled("bf16") or weight.dtype != torch.bfloat16 or weight.dim() != 2
        or weight.stride(-1) != 1
        or not weight.is_cuda or _capturing() or lookup(weight) is not None
    ):
        return False
    n = weight.shape[0]
    w16 = torch.zeros(-(-n // 8) * 8, weight.shape[1], dtype=torch.float16, device=weight.device)
    winv = torch.empty(n, dtype=torch.float32, device=weight.device)
    bads = []
    with torch.no_grad():
        for r0 in range(0, n, 8192):  # row chunks: bounded temporaries (lm_head: 152k rows)
            w = weight[r0 : r0 + 8192].float()
            bits = w.view(torch.int32) & 0x7FFFFFFF
            mx = bits.amax(1)
            e = (mx >> 23) - 127
            if bool((bits >= 0x7F800000).any()) or bool(
                ((mx != 0) & ((e < E_LO) | (e > E_HI))).any()
            ):
                return False
            s = torch.where(mx == 0, torch.zeros_like(e), 15 - e).clamp(-100, 100)
            p2 = ((s + 127) << 23).view(torch.float32)
            inv = ((127 - s) << 23).view(torch.float32)
            h = (w * p2[:, None]).to(torch.float16)
            ex = (bits == 0) | ((bits >= 0x00800000) & (h.float() * inv[:, None] == w))
            bad = (~ex).nonzero()
            if bad.shape[0]:
                bads.append((bad[:, 0] + r0, bad[:, 1], w[bad[:, 0], bad[:, 1]]))
            w16[r0 : r0 + w.shape[0]] = torch.where(ex, h, torch.zeros_like(h))
            winv[r0 : r0 + w.shape[0]] = inv
        nbad = sum(b[0].numel() for b in bads)
        if nbad > MAX_BAD_ELEMS:
            logger.warning("Olala Volta TC: %s keeps the stock path (%d non-exact weights)",
                           name, nbad)
            return False
        if bads:
            bn = torch.cat([b[0] for b in bads]).to(torch.int32).contiguous()
            bk = torch.cat([b[1] for b in bads]).to(torch.int32).contiguous()
            bv = torch.cat([b[2] for b in bads]).contiguous()
        else:
            bn = bk = bv = None
    _add(_Entry(weight.detach(), w16, max(1, min(MAX_TERMS, terms)), name,
                _new_site(name, weight.device), winv, bn, bk, bv))
    return True


def unregister(weight: torch.Tensor) -> _Entry | None:
    hit = lookup(weight)
    if hit is None:
        return None
    entry = hit[0]
    lst = _REG.get(entry.base.untyped_storage().data_ptr(), [])
    if entry in lst:
        lst.remove(entry)
    _HITS.clear()
    return entry


def concat_registered(parts: list[torch.Tensor], buf: torch.Tensor) -> None:
    """``buf`` (INT8) now holds ``parts`` row-concatenated (cat_int8_rows_shared): register it
    in their place (the parts' storage is about to be released)."""
    olds = [unregister(p) for p in parts]
    if not any(olds):
        return
    terms = max(e.terms for e in olds if e is not None)
    name = "+".join(e.name for e in olds if e is not None)
    register_int8(buf, terms, name)


def lookup(w: torch.Tensor) -> tuple[_Entry, int] | None:
    """(entry, first row) of a registered weight or of a row range of one, else None."""
    key = _key(w)
    if key in _HITS:
        return _HITS[key]
    hit = None
    for e in _REG.get(key[0], ()):
        b = e.base
        if (
            w.dtype != b.dtype or w.dim() != 2 or w.shape[1] != b.shape[1]
            or w.stride() != b.stride()
        ):
            continue
        d = w.storage_offset() - b.storage_offset()
        if d % b.stride(0):
            continue
        r = d // b.stride(0)
        if 0 <= r and r + w.shape[0] <= b.shape[0]:
            hit = (e, r)
            break
    _HITS[key] = hit
    return hit


# -- compute -------------------------------------------------------------------------------


def _epilogue_cfg(n: int) -> tuple[int, int, int]:
    """(BN, BK of the fallback, warps)."""
    if n <= 512:
        return 128, 64, 4
    return 256, 32, 4


def tc_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """``x @ (scale * weight).T`` (INT8 weight, per-channel ``scale``) or ``x @ weight.T`` (bf16
    weight, ``scale`` None) on the fp16 tensor cores for bf16 ``x [M, K]`` (unit inner stride),
    into ``out`` (an (M, N) view with unit column stride) or a new ``out_dtype`` tensor (x's
    dtype by default). None when ``weight`` is not registered: the caller keeps its path."""
    hit = lookup(weight)
    if hit is None or x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(-1) != 1:
        return None
    e, r = hit
    int8 = weight.dtype == torch.int8
    if int8 != (scale is not None) or (int8 and scale.dim() != 1):
        return None
    if x.shape[0] < min_rows(e.terms, not int8):
        return None
    M, K = x.shape
    N = weight.shape[0]
    full = r == 0 and N == e.base.shape[0]
    nbad = 0 if int8 or e.bad_n is None else e.bad_n.numel()
    if nbad and not full:
        return None  # a row range of a bf16 weight with a sparse term: not supported
    y = out if out is not None else torch.empty(M, N, dtype=out_dtype or x.dtype, device=x.device)
    if M == 0:
        return y
    w16 = e.w16 if full else e.w16[r : r + N]
    T = e.terms
    dev = x.device
    h = torch.empty(T * M, K, dtype=torch.float16, device=dev)
    sh = torch.empty(T, M, dtype=torch.int32, device=dev)
    ok = torch.empty(M, dtype=torch.int32, device=dev)
    stats = _STATS is not None and e.site >= 0
    st = _STATS[e.site] if stats else ok
    bk = triton.next_power_of_2(K)
    _tc_split_kernel[(M,)](
        x, h, sh, ok, st, M, K, x.stride(0),
        T=T, BK=bk, STATS=stats, E_MIN=E_LO, E_MAX=E_HI,
        num_warps=4 if bk <= 2048 else (8 if bk <= 4096 else 16),
    )
    acc = torch.mm(h, w16.t(), out_dtype=torch.float32)
    bn, bkf, nw = _epilogue_cfg(N)
    winv = None if int8 else (e.winv if full else e.winv[r : r + N])
    dummy = ok
    _tc_epilogue_kernel[(M, triton.cdiv(N, bn))](
        acc, sh, ok,
        scale if int8 else dummy,
        winv if winv is not None else dummy,
        e.bad_n if nbad else dummy, e.bad_k if nbad else dummy, e.bad_v if nbad else dummy, nbad,
        x, weight, y, M, N, K, acc.stride(0), x.stride(0), weight.stride(0), y.stride(0),
        T=T, SCALE=int8, WINV=not int8, BN=bn, BK=bkf, num_warps=nw,
    )
    return y


_lazy_warned = False


def bf16_linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    """``x @ weight.T`` for a bf16 weight on the tensor cores, or None (the caller keeps its
    path). Weights not registered by prepare_bf16 are registered here outside graph capture."""
    global _lazy_warned
    if not tc_enabled("bf16") or weight.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        return None
    if x.dim() != 2 or x.shape[0] < min(_bf16_rows_cfg()):
        return None
    if lookup(weight) is None:
        if _capturing() or not register_bf16(weight, 1, f"lazy {tuple(weight.shape)}"):
            return None
        if not _lazy_warned:
            _lazy_warned = True
            logger.warning("Olala Volta TC: bf16 weight %s registered at its first call",
                           tuple(weight.shape))
    return tc_linear(x, weight, out=out)


def dense_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` (bf16, no bias), on the fp16 tensor cores when OLALA_VOLTA_TC
    applies."""
    y = bf16_linear(x, weight) if tc_enabled("bf16") else None
    return y if y is not None else torch.nn.functional.linear(x, weight)


def prepare_bf16(model: torch.nn.Module) -> None:
    """Register the bf16 weights the Olala layers send to the tensor cores (at the end of
    loading: before the profiling run sizes the KV cache, and before graph capture), with the
    exact activation terms each needs: the block gate / output projections (Mamba-3 output
    rows: 3 terms), the Mamba-3 in_proj caches, the DiffTPA input projections, the latent MoE's
    [fc1 ; shared up] and [fc2 | shared down] (the ReLU² half: 2 terms), the lm_head."""
    from vllm.model_executor.layers.mamba.olala.diff_tpa import OlalaDiffTPAAttention
    from vllm.model_executor.layers.mamba.olala.mamba3 import OlalaMamba3Mixer
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm.model_executor.models.olala import OlalaLatentMoE, OlalaMonoBlock

    if not tc_enabled("bf16"):
        return
    for name, mod in model.named_modules():
        if isinstance(mod, OlalaMonoBlock):
            if mod.use_gate and mod._plain_gate_proj:
                register_bf16(mod.gate_proj.weight, 1, f"{name}.gate_proj")
            if mod._plain_mixer_proj:
                t = getattr(mod.mixer_proj, "olala_tc_terms", 1)
                register_bf16(mod.mixer_proj.weight, t, f"{name}.mixer_proj")
        elif isinstance(mod, OlalaMamba3Mixer):
            if mod._in_proj_cat_ok:
                register_bf16(mod._in_proj_cat_weight(), 1, f"{name}.in_proj+dyn")
                dev = mod.in_proj.weight.device
                if mod._hybrid_ok and mod._pf_split_ok(dev):
                    register_bf16(mod._pf_in_weight(), 1, f"{name}.in_proj prefill")
        elif isinstance(mod, OlalaDiffTPAAttention):
            if mod._proj_concat_ok:
                register_bf16(mod._decode_proj_weight(), 1, f"{name}.proj")
        elif isinstance(mod, OlalaLatentMoE):
            if not mod._int8 and mod._fused_proj:
                w_in, w_out = mod._small_path_weights()
                register_bf16(w_in, 1, f"{name}.w_in")
                register_bf16(w_out, 2, f"{name}.w_out")
    head = getattr(model, "lm_head", None)
    if (
        head is not None
        and isinstance(getattr(head, "quant_method", None), UnquantizedEmbeddingMethod)
        and head.weight.dtype == torch.bfloat16
    ):
        register_bf16(head.weight, 1, "lm_head")


def stats() -> dict[str, list[int]]:
    """Rows per linear: [zeros, 1 term, 2 terms, 3 terms, fallback] (OLALA_VOLTA_TC_STATS=1)."""
    if _STATS is None:
        return {}
    vals = _STATS[: len(_SITES)].tolist()
    out: dict[str, list[int]] = {}
    for name, v in zip(_SITES, vals):
        cur = out.setdefault(name, [0] * _NSTAT)
        out[name] = [a + b for a, b in zip(cur, v)]
    return out


# -- routed experts (kind "moe") --------------------------------------------------------------


@functools.cache
def _moe_ext():
    """csrc/moe_tc_volta.cu (hand-written mma.m8n8k4: volta-forever passes the fp16 product through
    to Volta's HMMA), or None if its JIT build fails."""
    from vllm.model_executor.layers.mamba.olala.arch import gencode, sm_arch
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    try:
        return load_extension(
            f"olala_moe_tc_volta_sm{sm_arch()}", "moe_tc_volta.cu", ["-O3", "-std=c++20", gencode()]
        )
    except Exception as e:  # noqa: BLE001 - any build failure keeps the stock kernels
        logger.warning("Olala Volta TC: expert kernel not built (%s); stock kernels instead.", e)
        return None


@functools.cache
def _moe_terms() -> tuple[int, int]:
    """Exact fp16 terms of the up input (the latent rows) and the down input (ReLU² rows).
    OLALA_VOLTA_TC_MOE_TERMS="up,down" overrides."""
    raw = os.environ.get("OLALA_VOLTA_TC_MOE_TERMS")
    if raw:
        a, b = (int(v) for v in raw.split(","))
        return max(1, min(2, a)), max(1, min(2, b))
    return 1, 2


# Rows per sorted block: 16 (two m8n8k4 row tiles share each weight fragment, which halves the
# per-product cost of the weight conversion) from 33 tokens, i.e. wherever this path runs; 8 below.
# V100 through volta-forever, whole routed-expert call at 128 tokens, 8 -> 16 rows: INT4 495 ->
# 370 us, INT8 452 -> 368 us. OLALA_VOLTA_TC_MOE_BM16="tokens" overrides (measurements).
MOE_BM16_MIN_TOKENS = int(os.environ.get("OLALA_VOLTA_TC_MOE_BM16", "33"))
_MOE_SITES: dict[str, int] = {}


def moe_ok(w13, w2) -> bool:
    """The routed experts can take the tensor-core kernel: INT8 per row (FMT 1) or INT4 affine g32
    (FMT 3) on both projections, K % 128 == 0, K <= 1024 and N % 128 == 0."""
    if not tc_enabled("moe"):
        return False
    for w in (w13, w2):
        if (
            w.fmt not in (1, 3) or w.k % 128 or w.k > 1024 or w.n % 128
            or (w.fmt == 3 and w.group != 32)
        ):
            return False
    return _moe_ext() is not None


def _moe_split(x: torch.Tensor, terms: int, name: str):
    M, K = x.shape
    h = torch.empty(terms * M, K, dtype=torch.float16, device=x.device)
    sh = torch.empty(terms, M, dtype=torch.int32, device=x.device)
    ok = torch.empty(M, dtype=torch.int32, device=x.device)
    site = -1
    if _stats_on():
        if name not in _MOE_SITES:
            if _capturing():
                site = -1
            else:
                _MOE_SITES[name] = _new_site(name, x.device)
        site = _MOE_SITES.get(name, -1)
    stats = _STATS is not None and site >= 0
    bk = triton.next_power_of_2(K)
    _tc_split_kernel[(M,)](
        x, h, sh, ok, _STATS[site] if stats else ok, M, K, x.stride(0),
        T=terms, BK=bk, STATS=stats, E_MIN=E_LO, E_MAX=E_HI,
        num_warps=4 if bk <= 2048 else 8,
    )
    return h, sh, ok


@triton.jit(do_not_specialize=["M", "stride_xm"])
def _xsum_kernel(x_ptr, xs_ptr, M, K, stride_xm, BK: tl.constexpr):
    """xs[m, g] = sum of x[m, 32g .. 32g + 31] in fp32 (the INT4 experts' bias term)."""
    m = tl.program_id(0)
    offs = tl.arange(0, BK)
    v = tl.load(x_ptr + m.to(tl.int64) * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
    g = tl.sum(tl.reshape(v, [BK // 32, 32]), axis=1)
    og = tl.arange(0, BK // 32)
    tl.store(xs_ptr + m.to(tl.int64) * (K // 32) + og, g, mask=og < K // 32)


def _xsum(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    xs = torch.empty(M, K // 32, dtype=torch.float32, device=x.device)
    _xsum_kernel[(M,)](x, xs, M, K, x.stride(0), BK=triton.next_power_of_2(K), num_warps=4)
    return xs


def moe_experts(
    x: torch.Tensor,  # [M, D1] bf16 latent rows
    ids: torch.Tensor,  # [M, top_k] int32 experts
    tw: torch.Tensor,  # [M * top_k] fp32 routing weights
    w13,
    w2,
    top_k: int,
    routed_scale: float,
) -> torch.Tensor:
    """The routed experts of latent_moe_int8.wq_routed_experts after routing, on the tensor cores:
    grouped up GEMM + ReLU², grouped down GEMM weighted by the routing weights (fp32), top-k sum
    times the routed scale. Returns the routed latent [M, D2] bf16."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import _topk_sum_kernel

    ext = _moe_ext()
    M, D1 = x.shape
    E, N1, D2 = w13.weight.shape[0], w13.n, w2.n
    MK = M * top_k
    mt = 2 if M >= MOE_BM16_MIN_TOKENS else 1
    sorted_ids, expert_ids, ntp = moe_align_block_size(ids, 8 * mt, E)
    t_up, t_down = _moe_terms()
    if x.stride(-1) != 1 or x.stride(0) != D1:
        x = x.contiguous()
    hx, shx, okx = _moe_split(x, t_up, "moe up input")
    xs = _xsum(x) if w13.fmt == 3 else okx
    h = torch.empty(MK, N1, dtype=torch.bfloat16, device=x.device)
    ext.moe_tc(hx, shx, okx, x, xs, w13.weight, w13.scale, h, tw, sorted_ids, expert_ids, ntp,
               MK, N1, D1, top_k, w13.fmt, t_up, True, mt)
    hh, shh, okh = _moe_split(h, t_down, "moe down input")
    xs = _xsum(h) if w2.fmt == 3 else okh
    part = torch.empty(MK, D2, dtype=torch.float32, device=x.device)
    ext.moe_tc(hh, shh, okh, h, xs, w2.weight, w2.scale, part, tw, sorted_ids, expert_ids, ntp,
               MK, D2, N1, top_k, w2.fmt, t_down, False, mt)
    out = torch.empty(M, D2, dtype=torch.bfloat16, device=x.device)
    BR = 8
    _topk_sum_kernel[(triton.cdiv(M, BR), D2 // 128)](
        part, out, routed_scale, M, D=D2, K=top_k, BR=BR, BD=128, num_warps=4
    )
    return out

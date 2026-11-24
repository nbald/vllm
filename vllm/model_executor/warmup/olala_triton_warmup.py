# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the Triton kernels that Olala's first requests would otherwise compile.

Triton compiles a kernel once per specialization: the 16-divisibility of its integer arguments
(the token count) and the 16-byte alignment of its pointers (slices of the batch layout, the
prefill slots after the decodes, ...), plus its constexprs. A specialization first met by a
serving batch compiles while the batch waits. On a V100 through volta-forever a cold compile and
translation costs 0.4-1.5 s: a 128-prompt burst on a fresh Triton cache spent 3.3 s of 19.5 s in
them, and the first 8-request burst after a start took 8.8 s instead of 4.3 s. On an RTX 5090 the
first requests of a new configuration stalled 0.6-1.9 s per compile. The profiling and
graph-capture runs meet only a few of these classes. The kernels:
  - the Mamba-3 hybrid prefill: prologue, ``_ssd_prep``, ``_ssd_state_pass_pool``,
    ``_ssd_pool_tail``, and ReplaySSM's reset of the prefilled slots
    (OlalaMamba3Mixer.warmup_prefill_kernels);
  - the TPA-factor prefill: ``_tpa_rows_kernel``, ``_tpa_pools_kernel``, ``_tpa_q_kernel``,
    ``_reconstruct_dense_kernel2``;
  - the lm_head kernels of compute_logits, which runs outside the CUDA graphs on the unpadded
    rows (e.g. ``_gemv_rows_many_kernel``), and the logprob kernels of the sampler and of prompt
    logprobs (``_topk_log_softmax_kernel`` per number of requested logprobs, ``_ranks_kernel``).
Each runs once per class on small dummy batches through the code the forward pass uses, on
dummy caches with the real caches' strides: no KV cache or output is touched. With a warm Triton
cache the warmup costs one module load per specialization.
"""

import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _dummy_like(t: torch.Tensor, n: int = 2) -> torch.Tensor:
    """Zeros with ``t``'s strides and dtype, ``n`` entries along dim 0 (blocks / slots)."""
    return torch.empty_strided((n, *t.shape[1:]), t.stride(), dtype=t.dtype, device=t.device).zero_()


def _factor_block_table(runner, prefix: str) -> torch.Tensor | None:
    """The block table the forward pass slices for the factor-cache group (its row stride is a
    kernel argument)."""
    groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None) or []
    gid = next((i for i, g in enumerate(groups) if prefix in g.layer_names), None)
    if gid is None:
        return None
    tables = getattr(getattr(runner, "block_tables", None), "input_block_tables", None)
    if tables is not None:  # model runner V2
        return tables[gid]
    bt = getattr(getattr(runner, "input_batch", None), "block_table", None)
    try:  # model runner V1
        return bt[gid].get_device_tensor(bt[gid].max_num_reqs)
    except Exception:  # noqa: BLE001 - unknown runner: no TPA warmup
        return None


def _tpa_projections(layer, h: torch.Tensor):
    """q, A_k, A_v, B_k, B_v and the shift alphas as OlalaDiffTPAAttention._forward_impl and
    _forward_factor compute them (views of the same buffers, same strides)."""
    from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
    from vllm.model_executor.layers.mamba.olala.decode_gemv import SKINNY_MAX_M, decode_gemv

    concat = os.environ.get("OLALA_TPA_CONCAT", "1") != "0"
    shift = None
    if layer._int8_concat_ok and concat:
        parts = layer._int8_proj(h)
        shift = tuple(parts[6:]) if layer.token_shift else None
    elif layer._proj_concat_ok and concat:
        w_all = layer._decode_proj_weight()
        if h.shape[0] <= SKINNY_MAX_M and gpu_family() is not GpuFamily.SM100:
            allp = decode_gemv(h, w_all)
        else:
            allp = F.linear(h, layer._decode_proj_w_padded)[:, : w_all.shape[0]]
        parts = torch.split(allp, layer._decode_proj_sizes, dim=-1)
        shift = tuple(parts[6:]) if layer.token_shift else None
    else:
        parts = [m(h)[0] for m in (layer.c_q, layer.W_A_k, layer.W_A_v, layer.W_B_k, layer.W_B_v)]
    if shift is None:
        shift = (layer.shift_proj_k(h)[0], layer.shift_proj_v(h)[0])
    return (*parts[:5], *shift)


@torch.inference_mode()
def _warm_tpa_prefill(runner, layer, max_tokens: int) -> int:
    """OlalaDiffTPAAttention._factor_prefill (rows, pools, q, dense reconstruction, attention) per
    class of its kernels' arguments: the decode count's parity (positions and slot-mapping slices,
    int64), the prefill request count mod 4 (layout slices), initial states or not, and the batch
    size regimes of the input projections (their output strides). Dummy factor planes and shift
    pools with the real strides; slot -1 (no row written to the planes)."""
    bt = _factor_block_table(runner, layer.factor_cache.prefix)
    if bt is None:
        logger.info("Olala TPA prefill warmup skipped: factor-cache block table not found.")
        return 0
    bt = torch.zeros_like(bt)  # the real table's strides; every block id 0 (a dummy plane block)
    kv = _dummy_like(layer.factor_cache.kv_cache)
    if kv.shape[0] != 2 or kv.dim() != 4 or kv.shape[1] == 1:  # block-major (as _forward_factor)
        nb, bs = kv.shape[0], kv.shape[-2]
        kv = kv.reshape(nb, 2, bs, kv.shape[-1] // 2).transpose(0, 1)
    kplane, vplane = kv[0], kv[1]
    # The token-shift pools, where _ts_kv reads them: the layer's own state, or (OLALA_TS_FOLD) views of the
    # page of its Mamba3 host. Dummies with the same strides take their place for the warmup.
    shift = layer.shift_state
    host = getattr(layer, "_ts_host", None) if shift.folded else None
    if shift.folded and host is None:
        logger.info("Olala TPA prefill warmup skipped: folded token shift without its Mamba3 host.")
        return 0
    real_pools = host.folded_token_shift_pools() if host is not None else shift.kv_cache
    dummies = tuple(_dummy_like(p) for p in real_pools)
    dev, dtype = kplane.device, kplane.dtype
    hidden = layer.c_q.input_size
    md = SimpleNamespace(block_table=bt)
    n = 0
    if host is not None:
        host.folded_token_shift_pools = lambda: dummies  # instance attribute: shadows the method
    else:
        shift.kv_cache = list(dummies)
    try:
        for total in (6, 200, 1100):  # decode kernels / mid / prefill GEMM regimes of the projections
            for ndt in (0, 1):
                for nreq in (1, 2, 3, 4):
                    npt = total - ndt
                    if npt < nreq or total > max_tokens or ndt + nreq > bt.shape[0]:
                        continue
                    lens = [npt // nreq] * nreq
                    lens[-1] += npt - sum(lens)
                    qsl = [0]
                    for length in lens:
                        qsl.append(qsl[-1] + length)
                    h = torch.zeros(total, hidden, dtype=dtype, device=dev)
                    proj = _tpa_projections(layer, h)
                    positions = torch.zeros(total, dtype=torch.int64, device=dev)
                    slots = torch.full((total,), -1, dtype=torch.int64, device=dev)
                    rows = slice(ndt, ndt + npt)
                    for init in (False, True):
                        ctx = 16 if init else 0
                        reqs = (nreq, qsl, [length + ctx for length in lens], [init] * nreq,
                                [r % 2 for r in range(nreq)], ndt)
                        layer._factor_prefill(md, None, positions[rows], *(p[rows] for p in proj),
                                              slots[rows], kplane, vplane, reqs=reqs)
                        n += 1
    finally:
        if host is not None:
            del host.folded_token_shift_pools
        else:
            shift.kv_cache = real_pools
    return n


@torch.inference_mode()
def _warm_logits(runner, model) -> int:
    """compute_logits at every row count a step can sample (it runs outside the CUDA graphs on the
    unpadded rows: the lm_head GEMV / GEMM kernels specialize on the row count), then
    compute_topk_scores (the sampler's and prompt logprobs' fast path) for every number of
    requested logprobs up to max_logprobs, on both row regimes of compute_logits."""
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

    vc = runner.vllm_config
    mc = vc.model_config
    max_lp = getattr(mc, "max_logprobs", 20)
    if max_lp is None or max_lp < 0:
        max_lp = 20
    dev = next(model.parameters()).device
    rows_max = max(64, vc.scheduler_config.max_num_seqs)
    h = torch.zeros(rows_max, mc.get_hidden_size(), dtype=mc.dtype, device=dev)
    n = 0
    for rows in range(1, rows_max + 1):
        logits = model.compute_logits(h[:rows])
        n += 1
        if logits is None or rows not in (1, 64):
            continue
        sampled = torch.zeros(rows, dtype=torch.int64, device=dev)
        for k in range(max_lp + 1):
            compute_topk_scores(logits, k, sampled)
            n += 1
    return n


def olala_triton_warmup(runner: "GPUModelRunner") -> None:
    """Compile the Olala prefill, logits and logprob Triton specializations before serving, on every
    CUDA GPU (OLALA_TRITON_WARMUP=0 skips it), direct or speculative. Every Mamba-3 (TPA) layer has
    the same shapes and kernels, so one layer of each kind covers them all."""
    if runner.device.type != "cuda" or os.environ.get("OLALA_TRITON_WARMUP", "1") == "0":
        return
    from vllm.model_executor.layers.mamba.olala.diff_tpa import OlalaDiffTPAAttention
    from vllm.model_executor.layers.mamba.olala.mamba3 import OlalaMamba3Mixer

    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    mamba = next((m for m in model.modules() if isinstance(m, OlalaMamba3Mixer)), None)
    tpa = next(
        (m for m in model.modules()
         if isinstance(m, OlalaDiffTPAAttention) and m.factor_mode and m.factor_cache is not None),
        None,
    )
    if mamba is None and tpa is None:
        return
    max_tokens = runner.vllm_config.scheduler_config.max_num_batched_tokens
    t0 = time.perf_counter()
    counts = {}
    if mamba is not None and getattr(mamba, "kv_cache", None):
        counts["mamba3 prefill"] = mamba.warmup_prefill_kernels(max_tokens)
    if tpa is not None:
        counts["tpa prefill"] = _warm_tpa_prefill(runner, tpa, max_tokens)
    counts["logits and logprobs"] = _warm_logits(runner, model)
    torch.cuda.synchronize()
    logger.info(
        "Warmed the Olala prefill, logits and logprob Triton kernels (%s calls) in %.2f s.",
        ", ".join(f"{k}: {v}" for k, v in counts.items()),
        time.perf_counter() - t0,
    )

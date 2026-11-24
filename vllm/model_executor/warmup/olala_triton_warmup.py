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
  - with a DFlash / DSpine drafter, DFlash's input preparation per span class;
  - the routed experts of an INT8 / INT4 checkpoint's prefill above the medium batch, at both token-count classes
    (``_route_topk_kernel``: eager steps above the PIECEWISE graphs' cap take any count);
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


def _int_class_size(n: int) -> int:
    """A dim-0 size Triton specializes like ``n`` (1, a multiple of 16, other): dummy pools keep the
    real pools' class for the kernels that take their slot count as an argument."""
    return 1 if n == 1 else (16 if n % 16 == 0 else 2)


@torch.inference_mode()
def _warm_tpa_verify(runner, layer, max_tokens: int) -> int:
    """F1's verify write (factor_decode_write_spec, OLALA_TPA_FACTOR_SPEC) specializes on its request
    count ``lanes`` and row count ``N`` (1, a multiple of 16, other), and the projections' output
    strides follow the step's row count (decode GEMV / GEMM regimes). Uniform CUDA-graph batches and
    the warm-up's steps meet only some of those classes: replaying the V100's dynamic-K schedule on
    an RTX 5090, a padded verify batch (lanes % 16 != 0, N % 16 == 0) compiled it while serving. One
    call per (lanes, N) class and projection regime, in this server's spec layout (light columns or
    column slots), on dummy token-shift pools and factor planes with the real strides (slot class
    kept), the real spec column table, and slot -1 for the factor rows."""
    try:
        from vllm.model_executor.layers.mamba.olala.tpa_factor import (
            factor_decode_write_spec,
            factor_spec_enabled,
        )
    except ImportError:
        return 0
    if (
        runner.vllm_config.speculative_config is None
        or not getattr(layer, "factor_mode", False)
        or not factor_spec_enabled()
        or not hasattr(layer, "_factor_verify")
    ):
        return 0
    cols_all = _factor_block_table(runner, layer._ts_md_key)
    if cols_all is None:
        logger.info("Olala TPA verify warmup skipped: token-shift block table not found.")
        return 0
    kv = layer.factor_cache.kv_cache
    kv = _dummy_like(kv, _int_class_size(kv.shape[0]))
    if kv.shape[0] != 2 or kv.dim() != 4 or kv.shape[1] == 1:  # block-major (as _forward_factor)
        nb, bs = kv.shape[0], kv.shape[-2]
        kv = kv.reshape(nb, 2, bs, kv.shape[-1] // 2).transpose(0, 1)
    kplane, vplane = kv[0], kv[1]
    shift = layer.shift_state
    host = getattr(layer, "_ts_host", None) if shift.folded else None
    if shift.folded and host is None:
        return 0
    real_pools = host.folded_token_shift_pools() if host is not None else shift.kv_cache
    dummies = tuple(_dummy_like(p, _int_class_size(p.shape[0])) for p in real_pools)
    dev, dtype = kplane.device, kplane.dtype
    num_spec = int(runner.vllm_config.speculative_config.num_speculative_tokens or 0)
    if num_spec <= 0:
        return 0
    qlen = 1 + num_spec  # a full verify per request; shorter requests change no argument class
    max_lanes = min(runner.max_num_reqs, cols_all.shape[0])
    wsize = float(getattr(layer.config, "sliding_window_size", 0) or 0)
    i32 = dict(dtype=torch.int32, device=dev)
    n = 0
    if host is not None:
        host.folded_token_shift_pools = lambda: dummies  # instance attribute: shadows the method
    else:
        shift.kv_cache = list(dummies)
    try:
        k_cols, v_cols, ak_cols, av_cols = layer._ts_kv(spec=True)
        cases = set()
        for lanes in sorted({1, 2, 16, max_lanes - 1, max_lanes}):
            if lanes < 1 or lanes > max_lanes:
                continue
            rows = lanes * qlen
            cases.add((lanes, rows + (-rows) % 16))  # N % 16 == 0 (padding rows past the requests)
            cases.add((lanes, rows if rows % 16 else rows + 1))  # N % 16 != 0
            if lanes == 1:
                cases.add((1, 1))  # a lone one-token verify (no draft)
        for lanes, N in sorted(cases):
            for total in sorted({N, 200, 1100}):  # projection regimes of the step's row count
                if total < N or total > max_tokens:
                    continue
                proj = _tpa_projections(layer, torch.zeros(total, layer.c_q.input_size, dtype=dtype, device=dev))
                per = min(qlen, max(1, N // lanes))
                qsl = torch.tensor([min(i * per, N) for i in range(lanes + 1)], **i32)
                factor_decode_write_spec(
                    *(p[:N] for p in proj), torch.zeros(N, dtype=torch.int64, device=dev),
                    qsl, torch.zeros_like(cols_all)[:lanes, :qlen], torch.ones(lanes, **i32),
                    torch.full((N,), -1, dtype=torch.int64, device=dev),
                    k_cols, v_cols, ak_cols, av_cols, layer.q_norm.norm.weight, layer.k_norm.norm.weight,
                    layer.q_norm.norm.eps, layer.softmax_scaler, wsize, kplane, vplane, layer.snr + 1,
                )
                n += 1
    finally:
        if host is not None:
            del host.folded_token_shift_pools
        else:
            shift.kv_cache = real_pools
    torch.cuda.synchronize()
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


@torch.inference_mode()
def _warm_dflash_inputs(runner) -> int:
    """DFlash's _prepare_dflash_inputs_kernel (spec_decode/dflash/speculator.py) specializes on BLOCK_SIZE, the
    next power of two of the step's longest request span (query + draft slots, at most 256): the graph-capture runs
    meet the decode spans only, so the first prefills of each larger class compiled while serving (0.44-0.46 s each
    on an RTX 5090). One call per class, for one dummy request, through prepare_dflash_inputs on the speculator's
    own buffers (the prepared inputs and slot mappings, which every step rewrites); the per-request inputs are
    dummies of the runner's dtypes."""
    from types import SimpleNamespace as NS

    import numpy as np

    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator, prepare_dflash_inputs

    spec = getattr(runner, "speculator", None)
    if not isinstance(spec, DFlashSpeculator) or not getattr(spec, "draft_kv_cache_group_ids", None):
        return 0
    dev = runner.device
    nq = spec.num_query_per_req
    i32 = dict(dtype=torch.int32, device=dev)
    n = 0
    for bs in (8, 16, 32, 64, 128, 256):
        q = bs - nq if bs > nq else 1  # the longest query of the step: next_power_of_2(q + nq) == bs
        if q + nq > spec.max_num_tokens:
            break
        batch = NS(num_reqs=1, num_scheduled_tokens=np.array([q], dtype=np.int32),
                   positions=torch.zeros(spec.max_num_tokens, dtype=torch.int64, device=dev),
                   query_start_loc=torch.tensor([0, q], **i32), idx_mapping=torch.zeros(1, dtype=torch.int64, device=dev))
        for i, gid in enumerate(spec.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                spec.input_buffers, spec.block_tables.slot_mappings[gid], spec.context_positions,
                spec._context_slot_mappings[i], spec.sample_indices, spec.sample_pos, spec.sample_idx_mapping,
                spec.temperature, spec.seeds, batch, torch.ones(1, **i32), torch.zeros(1, **i32),
                runner.req_states.last_sampled_tokens, runner.req_states.next_prefill_tokens,
                runner.sampler.sampling_states.temperature.gpu, runner.sampler.sampling_states.seeds.gpu,
                spec.block_tables.input_block_tables[gid], spec.block_tables.block_sizes[gid],
                spec.block_tables.cp_rank, spec.dcp_size, spec.block_tables.cp_interleave,
                spec.parallel_drafting_token_id, nq, spec.num_speculative_steps, spec.max_num_reqs,
                spec.max_num_tokens, spec.max_model_len, spec.sample_from_anchor, 1,
            )  # fmt: skip
            n += 1
    return n


@torch.inference_mode()
def _warm_dflash_context_kv(runner) -> int:
    """DFlash stores the context K/V of the target rows a FULL draft graph does not cover eagerly, through
    _precompute_context_kv(num_context, num_tokens), with num_context = requests x (1 + K) (speculator.py). With an
    odd 1 + K, e.g. K = 2 in a dynamic-K schedule, an odd request count starts those int64 slot-mapping and position
    slices off a 16-byte boundary: a Triton specialization the graph captures (which start at 0) never meet. A V100
    (K = 2, [[1, 2, 2], [3, 6, 0], [7, 16, 2]]) compiled the fused context store (_ctx_kv_store_kernel,
    OLALA_DSPINE_FUSED=3, at most OLALA_DSPINE_FUSED_MAX_ROWS rows) and the TRITON_ATTN cache write
    (reshape_and_cache_kernel_flash, more rows) while serving. Warmed here at both alignments and both row counts,
    through the speculator's own call, on PAD slots (no K/V is written; every step rewrites the slot mappings and
    positions)."""
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    spec = getattr(runner, "speculator", None)
    if not isinstance(spec, DFlashSpeculator) or not hasattr(spec, "_precompute_context_kv"):
        return 0
    try:
        from vllm.model_executor.models.qwen3_dspine_fast import HAND_MAX_ROWS
    except ImportError:
        HAND_MAX_ROWS = 64
    n = 0
    for rows in (2, HAND_MAX_ROWS + 2):
        for start in (0, 1):
            end = start + rows
            if end > spec.max_num_tokens:
                continue
            spec._context_slot_mappings[:, :end].fill_(PAD_SLOT_ID)
            spec.context_positions[:end].zero_()
            spec._precompute_context_kv(start, end)
            n += 1
    torch.cuda.synchronize()
    return n


@torch.inference_mode()
def _warm_moe_prefill(model) -> int:
    """The routed experts of an INT8 / INT4 checkpoint's prefill (latent_moe_int8.wq_routed_experts) at the two
    token-count classes of its steps above the medium batch. Up to the PIECEWISE graphs' cap a step is padded to a
    captured size (a multiple of 16); an eager step above it (a chunked-prefill budget past the cap, e.g. 8192 on an
    RTX 5090, or the offline LLM's 8192) can take any count, and compiled ``_route_topk_kernel`` while serving."""
    from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import wq_routed_experts
    from vllm.model_executor.layers.mamba.olala.latent_moe_medium import MEDIUM_BATCH_MAX_TOKENS
    from vllm.model_executor.models.olala import OlalaLatentMoE

    moe = next((m for m in model.modules() if isinstance(m, OlalaLatentMoE) and m._int8), None)
    if moe is None:
        return 0
    weights = moe._int8_weights()
    if not moe._int8_own_prefill(weights[0]):
        return 0
    bias = moe.gate.e_score_correction_bias
    n = 0
    for m in (MEDIUM_BATCH_MAX_TOKENS + 16, MEDIUM_BATCH_MAX_TOKENS + 17):
        latent = torch.zeros(m, moe.latent_size, dtype=torch.bfloat16, device=bias.device)
        logits = torch.zeros(m, moe.num_experts, dtype=torch.float32, device=bias.device)
        wq_routed_experts(latent, logits, bias, weights[0], weights[1], moe.top_k, moe.routed_scaling_factor)
        n += 1
    torch.cuda.synchronize()
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
        verify = _warm_tpa_verify(runner, tpa, max_tokens)
        if verify:
            counts["tpa verify"] = verify
    counts["logits and logprobs"] = _warm_logits(runner, model)
    moe_prefill = _warm_moe_prefill(model)
    if moe_prefill:
        counts["moe prefill"] = moe_prefill
    if runner.vllm_config.speculative_config is not None:
        counts["dflash inputs"] = _warm_dflash_inputs(runner)
        counts["dflash context kv"] = _warm_dflash_context_kv(runner)
    torch.cuda.synchronize()
    logger.info(
        "Warmed the Olala prefill, logits and logprob Triton kernels (%s calls) in %.2f s.",
        ", ".join(f"{k}: {v}" for k, v in counts.items()),
        time.perf_counter() - t0,
    )

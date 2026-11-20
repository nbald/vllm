# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 (sm_100 / sm_103) causal forward with a native ``softcap * tanh(s / softcap)``.

FA4 applies a softcap as a generic ``score_mod``: ``s * scale``, the mod's ``/ cap``, ``tanh``,
``* cap``, then the row max over the capped tile and ``exp2((s - max) * log2e)``. Here:

* the scales are folded: the tile is multiplied once by ``scale / cap`` and the softmax uses
  ``cap * log2e`` as its exp2 scale (two fewer FP32 ops per score);
* tanh is monotonic, so the row max is taken on the raw scores and only the max goes through
  ``tanh``; the max reduction no longer waits on the 128 tanh of the row, which overlap with
  it (the softmax warps are latency-bound, not MUFU-bound).

Same instructions per score as FA4's mod (one ``tanh.approx``), same results (bit-identical
outputs on the tests), 8-12% faster causal hd128 prefill attention on B300.

``OlalaSoftcapFwdSm100.softmax_step`` is FA4's ``softmax_step`` with only the score_mod / mask /
row-max block replaced. FA4 is a vendored build artifact: if its ``softmax_step`` source
changes, ``native_softcap_available()`` turns false and callers keep the score_mod path.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from functools import lru_cache

# sha256[:16] of the FA4 softmax_step source (from its def line) the override copies.
_BASE_SOFTMAX_STEP_SHA = "4e19c0823e5de7ea"
_LOG2_E = math.log2(math.e)


@lru_cache(maxsize=1)
def native_softcap_available() -> bool:
    """sm_10x and the vendored FA4 ``softmax_step`` matches the one the override copies."""
    from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family

    if gpu_family() is not GpuFamily.SM100:
        return False
    try:
        from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    except Exception:
        return False
    return _source_sha(FlashAttentionForwardSm100.softmax_step) == _BASE_SOFTMAX_STEP_SHA


def _source_sha(fn) -> str:
    # From the ``def`` line: once compiled, the DSL's wrapper drops the decorator line.
    src = inspect.getsource(inspect.unwrap(fn))
    src = src[src.index("def "):]
    return hashlib.sha256(src.encode()).hexdigest()[:16]


@lru_cache(maxsize=8)
def softcap_score_mod(softcap: float):
    """Marker score_mod for the native path. Under ``_setup``'s scale split the scores reach it
    already multiplied by ``scale / softcap`` and the softmax exp2 scale is ``softcap * log2e``,
    so ``tanh`` alone is the full softcap: also correct if FA4 applies it as a plain score_mod.
    """
    import cutlass.cute as cute

    @cute.jit
    def softcap_tanh(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return cute.math.tanh(acc_S_SSA, fastmath=True)

    softcap_tanh.__olala_softcap__ = softcap
    return softcap_tanh


@lru_cache(maxsize=8)
def softcap_prescaled_score_mod(softcap: float):
    """Marker for q already multiplied by ``scale / softcap``: the kernel skips that multiply."""
    import cutlass.cute as cute

    @cute.jit
    def softcap_tanh_prescaled(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info,
                               aux_tensors):
        return cute.math.tanh(acc_S_SSA, fastmath=True)

    softcap_tanh_prescaled.__olala_softcap__ = softcap
    softcap_tanh_prescaled.__olala_prescaled__ = True
    return softcap_tanh_prescaled


@lru_cache(maxsize=1)
def _setup():
    """FA4's interface module and our kernel class. Installs, once, a scale split that only
    differs from FA4's for the marker score_mod."""
    from vllm.vllm_flash_attn.cute import interface, utils

    from vllm.model_executor.layers.mamba.olala._fa4_softcap_kernel import OlalaSoftcapFwdSm100

    orig_scale = utils.compute_softmax_scale_log2

    def scale_split(softmax_scale, score_mod):
        cap = getattr(score_mod, "__olala_softcap__", None)
        if cap is None:
            return orig_scale(softmax_scale, score_mod)
        return cap * _LOG2_E, softmax_scale * (1.0 / cap)

    utils.compute_softmax_scale_log2 = scale_split
    return interface, OlalaSoftcapFwdSm100


def fa4_softcap_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           softmax_scale, softcap, prescaled=False):
    """Causal varlen FA4 forward with the native softcap (needs ``native_softcap_available()``).
    FA4's sm100 forward class is swapped for the call; the kernel it compiles is cached by FA4
    under the marker score_mod."""
    interface, kernel_cls = _setup()
    orig_cls = interface.FlashAttentionForwardSm100
    interface.FlashAttentionForwardSm100 = kernel_cls
    try:
        return interface._flash_attn_fwd(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
            causal=True,
            score_mod=(softcap_prescaled_score_mod if prescaled else softcap_score_mod)(float(softcap)),
        )[0]
    finally:
        interface.FlashAttentionForwardSm100 = orig_cls

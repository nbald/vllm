# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala Differential-TPA attention mixer.

Native port of ``OlalaDifferentialTensorProductAttentionV2``:

* TPA low-rank K/V reconstruction — ``key = (A_k @ B_k) / rank``, where
  ``A_k``/``A_v`` are per-head coefficients and ``B_k``/``B_v`` shared bases.
* Optional token-shift EMA mixing of the previous token's raw K/V. The
  one-token ``(k_last, v_last)`` buffer is owned by a side-channel
  ``OLALA_DIFF_TPA`` mamba backend so vLLM's state manager pages it.
* Softmax attention over vLLM's paged KV through ``Attention``.
* Differential recombination — heads split into ``snr`` signal heads plus one
  noise head per group, combined as ``sig - sigmoid(lambda) * noise``.

Not ported (the reference model's optional paths): ``token_conv1d_attn``,
``xsa``, ``intra_doc_masking``, value embeddings, per-layer sliding window,
and Olala's custom p-rope (vLLM's ``get_rope`` is used instead).
"""

from __future__ import annotations

import dataclasses
import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.olala.decode_gemv import (
    SKINNY_MAX_M,
    cat_rows_shared,
    decode_gemv,
)
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gpu_family
from vllm.model_executor.layers.mamba.olala.volta_tc import bf16_linear, tc_enabled
from vllm.model_executor.layers.mamba.olala.w8a16_layers import (
    cat_int8_rows_shared,
    is_int8_quant_method,
    mixer_quant_config,
)
from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
    Segment,
    int8_decode_linear,
    int8_decode_max_m,
    int8_prefill_mm,
)
from vllm.model_executor.layers.mamba.olala.fallback import warn_slow_path
from vllm.model_executor.layers.mamba.olala.diff_tpa_decode import (
    tpa_decode_qkv,
    tpa_decode_qkv_spec,
    tpa_diff_combine,
)
from vllm.model_executor.layers.mamba.olala.tpa_factor import (
    HKV as FACTOR_HKV,
    HQ as FACTOR_HQ,
    OlalaTPAFactorCache,
    factor_cache_enabled,
    factor_decode_attention,
    factor_decode_write,
    factor_decode_write_spec,
    factor_rows,
    factor_spec_enabled,
    num_splits_for_batch,
    num_splits_for_verify,
    reconstruct_dense_batched,
    reconstruct_dense_kv,
    write_rows,
)
from vllm.model_executor.layers.mamba.olala.tpa_prefill import (
    prefill_q_prescale,
    tpa_prefill_q,
    tpa_prefill_rows,
    tpa_varlen_attention,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.olala.norm import OlalaNorm
from vllm.model_executor.layers.mamba.olala.state import (
    spec_light_cols,
    token_shift_state_dtype,
    token_shift_state_shape,
    ts_fold_active,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.olala_diff_tpa_attn import OlalaDiffTPAMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


@triton.jit
def _token_shift_decode_kernel(
    k_ptr,
    v_ptr,  # (N, Hkv, D) current k/v
    ks_ptr,
    vs_ptr,  # (N, Hkv, D) shifted outputs
    kpool_ptr,
    vpool_ptr,  # (P, Hkv, D) one-token side pools
    ak_ptr,
    av_ptr,  # (N, Hkv) raw shift logits (pre-sigmoid)
    slots_ptr,  # (N,) int32 pool rows
    pos_ptr,  # (N,) positions; pos == 0 disables the shift
    H,
    P,
    stride_kn,
    stride_kh,
    stride_pn,
    stride_ph,
    stride_an,
    D: tl.constexpr,
):
    """Per (token, kv-head): ``shifted = sigmoid(a) * prev + (1 - sigmoid(a)) *
    cur``, then store ``cur`` as the next step's ``prev``.

    Blending is done in fp32; the eager equivalent rounded to bf16 per op.
    Cudagraph padding lanes carry ``NULL_BLOCK_ID`` (row 0, the reserved null
    block), so they read and write a row no live request owns; the clamp only
    guards against out-of-range slots in dummy capture batches.
    """
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H
    # int64: the pools are page-strided views (row stride ~4.4e5), so an int32
    # row * stride product overflows once block ids grow with uptime.
    slot = tl.load(slots_ptr + n)
    srow = tl.minimum(tl.maximum(slot, 0), P - 1).to(tl.int64)

    offs = tl.arange(0, D)
    k = tl.load(k_ptr + n * stride_kn + h * stride_kh + offs).to(tl.float32)
    v = tl.load(v_ptr + n * stride_kn + h * stride_kh + offs).to(tl.float32)
    kp = tl.load(kpool_ptr + srow * stride_pn + h * stride_ph + offs).to(tl.float32)
    vp = tl.load(vpool_ptr + srow * stride_pn + h * stride_ph + offs).to(tl.float32)
    ak = tl.sigmoid(tl.load(ak_ptr + n * stride_an + h).to(tl.float32))
    av = tl.sigmoid(tl.load(av_ptr + n * stride_an + h).to(tl.float32))
    # Document start: no previous token, so the shift is disabled.
    pos = tl.load(pos_ptr + n)
    ak = tl.where(pos == 0, 0.0, ak)
    av = tl.where(pos == 0, 0.0, av)

    tl.store(
        ks_ptr + n * stride_kn + h * stride_kh + offs,
        (ak * kp + (1.0 - ak) * k).to(ks_ptr.dtype.element_ty),
    )
    tl.store(
        vs_ptr + n * stride_kn + h * stride_kh + offs,
        (av * vp + (1.0 - av) * v).to(vs_ptr.dtype.element_ty),
    )
    tl.store(
        kpool_ptr + srow * stride_pn + h * stride_ph + offs,
        k.to(kpool_ptr.dtype.element_ty),
    )
    tl.store(
        vpool_ptr + srow * stride_pn + h * stride_ph + offs,
        v.to(vpool_ptr.dtype.element_ty),
    )



# Speculative verify batches run the fused decode preamble too
# (tpa_decode_qkv_spec); OLALA_TPA_SPEC_FUSED=0 keeps the eager per-op path.
_FUSED_SPEC = os.environ.get("OLALA_TPA_SPEC_FUSED", "1") != "0"
# OLALA_TPA_MIXED_FUSED=1 (opt-in): in a step that mixes prefill and verify rows,
# the verify rows keep the fused spec path (tpa_decode_qkv_spec, as in a pure
# verify step); only the prefill rows take the op-by-op path.
_MIXED_FUSED = os.environ.get("OLALA_TPA_MIXED_FUSED", "0") == "1"


class OlalaTokenShiftState(nn.Module, MambaBase):
    """Owner of the per-request one-token ``(k_last, v_last)`` buffer.

    Modelled as a ``MambaBase`` layer purely so vLLM's paged state manager
    allocates and shards it alongside the real recurrent states. The mixer
    never calls ``forward`` — it reads and writes ``self.kv_cache`` directly.
    """

    kv_cache: tuple[torch.Tensor, ...]

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        vllm_config: VllmConfig,
        prefix: str,
        rank: int = 0,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rank = rank  # > 0: also keep the previous token's A_k / A_v (factor cache)
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.prefix = prefix
        # Token-shift fold (state.py): the buffer lives in the page of the
        # Mamba3 layer right before this V layer, so this layer asks for no
        # KV cache of its own.
        self.folded = ts_fold_active(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config.pipeline_parallel_size,
        )
        # Speculative light columns (state.py): 1 + k buffers in one page.
        self.light_cols = spec_light_cols(vllm_config)
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        if self.folded:
            return None
        spec = super().get_kv_cache_spec(vllm_config)
        if self.light_cols and spec is not None:
            spec = dataclasses.replace(spec, num_speculative_blocks=0)
        return spec

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.OLALA_DIFF_TPA

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return token_shift_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.rank,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return token_shift_state_shape(
            tp_world_size=self.tp_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rank=self.rank,
            light_cols=self.light_cols,
        )


class OlalaDiffTPAAttention(PluggableLayer):
    """Differential tensor-product attention over vLLM's paged KV."""

    _factor_calls = 0  # Python executions of the factor path (debug: cudagraph coverage)

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        tp = get_tensor_model_parallel_world_size()
        self.tp_size = tp
        # Register under our own prefix so the vllm::olala_diff_tpa custom op
        # can resolve this layer from the forward context. Distinct from the
        # nested Attention (.attn) and OlalaTokenShiftState (.shift_state)
        # registrations.
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

        for flag in ("token_conv1d_attn", "xsa", "intra_doc_masking"):
            if getattr(config, flag, False):
                raise NotImplementedError(
                    f"OlalaDiffTPAAttention: config.{flag} is not supported"
                )

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rank = config.tpa_rank
        self.qk_norm = config.qk_norm
        self.softcap = float(config.softcap_attn) if config.softcap_attn else 0.0
        self.num_attention_heads = config.num_attention_heads
        self.num_signal_heads = (
            config.num_signal_heads_diff
            if config.num_signal_heads_diff
            else self.num_attention_heads // 2
        )
        self.num_noise_heads = self.num_attention_heads - self.num_signal_heads
        if self.num_noise_heads <= 0 or self.num_signal_heads % self.num_noise_heads:
            raise ValueError(
                f"OlalaDiffTPAAttention: num_signal_heads="
                f"{self.num_signal_heads} must be a positive multiple of "
                f"num_noise_heads={self.num_noise_heads}"
            )
        self.snr = self.num_signal_heads // self.num_noise_heads
        self.num_kv_heads = self.num_noise_heads

        for name, count in (
            ("num_attention_heads", self.num_attention_heads),
            ("num_noise_heads", self.num_noise_heads),
            ("num_kv_heads", self.num_kv_heads),
        ):
            if count % tp:
                raise ValueError(
                    f"OlalaDiffTPAAttention: {name}={count} is not divisible "
                    f"by tensor_parallel_size={tp}"
                )
        self.num_attention_heads_local = divide(self.num_attention_heads, tp)
        self.num_signal_heads_local = divide(self.num_signal_heads, tp)
        self.num_noise_heads_local = divide(self.num_noise_heads, tp)
        self.num_kv_heads_local = divide(self.num_kv_heads, tp)
        self.q_size_local = self.num_attention_heads_local * self.head_dim
        self.kv_size_local = self.num_kv_heads_local * self.head_dim

        self.token_shift = bool(getattr(config, "token_shift_attn", False))
        self.scalable_softmax = bool(getattr(config, "scalable_softmax", False))
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(
                torch.ones(self.num_attention_heads_local, dtype=torch.float32)
            )
            set_weight_attrs(
                self.softmax_scaler, {"weight_loader": sharded_weight_loader(0)}
            )

        # Head-axis projections are column-parallel so each rank owns its own
        # head slice; the shared TPA bases are replicated.
        qc = mixer_quant_config(vllm_config.quant_config)

        def column(out_features: int, name: str) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=out_features,
                bias=False,
                gather_output=False,
                quant_config=qc,
                prefix=f"{prefix}.{name}",
            )

        def replicated(out_features: int, name: str) -> ReplicatedLinear:
            return ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=out_features,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.{name}",
            )

        self.c_q = column(self.num_attention_heads * self.head_dim, "c_q")
        self.W_A_k = column(self.num_kv_heads * self.rank, "W_A_k")
        self.W_A_v = column(self.num_kv_heads * self.rank, "W_A_v")
        self.W_B_k = replicated(self.rank * self.head_dim, "W_B_k")
        self.W_B_v = replicated(self.rank * self.head_dim, "W_B_v")
        self.lambda_proj = column(self.num_noise_heads, "lambda_proj")

        # TPA-factorized paged KV cache (2.5x smaller than dense K/V): needs
        # the exact Olala V-layer recipe the decode kernel implements.
        self.factor_mode = False
        if factor_cache_enabled():
            unmet = [
                name
                for name, ok in (
                    (f"tensor_parallel_size == 1 (got {tp})", tp == 1),
                    ("token_shift", self.token_shift),
                    ("qk_norm", self.qk_norm),
                    ("scalable_softmax", bool(getattr(config, "scalable_softmax", False))),
                    ("rope_theta == 0", float(getattr(config, "rope_theta", 0.0) or 0.0) == 0.0),
                    (f"heads {FACTOR_HQ}/{FACTOR_HKV}/rank 4/head_dim 128",
                     (self.num_attention_heads, self.num_kv_heads, self.rank, self.head_dim)
                     == (FACTOR_HQ, FACTOR_HKV, 4, 128)),
                )
                if not ok
            ]
            self.factor_mode = not unmet
            if unmet:
                warn_slow_path(
                    "TPA factor KV cache disabled",
                    "this layer/deployment does not meet: " + ", ".join(unmet),
                    "dense paged KV cache: 2.5x more KV memory and slower "
                    "long-context decode (up to -40% throughput at 24k)",
                    "serve with --tensor-parallel-size 1 (data-parallel replicas "
                    "instead) if TP is the reason; otherwise the checkpoint's "
                    "attention recipe is not the one the kernel implements",
                )

        if self.token_shift:
            self.shift_proj_k = column(self.num_kv_heads, "shift_proj_k")
            self.shift_proj_v = column(self.num_kv_heads, "shift_proj_v")
            self.shift_state = OlalaTokenShiftState(
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                vllm_config=vllm_config,
                prefix=f"{prefix}.shift_state",
                rank=self.rank if self.factor_mode else 0,
            )
        else:
            self.shift_state = None

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        if self.qk_norm:
            self.q_norm = OlalaNorm(self.head_dim, eps=eps, zero_centered=zc)
            self.k_norm = OlalaNorm(self.head_dim, eps=eps, zero_centered=zc)

        rope_theta = float(getattr(config, "rope_theta", 0.0) or 0.0)
        if rope_theta > 0.0:
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                max_position=getattr(config, "max_position_embeddings", 131072),
                rope_parameters={
                    "rope_type": "default",
                    "base": rope_theta,
                    "rotary_dim": self.head_dim,
                },
            )
        else:
            self.rotary_emb = None

        # Olala scales by 1/head_dim under use_completed_p, else 1/sqrt(d).
        scale = (
            1.0 / self.head_dim
            if getattr(config, "use_completed_p", False)
            else self.head_dim**-0.5
        )
        self.scale = scale
        if self.factor_mode:
            self.attn = None
            self.factor_cache = OlalaTPAFactorCache(
                vllm_config=vllm_config, prefix=f"{prefix}.factor"
            )
        else:
            self.factor_cache = None
            self.attn = Attention(
                num_heads=self.num_attention_heads_local,
                head_size=self.head_dim,
                scale=scale,
                num_kv_heads=self.num_kv_heads_local,
                cache_config=vllm_config.cache_config,
                logits_soft_cap=self.softcap if self.softcap > 0.0 else None,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,  # (N,)
        hidden_states: torch.Tensor,  # (N, D)
    ) -> torch.Tensor:
        """Return the differential output, ``(N, num_signal_heads_local*D)``.

        Runs outside the compiled graph as the ``vllm::olala_diff_tpa``
        custom op: the token shift mutates paged state, and the inner paged
        attention is itself an out-of-graph op.
        """
        out = torch.empty(
            hidden_states.shape[0],
            self.num_signal_heads_local * self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        torch.ops.vllm.olala_diff_tpa(
            positions, hidden_states, out, _encode_layer_name(self.prefix)
        )
        return out

    def _forward_impl(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        n = hidden_states.shape[0]

        if self._int8_concat_ok and os.environ.get("OLALA_TPA_CONCAT", "1") != "0":
            # INT8 projections first, bf16 ones after: one launch / one GEMM
            parts = self._int8_proj(hidden_states)
            q_all, A_k_all, A_v_all, B_k_all, B_v_all, lam_all = parts[:6]
            shift_alphas = tuple(parts[6:]) if self.token_shift else None
        elif self._proj_concat_ok and os.environ.get("OLALA_TPA_CONCAT", "1") != "0":
            # The input projections share one GEMM over concatenated weights
            # streaming GEMV over their concatenated weights.
            w_all = self._decode_proj_weight()
            if n <= SKINNY_MAX_M and gpu_family() is not GpuFamily.SM100:
                allp = decode_gemv(hidden_states, w_all)
            else:
                # also batch 1 on datacenter Blackwell: cuBLAS streams the aligned width
                # faster than the Triton GEMV (5.8 vs 6.3 us; 9.1 unaligned); OLALA_VOLTA_TC:
                # the fp16 tensor cores (volta_tc.py)
                allp = bf16_linear(hidden_states, w_all) if tc_enabled("bf16") else None
                if allp is None:
                    allp = F.linear(hidden_states, self._decode_proj_w_padded)[:, : w_all.shape[0]]
            parts = torch.split(allp, self._decode_proj_sizes, dim=-1)
            q_all, A_k_all, A_v_all, B_k_all, B_v_all, lam_all = parts[:6]
            shift_alphas = tuple(parts[6:]) if self.token_shift else None
        else:
            q_all, _ = self.c_q(hidden_states)
            A_k_all, _ = self.W_A_k(hidden_states)
            A_v_all, _ = self.W_A_v(hidden_states)
            B_k_all, _ = self.W_B_k(hidden_states)
            B_v_all, _ = self.W_B_v(hidden_states)
            lam_all, _ = self.lambda_proj(hidden_states)
            shift_alphas = None

        if self.factor_mode:
            self._forward_factor(
                positions, hidden_states, out, q_all, A_k_all, A_v_all,
                B_k_all, B_v_all, lam_all, shift_alphas,
            )
            return

        md = None
        if shift_alphas is not None and self._fused_decode_ok:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is not None and self._ts_md_key in attn_metadata:
                md = attn_metadata[self._ts_md_key]
                if (
                    _MIXED_FUSED and _FUSED_SPEC and md.spec is not None
                    and md.num_prefills != 0 and md.num_decodes != 0
                ):
                    self._forward_mixed_fused(
                        md, positions, out, q_all, A_k_all, A_v_all, B_k_all, B_v_all,
                        lam_all, shift_alphas,
                    )
                    return
                if md.num_prefills != 0 or (md.spec is not None and not _FUSED_SPEC):
                    md = None
        if md is not None:
            # Pure decode batch: one kernel per (token, kv head) for the K/V
            # reconstruction, token shift, q/k norms and softmax scaling; for a
            # verify batch one program per (request, kv head) chains the
            # positions (column-slot protocol).
            k_pool, v_pool = self._ts_kv(spec=md.spec is not None)
            common = (
                k_pool, v_pool,
                self.q_norm.norm.weight, self.k_norm.norm.weight,
                self.q_norm.norm.eps, self.softmax_scaler,
                float(getattr(self.config, "sliding_window_size", 0) or 0),
                self.num_kv_heads_local, self.snr + 1, self.rank, self.head_dim,
            )
            if md.spec is not None:
                spec = md.spec
                q_flat, k_flat, v_flat = tpa_decode_qkv_spec(
                    q_all, A_k_all, A_v_all, B_k_all, B_v_all,
                    shift_alphas[0], shift_alphas[1], positions,
                    spec.query_start_loc_d, spec.state_cols,
                    spec.num_accepted_tokens, *common,
                )
            else:
                q_flat, k_flat, v_flat = tpa_decode_qkv(
                    q_all, A_k_all, A_v_all, B_k_all, B_v_all,
                    shift_alphas[0], shift_alphas[1], positions,
                    md.state_indices_tensor, *common,
                )
            attn_out = self.attn(q_flat, k_flat, v_flat)
            tpa_diff_combine(attn_out, lam_all, out, self.num_noise_heads_local,
                             self.snr, self.head_dim)
            return

        q = q_all.view(n, self.num_attention_heads_local, self.head_dim)
        A_k = A_k_all.view(n, self.num_kv_heads_local, self.rank)
        A_v = A_v_all.view(n, self.num_kv_heads_local, self.rank)
        B_k = B_k_all.view(n, self.rank, self.head_dim)
        B_v = B_v_all.view(n, self.rank, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank)  # (N, Hkv_local, D)
        v = torch.bmm(A_v, B_v).div_(self.rank)

        if self.token_shift:
            k, v = self._apply_token_shift(hidden_states, positions, k, v, shift_alphas)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q_flat = q.reshape(n, self.q_size_local)
        k_flat = k.reshape(n, self.kv_size_local)
        v_flat = v.reshape(n, self.kv_size_local)

        if self.rotary_emb is not None:
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)

        if self.scalable_softmax:
            # q <- s * log(max(pos + 1, wsize)) * q, per head. A zero
            # sliding_window_size disables the clamp.
            pos = (positions.float() + 1.0).clamp_min(1.0)
            wsize = float(getattr(self.config, "sliding_window_size", 0) or 0)
            log_pos = pos.clamp_max(wsize).log() if wsize > 0 else pos.log()
            scale = self.softmax_scaler.to(q_flat.dtype) * log_pos.to(
                q_flat.dtype
            ).unsqueeze(-1)
            q_flat = q_flat.view(n, self.num_attention_heads_local, self.head_dim)
            q_flat = (q_flat * scale.unsqueeze(-1)).reshape(n, self.q_size_local)

        attn_out = self.attn(q_flat, k_flat, v_flat)  # (N, H_local*D)

        attn_out = attn_out.view(
            n, self.num_noise_heads_local, self.snr + 1, self.head_dim
        )
        sig = attn_out[:, :, : self.snr, :]
        noi = attn_out[:, :, self.snr : self.snr + 1, :]

        lam = lam_all.view(n, self.num_noise_heads_local, 1, 1)
        diff = sig - torch.sigmoid(lam) * noi
        out.copy_(diff.reshape(n, self.num_signal_heads_local * self.head_dim))

    def _common_fused_args(self, k_pool, v_pool) -> tuple:
        return (
            k_pool, v_pool,
            self.q_norm.norm.weight, self.k_norm.norm.weight,
            self.q_norm.norm.eps, self.softmax_scaler,
            float(getattr(self.config, "sliding_window_size", 0) or 0),
            self.num_kv_heads_local, self.snr + 1, self.rank, self.head_dim,
        )

    def _mixed_fused_qkv(
        self, md, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all, shift_alphas
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q / k / v of a mixed (verify + prefill) batch: the verify rows through
        the fused spec kernel, the prefill rows (and any cudagraph padding rows
        after them) through the op-by-op path."""
        n = q_all.shape[0]
        ndt, npt = md.num_decode_tokens, md.num_prefill_tokens
        spec = md.spec
        k_cols, v_cols = self._ts_kv(spec=True)
        q_d, k_d, v_d = tpa_decode_qkv_spec(
            q_all[:ndt], A_k_all[:ndt], A_v_all[:ndt], B_k_all[:ndt], B_v_all[:ndt],
            shift_alphas[0][:ndt], shift_alphas[1][:ndt], positions[:ndt],
            spec.query_start_loc_d, spec.state_cols, spec.num_accepted_tokens,
            *self._common_fused_args(k_cols, v_cols),
        )
        rows = slice(ndt, n)
        m = n - ndt
        H, R, D = self.num_kv_heads_local, self.rank, self.head_dim
        q = q_all[rows].view(m, self.num_attention_heads_local, D)
        k = torch.bmm(A_k_all[rows].view(m, H, R), B_k_all[rows].view(m, R, D)).div_(R)
        v = torch.bmm(A_v_all[rows].view(m, H, R), B_v_all[rows].view(m, R, D)).div_(R)
        pos = positions[rows]
        if self.token_shift:
            k_pool, v_pool = self._ts_kv()
            k_s, v_s = self._alloc_shift(k, v, npt)
            self._shift_prefill_rows(
                md, k, v, md.state_indices_tensor, k_pool, v_pool,
                shift_alphas[0][rows], shift_alphas[1][rows], pos,
                md.num_decodes, md.num_prefills, 0, npt, k_s, v_s,
            )
            k, v = k_s, v_s
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q_p = q.reshape(m, self.q_size_local)
        k_p = k.reshape(m, self.kv_size_local)
        v_p = v.reshape(m, self.kv_size_local)
        if self.rotary_emb is not None:
            q_p, k_p = self.rotary_emb(pos, q_p, k_p)
        if self.scalable_softmax:
            p1 = (pos.float() + 1.0).clamp_min(1.0)
            wsize = float(getattr(self.config, "sliding_window_size", 0) or 0)
            log_pos = p1.clamp_max(wsize).log() if wsize > 0 else p1.log()
            scale = self.softmax_scaler.to(q_p.dtype) * log_pos.to(q_p.dtype).unsqueeze(-1)
            q_p = q_p.view(m, self.num_attention_heads_local, D)
            q_p = (q_p * scale.unsqueeze(-1)).reshape(m, self.q_size_local)
        return torch.cat([q_d, q_p]), torch.cat([k_d, k_p]), torch.cat([v_d, v_p])

    def _forward_mixed_fused(
        self, md, positions, out, q_all, A_k_all, A_v_all, B_k_all, B_v_all, lam_all,
        shift_alphas,
    ) -> None:
        n = q_all.shape[0]
        ndt = md.num_decode_tokens
        q_flat, k_flat, v_flat = self._mixed_fused_qkv(
            md, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all, shift_alphas
        )
        attn_out = self.attn(q_flat, k_flat, v_flat)  # (N, H_local*D)
        tpa_diff_combine(attn_out[:ndt], lam_all[:ndt], out[:ndt], self.num_noise_heads_local,
                         self.snr, self.head_dim)
        m = n - ndt
        a = attn_out[ndt:].view(m, self.num_noise_heads_local, self.snr + 1, self.head_dim)
        lam = lam_all[ndt:].view(m, self.num_noise_heads_local, 1, 1)
        diff = a[:, :, : self.snr, :] - torch.sigmoid(lam) * a[:, :, self.snr :, :]
        out[ndt:].copy_(diff.reshape(m, self.num_signal_heads_local * self.head_dim))

    # ------------------------------------------------------------------
    # TPA-factorized KV cache path
    # ------------------------------------------------------------------

    # -- token-shift buffer: own MambaSpec layer, or folded into a Mamba3 page -------
    def set_token_shift_host(self, host) -> None:
        """Fold: read this layer's token-shift buffer from ``host`` (the
        Mamba3 mixer right before it), whose page carries it, and its metadata
        under the host's prefix (same block table, so the same state slots,
        spec columns and prefill layout)."""
        object.__setattr__(self, "_ts_host", host)

    @property
    def _ts_md_key(self) -> str:
        host = getattr(self, "_ts_host", None)
        if self.shift_state.folded:
            assert host is not None, f"{self.prefix}: folded token shift without a Mamba3 host"
            return host.prefix
        return self.shift_state.prefix

    def _ts_kv(self, spec: bool = False) -> tuple[torch.Tensor, ...]:
        """(k_last, v_last[, A_k_last, A_v_last]) pools. With light columns
        the spec paths take the (slots, 1 + k, ...) tensors, the others
        column 0 (the committed buffer)."""
        if self.shift_state.folded:
            pools = self._ts_host.folded_token_shift_pools()
            pools = pools[:4] if self.factor_mode else pools[:2]
        else:
            pools = tuple(self.shift_state.kv_cache)
        if self.shift_state.light_cols and not spec:
            pools = tuple(p[:, 0] for p in pools)
        return pools

    def _q_for_attention(self, q_all: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Eager q pre-processing: zero-centered RMSNorm + scalable softmax scale."""
        n = q_all.shape[0]
        q = self.q_norm(q_all.view(n, self.num_attention_heads_local, self.head_dim))
        pos = (positions.float() + 1.0).clamp_min(1.0)
        wsize = float(getattr(self.config, "sliding_window_size", 0) or 0)
        log_pos = pos.clamp_max(wsize).log() if wsize > 0 else pos.log()
        scale = self.softmax_scaler.to(q.dtype) * log_pos.to(q.dtype).unsqueeze(-1)
        return q * scale.unsqueeze(-1)

    def _forward_factor(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        out: torch.Tensor,
        q_all: torch.Tensor,
        A_k_all: torch.Tensor,
        A_v_all: torch.Tensor,
        B_k_all: torch.Tensor,
        B_v_all: torch.Tensor,
        lam_all: torch.Tensor,
        shift_alphas: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        n = hidden_states.shape[0]
        OlalaDiffTPAAttention._factor_calls += 1
        if shift_alphas is None:
            alpha_k_all, _ = self.shift_proj_k(hidden_states)
            alpha_v_all, _ = self.shift_proj_v(hidden_states)
        else:
            alpha_k_all, alpha_v_all = shift_alphas
        fc = get_forward_context()
        attn_out = torch.empty(
            n, self.num_attention_heads_local * self.head_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        if fc.attn_metadata is None or self._ts_md_key not in fc.attn_metadata:
            # Profiling / dry run: no paged cache yet; dense causal attention
            # over the batch as one document (same activation footprint).
            self._factor_dry_run(positions, q_all, A_k_all, A_v_all, B_k_all,
                                 B_v_all, alpha_k_all, alpha_v_all, attn_out)
        else:
            md = fc.attn_metadata[self.factor_cache.prefix]
            ms: OlalaDiffTPAMetadata = fc.attn_metadata[self._ts_md_key]
            kv = self.factor_cache.kv_cache
            if kv.shape[0] != 2 or kv.dim() != 4 or kv.shape[1] == 1:
                # vLLM >= 0.33 hands a block-major view (blocks, 1, block_size,
                # 2 * ROWP). Each block's page is ours to lay out: view it as
                # [K rows | V rows] so the kernels keep a ROWP token stride, with
                # a 2x block stride (they all take stride(0) as the block stride).
                nb, bs = kv.shape[0], kv.shape[-2]
                kv = kv.reshape(nb, 2, bs, kv.shape[-1] // 2).transpose(0, 1)
            kplane, vplane = kv[0], kv[1]
            if os.environ.get("OLALA_TPA_DEBUG") and not getattr(self, "_dbg_once", False):
                self._dbg_once = True
                print(f"[tpa-factor] kv shape {tuple(kv.shape)} strides {kv.stride()} contiguous={kv.is_contiguous()}", flush=True)
            slot_mapping = fc.slot_mapping[self.factor_cache.prefix]
            nd, ndt = md.num_decodes, md.num_decode_tokens
            np_, npt = md.num_prefills, md.num_prefill_tokens
            if ms.spec is not None:
                if not factor_spec_enabled():
                    raise NotImplementedError(
                        "speculative decoding with OLALA_TPA_FACTOR=1 needs OLALA_TPA_FACTOR_SPEC=1")
                self._factor_verify(
                    md, ms, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all,
                    alpha_k_all, alpha_v_all, slot_mapping, kplane, vplane, attn_out,
                )
            elif nd > 0 and os.environ.get("OLALA_TPA_FACTOR_DECODE_DENSE") == "1":
                # Debug: route decode rows through the write + reconstruct + FA path.
                reqs = (nd, list(range(nd + 1)), md.seq_lens[:nd].tolist(), [True] * nd,
                        ms.state_indices_tensor[:nd].tolist(), 0)
                attn_out[:nd] = self._factor_prefill(
                    md, ms, positions[:nd], q_all[:nd], A_k_all[:nd], A_v_all[:nd],
                    B_k_all[:nd], B_v_all[:nd], alpha_k_all[:nd], alpha_v_all[:nd],
                    slot_mapping[:nd], kplane, vplane, reqs=reqs,
                )
            elif nd > 0:
                assert ndt == nd, "factor decode path expects one token per decode request"
                k_pool, v_pool, ak_pool, av_pool = self._ts_kv()
                q_d = factor_decode_write(
                    q_all[:nd], A_k_all[:nd], A_v_all[:nd], B_k_all[:nd], B_v_all[:nd],
                    alpha_k_all[:nd], alpha_v_all[:nd], positions[:nd],
                    ms.state_indices_tensor[:nd], slot_mapping[:nd],
                    k_pool, v_pool, ak_pool, av_pool,
                    self.q_norm.norm.weight, self.k_norm.norm.weight,
                    self.q_norm.norm.eps, self.softmax_scaler,
                    float(getattr(self.config, "sliding_window_size", 0) or 0),
                    kplane, vplane,
                    self.snr + 1,
                )
                nsplit = num_splits_for_batch(nd)
                factor_decode_attention(
                    q_d, kplane, vplane, md.block_table[:nd], md.seq_lens[:nd],
                    md.split_size(nsplit), nsplit, self.scale, self.softcap,
                    self.factor_cache.block_size, out=attn_out[:nd],
                )
                if os.environ.get("OLALA_TPA_DEBUG") == "2" and bool(torch.isnan(attn_out[:nd]).any()):
                    tps, nact = md.split_size(nsplit)
                    bad = torch.isnan(attn_out[:nd]).view(nd, -1).any(1).nonzero().flatten().tolist()
                    print(f"[tpa-factor NaN] layer={self.prefix} nd={nd} nsplit={nsplit} tps={int(tps)} nact={int(nact)} "
                          f"seq_lens={md.seq_lens[:nd].tolist()} bad_rows={bad} q_nan={bool(torch.isnan(q_d).any())}", flush=True)
            if np_ > 0:
                rows = slice(ndt, ndt + npt)
                attn_out[rows] = self._factor_prefill(
                    md, ms, positions[rows], q_all[rows], A_k_all[rows], A_v_all[rows],
                    B_k_all[rows], B_v_all[rows], alpha_k_all[rows], alpha_v_all[rows],
                    slot_mapping[rows], kplane, vplane,
                )
            if ndt + npt < n:
                attn_out[ndt + npt :] = 0  # cudagraph padding rows
        tpa_diff_combine(attn_out, lam_all, out, self.num_noise_heads_local,
                         self.snr, self.head_dim)

    def _factor_verify(
        self, md, ms, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all,
        alpha_k_all, alpha_v_all, slot_mapping, kplane, vplane, attn_out,
    ) -> None:
        """Verify rows of a spec step (OLALA_TPA_FACTOR_SPEC), pure or mixed: the
        factor row and token shift of every position (factor_decode_write_spec,
        column-slot protocol or light columns), then attention through the
        factor decode kernel on one virtual single-token row per position
        (seq_len = committed context + position + 1). Rejected positions leave
        rows past the accepted ones: the next step overwrites them, and the
        kernel reads only rows < seq_len."""
        ndt = md.num_decode_tokens
        spec = ms.spec
        assert md.num_decodes == ms.num_decodes, (
            f"{self.prefix}: factor and token-shift builders split the batch differently "
            f"({md.num_decodes} vs {ms.num_decodes} verify requests)"
        )
        k_cols, v_cols, ak_cols, av_cols = self._ts_kv(spec=True)
        q_d = factor_decode_write_spec(
            q_all[:ndt], A_k_all[:ndt], A_v_all[:ndt], B_k_all[:ndt], B_v_all[:ndt],
            alpha_k_all[:ndt], alpha_v_all[:ndt], positions[:ndt],
            spec.query_start_loc_d, spec.state_cols, spec.num_accepted_tokens,
            slot_mapping[:ndt], k_cols, v_cols, ak_cols, av_cols,
            self.q_norm.norm.weight, self.k_norm.norm.weight,
            self.q_norm.norm.eps, self.softmax_scaler,
            float(getattr(self.config, "sliding_window_size", 0) or 0),
            kplane, vplane, self.snr + 1,
        )
        bt_v, sl_v = md.virtual_decode_rows()
        nsplit = num_splits_for_verify(md.num_decodes)
        factor_decode_attention(
            q_d, kplane, vplane, bt_v, sl_v, md.split_size(nsplit), nsplit,
            self.scale, self.softcap, self.factor_cache.block_size, out=attn_out[:ndt],
        )

    def _factor_prefill(
        self, md, ms, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all,
        alpha_k_all, alpha_v_all, slots, kplane, vplane,
        reqs: tuple | None = None,
    ) -> torch.Tensor:
        """Write the chunk's factor rows, then attend over each request's
        full context reconstructed from the cache (dense FA varlen).

        ``reqs`` = (num_reqs, query_start_loc, seq_lens, has_initial_state,
        pool_slots, first_block_table_row) overrides the prefill metadata so
        decode rows can be routed through this path for debugging."""
        if reqs is None:
            reqs = (md.num_prefills, md.query_start_loc_p_cpu, md.seq_lens_p_cpu,
                    md.has_initial_state_p_cpu, ms.state_indices_p_cpu, md.num_decodes)
        nreq, qsl, seq_lens_cpu, has_init, pslots, bt0 = reqs
        npt = q_all.shape[0]
        H, R, D = self.num_kv_heads_local, self.rank, self.head_dim
        dev = q_all.device
        fused_glue = os.environ.get("OLALA_TPA_GLUE_TORCH") != "1"

        # Previous-token seeds: in-batch predecessor, or the pool (or zero) at a request start.
        # host-side filtering and one async upload (device boolean indexing / torch.tensor(device=)
        # would each synchronize the stream)
        keep = [r for r in range(nreq) if qsl[r + 1] > qsl[r]]
        lengths = [int(x) for x in seq_lens_cpu]
        cu_k = [0]
        for L in lengths:
            cu_k.append(cu_k[-1] + L)
        nk = len(keep)
        if pslots is None:
            # OLALA_PREFILL_NOSYNC: slots stay on the device (no host copy).
            dev_slots = ms.state_indices_tensor[bt0 : bt0 + nreq]
            if nk < nreq:
                dev_slots = dev_slots[torch.tensor(keep, dtype=torch.long).to(dev, non_blocking=True)]
            pslots_cpu = [0] * nk
        else:
            pslots_cpu = [int(pslots[r]) for r in keep]
        meta = torch.tensor([qsl[r] for r in keep] + [qsl[r + 1] for r in keep] + pslots_cpu
                            + [int(bool(has_init[r])) for r in keep] + [int(x) for x in qsl] + cu_k,
                            dtype=torch.int64).to(dev, non_blocking=True)
        starts, ends, pslots_t = meta[:nk], meta[nk:2 * nk], meta[2 * nk:3 * nk]
        if pslots is None:
            pslots_t = dev_slots.to(torch.int64)
        init_t = meta[3 * nk:4 * nk].bool()
        cu_q_dev = meta[4 * nk:4 * nk + nreq + 1].to(torch.int32)
        cu_k_dev = meta[4 * nk + nreq + 1:].to(torch.int32)
        k_pool, v_pool, ak_pool, av_pool = self._ts_kv()
        if fused_glue:
            tpa_prefill_rows(
                A_k_all, A_v_all, B_k_all, B_v_all, alpha_k_all, alpha_v_all, positions, slots,
                starts, ends, pslots_t, init_t, (k_pool, v_pool, ak_pool, av_pool),
                self.k_norm.norm.weight, self.k_norm.norm.eps, kplane, vplane, H, R, D,
                max_len=max((qsl[r + 1] - qsl[r] for r in keep), default=0),
            )
            q_scale = prefill_q_prescale(self.scale, self.softcap)
            q = tpa_prefill_q(
                q_all, positions, self.q_norm.norm.weight, self.softmax_scaler, self.q_norm.norm.eps,
                float(getattr(self.config, "sliding_window_size", 0) or 0),
                self.num_attention_heads_local, D, out_scale=q_scale,
            )
        else:
            q_scale = None
            q = self._factor_prefill_glue_torch(
                positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all, alpha_k_all, alpha_v_all, slots,
                kplane, vplane, starts, ends, pslots_t, init_t,
            )
        bt_rows = md.block_table[bt0 : bt0 + nreq]
        if os.environ.get("OLALA_TPA_RECON_TORCH") == "1":  # reference path (A/B checks)
            k_all = reconstruct_dense_batched(kplane, bt_rows, lengths)
            v_all = reconstruct_dense_batched(vplane, bt_rows, lengths)
        else:
            k_all, v_all = reconstruct_dense_kv(kplane, vplane, bt_rows, lengths)
        out = tpa_varlen_attention(
            q, k_all, v_all, cu_q_dev, cu_k_dev,
            max(qsl[r + 1] - qsl[r] for r in range(nreq)), max(lengths),
            self.scale, self.softcap, prescaled=q_scale is not None,
        )
        return out.reshape(npt, -1)

    def _factor_prefill_glue_torch(self, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all,
                                   alpha_k_all, alpha_v_all, slots, kplane, vplane,
                                   starts, ends, pslots_t, init_t) -> torch.Tensor:
        """Torch reference of tpa_prefill_rows + tpa_prefill_q (OLALA_TPA_GLUE_TORCH=1)."""
        npt = q_all.shape[0]
        H, R, D = self.num_kv_heads_local, self.rank, self.head_dim
        A_k = A_k_all.view(npt, H, R)
        A_v = A_v_all.view(npt, H, R)
        B_k = B_k_all.view(npt, R, D)
        B_v = B_v_all.view(npt, R, D)
        k = torch.bmm(A_k, B_k).div_(R)
        v = torch.bmm(A_v, B_v).div_(R)
        k_pool, v_pool, ak_pool, av_pool = self._ts_kv()

        def seeded(cur: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
            prev = torch.empty_like(cur)
            prev[1:] = cur[:-1]
            seed = pool[pslots_t].to(cur.dtype)
            prev[starts] = torch.where(init_t.view(-1, *([1] * (cur.dim() - 1))), seed, torch.zeros_like(seed))
            return prev

        k_prev, v_prev = seeded(k, k_pool), seeded(v, v_pool)
        ak_prev, av_prev = seeded(A_k, ak_pool), seeded(A_v, av_pool)
        last = ends - 1
        k_pool[pslots_t] = k[last].to(k_pool.dtype)
        v_pool[pslots_t] = v[last].to(v_pool.dtype)
        ak_pool[pslots_t] = A_k[last].to(ak_pool.dtype)
        av_pool[pslots_t] = A_v[last].to(av_pool.dtype)

        doc_start = (positions == 0).view(-1, 1)
        a_k = torch.sigmoid(alpha_k_all.float()).masked_fill(doc_start, 0)
        a_v = torch.sigmoid(alpha_v_all.float()).masked_fill(doc_start, 0)
        ks = (a_k.unsqueeze(-1) * k_prev.float() + (1 - a_k.unsqueeze(-1)) * k.float()).to(k.dtype).float()
        inv = torch.rsqrt(ks.pow(2).mean(-1) + self.k_norm.norm.eps)  # (npt, H)
        kw = (1.0 + self.k_norm.norm.weight.float()).to(k.dtype)
        write_rows(kplane, slots, factor_rows(A_k, ak_prev, B_k, a_k, inv, kw))
        write_rows(vplane, slots, factor_rows(A_v, av_prev, B_v, a_v, None, None))
        return self._q_for_attention(q_all, positions)

    def _factor_dry_run(self, positions, q_all, A_k_all, A_v_all, B_k_all, B_v_all,
                        alpha_k_all, alpha_v_all, attn_out) -> None:
        n = q_all.shape[0]
        H, R, D = self.num_kv_heads_local, self.rank, self.head_dim
        k = torch.bmm(A_k_all.view(n, H, R), B_k_all.view(n, R, D)).div_(R)
        v = torch.bmm(A_v_all.view(n, H, R), B_v_all.view(n, R, D)).div_(R)
        a_k = torch.sigmoid(alpha_k_all.float()).unsqueeze(-1).to(k.dtype)
        a_v = torch.sigmoid(alpha_v_all.float()).unsqueeze(-1).to(v.dtype)
        k = self.k_norm((1 - a_k) * k)
        v = (1 - a_v) * v
        q = self._q_for_attention(q_all, positions)
        cu = torch.tensor([0, n], dtype=torch.int32, device=q.device)
        out = tpa_varlen_attention(q, k, v, cu, cu, n, n, self.scale, self.softcap)
        attn_out.copy_(out.reshape(n, -1))

    @property
    def _fused_decode_ok(self) -> bool:
        return (
            self.token_shift
            and self.qk_norm
            and self.rotary_emb is None
            and self.scalable_softmax
            and self.q_norm.norm.zero_centered
            and self.k_norm.norm.zero_centered
            and self.head_dim & (self.head_dim - 1) == 0
        )

    def _proj_modules(self) -> tuple:
        mods = (
            self.c_q,
            self.W_A_k,
            self.W_A_v,
            self.W_B_k,
            self.W_B_v,
            self.lambda_proj,
        )
        if self.token_shift:
            mods = mods + (self.shift_proj_k, self.shift_proj_v)
        return mods

    def _n_quantized(self) -> int:
        """Leading projections with a quantized weight (-1: not a prefix)."""
        q = [
            not isinstance(m.quant_method, UnquantizedLinearMethod)
            for m in self._proj_modules()
        ]
        k = q.index(False) if False in q else len(q)
        return k if not any(q[k:]) else -1

    @property
    def _proj_concat_ok(self) -> bool:
        return (
            get_tensor_model_parallel_world_size() == 1
            and all(m.bias is None for m in self._proj_modules())
            and self._n_quantized() == 0
        )

    @property
    def _int8_concat_ok(self) -> bool:
        return (
            is_int8_quant_method(self.c_q)
            and get_tensor_model_parallel_world_size() == 1
            and all(m.bias is None for m in self._proj_modules())
            and self._n_quantized() > 0
            and self._int8_proj_state() is not None
        )

    def _int8_proj_state(self) -> tuple | None:
        """(decode segments, INT8 weight, bf16 weight, sizes) of the concatenated
        projections ``[INT8 ... | bf16 ...]``, sharing their parameters' storage
        (W8A16 INT8 checkpoints); built on the first call (the profiling run).
        None if a layout is not supported (each projection runs its own linear)."""
        st = getattr(self, "_int8_st", None)
        if st is None:
            mods = self._proj_modules()
            k = self._n_quantized()
            mods8, mods16 = mods[:k], mods[k:]
            w8 = (
                cat_int8_rows_shared(list(mods8))
                if all(is_int8_quant_method(m) for m in mods8)
                else None
            )
            if w8 is None:
                st = ()
            else:
                segs = [Segment(w8.weight, w8.scale)]
                w16 = None
                if mods16:
                    w16 = cat_rows_shared([m.weight for m in mods16])
                    segs.append(Segment(w16, col0=w8.n))
                st = (segs, w8, w16, [m.output_size_per_partition for m in mods])
            self._int8_st = st
        return st or None

    def _int8_proj(self, h: torch.Tensor) -> list[torch.Tensor]:
        """``[q, A_k, A_v, B_k, B_v, lambda, shift ...]`` views of one ``(n, W)``
        buffer: W8A16 decode kernels up to int8_decode_max_m rows, one W8A16
        prefill GEMM over the concatenated INT8 weight beyond (+ the bf16 rows)."""
        segs, w8, w16, sizes = self._int8_proj_state()
        n = h.shape[0]
        width = sum(sizes)
        n16 = 0 if w16 is None else w16.shape[0]
        if 0 < n <= int8_decode_max_m(width, w8.k, n16):
            out = int8_decode_linear(h, segs, width=width)
        elif w16 is None:
            out = int8_prefill_mm(h, w8)
        else:
            out = torch.empty(n, width, dtype=h.dtype, device=h.device)
            out[:, : w8.n] = int8_prefill_mm(h, w8)
            out[:, w8.n :] = F.linear(h, w16)
        return list(torch.split(out, sizes, dim=-1))

    def _decode_proj_weight(self) -> torch.Tensor:
        """Concatenated [c_q; W_A_k; W_A_v; W_B_k; W_B_v; lambda_proj] weights
        for the single-token GEMV; built once, refreshed in place by
        refresh_weight_caches() after an in-place weight reload."""
        w = getattr(self, "_decode_proj_w", None)
        if w is None:
            mods = (self.c_q, self.W_A_k, self.W_A_v, self.W_B_k,
                    self.W_B_v, self.lambda_proj)
            if self.token_shift:
                mods = mods + (self.shift_proj_k, self.shift_proj_v)
            self._decode_proj_sizes = [m.weight.shape[0] for m in mods]
            n = sum(self._decode_proj_sizes)
            w = cat_rows_shared([m.weight for m in mods], pad_to=128)
            self._decode_proj_w_padded = w  # prefill GEMM: aligned width (Hopper cuBLAS kernels)
            w = w[:n]
            self._decode_proj_w = w
        return w

    def refresh_weight_caches(self) -> None:
        """Recompute the concatenated-projection snapshot into its existing
        storage: FULL-mode CUDA graphs bake its device address into the
        captured decode graphs (same failure mode as the Mamba3 constant
        caches under RL weight resync)."""
        w = getattr(self, "_decode_proj_w", None)
        if w is None:
            return
        mods = (self.c_q, self.W_A_k, self.W_A_v, self.W_B_k,
                self.W_B_v, self.lambda_proj)
        if self.token_shift:
            mods = mods + (self.shift_proj_k, self.shift_proj_v)
        with torch.inference_mode():
            off = 0
            for m in mods:
                n = m.weight.shape[0]
                w[off : off + n].copy_(m.weight)
                off += n

    def _apply_token_shift(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alphas: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix the previous token's raw K/V into the current token."""
        if alphas is not None:
            alpha_k_all, alpha_v_all = alphas
        else:
            alpha_k_all, _ = self.shift_proj_k(hidden_states)
            alpha_v_all, _ = self.shift_proj_v(hidden_states)

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None or self._ts_md_key not in attn_metadata:
            # Profiling / dry run (or a warmup building attention metadata only): treat every token as a sequence start.
            a_k = torch.sigmoid(alpha_k_all.float()).unsqueeze(-1).to(k.dtype)
            a_v = torch.sigmoid(alpha_v_all.float()).unsqueeze(-1).to(v.dtype)
            return (1 - a_k) * k, (1 - a_v) * v

        md: OlalaDiffTPAMetadata = attn_metadata[self._ts_md_key]
        k_pool, v_pool = self._ts_kv()
        k_cols, v_cols = self._ts_kv(spec=True) if md.spec is not None else (k_pool, v_pool)
        slots = md.state_indices_tensor
        nd, np_ = md.num_decodes, md.num_prefills
        ndt, npt = md.num_decode_tokens, md.num_prefill_tokens

        if md.spec is not None and nd > 0:
            k_shift, v_shift = self._alloc_shift(k, v, ndt + npt)
            self._shift_spec_rows(
                md.spec,
                k,
                v,
                k_cols,
                v_cols,
                alpha_k_all,
                alpha_v_all,
                positions,
                k_shift,
                v_shift,
                ndt,
            )
            if np_ > 0:
                self._shift_prefill_rows(
                    md,
                    k,
                    v,
                    slots,
                    k_pool,
                    v_pool,
                    alpha_k_all,
                    alpha_v_all,
                    positions,
                    nd,
                    np_,
                    ndt,
                    npt,
                    k_shift,
                    v_shift,
                )
            return k_shift, v_shift

        head_dim = k.shape[-1]
        if nd > 0 and head_dim & (head_dim - 1) == 0:
            # One kernel for the decode rows: gather previous, blend, store
            # current — replacing ~10 eager kernels per layer per step.
            k_shift, v_shift = self._alloc_shift(k, v, ndt + npt)
            num_kv_heads = k.shape[1]
            _token_shift_decode_kernel[(ndt * num_kv_heads,)](
                k,
                v,
                k_shift,
                v_shift,
                k_pool,
                v_pool,
                alpha_k_all,
                alpha_v_all,
                slots[:nd].to(torch.int32),
                positions,
                num_kv_heads,
                k_pool.shape[0],
                k.stride(0),
                k.stride(1),
                k_pool.stride(0),
                k_pool.stride(1),
                alpha_k_all.stride(0),
                D=head_dim,
            )
            if np_ > 0:
                self._shift_prefill_rows(
                    md,
                    k,
                    v,
                    slots,
                    k_pool,
                    v_pool,
                    alpha_k_all,
                    alpha_v_all,
                    positions,
                    nd,
                    np_,
                    ndt,
                    npt,
                    k_shift,
                    v_shift,
                )
            return k_shift, v_shift

        k_shift, v_shift = self._alloc_shift(k, v, ndt + npt)
        if nd > 0:
            dslots = slots[:nd]
            k_prev = k_pool[dslots].to(k.dtype)
            v_prev = v_pool[dslots].to(v.dtype)
            k_pool[dslots] = k[:ndt].to(k_pool.dtype)
            v_pool[dslots] = v[:ndt].to(v_pool.dtype)
            self._blend(
                k_shift[:ndt],
                v_shift[:ndt],
                k[:ndt],
                v[:ndt],
                k_prev,
                v_prev,
                alpha_k_all[:ndt],
                alpha_v_all[:ndt],
                positions[:ndt],
            )
        if np_ > 0:
            self._shift_prefill_rows(
                md,
                k,
                v,
                slots,
                k_pool,
                v_pool,
                alpha_k_all,
                alpha_v_all,
                positions,
                nd,
                np_,
                ndt,
                npt,
                k_shift,
                v_shift,
            )
        return k_shift, v_shift

    def _shift_spec_rows(
        self,
        spec,  # OlalaSpecMetadata
        k: torch.Tensor,
        v: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        alpha_k_all: torch.Tensor,
        alpha_v_all: torch.Tensor,
        positions: torch.Tensor,
        k_shift: torch.Tensor,
        v_shift: torch.Tensor,
        ndt: int,
    ) -> None:
        """Token shift for spec-decode verify rows (column-slot protocol).

        Previous K/V for a request's first token comes from the pool column
        selected by last step's acceptance; later positions shift in-batch.
        Every position's raw K/V is stored to its own column so next step's
        acceptance picks the right one. Inactive (row, t) writes land on the
        reserved null block row 0.
        """
        qsl = spec.query_start_loc_d
        starts = qsl[:-1].long()
        qlens = torch.diff(qsl)
        rows = torch.arange(ndt, device=k.device)
        prev_rows = (rows - 1).clamp_(min=0)
        k_prev = k.index_select(0, prev_rows)
        v_prev = v.index_select(0, prev_rows)
        light = k_pool.dim() == 4  # (slots, 1 + k, kv, D): light columns (state.py)
        if light:
            slot0 = spec.state_cols[:, 0].long()
            col0 = (spec.num_accepted_tokens.long() - 1).clamp_(min=0, max=k_pool.shape[1] - 1)
            k_prev.index_copy_(0, starts, k_pool[slot0, col0].to(k.dtype))
            v_prev.index_copy_(0, starts, v_pool[slot0, col0].to(v.dtype))
        else:
            init = spec.initial_slots().long()
            k_prev.index_copy_(0, starts, k_pool[init].to(k.dtype))
            v_prev.index_copy_(0, starts, v_pool[init].to(v.dtype))
        self._blend(
            k_shift[:ndt],
            v_shift[:ndt],
            k[:ndt],
            v[:ndt],
            k_prev,
            v_prev,
            alpha_k_all[:ndt],
            alpha_v_all[:ndt],
            positions[:ndt],
        )
        # Store position t's raw K/V into column t.
        T = spec.max_qlen
        t_idx = torch.arange(T, device=k.device)
        active = t_idx.unsqueeze(0) < qlens.unsqueeze(1)  # (nd, T)
        src = torch.where(active, starts.unsqueeze(1) + t_idx.unsqueeze(0), 0).view(-1)
        if light:
            T = min(T, k_pool.shape[1])
            active = active[:, :T]
            src = torch.where(active, starts.unsqueeze(1) + t_idx[:T].unsqueeze(0), 0).view(-1)
            dslot = torch.where(active, spec.state_cols[:, :1].long(), 0).view(-1)
            dcol = torch.where(active, t_idx[:T].unsqueeze(0), 0).view(-1)
            k_pool[dslot, dcol] = k.index_select(0, src).to(k_pool.dtype)
            v_pool[dslot, dcol] = v.index_select(0, src).to(v_pool.dtype)
            return
        dest = torch.where(active, spec.state_cols[:, :T].long(), 0).view(-1)
        k_pool[dest] = k.index_select(0, src).to(k_pool.dtype)
        v_pool[dest] = v.index_select(0, src).to(v_pool.dtype)

    @staticmethod
    def _alloc_shift(
        k: torch.Tensor, v: torch.Tensor, num_covered: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Output buffers for the shift, with any uncovered tail passed through.

        Decode and prefill rows between them normally cover the whole batch,
        but a cudagraph-padded batch can carry trailing rows that neither
        branch writes. Leaving those uninitialized would feed uninitialized
        memory to attention, so copy the unshifted k/v into them; their
        outputs are discarded either way.
        """
        k_shift = torch.empty_like(k)
        v_shift = torch.empty_like(v)
        if num_covered < k.shape[0]:
            k_shift[num_covered:] = k[num_covered:]
            v_shift[num_covered:] = v[num_covered:]
        return k_shift, v_shift

    @staticmethod
    def _blend(
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_prev: torch.Tensor,
        v_prev: torch.Tensor,
        alpha_k: torch.Tensor,
        alpha_v: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        doc_start = (positions == 0).view(-1, 1, 1)
        a_k = torch.sigmoid(alpha_k.float()).unsqueeze(-1).masked_fill(doc_start, 0)
        a_v = torch.sigmoid(alpha_v.float()).unsqueeze(-1).masked_fill(doc_start, 0)
        k_prev = k_prev.masked_fill(doc_start, 0)
        v_prev = v_prev.masked_fill(doc_start, 0)
        k_out.copy_((a_k * k_prev.float() + (1 - a_k) * k.float()).to(k_out.dtype))
        v_out.copy_((a_v * v_prev.float() + (1 - a_v) * v.float()).to(v_out.dtype))

    def _shift_prefill_rows(
        self,
        md: OlalaDiffTPAMetadata,
        k: torch.Tensor,
        v: torch.Tensor,
        slots: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        alpha_k_all: torch.Tensor,
        alpha_v_all: torch.Tensor,
        positions: torch.Tensor,
        nd: int,
        np_: int,
        ndt: int,
        npt: int,
        k_shift: torch.Tensor,
        v_shift: torch.Tensor,
    ) -> None:
        """Shift-within-chunk for prefill rows, seeded from the cached K/V.

        Also stores each chunk's last raw K/V for the next chunk or decode.
        """
        rows = slice(ndt, ndt + npt)
        k_pref, v_pref = k[rows], v[rows]
        k_prev = torch.empty_like(k_pref)
        v_prev = torch.empty_like(v_pref)

        if md.state_indices_p_cpu is None:
            # OLALA_PREFILL_NOSYNC: no host copy of the slots; the same copies
            # with device indices, for every request at once.
            self._seed_prefill_rows_dev(
                md, k_pref, v_pref, k_prev, v_prev, slots, k_pool, v_pool, nd, np_
            )
            self._blend(
                k_shift[rows], v_shift[rows], k_pref, v_pref, k_prev, v_prev,
                alpha_k_all[rows], alpha_v_all[rows], positions[rows],
            )
            return

        qsl = md.query_start_loc_p_cpu.tolist()
        has_init = md.has_initial_state_cpu.tolist()
        pslots = md.state_indices_p_cpu
        for r in range(np_):
            s, e = qsl[r], qsl[r + 1]
            if e == s:
                continue
            slot = int(pslots[r])
            if has_init[r]:
                k_prev[s : s + 1] = k_pool[slot].to(k.dtype).unsqueeze(0)
                v_prev[s : s + 1] = v_pool[slot].to(v.dtype).unsqueeze(0)
            else:
                k_prev[s : s + 1] = 0
                v_prev[s : s + 1] = 0
            if e - s > 1:
                k_prev[s + 1 : e] = k_pref[s : e - 1]
                v_prev[s + 1 : e] = v_pref[s : e - 1]
            k_pool[slot] = k_pref[e - 1].to(k_pool.dtype)
            v_pool[slot] = v_pref[e - 1].to(v_pool.dtype)

        self._blend(
            k_shift[rows],
            v_shift[rows],
            k_pref,
            v_pref,
            k_prev,
            v_prev,
            alpha_k_all[rows],
            alpha_v_all[rows],
            positions[rows],
        )

    @staticmethod
    def _seed_prefill_rows_dev(
        md, k_pref, v_pref, k_prev, v_prev, slots, k_pool, v_pool, nd: int, np_: int
    ) -> None:
        """Previous-token K/V of the prefill rows and the pool update, from device
        slots: row s of a request is seeded from its pool slot (or zero without an
        initial state), every other row from the row before; each request's last
        raw K/V is then stored to its slot. Same values as the per-request loop
        (copies only; the slots of one step are distinct)."""
        qsl = md.query_start_loc_p_cpu.tolist()
        keep = [r for r in range(np_) if qsl[r + 1] > qsl[r]]
        if not keep:
            return
        qsl_d = md.query_start_loc_p
        pslots = slots[nd : nd + np_]
        init = md.has_initial_state[:np_]
        starts, ends = qsl_d[:np_], qsl_d[1 : np_ + 1]
        if len(keep) < np_:
            idx = torch.tensor(keep, dtype=torch.long).to(k_pref.device, non_blocking=True)
            pslots, init, starts, ends = pslots[idx], init[idx], starts[idx], ends[idx]
        pslots, starts, ends = pslots.long(), starts.long(), ends.long()
        if k_pref.shape[0] > 1:
            k_prev[1:] = k_pref[:-1]
            v_prev[1:] = v_pref[:-1]
        m = init.view(-1, *([1] * (k_pref.dim() - 1)))
        k_prev.index_copy_(0, starts, torch.where(m, k_pool[pslots].to(k_pref.dtype), 0))
        v_prev.index_copy_(0, starts, torch.where(m, v_pool[pslots].to(v_pref.dtype), 0))
        k_pool[pslots] = k_pref[ends - 1].to(k_pool.dtype)
        v_pool[pslots] = v_pref[ends - 1].to(v_pool.dtype)


def olala_diff_tpa(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    self = get_forward_context().no_compile_layers[layer_name]
    self._forward_impl(positions, hidden_states, output)


def olala_diff_tpa_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="olala_diff_tpa",
    op_func=olala_diff_tpa,
    mutates_args=["output"],
    fake_impl=olala_diff_tpa_fake,
)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Olala model for vLLM.

Olala is a hybrid stack:

* ``M`` layers — :class:`OlalaMamba3Mixer`, an SSM with four temporal states
  paged through the MAMBA3 backend.
* ``V`` layers — :class:`OlalaDiffTPAAttention`, differential tensor-product
  softmax attention over paged KV, plus a one-token shift buffer paged through
  the OLALA_DIFF_TPA side-channel backend.
* MLP — :class:`OlalaLatentMoE` (latent projection, 256 routed experts, one
  shared expert) or the dense :class:`OlalaMLP`.

Known fidelity gaps, flagged rather than papered over:

* vLLM's ``get_rope`` stands in for Olala's custom p-rope.
* ``scalable_softmax`` pre-scales q but the softmax scale itself is vLLM's;
  sliding-window bounds only the log position, not the attention mask.
* ``token_conv1d_attn``, ``xsa``, ``use_value_embedding`` and ``cosnet`` are
  not ported.
* TP splits the head-interleaved projections on the head axis. The Mamba3
  post-gate norm, when enabled, is a per-rank shard norm, which differs
  numerically from a global RMSNorm.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.mamba.olala.decode_gemv import decode_gemv, olala_linear
from vllm.model_executor.layers.mamba.olala.geo_linear import (
    GEO_FUSE,
    GEO_LINEAR_MAX_TOKENS,
    geo_linear,
)
from vllm.model_executor.layers.mamba.olala.latent_moe_decode import (
    FUSED_PROJ_MAX_TOKENS,
    SMALL_BATCH_MAX_TOKENS,
    SMALL_BATCH_MAX_TOKENS_WITH_MEDIUM,
    small_path_max_tokens,
    olala_moe_small,
    moe_cat_shared,
)
from vllm.model_executor.layers.mamba.olala.latent_moe_fused import olala_moe_fused
from vllm.model_executor.layers.mamba.olala.latent_moe_updown import (
    UPDOWN_ENABLED,
    olala_moe_updown,
)
from vllm.model_executor.layers.mamba.olala.latent_moe_medium import (
    MEDIUM_BATCH_MAX_TOKENS,
    olala_moe_medium,
)
from vllm.model_executor.layers.mamba.olala.w8a16_layers import (
    int8_apply,
    int8_linear_weight,
    is_int8_linear,
    is_int8_quant_method,
)
from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
    Segment,
    int8_decode_linear,
    int8_decode_max_m,
)
from vllm.model_executor.layers.mamba.olala.latent_moe_int8 import (
    INT8_SMALL_MAX_TOKENS,
    WqWeight,
    olala_moe_medium_int8,
    olala_moe_small_int8,
    wq_routed_experts,
)
from vllm.utils.torch_utils import _encode_layer_name, direct_register_custom_op
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    activation_without_mul,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.olala import (
    OlalaDiffTPAAttention,
    OlalaMamba3Mixer,
    OlalaNorm,
)
from vllm.model_executor.layers.mamba.olala.state import (
    mamba3_state_dtype,
    mamba3_state_shape,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

SUPPORTED_LAYER_TYPES = frozenset({"M", "V"})


class OlalaMLP(nn.Module):
    """Dense ReLU² MLP (``fc_1 -> relu² -> fc_2``) in Olala's naming."""

    def __init__(self, config, prefix: str = "", quant_config=None):
        super().__init__()
        self.fc_1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_1",
        )
        self.fc_2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_2",
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_1(x)
        x = self.act_fn(x)
        x, _ = self.fc_2(x)
        return x


class OlalaSharedMLP(nn.Module):
    """Shared-expert ReLU² MLP.

    The checkpoint's ``fc_1``/``fc_2`` are remapped to ``up_proj``/``down_proj``
    in :meth:`OlalaForCausalLM.load_weights`. Results are not reduced here —
    the MoE runner all-reduces the combined shared plus routed output once.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        quant_config=None,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


def _int8_moe_config(quant_config, prefix: str) -> bool:
    """ModelOpt MIXED_PRECISION layer with W8A16 INT8 (or W4A16_AFFINE INT4)
    routed experts and a W8A16 or bf16 shared expert (both projections alike)
    and latent projections: the weight-only paths of latent_moe_int8.py."""
    if quant_config is None or quant_config.get_name() != "modelopt_mixed":
        return False
    algo = quant_config._resolve_quant_algo
    shared = {algo(f"{prefix}.shared_experts.{p}") for p in ("up_proj", "down_proj")}
    return (
        algo(f"{prefix}.experts") in ("W8A16", "W4A16_AFFINE")
        and len(shared) == 1
        and shared.pop() in ("W8A16", None)
        and all(
            algo(f"{prefix}.{p}") in ("W8A16", None)
            for p in ("fc1_latent_proj", "fc2_latent_proj")
        )
    )


def _wq_weight(layer) -> WqWeight | None:
    """A W8A16 INT8 or bf16 linear as the INT8 MoE kernels read it."""
    if is_int8_linear(layer):
        w = int8_linear_weight(layer)
        return WqWeight(w.weight, w.scale) if w is not None else None
    if (
        isinstance(layer.quant_method, UnquantizedLinearMethod)
        and layer.weight.is_contiguous()
    ):
        return WqWeight(layer.weight)
    return None


@functools.cache
def _int8_skip_layers() -> frozenset[int]:
    raw = os.environ.get("OLALA_INT8_SKIP_LAYERS", "")
    return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x)


def _olala_router_logits(
    hidden_states: torch.Tensor, weight: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """fp32 router logits from bf16 activations and a bf16 router weight
    holding one ([E, K]) or two ([hi ; lo], [2E, K]) bf16 parts."""
    out = torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
    if weight.shape[0] == num_experts:
        return out
    return out[:, :num_experts] + out[:, num_experts:]


def _olala_router_logits_fake(
    hidden_states: torch.Tensor, weight: torch.Tensor, num_experts: int
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], num_experts), dtype=torch.float32
    )


direct_register_custom_op(
    op_name="olala_router_logits",
    op_func=_olala_router_logits,
    mutates_args=[],
    fake_impl=_olala_router_logits_fake,
)


class OlalaLatentMoE(nn.Module):
    """Latent MoE with sigmoid routing and non-gated ReLU² experts.

    Tokens are projected down to ``moe_routed_input_dim`` before the experts
    and back up afterwards; the MoE runner owns both transforms, the routed
    scaling factor, the shared-expert add and the all-reduce.
    """

    def __init__(self, config, parallel_config, prefix: str = "", quant_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.latent_size = config.moe_routed_input_dim
        self.num_experts = config.moe_num_routed_experts
        self.top_k = config.moe_num_active_experts
        if not self.latent_size:
            raise ValueError("OlalaLatentMoE requires moe_routed_input_dim")
        # --quantization experts_int8: routed experts W8A16 (per-channel int8
        # weights, bf16 activations), every linear stays bf16. Layers listed
        # in OLALA_INT8_SKIP_LAYERS keep bf16 experts too.
        experts_int8 = quant_config is not None and quant_config.get_name() == "experts_int8"
        experts_quant_config = quant_config
        if experts_int8 and extract_layer_index(prefix) in _int8_skip_layers():
            experts_quant_config = None

        self.gate = GateLinear(
            self.hidden_size,
            self.num_experts,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32)
        )

        shared_intermediate = config.moe_shared_intermediate_size
        if shared_intermediate and shared_intermediate > 0:
            self.shared_experts = OlalaSharedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                prefix=f"{prefix}.shared_experts",
                quant_config=quant_config,
            )
        else:
            self.shared_experts = None

        # Registered before `experts` so the latent projections keep their own
        # parameter names: the runner holds the same module objects, and
        # named_parameters reports whichever path it reaches first.
        self.fc1_latent_proj = ReplicatedLinear(
            input_size=self.hidden_size,
            output_size=self.latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.fc2_latent_proj = ReplicatedLinear(
            input_size=self.latent_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )

        self._int8 = (
            _int8_moe_config(quant_config, prefix)
            and self.shared_experts is not None
            and all(
                isinstance(lin.quant_method, UnquantizedLinearMethod)
                or is_int8_quant_method(lin)
                for lin in (
                    self.fc1_latent_proj,
                    self.fc2_latent_proj,
                    self.shared_experts.up_proj,
                    self.shared_experts.down_proj,
                )
            )
        )
        self._fc1_int8 = self._int8 and is_int8_quant_method(self.fc1_latent_proj)
        self._int8_entry_st = None
        self._int8_w = None

        # Single rank, unquantized: the latent projections and the shared
        # expert are handled here around a bare routed-experts FusedMoE — one
        # GEMM on [fc1_latent ; shared_up], routed experts on the latent slice,
        # one K-concatenated GEMM on [fc2_latent | shared_down] that also
        # absorbs the shared+routed add; the routed scale folds into the
        # router weights. Otherwise the runner owns transforms and shared.
        self._fused_proj = (
            self.shared_experts is not None
            and (quant_config is None or experts_int8 or self._int8)
            and get_tensor_model_parallel_world_size() == 1
            and not parallel_config.enable_eplb
            # EP dispatch is collective: every rank must take the runner path.
            and not parallel_config.enable_expert_parallel
        )
        moe_kwargs = dict(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.latent_size,
            intermediate_size=config.moe_routed_intermediate_size,
            renormalize=self.top_k > 1,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation=activation_without_mul("relu2"),
            # Non-gated experts: no separate up projection (vLLM >= 0.33 checks).
            ckpt_names=("up_proj", "down_proj", None),
            routed_scaling_factor=config.moe_routed_scaling_factor,
            router_logits_dtype=self.gate.out_dtype,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            quant_config=experts_quant_config,
            prefix=f"{prefix}.experts",
        )
        if self._fused_proj:
            self.experts = FusedMoEFactory(
                shared_experts=None, apply_routed_scale_to_output=False, **moe_kwargs
            )
        else:
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                routed_input_transform=self.fc1_latent_proj,
                routed_output_transform=self.fc2_latent_proj,
                apply_routed_scale_to_output=True,
                **moe_kwargs,
            )

        # Small-batch (decode) path: two gather-GEMV kernels instead of the
        # generic permute/grouped-gemm pipeline. Single GPU, unquantized only;
        # OLALA_MOE_SMALL=0 disables it.
        self.routed_scaling_factor = float(config.moe_routed_scaling_factor)

        # Router GEMM. The checkpoint router is fp32 and, off Hopper/Blackwell,
        # GateLinear keeps it fp32 (an SGEMM: 182 us at 2048 tokens on an RTX
        # A4500, 3% of a 2k-token prefill). bf16x2 (default) splits the fp32
        # weight into bf16 hi + lo halves and runs one bf16 tensor-core GEMM
        # with fp32 output: 16 mantissa bits of weight times exact bf16
        # activations, ~fp32 logits (test-set NLL unchanged to 1e-5), 2.3x
        # faster. bf16 keeps the hi half only, as the Hopper path does; fp32
        # is the GateLinear SGEMM. Only applies where GateLinear stores fp32.
        self._router_mode = os.environ.get("OLALA_ROUTER_GEMM", "bf16x2")
        if self._router_mode not in ("fp32", "bf16", "bf16x2"):
            raise ValueError(f"OLALA_ROUTER_GEMM={self._router_mode!r}")
        self._router_w: torch.Tensor | None = None
        self._small_w: tuple[torch.Tensor, torch.Tensor] | None = None
        self._small_path = (
            self._fused_proj
            and os.environ.get("OLALA_MOE_SMALL", "1") != "0"
            and (not self._int8 or os.environ.get("OLALA_MOE_INT8", "1") != "0")
        )

        # Medium batches (decode): fused routing/sort, relu^2 in the up GEMM,
        # top-k sum fused with the shared concat. Unquantized experts only;
        # OLALA_MOE_MEDIUM=0 disables it.
        self._medium_path = (
            self._small_path
            and (quant_config is None or self._int8)
            and os.environ.get("OLALA_MOE_MEDIUM", "1") != "0"
        )
        # Opt-in: medium batches through one persistent kernel (routing,
        # compaction, up, relu^2, down; latent_moe_fused.py).
        self._fused_moe_path = (
            self._medium_path
            and not self._int8
            and os.environ.get("OLALA_MOE_FUSED", "0") == "1"
        )

        # Decode batches (small / medium paths): the router GEMM is merged
        # into the input GEMM inside the op. Needs the bf16 router weight that
        # GateLinear keeps on Hopper / Blackwell. OLALA_MOE_ENTRY=0 disables.
        self._entry_w: torch.Tensor | None = None
        self._entry_path = (
            self._small_path
            and self.gate.weight.dtype == torch.bfloat16
            and self.gate.weight.dtype == self.fc1_latent_proj.weight.dtype
            and os.environ.get("OLALA_MOE_ENTRY", "1") != "0"
        )

        self._entry_router_parts = 1
        if (
            self._int8
            and self._small_path
            and not self._entry_path
            and self.gate.weight.dtype == torch.float32
            and (self.fc1_latent_proj.weight.dtype == torch.bfloat16 or self._fc1_int8)
            and self._router_mode in ("bf16", "bf16x2")
            and os.environ.get("OLALA_MOE_ENTRY", "1") != "0"
        ):
            self._entry_path = True
            self._entry_router_parts = 2 if self._router_mode == "bf16x2" else 1

        # Registered like the mixers so the MoE call is one opaque custom op
        # and the small/generic dispatch runs inside it (see forward_dispatch).
        self.registry_name = f"{prefix}.latent_moe"
        compilation = get_current_vllm_config().compilation_config
        if self.registry_name in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.registry_name}")
        compilation.static_forward_context[self.registry_name] = self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        hidden_states = x.reshape(-1, orig_shape[-1])
        if self._entry_path:
            # The router runs inside the op, merged with the input GEMM.
            out = torch.ops.vllm.olala_latent_moe(
                hidden_states, None, _encode_layer_name(self.registry_name)
            )
            return out.reshape(orig_shape)
        if self._router_w is not None:
            router_logits = torch.ops.vllm.olala_router_logits(
                hidden_states, self._router_w, self.num_experts
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        out = torch.ops.vllm.olala_latent_moe(
            hidden_states, router_logits, _encode_layer_name(self.registry_name)
        )
        return out.reshape(orig_shape)

    def forward_geo(
        self, residual: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h = self._geo(residual, g)`` and the MoE of ``h``, with the
        geodesic update fused into the entry GEMM at decode sizes."""
        h = torch.empty_like(residual)
        out = torch.ops.vllm.olala_latent_moe(
            residual, None, _encode_layer_name(self.registry_name), g, h
        )
        return h, out

    def forward_dispatch_geo(
        self, residual: torch.Tensor, g: torch.Tensor, h_out: torch.Tensor
    ) -> torch.Tensor:
        if residual.shape[0] <= GEO_LINEAR_MAX_TOKENS:
            y = geo_linear(
                residual, g, self._geo, self._entry_weight(), h_out, torch.float32
            )
            n_in = y.shape[1] - self.num_experts
            return self.forward_dispatch(h_out, y[:, n_in:], x_in=y[:, :n_in])
        self._geo(residual, g, out=h_out)
        return self.forward_dispatch(h_out, None)

    def forward_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        x_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._entry_path and self._entry_w is None:
            # First call is the profiling run, before CUDA graph capture: a
            # cache built during capture would live in a graph's private pool.
            self._entry_weight()
        if self._int8:
            return self._forward_int8(hidden_states, router_logits, x_in)
        re = self.experts.routed_experts if self._small_path else None
        # int8 experts on sm8x / sm90: the CUDA small path covers up to 32 tokens
        small_max = (
            SMALL_BATCH_MAX_TOKENS_WITH_MEDIUM
            if self._medium_path
            else small_path_max_tokens(re.w13_weight)
            if re is not None
            else SMALL_BATCH_MAX_TOKENS
        )
        if router_logits is None:
            m = hidden_states.shape[0]
            if m <= small_max or (self._medium_path and m <= MEDIUM_BATCH_MAX_TOKENS):
                # One GEMM for [fc1_latent ; shared_up ; router] with fp32
                # output: the logits match GateLinear's bf16 x bf16 -> fp32,
                # and the kernels round x_in to bf16 on load, i.e. exactly a
                # bf16 GEMM output.
                y = torch.mm(
                    hidden_states, self._entry_weight().T, out_dtype=torch.float32
                )
                n_in = y.shape[1] - self.num_experts
                x_in, router_logits = y[:, :n_in], y[:, n_in:]
            else:
                router_logits, _ = self.gate(hidden_states)
        if re is not None and hidden_states.shape[0] <= small_max:
            w_in, w_out = self._small_path_weights()
            return olala_moe_small(
                hidden_states,
                router_logits,
                self.gate.e_score_correction_bias,
                w_in,
                w_out,
                re.w13_weight,
                re.w2_weight,
                self.top_k,
                self.routed_scaling_factor,
                w13_scale=getattr(re, "w13_scale", None),
                w2_scale=getattr(re, "w2_scale", None),
                x_in=x_in,
            )
        if self._medium_path and hidden_states.shape[0] <= MEDIUM_BATCH_MAX_TOKENS:
            w_in, w_out = self._small_path_weights()
            re = self.experts.routed_experts
            if UPDOWN_ENABLED:
                medium = olala_moe_updown
            elif self._fused_moe_path:
                medium = olala_moe_fused
            else:
                medium = olala_moe_medium
            return medium(
                hidden_states,
                router_logits,
                self.gate.e_score_correction_bias,
                w_in,
                w_out,
                re.w13_weight,
                re.w2_weight,
                self.top_k,
                self.routed_scaling_factor,
                x_in=x_in,
            )
        if self._fused_proj and hidden_states.shape[0] <= FUSED_PROJ_MAX_TOKENS:
            w_in, w_out = self._small_path_weights()
            x_in = decode_gemv(hidden_states, w_in)
            latent = x_in[:, : self.latent_size].contiguous()
            routed = self.experts(hidden_states=latent, router_logits=router_logits)
            cat = moe_cat_shared(routed, x_in[:, self.latent_size :])
            return decode_gemv(cat, w_out)
        if self._fused_proj:
            # Large prefill batches: the concatenation traffic outweighs the
            # saved launches, so run the projections separately.
            latent, _ = self.fc1_latent_proj(hidden_states)
            routed = self.experts(hidden_states=latent, router_logits=router_logits)
            out, _ = self.fc2_latent_proj(routed)
            if current_platform.is_cuda():
                # shared expert: one relu^2 kernel (this runs eagerly inside the opaque MoE op, where the
                # native activation is two kernels) and the shared + routed add in the down GEMM (beta = 1)
                s, _ = self.shared_experts.up_proj(hidden_states)
                torch.ops._C.relu_squared(s, s)
                return torch.addmm(out, s, self.shared_experts.down_proj.weight.t())
            return out + self.shared_experts(hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)

    def _forward_int8(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        x_in: torch.Tensor | None,
    ) -> torch.Tensor:
        """W8A16 INT8 (or W4A16 INT4 affine) routed + W8A16 shared experts: the
        small / medium paths of latent_moe_int8.py up to MEDIUM_BATCH_MAX_TOKENS
        tokens, its sorted grouped GEMMs (or the FusedMoE, Triton int8_w8a16,
        for per-row INT8) and W8A16 linears beyond (prefill)."""
        m = hidden_states.shape[0]
        weights = self._int8_weights()
        small = m <= INT8_SMALL_MAX_TOKENS
        if self._small_path and (
            small or (self._medium_path and m <= MEDIUM_BATCH_MAX_TOKENS)
        ):
            logits_lo = None
            if router_logits is None:
                if self._fc1_int8:
                    y = self._int8_entry(hidden_states)
                else:
                    y = torch.mm(
                        hidden_states, self._entry_weight().t(), out_dtype=torch.float32
                    )
                d, e = self.latent_size, self.num_experts
                x_in, router_logits = y[:, :d], y[:, d : d + e]
                if self._entry_router_parts == 2:
                    logits_lo = y[:, d + e :]
                    if not small:
                        router_logits = router_logits + logits_lo
            elif x_in is None:
                x_in = (
                    int8_apply(self.fc1_latent_proj, hidden_states)
                    if self._fc1_int8
                    else decode_gemv(hidden_states, self.fc1_latent_proj.weight)
                )
            args = (
                hidden_states,
                x_in,
                router_logits,
                self.gate.e_score_correction_bias,
                *weights,
                self.top_k,
                self.routed_scaling_factor,
            )
            if small:
                return olala_moe_small_int8(*args, router_logits_lo=logits_lo)
            return olala_moe_medium_int8(*args)
        if router_logits is None:
            if self._router_w is not None:
                router_logits = torch.ops.vllm.olala_router_logits(
                    hidden_states, self._router_w, self.num_experts
                )
            else:
                router_logits, _ = self.gate(hidden_states)
        latent = self.fc1_latent_proj(hidden_states)[0]
        if self._int8_own_prefill(weights[0]):
            # Grouped weights require the Olala expert kernel.
            routed = wq_routed_experts(
                latent,
                router_logits,
                self.gate.e_score_correction_bias,
                weights[0],
                weights[1],
                self.top_k,
                self.routed_scaling_factor,
            )
        else:
            routed = self.experts(hidden_states=latent, router_logits=router_logits)
        out = self.fc2_latent_proj(routed)[0]
        return out + self.shared_experts(hidden_states)

    @staticmethod
    def _int8_own_prefill(w13: WqWeight) -> bool:
        """Prefill routed experts through latent_moe_int8.wq_routed_experts
        (always for per-group scales and INT4; per-row ones too by default:
        OLALA_INT8_MOE_PREFILL=vllm takes vLLM's Triton int8_w8a16 FusedMoE,
        2 % slower on 64 x 1k / 8 x 8k prefills on an RTX 5090)."""
        mode = os.environ.get("OLALA_INT8_MOE_PREFILL", "olala")
        return w13.scale is not None and (
            w13.scale.dim() == 3
            or (mode == "olala" and os.environ.get("OLALA_MOE_INT8", "1") != "0")
        )

    def _int8_entry(self, h: torch.Tensor) -> torch.Tensor:
        """``[fc1_latent | router parts]`` in fp32 with an INT8 fc1_latent: one
        two-segment W8A16 launch (INT8 fc1 rows + bf16 router rows) at decode
        sizes; the W8A16 linear and an fp32-output router GEMM beyond."""
        if self._int8_entry_st is None:
            w8 = int8_linear_weight(self.fc1_latent_proj)
            if w8 is None:
                raise RuntimeError("fc1_latent_proj: unsupported INT8 weight layout")
            router = self._entry_weight()
            segs = [Segment(w8.weight, w8.scale), Segment(router, col0=w8.n)]
            self._int8_entry_st = (segs, w8, router)
        segs, w8, router = self._int8_entry_st
        m, width = h.shape[0], w8.n + router.shape[0]
        out = torch.empty(m, width, dtype=torch.float32, device=h.device)
        if 0 < m <= int8_decode_max_m(width, w8.k, router.shape[0]):
            return int8_decode_linear(h, segs, out=out)
        out[:, : w8.n] = int8_apply(self.fc1_latent_proj, h)
        out[:, w8.n :] = torch.mm(h, router.t(), out_dtype=torch.float32)
        return out

    def _int8_weights(self) -> tuple[WqWeight, ...]:
        """(w13, w2, shared_up, shared_down, fc2_latent) as the INT8 MoE kernels
        read them, built on the first call (the profiling run, before graph
        capture)."""
        if self._int8_w is None:
            re = self.experts.routed_experts
            w13, w2 = re.w13_weight, re.w2_weight
            int4 = hasattr(re, "w4a16_affine_group")
            if int4:  # W4A16_AFFINE: interleaved scale / bias (modelopt_w8a16)
                s13, s2 = getattr(re, "w13_sb", None), getattr(re, "w2_sb", None)
            else:
                s13, s2 = getattr(re, "w13_scale", None), getattr(re, "w2_scale", None)
            ok = (
                w13.dtype == w2.dtype == (torch.uint8 if int4 else torch.int8)
                and w13.is_contiguous()
                and w2.is_contiguous()
                and s13 is not None
                and s2 is not None
            )
            dense = [
                _wq_weight(lin)
                for lin in (
                    self.shared_experts.up_proj,
                    self.shared_experts.down_proj,
                    self.fc2_latent_proj,
                )
            ]
            if not ok or None in dense:
                raise ValueError("Unsupported Olala weight-only MoE parameter layout.")
            elif int4:
                self._int8_w = (WqWeight(w13, s13), WqWeight(w2, s2), *dense)
            else:
                # per-row scales fp32, per-group ones as stored (fp16)
                self._int8_w = (
                    WqWeight(
                        w13, (s13 if s13.dim() == 3 else s13.float()).contiguous()
                    ),
                    WqWeight(w2, (s2 if s2.dim() == 3 else s2.float()).contiguous()),
                    *dense,
                )
        return self._int8_w

    def _refresh_router_weight(self) -> None:
        w = self.gate.weight
        if self._router_mode == "fp32" or w.dtype != torch.float32 or not w.is_cuda:
            return
        with torch.inference_mode():
            hi = w.to(torch.bfloat16)
            parts = [hi]
            if self._router_mode == "bf16x2":
                parts.append((w - hi.float()).to(torch.bfloat16))
            new = torch.cat(parts, 0).contiguous()
            if self._router_w is None:
                self._router_w = new
            else:
                self._router_w.copy_(new)

    def _small_path_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenated projections for the small-batch path.

        ``w_in = [fc1_latent ; shared_up]`` shares the input GEMV and
        ``w_out = [fc2_latent | shared_down]`` folds the shared+routed add into
        one K-concatenated GEMV. Built once from the live parameters; call
        refresh_weight_caches() after an in-place weight reload.
        """
        if self._small_w is None:
            w_in = torch.cat(
                [self.fc1_latent_proj.weight, self.shared_experts.up_proj.weight], 0
            ).contiguous()
            w_out = torch.cat(
                [self.fc2_latent_proj.weight, self.shared_experts.down_proj.weight], 1
            ).contiguous()
            self._small_w = (w_in, w_out)
        return self._small_w

    def _entry_weight(self) -> torch.Tensor:
        """``[fc1_latent ; shared_up ; router]`` for the merged entry GEMM
        (``[fc1_latent ; router]`` with a weight-only shared expert)."""
        if self._entry_w is None:
            router = self._entry_router_rows()
            if self._fc1_int8:
                # the router rows only: _int8_entry runs the INT8 fc1 next to them
                self._entry_w = router.clone()
                return self._entry_w
            if self._int8:
                w_in = self.fc1_latent_proj.weight
            else:
                w_in, _ = self._small_path_weights()
            self._entry_w = torch.cat([w_in, router], 0).contiguous()
        return self._entry_w

    def _entry_router_rows(self) -> torch.Tensor:
        if self.gate.weight.dtype == torch.bfloat16:
            return self.gate.weight
        if self._router_w is None:
            self._refresh_router_weight()
        return self._router_w

    def refresh_weight_caches(self) -> None:
        """Recompute the concatenated small-path projections into their
        existing storage (CUDA graphs bake their addresses; see the Mamba3
        mixer's refresh_weight_caches)."""
        self._refresh_router_weight()
        if self._entry_w is not None:
            with torch.inference_mode():
                router = self._entry_router_rows()
                n_in = self._entry_w.shape[0] - router.shape[0]
                w_fc1 = self.fc1_latent_proj.weight
                if self._int8:
                    if n_in:
                        self._entry_w[:n_in].copy_(w_fc1)
                else:
                    w_up = self.shared_experts.up_proj.weight
                    self._entry_w[:n_in].copy_(torch.cat([w_fc1, w_up]))
                self._entry_w[n_in:].copy_(router)
        if self._small_w is None:
            return
        w_in, w_out = self._small_w
        with torch.inference_mode():
            n1 = self.fc1_latent_proj.weight.shape[0]
            w_in[:n1].copy_(self.fc1_latent_proj.weight)
            w_in[n1:].copy_(self.shared_experts.up_proj.weight)
            k1 = self.fc2_latent_proj.weight.shape[1]
            w_out[:, :k1].copy_(self.fc2_latent_proj.weight)
            w_out[:, k1:].copy_(self.shared_experts.down_proj.weight)


@triton.jit
def _geodesic_norm_kernel(
    x_ptr,
    g_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    inv_depth,
    clamp_val,
    D,
    stride_x,
    stride_g,
    stride_o,
    BLOCK: tl.constexpr,
):
    """Fused geodesic residual update, one program per token row.

    A single load of ``x``/``g``, fp32 reductions and a single store replace
    the ~15 small elementwise/reduction kernels of the eager reference, which
    otherwise dominate decode launch overhead at two calls per layer. Uses
    ``||g - c*x||² = ||g||² - 2c(x·g) + c²||x||²`` to avoid a second pass.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)

    xx = tl.sum(x * x, axis=0)
    xg = tl.sum(x * g, axis=0)
    gg = tl.sum(g * g, axis=0)

    coeff = xg / tl.maximum(xx, 1e-12)
    tangent_sq = gg - 2.0 * coeff * xg + coeff * coeff * xx
    safe_tangent = tl.maximum(tl.sqrt(tl.maximum(tangent_sq, 0.0)), 1e-8)
    R = tl.sqrt(xx)
    safe_R = tl.maximum(R, 1e-6)

    scale = tl.load(scale_ptr).to(tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    theta = tl.minimum(safe_tangent / safe_R, clamp_val)
    # Inductor passes the Python-float args as fp64; keep the row update fp32
    # (fp64 is ~4x slower here, far more on GPUs with little fp64 throughput).
    theta = tl.minimum((theta * scale + bias) * inv_depth, clamp_val).to(tl.float32)

    out = x * tl.cos(theta) + (g - coeff * x) * (safe_R * tl.sin(theta) / safe_tangent)
    tl.store(
        out_ptr + row * stride_o + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# OLALA_GEODESIC_OP=0: launch the kernel inline in the compiled graph (old behavior, for A/B)
_GEODESIC_INLINE = os.environ.get("OLALA_GEODESIC_OP", "1") == "0"


def _olala_geodesic_norm(
    x: torch.Tensor,
    g: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    inv_depth: float,
    clamp: float,
) -> torch.Tensor:
    out = torch.empty_like(x)
    D = x.shape[-1]
    _geodesic_norm_kernel[(x.numel() // D,)](
        x,
        g,
        out,
        scale,
        bias,
        inv_depth,
        clamp,
        D,
        x.stride(-2) if x.dim() > 1 else 0,
        g.stride(-2) if g.dim() > 1 else 0,
        out.stride(-2) if out.dim() > 1 else 0,
        BLOCK=triton.next_power_of_2(D),
        num_warps=2,  # measured fastest at batch 1 (1.56 us) and batch 256
    )
    return out


def _olala_geodesic_norm_fake(
    x: torch.Tensor,
    g: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    inv_depth: float,
    clamp: float,
) -> torch.Tensor:
    return torch.empty_like(x)


# Opaque to torch.compile: Inductor re-generates a user Triton kernel traced
# inside the graph, and its copy of this one ran 7x slower (260 us instead of
# 37 us for 2048 x 1536 on an RTX A4500, 12% of a 2k-token prefill). As a
# custom op the kernel is JIT-specialized on the real arguments like in eager.
direct_register_custom_op(
    op_name="olala_geodesic_norm",
    op_func=_olala_geodesic_norm,
    mutates_args=[],
    fake_impl=_olala_geodesic_norm_fake,
)


class OlalaGeodesicNorm(nn.Module):
    """Geodesic residual update, port of Olala's ``OlalaGeodesicNorm``.

    Blends the residual ``x`` with the mixer/MLP output ``g`` along the tangent
    direction on the hypersphere, bounded by a per-layer rotation angle
    ``theta = clamp(tangent_norm / R) * scale + bias``, divided by
    ``layer_idx + 1``. Checkpoint tensors: ``scale``, ``bias``, and the
    ``prosres_scalar`` buffer.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # 1-D ([1]) rather than 0-dim: matches the FSDP-compatible modeling
        # code (FSDP rejects 0-dim params), so RL weight syncs can copy the
        # trainer's [1]-shaped tensors in place (a 0-dim param cannot receive
        # a [1] tensor: "output with shape [] doesn't match the broadcast
        # shape [1]"). copy_ broadcasts 0-dim checkpoint tensors into [1], so
        # loading legacy checkpoints still works. Numerically identical —
        # only used as ``theta * scale + bias`` with theta (..., 1).
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("prosres_scalar", torch.tensor(1.0))
        self.clamp = torch.pi / 4.0

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not (x.is_cuda and x.stride(-1) == 1 and g.stride(-1) == 1):
            y = self.forward_native(x, g)
            return y if out is None else out.copy_(y)
        if out is None:
            # an opaque op inside the compiled graph (see olala_geodesic_norm)
            if _GEODESIC_INLINE:
                return _olala_geodesic_norm(
                    x, g, self.scale, self.bias, 1.0 / (self.layer_idx + 1), self.clamp
                )
            return torch.ops.vllm.olala_geodesic_norm(
                x, g, self.scale, self.bias, 1.0 / (self.layer_idx + 1), self.clamp
            )
        # into the caller's buffer: called from inside the opaque MoE op (eager)
        D = x.shape[-1]
        _geodesic_norm_kernel[(x.numel() // D,)](
            x,
            g,
            out,
            self.scale,
            self.bias,
            1.0 / (self.layer_idx + 1),
            self.clamp,
            D,
            x.stride(-2) if x.dim() > 1 else 0,
            g.stride(-2) if g.dim() > 1 else 0,
            out.stride(-2) if out.dim() > 1 else 0,
            BLOCK=triton.next_power_of_2(D),
            num_warps=2,  # measured fastest at batch 1 (1.56 us) and batch 256
        )
        return out

    def forward_native(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Eager reference. Kept as the non-CUDA path and the parity baseline."""
        x_norm_sq = x.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proj_coeff = (x * g).sum(dim=-1, keepdim=True) / x_norm_sq
        gradient = g - proj_coeff * x
        safe_tangent_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        unit_tangent = gradient / safe_tangent_norm
        safe_R = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        theta = (safe_tangent_norm / safe_R).clamp_max(self.clamp)
        theta = ((theta * self.scale + self.bias) / (self.layer_idx + 1)).clamp_max(
            self.clamp
        )
        return x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)


class OlalaMonoBlock(nn.Module):
    """Single-mixer block: pre-norm, mixer, optional gate, projection, MLP."""

    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if layer_type not in SUPPORTED_LAYER_TYPES:
            raise NotImplementedError(
                f"OlalaMonoBlock: layer type {layer_type!r} is not ported "
                f"(supported: {sorted(SUPPORTED_LAYER_TYPES)})."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        # Quantization applies only to the dense GEMMs — the block gate and
        # output projections plus the MLP/MoE. The mixers stay bf16: their
        # projections feed the mamba3/rotary/attention kernels, which are
        # numerically sensitive and have no fp8 path.
        quant_config = vllm_config.quant_config

        if layer_type == "M":
            self.mixer: nn.Module = OlalaMamba3Mixer(
                config, vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )
            head_dim = self.mixer.headdim
            num_heads = self.mixer.nheads
            self.use_gate = False  # mamba3 gates internally via z
        else:
            self.mixer = OlalaDiffTPAAttention(
                config, vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )
            head_dim = self.mixer.head_dim
            num_heads = self.mixer.num_signal_heads
            self.use_gate = bool(config.gate_attn)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mixer_out_dim = head_dim * num_heads

        tp = get_tensor_model_parallel_world_size()
        if self.num_heads % tp:
            raise ValueError(
                f"OlalaMonoBlock: num_heads={self.num_heads} is not divisible "
                f"by tensor_parallel_size={tp}"
            )
        self.num_heads_local = self.num_heads // tp
        self.mixer_out_dim_local = self.head_dim * self.num_heads_local

        if self.use_gate:
            if config.gate_type != "elementwise":
                raise NotImplementedError(
                    f"OlalaMonoBlock: gate_type={config.gate_type!r} is not "
                    f"ported (only 'elementwise' is supported)."
                )
            # Column-parallel on the output head axis, so each rank produces
            # its own (num_heads_local * head_dim) gate.
            self.gate_proj = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=self.mixer_out_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
            self.gate_bias = 1.15 if config.zero_centered_gate else 0.0
            if config.gate_act == "silu":
                self.gate_act: Callable[[torch.Tensor], torch.Tensor] = (
                    torch.nn.functional.silu
                )
            elif config.gate_act == "sigmoid":
                self.gate_act = torch.sigmoid
            else:
                raise NotImplementedError(
                    f"OlalaMonoBlock: gate_act={config.gate_act!r} is not ported."
                )

        # Row-parallel: each rank matmuls its own head slice, then all-reduces.
        self.mixer_proj = RowParallelLinear(
            input_size=self.mixer_out_dim,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer_proj",
        )

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        self.is_geodesic = bool(config.geodesic_update)
        if self.is_geodesic:
            self.input_norm: nn.Module = nn.Identity()
            self.postmixer_norm: nn.Module = nn.Identity()
            self.geodesic_mixer = OlalaGeodesicNorm(layer_idx)
            self.geodesic_mlp = OlalaGeodesicNorm(layer_idx)
        else:
            self.input_norm = OlalaNorm(config.hidden_size, eps=eps, zero_centered=zc)
            self.postmixer_norm = OlalaNorm(
                config.hidden_size, eps=eps, zero_centered=zc
            )

        if config.moe:
            self.mlp: nn.Module = OlalaLatentMoE(
                config,
                vllm_config.parallel_config,
                prefix=f"{prefix}.mlp",
                quant_config=quant_config,
            )
        else:
            if config.mlp_type != "simple":
                raise NotImplementedError(
                    f"OlalaMonoBlock: mlp_type={config.mlp_type!r} is not "
                    f"ported (only 'simple' is supported)."
                )
            self.mlp = OlalaMLP(
                config, prefix=f"{prefix}.mlp", quant_config=quant_config
            )

        # Residual scales. lns == 1.0 and the identity norms of geodesic mode
        # would still launch a scalar-mul kernel per call — 72 wasted kernels
        # per step across the stack — so the multiply is skipped when it is a
        # no-op (see forward).
        # Unquantized single-rank mixer_proj: opaque op with the batch-1
        # streaming GEMV inside (see decode_gemv).
        # experts_int8 leaves it unquantized too (OLALA_PLAIN_MIXER=0: old quant_config gate, for A/B);
        # through the compiled graph cuBLAS picks 2-3x slower kernels at 16-64 tokens under CUDA graphs.
        unquantized = (
            isinstance(self.mixer_proj.quant_method, UnquantizedLinearMethod)
            if os.environ.get("OLALA_PLAIN_MIXER", "1") == "1"
            else quant_config is None
        )
        self._plain_mixer_proj = (
            unquantized
            and get_tensor_model_parallel_world_size() == 1
            and self.mixer_proj.bias is None
        )
        self._plain_gate_proj = (
            self.use_gate
            and self._plain_mixer_proj
            and isinstance(self.gate_proj.quant_method, UnquantizedLinearMethod)
            and self.gate_proj.bias is None
        )
        self.lns = 1.0 / math.sqrt(layer_idx + 1) if config.layer_norm_scaling else 1.0
        # OLALA_GEO_FUSE=1: the geodesic updates run inside the next
        # projection (MoE entry GEMM; next Mamba layer's in_proj), see
        # geo_linear.py. The block then returns the pending (residual, g).
        self.geo_fuse = (
            GEO_FUSE
            and self.is_geodesic
            and self.lns == 1.0
            and isinstance(self.mlp, OlalaLatentMoE)
            and self.mlp._entry_path
            and self.mlp._medium_path
            and not self.mlp._int8
            and get_pp_group().world_size == 1
        )
        if self.geo_fuse:
            # Plain reference, not a submodule (keeps parameter names unique).
            object.__setattr__(self.mlp, "_geo", self.geodesic_mixer)
        if config.use_completed_p:
            depth_ratio = len(config.layers_config) / config.base_depth
            self.a = float(depth_ratio ** (-config.completed_p_alpha))
        else:
            self.a = 1.0
        self.b = 1.0

    def _mix(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        if self.layer_type == "M":
            return self.mixer(hidden_states)
        return self.mixer(positions, hidden_states)

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pending: tuple[torch.Tensor, "OlalaGeodesicNorm"] | None = None,
    ):
        """Flat-token forward: ``(N, D)`` in, ``(N, D)`` out. With geo_fuse,
        ``hidden_states`` is the previous block's residual, ``pending`` its
        (MLP output, geodesic module) still to apply, and the block returns
        ``(residual, pending)`` the same way."""
        n = hidden_states.shape[0]

        if pending is not None and self.layer_type == "M" and self.geo_fuse:
            residual, y_mix = self.mixer.forward_geo(hidden_states, pending[0])
        else:
            if pending is not None:
                hidden_states = pending[1](hidden_states, pending[0])
            residual = hidden_states
            x = self.input_norm(hidden_states)
            if self.lns != 1.0:
                x = self.lns * x
            y_mix = self._mix(x, positions)  # (N, H_local * D)
        if self.use_gate:
            if self._plain_gate_proj:
                g_all = olala_linear(x, self.gate_proj.weight)
            else:
                g_all, _ = self.gate_proj(x)
            g = self.gate_act(
                g_all.view(n, self.num_heads_local, self.head_dim) + self.gate_bias
            ).to(y_mix.dtype)
            y_mix = (y_mix.view(n, self.num_heads_local, self.head_dim) * g).reshape(
                n, self.mixer_out_dim_local
            )
        if self._plain_mixer_proj:
            y_mix = olala_linear(y_mix, self.mixer_proj.weight)
        else:
            y_mix, _ = self.mixer_proj(y_mix)

        if self.geo_fuse:
            residual, y_mlp = self.mlp.forward_geo(residual, y_mix)
            return residual, (y_mlp, self.geodesic_mlp)
        if self.is_geodesic:
            hidden_states = self.geodesic_mixer(residual, y_mix)
        else:
            hidden_states = self.b * residual + self.a * y_mix

        residual = hidden_states
        x = self.postmixer_norm(hidden_states)
        if self.lns != 1.0:
            x = self.lns * x
        y_mlp = self.mlp(x)
        if self.is_geodesic:
            return self.geodesic_mlp(residual, y_mlp)
        return self.b * residual + self.a * y_mlp


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class OlalaModel(nn.Module):
    """Token embedding, a stack of :class:`OlalaMonoBlock`, optional norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embeddings_prenormalized = False

        self.embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        def build_block(prefix: str) -> OlalaMonoBlock:
            idx = int(prefix.rsplit(".", 1)[1])
            return OlalaMonoBlock(
                config=config,
                vllm_config=vllm_config,
                layer_idx=idx,
                layer_type=config.layers_config[idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.layers_config), build_block, prefix=f"{prefix}.layers"
        )
        # geo_fuse: a Mamba mixer applies the previous block's MLP geodesic
        # update inside its in_proj (see OlalaMonoBlock.forward).
        for prev, layer in zip(
            self.layers[self.start_layer : self.end_layer - 1],
            self.layers[self.start_layer + 1 : self.end_layer],
        ):
            if prev.geo_fuse and layer.geo_fuse and layer.layer_type == "M":
                object.__setattr__(layer.mixer, "_pre_geo", prev.geodesic_mlp)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

        if not get_pp_group().is_last_rank:
            self.final_norm: nn.Module = PPMissingLayer()
        elif config.final_norm:
            self.final_norm = OlalaNorm(
                config.hidden_size,
                eps=config.norm_epsilon,
                zero_centered=getattr(config, "zero_centered_gamma", False),
            )
        else:
            self.final_norm = nn.Identity()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            from_table = inputs_embeds is None
            hidden_states = (
                self.embed_input_ids(input_ids) if from_table else inputs_embeds
            )
            if self.config.normalize_embeddings_ngpt and not (
                from_table and self.embeddings_prenormalized
            ):
                # Table rows are pre-normalized at load time where possible;
                # externally supplied embeddings still normalize here.
                hidden_states = torch.nn.functional.normalize(hidden_states, dim=-1)
            if self.config.normalize_embeddings:
                hidden_states = torch.nn.functional.normalize(
                    hidden_states, dim=-1
                ) * math.sqrt(self.config.hidden_size)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        pending = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            if layer.geo_fuse:
                hidden_states, pending = layer(
                    positions=positions, hidden_states=hidden_states, pending=pending
                )
            else:
                if pending is not None:
                    hidden_states = pending[1](hidden_states, pending[0])
                    pending = None
                hidden_states = layer(positions=positions, hidden_states=hidden_states)
        if pending is not None:
            hidden_states = pending[1](hidden_states, pending[0])

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.final_norm(hidden_states)


def moe_layer_indices(config) -> list[int]:
    """Indices of the layers that carry a MoE MLP.

    ``layers_mlp_config`` is either empty — every layer follows ``config.moe``
    — or a per-layer ``'m'``/``'d'`` string.
    """
    if config.layers_mlp_config:
        return [i for i, c in enumerate(config.layers_mlp_config) if c == "m"]
    return list(range(len(config.layers_config))) if config.moe else []


class OlalaForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
    SupportsLoRA,
):
    """Olala causal LM with paged KV and paged recurrent state."""

    # Olala's projection names (c_q / W_A_k / in_proj / in_proj_dyn / ...) do
    # not map onto vLLM's qkv_proj / gate_up_proj packing, so the default
    # packed-module handling is disabled.
    packed_modules_mapping: dict[str, list[str]] = {}

    embedding_modules = {
        "embedding": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        # Representative: the larger Mamba3 state.
        return mamba3_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        hf_config = vllm_config.model_config.hf_config
        d_inner = 2 * hf_config.hidden_size
        split = int(hf_config.mamba_d_state * 0.5)  # rope_fraction
        if split % 2:
            split -= 1
        return mamba3_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            num_heads=d_inner // hf_config.mamba_headdim,
            head_dim=hf_config.mamba_headdim,
            d_state=hf_config.mamba_d_state,
            mimo_dim=hf_config.mamba_mimo_dim,
            num_rope_angles=split // 2,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        # Align mode assumes every mamba layer exposes the same number of
        # state tensors; Olala's M layers have four and its V layers two, so
        # the single flat tuple this protocol expects cannot describe it.
        # OlalaForCausalLMConfig rejects --mamba-cache-mode align up front.
        raise NotImplementedError(
            "Olala does not support mamba prefix caching (align mode): its M "
            "and V layers hold different numbers of state tensors."
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if getattr(config, "cosnet", False):
            # The CosNet sidecar branch is not ported; accepting the flag
            # would silently drop a trained residual path.
            raise NotImplementedError(
                "OlalaForCausalLM: config.cosnet is not supported."
            )
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        self.model = OlalaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_lm_head or getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embedding.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self._init_moe_metadata()

    def _init_moe_metadata(self) -> None:
        """Populate the MixtureOfExperts bookkeeping surface."""
        self.expert_weights: list[object] = []
        self.moe_layers: list[nn.Module] = []
        example: OlalaLatentMoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, OlalaMonoBlock) and isinstance(
                layer.mlp, OlalaLatentMoE
            ):
                example = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        routed = example.experts.routed_experts
        self.num_expert_groups = 1
        self.num_shared_experts = 1 if example.shared_experts is not None else 0
        self.num_logical_experts = example.num_experts
        self.num_physical_experts = routed.global_num_experts
        self.num_local_physical_experts = routed.local_num_experts
        self.num_routed_experts = example.num_experts
        self.num_redundant_experts = routed.global_num_experts - example.num_experts

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Olala's experts are non-gated (w1/w2 only) and arrive packed in a
        # single tensor per layer, which load_weights scatters directly. No
        # checkpoint-name mapping applies, and an empty list keeps the generic
        # machinery from picking up RoutedExperts' gated default.
        return []

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for runner in self.moe_layers:
            routed = runner.routed_experts
            routed.local_num_experts = num_local_physical_experts
            routed.global_num_experts = num_physical_experts
            runner.update_expert_map()

    def _expert_params(self) -> dict[int, tuple[nn.Parameter, nn.Parameter]]:
        """Map layer index to that layer's packed ``(w13, w2)`` parameters.

        Resolved by walking the modules rather than by name: the expert
        weights live on the runner's ``RoutedExperts`` submodule, whose path
        is an implementation detail of the MoE composition.
        """
        params: dict[int, tuple[nn.Parameter, nn.Parameter]] = {}
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, OlalaMonoBlock) and isinstance(
                layer.mlp, OlalaLatentMoE
            ):
                routed = layer.mlp.experts.routed_experts
                params[idx] = (routed.w13_weight, routed.w2_weight)
        return params

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load a Olala safetensors checkpoint.

        Remaps, for MoE layers:
            ``mlp.moe_gate.weight``            -> ``mlp.gate.weight``
            ``mlp.down_proj.weight``           -> ``mlp.fc1_latent_proj.weight``
            ``mlp.up_proj.weight``             -> ``mlp.fc2_latent_proj.weight``
            ``mlp.expert_bias``                -> ``mlp.gate.e_score_correction_bias``
            ``mlp.shared_experts.fc_1/fc_2``   -> ``shared_experts.up_proj/down_proj``
            ``mlp.experts.experts``            -> packed ``w13`` (w1 shard)
            ``mlp.experts.output_experts``     -> packed ``w2``
            ``mlp.tokens_per_expert``          -> dropped (non-persistent buffer)
        """
        moe_layers = set(moe_layer_indices(self.config))
        expert_params = self._expert_params()
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded: set[str] = set()

        renames = {
            "moe_gate.weight": "gate.weight",
            "down_proj.weight": "fc1_latent_proj.weight",
            "up_proj.weight": "fc2_latent_proj.weight",
            "shared_experts.fc_1.weight_scale": "shared_experts.up_proj.weight_scale",
            "shared_experts.fc_2.weight_scale": "shared_experts.down_proj.weight_scale",
            "shared_experts.fc_1.weight": "shared_experts.up_proj.weight",
            "shared_experts.fc_2.weight": "shared_experts.down_proj.weight",
        }

        for name, loaded_weight in weights:
            handled = False
            if name.startswith("model.layers.") and ".mlp." in name:
                head, tail = name.split(".mlp.", 1)
                try:
                    layer_idx = int(head.rsplit(".", 1)[1])
                except ValueError:
                    layer_idx = -1
                if layer_idx in moe_layers:
                    target_prefix = f"model.layers.{layer_idx}.mlp"
                    routed = self.model.layers[layer_idx].mlp.experts.routed_experts
                    int4_tail = tail.startswith(
                        ("experts.experts.", "experts.output_experts.")
                    ) and hasattr(routed, "w4a16_affine_group")
                    if int4_tail:
                        # ModelOpt W4A16_AFFINE experts (single rank): packed codes,
                        # bf16 group scales and biases, every expert of the layer.
                        is_w1 = tail.startswith("experts.experts.")
                        suffix = tail.split(".", 2)[2]
                        pname = (
                            f"{'w13' if is_w1 else 'w2'}_"
                            + {
                                "weight": "weight",
                                "weight_scale": "scale",
                                "weight_bias": "qbias",
                            }[suffix]
                        )
                        param = getattr(routed, pname)
                        if (
                            param.shape != loaded_weight.shape
                            or param.dtype != loaded_weight.dtype
                        ):
                            raise ValueError(
                                f"Olala load_weights: {name} is {loaded_weight.dtype} "
                                f"{tuple(loaded_weight.shape)}, expected {param.dtype} "
                                f"{tuple(param.shape)}."
                            )
                        param.data.copy_(loaded_weight)
                        loaded.add(f"{target_prefix}.experts.{pname}")
                        handled = True
                    elif tail in (
                        "experts.experts.weight",
                        "experts.output_experts.weight",
                    ):
                        # Olala packs every expert into one tensor; scatter it
                        # into the fused w13/w2 parameters expert by expert.
                        # NOTE: RoutedExperts.weight_loader dispatches on the
                        # weight NAME ("weight" substring), so the target must
                        # carry the real parameter name.
                        is_w1 = tail.endswith("experts.experts.weight")
                        param = expert_params[layer_idx][0 if is_w1 else 1]
                        shard_id = "w1" if is_w1 else "w2"
                        target = (
                            f"{target_prefix}.experts."
                            f"{'w13_weight' if is_w1 else 'w2_weight'}"
                        )
                        # With expert parallelism the loader declines the
                        # experts of other ranks; at least one must land here.
                        accepted = 0
                        for expert_id in range(loaded_weight.shape[0]):
                            accepted += bool(param.weight_loader(
                                param,
                                loaded_weight[expert_id],
                                target,
                                shard_id=shard_id,
                                expert_id=expert_id,
                                return_success=True,
                            ))
                        if not accepted:
                            raise RuntimeError(
                                f"Olala load_weights: no expert of {target} "
                                f"was accepted by the loader."
                            )
                        loaded.add(target)
                        handled = True
                    elif tail in (
                        "experts.experts.weight_scale",
                        "experts.output_experts.weight_scale",
                    ):
                        # Serialized experts_int8 checkpoint: the per-row
                        # scales of every expert of the layer (single rank).
                        is_w1 = tail.startswith("experts.experts.")
                        routed = self.model.layers[layer_idx].mlp.experts.routed_experts
                        param = routed.w13_scale if is_w1 else routed.w2_scale
                        if param.shape != loaded_weight.shape:
                            raise ValueError(
                                f"Olala load_weights: {name} has shape "
                                f"{tuple(loaded_weight.shape)}, expected "
                                f"{tuple(param.shape)}."
                            )
                        param.data.copy_(loaded_weight)
                        loaded.add(
                            f"{target_prefix}.experts."
                            f"{'w13_scale' if is_w1 else 'w2_scale'}"
                        )
                        handled = True
                    elif tail == "expert_bias":
                        target = f"{target_prefix}.gate.e_score_correction_bias"
                        param = params_dict[target]
                        param.data.copy_(loaded_weight.to(param.dtype).to(param.device))
                        loaded.add(target)
                        handled = True
                    elif tail in renames:
                        target = f"{target_prefix}.{renames[tail]}"
                        param = params_dict[target]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded.add(target)
                        handled = True
                    elif tail == "tokens_per_expert":
                        handled = True  # non-persistent buffer, dropped

            if handled:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
            elif name in buffers_dict:
                buf = buffers_dict[name]
                buf.data.copy_(loaded_weight.to(buf.dtype).to(buf.device))
                loaded.add(name)
            else:
                logger.warning_once(
                    "Olala load_weights: parameter %s not found, skipping.", name
                )

        self._maybe_prenormalize_embeddings()
        for module in self.modules():
            if isinstance(module, (OlalaMamba3Mixer, OlalaDiffTPAAttention, OlalaLatentMoE)):
                module.refresh_weight_caches()
        return loaded

    def _maybe_prenormalize_embeddings(self) -> None:
        """Fold the nGPT embedding normalization into the table.

        The lookup only selects rows, so L2-normalizing them offline in fp32 is
        exactly equivalent to ``F.normalize`` after lookup — and slightly more
        precise than normalizing bf16 activations every step. Only valid when
        the lm_head is untied, since normalizing would otherwise change the
        logits.
        """
        if not getattr(self.config, "normalize_embeddings_ngpt", False):
            return
        if getattr(self.config, "tie_lm_head", False) or getattr(
            self.config, "tie_word_embeddings", False
        ):
            return
        weight = self.model.embedding.weight
        with torch.no_grad():
            weight.copy_(
                torch.nn.functional.normalize(weight.float(), dim=-1).to(weight.dtype)
            )
        self.model.embeddings_prenormalized = True

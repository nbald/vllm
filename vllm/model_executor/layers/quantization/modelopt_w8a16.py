# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt MIXED_PRECISION ``W8A16`` (INT8 weight-only) layers, and the
``W4A16_AFFINE`` (INT4 affine weight-only) routed experts of Olala models."""

from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.online.int8 import Int8SerializedMoEMethod
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import set_weight_attrs

W8A16_ALGO = "W8A16"
W4A16_AFFINE_ALGO = "W4A16_AFFINE"


def w8a16_group_size(layer_info: dict[str, Any] | None) -> int:
    """0 (per output channel) or the group size along K of a W8A16 entry."""
    g = int((layer_info or {}).get("group_size") or 0)
    if g not in (0, 32):
        raise ValueError(f"W8A16 supports per-channel or group-32 scales, got {g}")
    return g


@register_weight_loader_v2_supported_method
class ModelOptW8A16LinearMethod(LinearMethodBase):
    """INT8 weight-only linear: ``x @ (scale * q).T`` through the opaque op
    ``vllm::w8a16_int8_linear`` (decode kernels up to a shape-dependent row
    count, prefill GEMMs beyond; both W8A16).

    Checkpoint tensors: ``weight`` int8 ``[N, K]`` and ``weight_scale`` fp32
    ``[N]`` (per channel) or ``[N, K // group_size]`` (per group, fp16 or
    fp32)."""

    def __init__(self, group_size: int = 0, scale_dtype: torch.dtype | None = None):
        self.group_size = group_size
        self.scale_dtype = scale_dtype or (
            torch.float16 if group_size else torch.float32
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if params_dtype != torch.bfloat16:
            raise ValueError(
                f"W8A16 layers require bf16 activations, got {params_dtype}"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        n = sum(output_partition_sizes)
        k = input_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(n, k, dtype=torch.int8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        g = self.group_size
        if g:
            if k % g:
                raise ValueError(f"W8A16 group size {g} does not divide K = {k}")
            scale = GroupQuantScaleParameter(
                data=torch.empty(n, k // g, dtype=self.scale_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            scale = ChannelQuantScaleParameter(
                data=torch.empty(n, dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Plain contiguous tensors for the kernels.
        w = layer.weight.data.contiguous()
        s = layer.weight_scale.data
        s = (
            s.reshape(-1).float().contiguous()
            if not self.group_size
            else s.contiguous()
        )
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        from vllm.model_executor.layers.quantization.utils.w8a16_utils import volta_tc

        if (tc := volta_tc()) is not None:
            # SM70 family, OLALA_VOLTA_TC (on by default): the fp16 copy of the tensor-core edition
            tc.register_int8_layer(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
            w8a16_int8_linear,
        )

        y = w8a16_int8_linear(x, layer.weight, layer.weight_scale)
        if bias is not None:
            y = y + bias
        return y

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("W8A16 embeddings are not supported")


def w8a16_quant_method(layer: torch.nn.Module, layer_info: dict[str, Any] | None):
    """The quant method of a ModelOpt MIXED_PRECISION ``W8A16`` module."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    g = w8a16_group_size(layer_info)
    if isinstance(layer, (LinearBase, ParallelLMHead)):
        return ModelOptW8A16LinearMethod(group_size=g)
    if isinstance(layer, RoutedExperts):
        # OlalaLatentMoE runs W8A16 experts with its own kernels (and, for
        # per-channel ones, optionally the serialized int8_w8a16 fused MoE).
        if not _is_olala_model():
            raise NotImplementedError("W8A16 routed experts are not supported")
        if g:
            return ModelOptW8A16GroupMoEMethod(moe=layer.moe_config)
        return Int8SerializedMoEMethod(moe=layer.moe_config)
    return None


def _is_olala_model() -> bool:
    config = get_current_vllm_config_or_none()
    model_config = config.model_config if config is not None else None
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "olala"


class ModelOptW8A16GroupMoEMethod(Int8SerializedMoEMethod):
    """Group-32 W8A16 routed experts of an Olala model: int8 weights as
    Int8SerializedMoEMethod creates them, fp16 scales ``[E, N, K / 32]``.
    OlalaLatentMoE runs them with its own kernels (latent_moe_int8.py)."""

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        rows13 = layer.w13_weight.shape[1]
        for name, rows, k in (
            ("w13_scale", rows13, hidden_size),
            ("w2_scale", hidden_size, intermediate_size_per_partition),
        ):
            if k % 32:
                raise ValueError(f"W8A16 group size 32 does not divide {k}")
            delattr(layer, name)
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(num_experts, rows, k // 32, dtype=torch.float16),
                    requires_grad=False,
                ),
            )
        layer.w8a16_group_size = 32

    def process_weights_after_loading(self, layer) -> None:
        layer._already_called_process_weights_after_loading = True

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, *args, **kwargs):
        raise NotImplementedError(
            "Group-32 W8A16 routed experts only run inside Olala's latent MoE "
            "(OlalaLatentMoE), which calls its own kernels"
        )


def w4a16_affine_quant_method(
    layer: torch.nn.Module, layer_info: dict[str, Any] | None
):
    """The quant method of a ModelOpt MIXED_PRECISION ``W4A16_AFFINE`` module
    (routed experts of an Olala model only)."""
    if not isinstance(layer, RoutedExperts) or not _is_olala_model():
        raise NotImplementedError(
            f"{W4A16_AFFINE_ALGO} is only supported for the routed experts of "
            f"Olala models, got {type(layer).__name__}"
        )
    g = int((layer_info or {}).get("group_size", 32))
    if g != 32:
        raise ValueError(f"{W4A16_AFFINE_ALGO} requires group_size=32, got {g}")
    return ModelOptW4A16AffineMoEMethod(moe=layer.moe_config)


class ModelOptW4A16AffineMoEMethod(Int8SerializedMoEMethod):
    """INT4 affine routed experts, group 32: uint8 codes ``[E, N, K / 2]`` (two
    per byte, low nibble first), bf16 ``scale`` and ``qbias`` ``[E, N, K / 32]``,
    ``w = scale * q + qbias``. After loading, scale and bias live interleaved in
    one int32 buffer per weight (``w13_sb`` / ``w2_sb``) that OlalaLatentMoE's
    kernels read; the named parameters stay as write-through views of it."""

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if self.moe.has_bias or self.moe.is_act_and_mul:
            raise NotImplementedError(
                f"{W4A16_AFFINE_ALGO}: non-gated experts without biases only"
            )
        if self.moe.tp_size != 1 or self.moe.ep_size != 1:
            raise NotImplementedError(f"{W4A16_AFFINE_ALGO}: single-rank experts only")
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        g = 32
        i = intermediate_size_per_partition
        for prefix, rows, k in (("w13", i, hidden_size), ("w2", hidden_size, i)):
            if k % g:
                raise ValueError(
                    f"{W4A16_AFFINE_ALGO} group size {g} does not divide {k}"
                )
            w = torch.nn.Parameter(
                torch.empty(num_experts, rows, k // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter(f"{prefix}_weight", w)
            set_weight_attrs(w, extra_weight_attrs)
            for name in ("scale", "qbias"):
                layer.register_parameter(
                    f"{prefix}_{name}",
                    torch.nn.Parameter(
                        torch.empty(num_experts, rows, k // g, dtype=torch.bfloat16),
                        requires_grad=False,
                    ),
                )
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.w4a16_affine_group = g

    def process_weights_after_loading(self, layer) -> None:
        # Keep the checkpoint parameter names live. Their BF16 views write
        # directly into the interleaved scale/bias read by captured kernels.
        for prefix in ("w13", "w2"):
            scale = getattr(layer, f"{prefix}_scale")
            bias = getattr(layer, f"{prefix}_qbias")
            name = f"{prefix}_sb"
            packed = getattr(layer, name, None)
            if packed is None or packed.device != scale.device:
                packed = torch.empty_like(
                    scale, dtype=torch.int32, memory_format=torch.contiguous_format
                )
                layer.register_buffer(name, packed, persistent=False)
            halves = packed.view(torch.bfloat16).view(*scale.shape, 2)
            for parameter, view in ((scale, halves[..., 0]), (bias, halves[..., 1])):
                if not parameter.data.is_set_to(view):
                    view.copy_(parameter.data)
                    parameter.data = view
        layer._already_called_process_weights_after_loading = True

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, *args, **kwargs):
        raise NotImplementedError(
            f"{W4A16_AFFINE_ALGO} routed experts only run inside Olala's latent "
            "MoE (OlalaLatentMoE), which calls its own kernels"
        )

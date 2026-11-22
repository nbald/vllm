# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt MIXED_PRECISION ``W8A16`` (INT8 weight-only) layers."""

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

W8A16_ALGO = "W8A16"


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
        raise NotImplementedError("W8A16 routed experts are not supported")
    return None

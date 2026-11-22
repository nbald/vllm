# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt MIXED_PRECISION W8A16 linears: selection from quantized_layers,
checkpoint loading through the weight loaders, and the forward pass against
the dequantized weights in fp32."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.layers.quantization.modelopt_w8a16 import (
    ModelOptW8A16LinearMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="W8A16 layers need CUDA SM80+",
)


def _config(entries: dict) -> ModelOptMixedPrecisionConfig:
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": entries,
                "exclude_modules": [],
            }
        }
    )


def _quantize(w: torch.Tensor, group: int):
    """Symmetric round to nearest, |q| <= 127: (q, scale, q * scale)."""
    n, k = w.shape
    if group:
        wg = w.view(n, k // group, group)
        s = (wg.abs().amax(-1) / 127).half()
        q = torch.round(wg / s.float()[..., None]).clamp(-127, 127)
        return q.to(torch.int8).view(n, k), s, (q * s.float()[..., None]).view(n, k)
    s = w.abs().amax(1) / 127
    q = torch.round(w / s[:, None]).clamp(-127, 127)
    return q.to(torch.int8), s, q * s[:, None]


@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize("m", [1, 7, 64, 300])
def test_w8a16_linear(dist_init, group, m):
    n, k = 1536, 1024
    entry = {"quant_algo": "W8A16"} | ({"group_size": group} if group else {})
    quant_config = _config({"proj": entry})
    with (
        set_current_vllm_config(VllmConfig()),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layer = ReplicatedLinear(
            k, n, bias=False, quant_config=quant_config, prefix="proj"
        )
        other = ReplicatedLinear(
            k, n, bias=False, quant_config=quant_config, prefix="other"
        )
    assert isinstance(layer.quant_method, ModelOptW8A16LinearMethod)
    assert isinstance(other.quant_method, UnquantizedLinearMethod)

    torch.manual_seed(m + group)
    q, scale, ref_w = _quantize(torch.randn(n, k, device="cuda") * 0.02, group)
    layer.weight.weight_loader(layer.weight, q)
    layer.weight_scale.weight_loader(layer.weight_scale, scale)
    layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y, _ = layer(x)
    ref = x.float() @ ref_w.T
    assert y.dtype == torch.bfloat16 and y.shape == (m, n)
    assert ((y.float() - ref).abs().max() / ref.abs().max()) < 8e-3


def test_w8a16_rejects_other_group_sizes(dist_init):
    quant_config = _config({"proj": {"quant_algo": "W8A16", "group_size": 64}})
    with (
        set_current_vllm_config(VllmConfig()),
        set_default_torch_dtype(torch.bfloat16),
        pytest.raises(ValueError, match="group-32"),
    ):
        ReplicatedLinear(64, 64, bias=False, quant_config=quant_config, prefix="proj")

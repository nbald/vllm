# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the grouped-expert fallback with a real Olala MoE and known weights."""

import os

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.models.olala import OlalaLatentMoE
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="Olala weight-only MoE requires CUDA BF16",
)
def test_grouped_experts_with_small_path_disabled(monkeypatch, dist_init):
    monkeypatch.setenv("OLALA_MOE_SMALL", "0")
    monkeypatch.setenv("OLALA_INT8_MOE_PREFILL", "vllm")
    # Load the public model configuration only. The test supplies every weight.
    config = EngineArgs(
        model=os.getenv("VLLM_TEST_OLALA_CONFIG", "OVHaiLLM/Olala-v1.0-mini-7A1B"),
        trust_remote_code=True,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=256,
        enforce_eager=True,
        hf_overrides=dict(
            moe_num_routed_experts=8,
            moe_num_active_experts=2,
            moe_routed_scaling_factor=1.0,
        ),
    ).create_engine_config()
    prefix = "model.layers.0.mlp"
    quant = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    f"{prefix}.experts": {"quant_algo": "W8A16", "group_size": 32}
                },
                "exclude_modules": [],
            }
        }
    )
    hf_config = config.model_config.hf_config
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
        torch.inference_mode(),
    ):
        layer = OlalaLatentMoE(hf_config, config.parallel_config, prefix, quant)
        for parameter in layer.parameters():
            parameter.zero_()
        for proj in (layer.fc1_latent_proj, layer.fc2_latent_proj):
            proj.weight[0, 0] = 1
        # Each expert passes channel zero through its ReLU². The shared output
        # and all other channels are zero. Both selected experts contribute 1/2.
        routed = layer.experts.routed_experts
        for name in ("w13", "w2"):
            getattr(routed, f"{name}_scale").fill_(1)
            getattr(routed, f"{name}_weight")[:, 0, 0] = 1
        routed.quant_method.process_weights_after_loading(routed)
        inputs = torch.zeros(2, hf_config.hidden_size)
        inputs[:, 0] = torch.tensor([2.0, -2.0])
        expected = torch.zeros_like(inputs)
        expected[0, 0] = 4
        with set_forward_context(None, config):
            torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)

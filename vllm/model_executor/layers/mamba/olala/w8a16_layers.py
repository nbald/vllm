# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala layers with ModelOpt W8A16 (INT8 weight-only) projections.

Helpers that let the Mamba-3, DiffTPA and latent-MoE layers read their
W8A16 linears as the Triton kernels of quantization/utils/w8a16_utils.py
take them, and concatenate projections that share an input into one buffer.
"""

import torch

from vllm.model_executor.layers.quantization.modelopt_w8a16 import (
    ModelOptW8A16LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.w8a16_utils import (
    Int8Weight,
    int8_mm,
)


def is_int8_quant_method(layer) -> bool:
    """The layer's quant method is the ModelOpt W8A16 one (known before
    loading)."""
    return isinstance(getattr(layer, "quant_method", None), ModelOptW8A16LinearMethod)


def is_int8_linear(layer) -> bool:
    w = getattr(layer, "weight", None)
    return (
        w is not None
        and w.dtype == torch.int8
        and is_int8_quant_method(layer)
        and getattr(layer, "weight_scale", None) is not None
    )


def int8_linear_weight(layer) -> Int8Weight | None:
    """A loaded W8A16 INT8 linear as the kernels read it (no bias), else None."""
    if not is_int8_linear(layer) or getattr(layer, "bias", None) is not None:
        return None
    w, s = layer.weight, layer.weight_scale
    if not w.is_contiguous() or w.dim() != 2:
        return None
    return Int8Weight(w, s)


def int8_apply(layer, x: torch.Tensor) -> torch.Tensor:
    """``layer(x)`` for an INT8 linear (eager callers: the Olala opaque ops)."""
    return int8_mm(x, Int8Weight(layer.weight, layer.weight_scale))


def cat_int8_rows_shared(layers: list, pad_rows: int = 0) -> Int8Weight | None:
    """Row-concatenate the INT8 weights (and scales) of ``layers`` into one
    buffer plus ``pad_rows`` zero rows (scale 1), and rebind each layer's weight
    and scale to its slice (no extra memory); None if the layers do not share
    one scale layout."""
    ws = [int8_linear_weight(layer) for layer in layers]
    if (
        any(w is None for w in ws)
        or len({(w.k, w.group, w.scale.dtype) for w in ws}) != 1
    ):
        return None
    with torch.no_grad():
        ref = ws[0]
        n_rows = sum(w.n for w in ws)
        buf = torch.zeros(
            n_rows + pad_rows, ref.k, dtype=torch.int8, device=ref.weight.device
        )
        torch.cat([w.weight for w in ws], 0, out=buf[:n_rows])
        sshape = (n_rows + pad_rows,) + tuple(ref.scale.shape[1:])
        sbuf = torch.ones(sshape, dtype=ref.scale.dtype, device=ref.scale.device)
        torch.cat([w.scale for w in ws], 0, out=sbuf[:n_rows])
        from vllm.model_executor.layers.quantization.utils.w8a16_utils import volta_tc

        if (tc := volta_tc()) is not None:
            tc.concat_registered([w.weight for w in ws], buf)
        off = 0
        for layer, w in zip(layers, ws):
            layer.weight.data = buf[off : off + w.n]
            layer.weight_scale.data = sbuf[off : off + w.n]
            off += w.n
    return Int8Weight(buf, sbuf)


def mixer_quant_config(quant_config):
    """Only enable mixer quantization for the Olala weight-only recipes."""
    if quant_config is None or quant_config.get_name() != "modelopt_mixed":
        return None
    formats = {
        info["quant_algo"].upper() for info in quant_config.quantized_layers.values()
    }
    if "W8A16" in formats and formats <= {"W8A16", "W4A16_AFFINE"}:
        return quant_config
    return None


def mixer_int8_rows(quant_config, prefix: str):
    """Return the weight-only config if this mixer projection is INT8."""
    quant_config = mixer_quant_config(quant_config)
    if quant_config is None or quant_config.is_layer_excluded(prefix):
        return None
    return quant_config if quant_config._resolve_quant_algo(prefix) == "W8A16" else None

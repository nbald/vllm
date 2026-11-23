# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA editions of Triton kernels for Volta (V100, natively or through volta-forever) that return the Triton
kernels' bits: the geodesic norm (csrc/geodesic_volta.cu). It reproduces its Triton kernel's per-output arithmetic
(the same reductions in the same tree as the Triton layout, the same rounding instructions) with another work
decomposition, so it is on by default on the SM70 family; OLALA_GEODESIC_CUDA=0 keeps the Triton kernel (part of the
compile-cache key like every OLALA_* switch)."""

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.olala.arch import (
    GpuFamily,
    gencode,
    gpu_family,
    has_bf16_mma,
    sm_arch,
)

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def _on(name: str) -> bool:
    """The switch, on the SM70 family told sm_80+ (through volta-forever: the kernels use sm_80 bf16
    conversions, which volta-forever lowers to Volta)."""
    return os.environ.get(name, "1") != "0" and gpu_family() is GpuFamily.SM70 and has_bf16_mma()


@lru_cache(maxsize=1)
def _geodesic_ext():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(f"olala_geodesic_volta_sm{sm_arch()}", "geodesic_volta.cu", ["-O3", "-std=c++20", gencode()])


def _aligned(*ts: torch.Tensor) -> bool:
    return all(t.data_ptr() % 16 == 0 for t in ts)


def geodesic_ok(x: torch.Tensor, g: torch.Tensor, out: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> bool:
    """The CUDA edition takes rows of 1024 < D <= 2048 bf16 elements (D % 8 == 0, 16-byte aligned rows) and bf16
    scale / bias: the shapes and dtypes of the Triton kernel's BLOCK = 2048 specialization it reproduces."""
    if not _on("OLALA_GEODESIC_CUDA"):
        return False
    D = x.shape[-1]
    if not (1024 < D <= 2048 and D % 8 == 0):
        return False
    if any(t.dtype != torch.bfloat16 for t in (x, g, out, scale, bias)):
        return False
    if any(t.stride(-1) != 1 or (t.dim() > 1 and t.stride(-2) % 8) for t in (x, g, out)):
        return False
    return _aligned(x, g, out) and g.shape == x.shape and out.shape == x.shape


def geodesic(x, g, out, scale, bias, inv_depth: float, clamp: float) -> torch.Tensor:
    logger.info_once("Olala Volta: geodesic norm through its CUDA edition (OLALA_GEODESIC_CUDA=0: Triton)")
    _geodesic_ext().geodesic(x, g, out, scale, bias, inv_depth, clamp)
    return out

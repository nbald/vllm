# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent-MoE router logits above decode sizes on the SM70 family (csrc/router_chain_volta.cu).

The bf16x2 router (olala.py, ``OLALA_ROUTER_GEMM=bf16x2``) is ``mm(x, [hi ; lo].T, out_dtype=fp32)``
followed by the add of the two halves. On a V100 that bf16 GEMM is cuBLAS 12.9's fp32 SIMT
magma_sgemmEx (through volta-forever), ~235-290 us per layer from 33 to 512 rows. Its result is a
sequential fp32 FMA chain over k per output: the kernel computes the same chains (k ascending, from
+0) and the same add, so the logits are bit for bit today's (checked at every row count from 33 to
2048), 6-8.5x faster at 33-128 rows and 1.17x at 2048 rows (V100 through volta-forever, graphed).
Decode sizes (<= 32 rows) keep their paths (the Volta FMA GEMV over hi + lo, or the W8A16 MoE
entry launch). ``OLALA_ROUTER_CHAIN=0`` keeps magma.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily, gencode, gpu_family, sm_arch

logger = init_logger(__name__)

# Rows the kernel takes: above the decode paths (32 rows), up to the largest batch checked bit for bit
# against cuBLAS (2048 rows: vLLM's default token budget for this model).
CHAIN_MIN_M = 33
CHAIN_MAX_M = 2048

_ENABLED = os.environ.get("OLALA_ROUTER_CHAIN", "1") != "0"


@lru_cache(maxsize=1)
def _ext():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(f"olala_router_chain_volta_sm{sm_arch()}", "router_chain_volta.cu",
                          ["-O3", "-std=c++20", gencode()])


@lru_cache(maxsize=1)
def _available() -> bool:
    try:
        _ext()
    except Exception as e:  # noqa: BLE001 - a failed build keeps the cuBLAS router
        logger.warning("Olala: Volta router kernel unavailable (%s); the router keeps cuBLAS.", e)
        return False
    return True


def router_chain_ok(x: torch.Tensor, weight: torch.Tensor, num_experts: int) -> bool:
    """The chain kernel applies: SM70 family, bf16x2 weight ([hi ; lo], [2E, K]), 33..2048 rows."""
    return (
        _ENABLED
        and x.is_cuda
        and x.dim() == 2
        and CHAIN_MIN_M <= x.shape[0] <= CHAIN_MAX_M
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.shape[0] == 2 * num_experts
        and num_experts % 64 == 0
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % 64 == 0
        and x.stride(1) == 1
        and (x.stride(0) * 2) % 16 == 0
        and x.data_ptr() % 16 == 0
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and gpu_family() is GpuFamily.SM70
        and _available()
    )


def router_chain(x: torch.Tensor, weight: torch.Tensor, num_experts: int) -> torch.Tensor:
    """fp32 logits ``x @ hi.T + x @ lo.T`` (each a sequential chain), as mm + the slice add."""
    y = torch.empty(x.shape[0], num_experts, dtype=torch.float32, device=x.device)
    _ext().router_chain(x, weight, y, num_experts)
    return y

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU families and features for the Olala kernel dispatch.

Kernels are chosen by instruction set and memory system, not by product generation:
datacenter Blackwell (sm_100 / sm_103: tcgen05, TMEM, HBM) and consumer Blackwell
(sm_120 / sm_121: mma.sync, GDDR7) take different paths, and Ada shares Ampere's.
Tuned paths test the family; kernels that only need one instruction test the feature.
"""

import enum
from functools import cache

import torch


class GpuFamily(enum.Enum):
    UNSUPPORTED = "unsupported"  # no CUDA GPU, or no family the Olala kernels target
    SM80 = "sm80"  # Ampere / Ada (sm_80, 86, 87, 89): mma.sync, ldmatrix, cp.async
    SM90 = "sm90"  # Hopper: + wgmma, TMA (sm_90a kernels)
    SM100 = "sm100"  # datacenter Blackwell (B200 / B300): + tcgen05, TMEM (sm_10xa kernels)
    SM120 = "sm120"  # consumer Blackwell (RTX 50xx, RTX PRO): mma.sync, GDDR7, no tcgen05


_FAMILY_BY_MAJOR = {
    8: GpuFamily.SM80,
    9: GpuFamily.SM90,
    10: GpuFamily.SM100,
    12: GpuFamily.SM120,
}


@cache
def capability(device=None) -> tuple[int, int]:
    """Compute capability of ``device`` (current device if None), cached: the
    dispatch checks run several times per layer and step, and
    torch.cuda.get_device_capability costs ~10 us of host time."""
    return torch.cuda.get_device_capability(device)


@cache
def gpu_family(device=None) -> GpuFamily:
    if not torch.cuda.is_available():
        return GpuFamily.UNSUPPORTED
    return _FAMILY_BY_MAJOR.get(capability(device)[0], GpuFamily.UNSUPPORTED)


def has_bf16_mma(device=None) -> bool:
    """mma.sync on bf16 operands (sm_80+)."""
    return torch.cuda.is_available() and capability(device) >= (8, 0)


def has_fp8_cvt(device=None) -> bool:
    """e4m3 / e5m2 conversion instructions (sm_89+)."""
    return torch.cuda.is_available() and capability(device) >= (8, 9)


def has_pdl(device=None) -> bool:
    """Programmatic dependent launch (griddepcontrol, sm_90+)."""
    return torch.cuda.is_available() and capability(device) >= (9, 0)


def sm_arch(device=None) -> str:
    """Arch number of ``device``, e.g. "86", "90", "120"."""
    major, minor = capability(device)
    return f"{major}{minor}"


def gencode(device=None, arch_specific: bool = False) -> str:
    """nvcc flag building for ``device`` only; ``arch_specific`` targets sm_XXa
    (wgmma, tcgen05, hardware cvt.rs need the arch-specific target)."""
    a = sm_arch(device) + ("a" if arch_specific else "")
    return f"-gencode=arch=compute_{a},code=sm_{a}"


def mma_sync_class(device=None) -> bool:
    """Ampere / Ada and consumer Blackwell: mma.sync + ldmatrix + cp.async kernels, no wgmma /
    tcgen05, ~100 KB of smem per SM (sm_86 / 89 / 120). The sm80 kernels build for the running
    arch and run unchanged on both."""
    return gpu_family(device) in (GpuFamily.SM80, GpuFamily.SM120)

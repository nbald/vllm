# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU families and features for the Olala kernel dispatch.

Kernels are chosen by instruction set and memory system, not by product generation:
datacenter Blackwell (sm_100 / sm_103: tcgen05, TMEM, HBM) and consumer Blackwell
(sm_120 / sm_121: mma.sync, GDDR7) take different paths, and Ada shares Ampere's.
Tuned paths test the family; kernels that only need one instruction test the feature.

Volta (V100, sm_70) is recognized before the compute capability: volta-forever (VF) runs
CUDA 12/13 programs on a V100 and tells them a later capability (``VF_CAPABILITY``; 8.0 keeps
vLLM's and Olala's bf16 paths), so torch reports e.g. (8, 0). VF lowers the sm_80 code to
sm_70 and emulates the bf16 tensor-core instructions it lacks (``mma.sync`` m16n8k16,
``ldmatrix``, ``cp.async``) with scalar code, ~200 instructions per bf16 m16n8k16 mma of which
64 are the FMAs themselves, while plain fp32 FMA code runs natively. So the features read as
the told capability's, Volta paths test the SM70 family, and Volta runs the SM80 kernels
wherever it has no path of its own (``sm80_class``).
"""

import ctypes
import enum
import os
from functools import cache

import torch


class GpuFamily(enum.Enum):
    UNSUPPORTED = "unsupported"  # no CUDA GPU, or no family the Olala kernels target
    SM70 = "sm70"  # Volta (V100): fp32 FMA, no bf16 tensor cores; native or through volta-forever
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
def volta_forever() -> bool:
    """True when this process runs through volta-forever: its launcher exports VOLTA_FOREVER,
    and its ``libcuda.so.1`` (the layer's proxy) exports ``volta_forever_version``."""
    if os.environ.get("VOLTA_FOREVER"):
        return True
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    return hasattr(lib, "volta_forever_version")


def _volta(device) -> bool:
    """A V100-class GPU (sm_70), whatever capability the process is told. ``OLALA_VOLTA=1`` /
    ``0`` forces the choice either way (e.g. to measure the SM80 dispatch through VF)."""
    forced = os.environ.get("OLALA_VOLTA")
    if forced in ("0", "1"):
        return forced == "1"
    return capability(device) == (7, 0) or volta_forever()


@cache
def gpu_family(device=None) -> GpuFamily:
    if not torch.cuda.is_available():
        return GpuFamily.UNSUPPORTED
    if _volta(device):  # first: through volta-forever the capability reads 8.0
        return GpuFamily.SM70
    return _FAMILY_BY_MAJOR.get(capability(device)[0], GpuFamily.UNSUPPORTED)


def has_bf16_mma(device=None) -> bool:
    """mma.sync on bf16 operands (sm_80+; emulated on Volta through volta-forever)."""
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


def sm80_class(device=None) -> bool:
    """SM80, and Volta through volta-forever (told sm_80+, it runs the SM80 kernels with
    their tensor-core instructions emulated): the SM80 paths, where Volta has none of its
    own."""
    family = gpu_family(device)
    return family is GpuFamily.SM80 or (family is GpuFamily.SM70 and has_bf16_mma(device))


def mma_sync_class(device=None) -> bool:
    """Ampere / Ada and consumer Blackwell: mma.sync + ldmatrix + cp.async kernels, no wgmma /
    tcgen05, ~100 KB of smem per SM (sm_86 / 89 / 120). The sm80 kernels build for the running
    arch and run unchanged on both, and on Volta through volta-forever (sm80_class)."""
    return sm80_class(device) or gpu_family(device) is GpuFamily.SM120

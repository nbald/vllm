# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch (PDL) for the Olala decode kernels.

A kernel launched with ``launch_pdl=True`` may start while its predecessor in
the stream drains; it must call ``tl.extra.cuda.gdc_wait()`` before touching
anything the predecessor writes (the wait covers the whole chain, as every
PDL kernel waits on its own predecessor). Saves ~0.2-0.45 us per launch
boundary on B300 under CUDA graphs. ``OLALA_PDL=0`` disables it.

Only for kernels launched eagerly from the opaque custom ops: a PDL kernel
inside the torch.compile graph (the geodesic norm) gave wrong results ahead
of the cuBLAS entry GEMM, and Inductor cannot analyze ``gdc_wait``.
"""

import os
from functools import cache

from vllm.model_executor.layers.mamba.olala.arch import has_pdl


@cache
def pdl_enabled() -> bool:
    return (
        os.environ.get("OLALA_PDL", "1") != "0"
        and has_pdl()
    )


def pdl_kwargs() -> dict:
    """Launch kwargs: the kernel's ``PDL`` constexpr plus Triton's flag."""
    on = pdl_enabled()
    return {"PDL": on, "launch_pdl": on}

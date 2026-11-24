# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala's SupportsEagle3 aux hidden states (the drafter's target features).

Index l + 1 in aux_hidden_state_layers captures the residual stream after
layer l. Checked on a prefill of the real checkpoint (GPU only; the model is
VLLM_TEST_OLALA_CONFIG, a local path or a Hub id):

* the captured states are the outputs of those blocks, taken by forward hooks;
* turning the capture on leaves the model output bit-identical;
* with OLALA_GEO_FUSE=1 (geodesic updates deferred into the next projection),
  the capture still returns the materialized residual stream.

Each configuration runs in a fresh interpreter: OLALA_GEO_FUSE is read at
import time.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL = os.getenv("VLLM_TEST_OLALA_CONFIG", "OVHaiLLM/Olala-v1.0-mini-7A1B")
AUX = (2, 10, 18, 26, 34)
# Beyond the weights: activations of the eager 2048-token prefill, the sampler,
# the CUDA context and a small KV cache.
HEADROOM = 3 * 2**30


def _weight_bytes(model: str) -> int | None:
    """Size of the checkpoint's safetensors files (a local directory, or the
    Hub cache), None if they cannot be found without a download."""
    path = Path(model)
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model, local_files_only=True))
        except Exception:  # noqa: BLE001 - not in the local cache
            return None
    files = list(path.glob("*.safetensors"))
    return sum(f.stat().st_size for f in files) if files else None


def _gpu_memory() -> tuple[int, int]:
    """(free, total) bytes of the GPU the engine will use, from NVML: a CUDA
    context in this process would hold memory the engine subprocess needs."""
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import import_pynvml

    pynvml = import_pynvml()
    pynvml.nvmlInit()
    try:
        index = current_platform.device_id_to_physical_device_id(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index))
        return int(info.free), int(info.total)
    finally:
        pynvml.nvmlShutdown()


def _gpu_memory_utilization(model: str) -> float:
    """The weights plus HEADROOM as a fraction of this GPU (at least the
    former fixed 0.6): a bf16 checkpoint (12.7 GiB) does not fit in 0.6 of a
    20 GB GPU. Skips when that much memory is not free."""
    free, total = _gpu_memory()
    weights = _weight_bytes(model)
    if weights is None:
        return 0.6
    need = weights + HEADROOM
    if need > free:
        pytest.skip(
            f"{model} needs ~{need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free"
        )
    return min(0.95, max(0.6, need / total))


def test_default_aux_layers():
    """The default layers are the DFlash spread for 36 layers: outputs of
    layers 1, 9, 17, 25 and 33, the target_layer_ids of a DFlash or DSpine
    drafter (eagle3_utils adds 1)."""
    from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3
    from vllm.model_executor.models.olala import OlalaForCausalLM, OlalaModel

    assert supports_eagle3(OlalaForCausalLM)
    assert issubclass(OlalaModel, EagleModelMixin)
    # Capture is off unless a drafter sets the layers.
    assert OlalaModel.aux_hidden_state_layers == ()
    fake = SimpleNamespace(model=SimpleNamespace(layers=[None] * 36))
    layers = OlalaForCausalLM.get_eagle3_default_aux_hidden_state_layers(fake)
    assert layers == AUX
    assert tuple(i - 1 for i in layers) == (1, 9, 17, 25, 33)


_SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from vllm import LLM, SamplingParams

    model_dir, out = sys.argv[1], sys.argv[2]
    aux_layers = tuple(json.loads(sys.argv[3]))
    llm = LLM(model_dir, trust_remote_code=True, enforce_eager=True,
              max_model_len=2048, enable_prefix_caching=False,
              gpu_memory_utilization=float(sys.argv[4]),
              mamba_ssm_cache_dtype="float32", moe_backend="triton")
    model = (llm.llm_engine.engine_core.engine_core.model_executor
             .driver_worker.worker.model_runner.model)
    inner = model.model
    got = {}

    def model_hook(mod, args, kwargs, output):
        if isinstance(output, tuple):
            hidden, aux = output
            got["aux"] = [a.float().cpu() for a in aux]
        else:
            hidden = output
        got.setdefault("hidden", []).append(hidden.float().cpu())
        return hidden

    def block_hook(idx):
        def hook(mod, args, kwargs, output):
            if not isinstance(output, tuple):  # (residual, pending) under geo fusion
                got.setdefault("blocks", {})[idx] = output.float().cpu()
        return hook

    # top-level module: the support_torch_compile __call__ of OlalaModel skips
    # nn.Module hooks
    model.register_forward_hook(model_hook, with_kwargs=True)
    for l in aux_layers:
        inner.layers[l - 1].register_forward_hook(block_hook(l - 1), with_kwargs=True)
    ids = list(range(1000, 1000 + 300))
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)  # capture off
    inner._set_aux_hidden_state_layers(aux_layers)
    llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)  # capture on
    torch.save({"hidden": got["hidden"], "aux": got.get("aux"),
                "blocks": got.get("blocks", {})}, out)
    """
)


def _run(tmp_path, geo_fuse: bool) -> dict:
    out = tmp_path / f"aux_geo{int(geo_fuse)}.pt"
    script = tmp_path / "run.py"
    script.write_text(_SCRIPT)
    env = dict(os.environ, OLALA_GEO_FUSE="1" if geo_fuse else "0")
    util = _gpu_memory_utilization(MODEL)
    subprocess.run(
        [sys.executable, str(script), MODEL, str(out), json.dumps(AUX), str(util)],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    return torch.load(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_aux_hidden_states(tmp_path):
    plain = _run(tmp_path, geo_fuse=False)
    h_off, h_on = plain["hidden"]
    assert torch.equal(h_off, h_on)
    assert len(plain["aux"]) == len(AUX)
    for layer, aux in zip(AUX, plain["aux"]):
        assert torch.equal(aux, plain["blocks"][layer - 1])

    fused = _run(tmp_path, geo_fuse=True)
    f_off, f_on = fused["hidden"]
    assert torch.equal(f_off, f_on)
    for a, b in zip(fused["aux"], plain["aux"]):
        # same residual stream through a differently fused (bf16) path
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        assert cos.min().item() > 0.99, cos.min().item()

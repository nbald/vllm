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
from types import SimpleNamespace

import pytest
import torch

MODEL = os.getenv("VLLM_TEST_OLALA_CONFIG", "OVHaiLLM/Olala-v1.0-mini-7A1B")
AUX = (2, 10, 18, 26, 34)


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
              gpu_memory_utilization=0.6, mamba_ssm_cache_dtype="float32",
              moe_backend="triton")
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
    subprocess.run(
        [sys.executable, str(script), MODEL, str(out), json.dumps(AUX)],
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

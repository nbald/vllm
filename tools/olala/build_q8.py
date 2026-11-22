# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export an Olala bf16 checkpoint as a W8A16 (INT8 weight-only) checkpoint.

The output is a ModelOpt MIXED_PRECISION checkpoint in which every linear
weight declares ``"quant_algo": "W8A16"`` (format:
docs/features/quantization/olala.md). Quantization is round-to-nearest with
no calibration data:

- quantized: every 2-D / 3-D ``*.weight`` except the token embedding and the
  router (``mlp.moe_gate``): routed experts (per expert, per output row),
  shared expert, latent projections, ``mixer_proj``, ``gate_proj``, Mamba-3
  ``in_proj`` / ``in_proj_dyn``, DiffTPA projections and ``lm_head``;
- kept as stored: the embedding, the fp32 router and expert bias, norms,
  Mamba-3 / DiffTPA per-head parameters and the geodesic scalars.

Scales:

- ``--group 0`` (default): one fp32 scale per output row, ``amax / 127``;
  ``weight_scale`` is ``[N]`` (``[E, N]`` for experts).
- ``--group 32``: one scale per 32 input channels, ``d = amax / 127``,
  ``q = round(x / d)`` (ggml Q8_0), stored as fp16 ``[N, K / 32]``.

``q`` is rounded half away from zero and clipped to [-127, 127].

Usage::

    python tools/olala/build_q8.py --base BF16_DIR --out Q8_DIR [--group 32]
    python tools/olala/build_q8.py --base BF16_DIR --verify Q8_DIR [--group 32]

``--verify`` re-quantizes ``--base`` tensor by tensor and checks that the
checkpoint in ``Q8_DIR`` holds the same tensors bit for bit and the same
``quantized_layers``. The export needs about 1.5x the bf16 checkpoint size in
host memory.
"""

import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# HF names of the latent projections -> the module names vLLM uses (and that
# quantized_layers keys must match).
_RENAMES = (
    (".mlp.down_proj.", ".mlp.fc1_latent_proj."),
    (".mlp.up_proj.", ".mlp.fc2_latent_proj."),
)
_SHARED_RENAMES = (
    (".shared_experts.fc_1", ".shared_experts.up_proj"),
    (".shared_experts.fc_2", ".shared_experts.down_proj"),
)
_CONFIG_FILES = ("model.safetensors", "config.json", "hf_quant_config.json")


def roundf(x: torch.Tensor) -> torch.Tensor:
    """C roundf: half away from zero."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[..., N, K]`` -> (int8 q, scale)."""
    if group == 0:
        s = (w.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
        q = roundf(w / s).clamp(-127, 127)
        return q.to(torch.int8), s.squeeze(-1).float()
    k = w.shape[-1]
    if k % group:
        raise ValueError(f"group {group} does not divide K = {k}")
    wg = w.reshape(*w.shape[:-1], k // group, group)
    d = wg.abs().amax(-1, keepdim=True) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = roundf(wg * inv).clamp(-127, 127)
    return q.to(torch.int8).reshape(w.shape), d.half().squeeze(-1)


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if s.dim() == q.dim() - 1:
        return q.float() * s.float()[..., None]
    qg = q.float().reshape(*q.shape[:-1], s.shape[-1], -1)
    return (qg * s.float()[..., None]).reshape(q.shape)


def is_linear_weight(name: str, ndim: int) -> bool:
    return (
        name.endswith(".weight")
        and ndim >= 2
        and name != "model.embedding.weight"
        and not name.endswith(".mlp.moe_gate.weight")
    )


def out_name(name: str) -> str:
    """Checkpoint name of a bf16 release tensor."""
    for hf, vllm_name in _RENAMES:
        if hf in name:
            return name.replace(hf, vllm_name)
    return name


def module_key(name: str) -> str:
    """quantized_layers key (vLLM module prefix) of a quantized weight."""
    m = out_name(name)[: -len(".weight")]
    if m.endswith((".mlp.experts.experts", ".mlp.experts.output_experts")):
        return m.rsplit(".", 1)[0]
    for ckpt, vllm_name in _SHARED_RENAMES:
        m = m.replace(ckpt, vllm_name)
    return m


def _weight_files(base: str) -> list[str]:
    index = os.path.join(base, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(base, fn) for fn in files]
    return [os.path.join(base, "model.safetensors")]


def _iter_tensors(base: str):
    for path in _weight_files(base):
        with safe_open(path, "pt") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                yield name, f.get_tensor(name)


def quant_config(qlayers: dict, group: int) -> tuple[dict, dict]:
    """(config.json quantization_config, hf_quant_config.json)."""
    tag = f"g{group}" if group else "ch"
    producer = {"name": "modelopt", "version": f"olala-q8-int8{tag}-rtn-1.0"}
    qc = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": qlayers,
        "exclude_modules": [],
        "ignore": [],
        "producer": producer,
    }
    hf = {
        "producer": producer,
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": qlayers,
            "exclude_modules": [],
        },
    }
    return qc, hf


def convert(base: str, group: int):
    """Yield (checkpoint name, tensor, quantized_layers key, bf16 source); the
    key and the source are set on quantized weights only."""
    for name, w in _iter_tensors(base):
        if is_linear_weight(name, w.dim()):
            q, s = quantize(w.float(), group)
            stem = out_name(name)[: -len(".weight")]
            yield f"{stem}.weight", q.contiguous(), module_key(name), w
            yield f"{stem}.weight_scale", s.contiguous(), None, None
        else:
            yield out_name(name), w, None, None


def _entry(group: int) -> dict:
    return {"quant_algo": "W8A16"} | ({"group_size": group} if group else {})


def build(base: str, out: str, group: int) -> None:
    t0 = time.time()
    tensors, qlayers = {}, {}
    for name, t, key, _ in convert(base, group):
        tensors[name] = t
        if key is not None:
            qlayers[key] = _entry(group)
    os.makedirs(out, exist_ok=True)
    save_file(
        tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"}
    )
    for fn in os.listdir(base):
        src = os.path.join(base, fn)
        if os.path.isfile(src) and fn not in _CONFIG_FILES and ".safetensors" not in fn:
            shutil.copy2(src, os.path.join(out, fn))
    with open(os.path.join(base, "config.json")) as f:
        cfg = json.load(f)
    qc, hf = quant_config(qlayers, group)
    cfg["quantization_config"] = qc
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "hf_quant_config.json"), "w") as f:
        json.dump(hf, f, indent=2)
    size = os.path.getsize(os.path.join(out, "model.safetensors")) / 1e9
    print(
        f"{len(qlayers)} W8A16 modules (group {group}) -> {out} ({size:.2f} GB) "
        f"in {time.time() - t0:.0f} s"
    )


def verify(base: str, ckpt: str, group: int) -> None:
    """``ckpt`` must hold exactly what build(base, group) writes."""
    seen, qlayers, err = set(), {}, 0.0
    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as g:
        names = set(g.keys())
        for name, t, key, src in convert(base, group):
            if name not in names:
                raise SystemExit(f"missing tensor {name}")
            stored = g.get_tensor(name)
            if stored.dtype != t.dtype or not torch.equal(stored, t):
                raise SystemExit(f"tensor {name} differs from the re-quantized base")
            seen.add(name)
            if key is not None:
                qlayers[key] = _entry(group)
                s = g.get_tensor(name[: -len(".weight")] + ".weight_scale")
                w = src.float()
                err = max(err, float((dequantize(stored, s) - w).norm() / w.norm()))
    if names - seen:
        raise SystemExit(f"unexpected tensors, e.g. {sorted(names - seen)[:3]}")
    with open(os.path.join(ckpt, "config.json")) as f:
        stored_qc = json.load(f)["quantization_config"]
    qc, _ = quant_config(qlayers, group)
    if stored_qc["quantized_layers"] != qc["quantized_layers"]:
        raise SystemExit("quantized_layers differ from the export of the base")
    print(
        f"verify {ckpt}: {len(seen)} tensors and {len(qlayers)} quantized_layers "
        f"entries identical to the export of {base}; max relative weight error "
        f"{err:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="bf16 Olala checkpoint directory")
    ap.add_argument("--group", type=int, default=0, choices=(0, 32))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="output directory")
    mode.add_argument("--verify", metavar="DIR", help="checkpoint to check")
    a = ap.parse_args()
    if a.verify:
        verify(a.base, a.verify, a.group)
    else:
        build(a.base, a.out, a.group)


if __name__ == "__main__":
    main()

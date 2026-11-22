# Olala weight-only checkpoints

This page specifies the weight-only quantized checkpoints of
[Olala-v1.0-mini-7A1B](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)
(OVHaiLLM) that vLLM serves, independently of vLLM: any engine can load them
from this description. The checkpoints keep the bf16 release's tensors,
except that linear weights are stored as INT8 with their scales.

| Recipe | Linear weights | Size |
|---|---|---|
| Q8 | `W8A16` per channel, every linear including `lm_head` | 7.1 GB |
| Q8 group-32 | `W8A16` with `group_size: 32` | 7.5 GB |

## Container

A checkpoint is a Hugging Face model directory: `config.json`,
`hf_quant_config.json`, one `model.safetensors`, plus the tokenizer, chat
template and `trust_remote_code` files of the bf16 release. `config.json`
is the release's, with a `quantization_config` in the ModelOpt
`MIXED_PRECISION` layout:

```json
{
  "quant_method": "modelopt",
  "quant_algo": "MIXED_PRECISION",
  "quantized_layers": {
    "lm_head": {"quant_algo": "W8A16"},
    "model.layers.0.mixer.in_proj": {"quant_algo": "W8A16"},
    "model.layers.0.mlp.experts": {"quant_algo": "W8A16"}
  },
  "exclude_modules": [],
  "ignore": [],
  "producer": {"name": "modelopt", "version": "olala-q8-int8ch-rtn-1.0"}
}
```

`hf_quant_config.json` holds the same `producer` and, under `quantization`,
the same `quant_algo`, `quantized_layers` and `exclude_modules`. The
checkpoints are not produced by NVIDIA Model Optimizer; they follow its
mixed-precision layout, and `producer.version` names the exporter and recipe
(`olala-q8-int8ch-rtn-1.0`: INT8 per channel, round to nearest;
`olala-q8-int8g32-rtn-1.0`: group 32).

`quantized_layers` has one entry per quantized module, keyed by module name
(next section). An entry carries `quant_algo` and, for grouped scales,
`group_size`. A module without an entry keeps the dtype it has in the bf16
release.

## Modules and tensor names

For a module key `M`, the tensors are `M'.weight` and `M'.weight_scale`,
where `M'` is the tensor prefix below (equal to `M` except for the shared
expert and the routed experts). Shapes are those of the 7A1B model: hidden
size 1536, 36 layers (`layers_config`: `M` = Mamba-3 layer, `V` = DiffTPA
layer at indices 4, 9, ..., 34), 256 routed experts of width 768 on a
384-wide latent, shared expert width 1536.

| Module key `M` | Tensor prefix `M'` | Weight shape `[N, K]` | Layers |
|---|---|---|---|
| `lm_head` | `lm_head` | `[151936, 1536]` | |
| `model.layers.i.mixer.in_proj` | same | `[6288, 1536]` | Mamba-3 |
| `model.layers.i.mixer.in_proj_dyn` | same | `[1056, 1536]` | Mamba-3 |
| `model.layers.i.mixer.c_q` | same | `[6144, 1536]` | DiffTPA |
| `model.layers.i.mixer.W_A_k`, `W_A_v` | same | `[48, 1536]` | DiffTPA |
| `model.layers.i.mixer.W_B_k`, `W_B_v` | same | `[512, 1536]` | DiffTPA |
| `model.layers.i.mixer.lambda_proj`, `shift_proj_k`, `shift_proj_v` | same | `[12, 1536]` | DiffTPA |
| `model.layers.i.gate_proj` | same | `[4608, 1536]` | DiffTPA |
| `model.layers.i.mixer_proj` | same | `[1536, 3072]` (Mamba-3), `[1536, 4608]` (DiffTPA) | all |
| `model.layers.i.mlp.fc1_latent_proj` | same | `[384, 1536]` | all |
| `model.layers.i.mlp.fc2_latent_proj` | same | `[1536, 384]` | all |
| `model.layers.i.mlp.shared_experts.up_proj` | `model.layers.i.mlp.shared_experts.fc_1` | `[1536, 1536]` | all |
| `model.layers.i.mlp.shared_experts.down_proj` | `model.layers.i.mlp.shared_experts.fc_2` | `[1536, 1536]` | all |
| `model.layers.i.mlp.experts` | `model.layers.i.mlp.experts.experts` (up) and `model.layers.i.mlp.experts.output_experts` (down) | `[256, 768, 384]` and `[256, 384, 768]` | all |

The latent projections are stored under the names above; the bf16 release
calls them `mlp.down_proj` (`fc1_latent_proj`, hidden to latent) and
`mlp.up_proj` (`fc2_latent_proj`, latent to hidden). The routed experts are
not gated: `down(relu(up(x))^2)`.

Every other tensor is stored as in the bf16 release:

| Tensors | Dtype |
|---|---|
| `model.embedding.weight` | bf16 |
| `mlp.moe_gate.weight` (router, `[256, 1536]`), `mlp.expert_bias` | fp32 |
| `mixer.B_norm`, `mixer.C_norm`, `mixer.q_norm`, `mixer.k_norm` (`.norm.weight`) | bf16 |
| Mamba-3: `mixer.B_bias`, `C_bias`, `D`, `dt_bias`, `in_proj_mimo_x`, `in_proj_mimo_z`, `out_proj_mimo` | fp32 |
| DiffTPA: `mixer.softmax_scaler` | fp32 |
| `geodesic_mixer.*`, `geodesic_mlp.*` (scalars) | fp32 |

## W8A16: INT8 weights, BF16 activations

Per channel (no `group_size`):

- `M'.weight`: `int8` `[N, K]`, row-major (K contiguous); experts
  `[E, N, K]`. Values lie in `[-127, 127]`.
- `M'.weight_scale`: `float32` `[N]`; experts `[E, N]`.
- `w[n, k] = weight_scale[n] * weight[n, k]`.

Per group (`"group_size": 32`):

- `M'.weight_scale`: `float16` `[N, K / 32]`; experts `[E, N, K / 32]`.
- `w[n, k] = weight_scale[n, k // 32] * weight[n, k]`.

A layer computes `y = x @ w.T` with BF16 activations `x`. vLLM's kernels
accumulate in fp32: per-channel scales multiply the accumulator, grouped
weight tiles are dequantized to bf16 before the multiply (fp32 in the
single-row GEMV), and some prefill shapes dequantize the whole weight to bf16
for cuBLAS. Each of these matches a bf16 GEMM over the dequantized weights
within bf16 rounding. The router stays fp32 and its logits are fp32.

## Export

`tools/olala/build_q8.py` writes these checkpoints from the bf16 release with
round-to-nearest quantization and no calibration data:

```bash
python tools/olala/build_q8.py --base /path/to/Olala-v1.0-mini-7A1B --out /path/to/Q8
python tools/olala/build_q8.py --base /path/to/Olala-v1.0-mini-7A1B --out /path/to/Q8-g32 --group 32
# re-quantize the base and compare every tensor and quantized_layers entry
python tools/olala/build_q8.py --base /path/to/Olala-v1.0-mini-7A1B --verify /path/to/Q8
```

Per channel, `scale = max|w_row| / 127`; per group,
`d = max|w_group| / 127` stored as fp16. `q = round(w / scale)`, rounded half
away from zero and clipped to `[-127, 127]`.

## Serving with vLLM

```bash
vllm serve /path/to/Q8 --trust-remote-code
```

The checkpoint's dtype is `bfloat16`; the weight-only paths need a single CUDA
GPU with compute capability 8.0 or later, and were tuned on an RTX 5090. The
environment variables that switch the Olala and W8A16 fast paths off are
listed in the repository README.

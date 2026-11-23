<!-- markdownlint-disable MD001 MD041 -->
## Olala fork: environment variables

This branch (`dragon-v0.26`) adds the Olala hybrid model (Mamba-3 MIMO + Differential TPA + latent MoE).
Its kernels and fast paths are controlled by the variables below; every default is the tuned production setting.

| Variable | Default | Effect |
|---|---|---|
| `OLALA_TPA_FACTOR` | `1` | TPA-factorized paged KV cache for the DiffTPA layers (stores the rank-4 factors, 2.5x smaller than dense K/V) with the wgmma decode-attention kernel. `0` = dense paged KV + standard attention backend. Not compatible with speculative decoding (use `0` with spec decode). |
| `OLALA_MAMBA3_STEP` | `cuda` | Mamba-3 decode step: `cuda` = persistent CUDA kernel (bf16 and fp32 state); any other value = CuteDSL step (`ops/mamba3/step_cute.py`). |
| `OLALA_GROUPED_PREFILL` | `1` | Group-parallel exact Mamba-3 varlen prefill for long prompts; `0` = single-pass kernel. |
| `OLALA_GEMV` | `1` | Single-token Triton GEMV for decode projections; `0` = regular GEMM. |
| `OLALA_TPA_CONCAT` | `1` | One GEMM over the concatenated DiffTPA input projections; `0` = separate projections. |
| `OLALA_MOE_SMALL` | `1` | Small-batch latent-MoE decode path (gather-GEMV kernels); `0` = generic fused-MoE pipeline. |
| `OLALA_MOE_INT8` | `1` | W8A16 checkpoints: latent-MoE decode through the weight-only small and medium kernels (`latent_moe_int8.py`), per-channel routed-expert prefill through Olala's grouped GEMMs. `0` = per-channel INT8 experts through vLLM's `int8_w8a16` fused MoE at every batch size; group-32 and INT4 affine (`W4A16_AFFINE`) experts keep Olala's grouped GEMMs. |
| `OLALA_INT8_MOE_PREFILL` | `olala` | W8A16 checkpoints, prefill of per-channel INT8 routed experts: `olala` = Olala's grouped GEMMs, `vllm` = vLLM's `int8_w8a16` fused MoE (2 % slower on an RTX 5090). |
| `VLLM_W8A16_INT8_DECODE` | `1` | W8A16 checkpoints: Triton decode kernels (GEMV, skinny GEMM) for the INT8 linears; `0` = prefill kernels at every batch size. Declared in `vllm/envs.py`, like `VLLM_W8A16_INT8_DEQ_SCRATCH_MB` (default `48`, the dequantization buffer of the prefill path). |
| `OLALA_TPA_BUILD_DIR` | `~/.cache/olala_tpa_factor` | JIT build directory of the CUDA extensions (TPA decode attention, Mamba-3 step). |
| `OLALA_JIT_CUDA_HOME` | auto | CUDA toolkit used to JIT-build those extensions. Auto = torch's `CUDA_HOME` if its nvcc major matches `torch.version.cuda`, else the newest matching `/usr/local/cuda-<major>.*` (e.g. a cu128 torch on a host whose default toolkit is CUDA 13). The extensions are built for the running GPU (SM80 or later, see GPU support below); if the build fails, vLLM warns and falls back to dense KV + CuteDSL step. |
| `OLALA_VOLTA` | auto | Volta (V100) detection, see GPU support below: auto = a native sm_70 GPU, or a process running through volta-forever (`VOLTA_FOREVER` set, or its `libcuda.so.1` proxy loaded); `1` / `0` forces the SM70 family on / off (`0` = the dispatch of the capability the process is told, e.g. SM80 through volta-forever). |
| `OLALA_TPA_DEBUG` | unset | Debug: `1` prints the factor-cache layout once; `2` also reports NaNs in the decode-attention output. |
| `OLALA_TPA_FACTOR_DECODE_DENSE` | unset | Debug: `1` routes factor-mode decode through the reference reconstruct + dense-attention path. |

### GPU support

The Olala kernels dispatch on GPU families (`vllm/model_executor/layers/mamba/olala/arch.py`): SM70 (Volta:
V100), SM80 (Ampere, Ada), SM90 (Hopper), SM100 (B200, B300) and SM120 (consumer Blackwell: RTX 50xx, RTX PRO
Blackwell). SM120 has its own paths, tuned on an RTX 5090:

- the Mamba-3 prefill runs the CUDA mma.sync chunk-state and output-pass kernels, as on SM80;
- the decode projections use a dedicated GEMV dispatch: the full-K or K-loop GEMV at one row, the skinny kernel
  up to 32 rows (64 on wide outputs), cuBLAS beyond;
- the medium-batch latent-MoE projections take the same dispatch.

SM70 serves bf16 checkpoints through volta-forever, which runs CUDA 12/13 programs on a V100 and tells them
compute capability 8.0 (`VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1`: vLLM keeps bf16, the Olala kernels build for
sm_80 and the layer lowers them to sm_70, emulating the bf16 tensor-core instructions). The family is recognized
before that capability (`OLALA_VOLTA`), and Volta runs the SM80 kernels wherever it has no path of its own. Its
own paths replace emulated tensor-core code with fp32 FMA code:

- the decode projections take FMA GEMVs up to 32 rows, and so do the router at decode sizes (the hi + lo halves
  of its bf16x2 split as one fp32 weight) and the LM head up to 16 sequences;
- the Mamba-3 decode step runs its scalar-FMA update (`-DSTEP_VOLTA`, two CTAs per SM);
- the TPA-factor decode attention has an FMA kernel (`tpa_factor_decode_volta.cu`), with one-tile splits and a
  4-head split merge;
- the latent-MoE small path covers up to 32 tokens; above, the medium path's projections take the same GEMV
  dispatch and its grouped GEMMs one 16x64x32 single-stage tile;
- the Mamba-3 prefill's chunk-state and output-pass kernels and the DiffTPA prefill attention have FMA editions
  (`mamba3_chunk_state_volta.cu`, `mamba3_scan_prefill_volta.cu`, `tpa_prefill_attn_volta.cu`) that return the SM80
  kernels' and FlashAttention-2's values, bit for bit through volta-forever (`OLALA_VOLTA_PREFILL=0` keeps those).

On a V100-SXM2-32GB through volta-forever this takes the bf16 model from 55 to 174 tok/s of decode at one
request, from 12.6 to 72 tok/s per request at eight, and its prefill (480-token prompts) from ~1590 to ~2700
tok/s; `OLALA_VOLTA=0` gives the SM80 dispatch back, token for token.

ReplaySSM (`OLALA_REPLAY_SSM`) needs the fp8 conversion instructions of sm_89 or later.

### Quantized checkpoints

Single-GPU W8A16 (INT8 weight-only) checkpoints, and Q4A2 checkpoints with INT4 affine (`W4A16_AFFINE`) routed
experts, in the ModelOpt `MIXED_PRECISION` layout are served with `vllm serve <dir> --trust-remote-code`. Their format and the exporter, `tools/olala/build_q8.py`, are documented
in [docs/features/quantization/olala.md](docs/features/quantization/olala.md).

Chunked prefill is on by default, as for any vLLM model (`OLALA_CHUNKED_PREFILL` is no longer read): it bounds
the TTFT of long prompts. `--no-enable-chunked-prefill` keeps whole-prompt steps and the FULL-decode CUDA-graph
regime (more throughput under concurrency on short prompts); `--max-num-batched-tokens` must then be at least
`--max-model-len`.

The Mamba-3 SSM state is stored in **fp32** by default (`--mamba-ssm-cache-dtype auto`): bf16 storage drifts over
long generations (repetition loops). `--mamba-ssm-cache-dtype bfloat16` is ~15% faster at high concurrency.
Checkpoints converted before the Olala rename (`DragonForCausalLM`) still load.

The Mamba-3 MIMO inference kernels are vendored in `vllm/model_executor/layers/mamba/ops/mamba3/`
(forward only, next to the Mamba-2 ops): **`mamba_ssm` is not needed**. They use `tilelang`,
`nvidia-cutlass-dsl` and `quack-kernels`, already in vLLM's CUDA requirements.

### Docker image

The fork changes no `csrc/`, so the image reuses the precompiled upstream binaries of its base commit:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --target vllm-openai \
  --build-arg VLLM_USE_PRECOMPILED=1 \
  --build-arg VLLM_MERGE_BASE_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0 \
  --build-arg VLLM_VERSION_OVERRIDE=0.26.0 \
  --build-arg RUN_WHEEL_CHECK=false \
  -t olala-vllm:dragon-v0.26 .

docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v /path/to/checkpoint:/model:ro -v olala-jit:/root/.cache \
  olala-vllm:dragon-v0.26 /model --trust-remote-code --served-model-name olala
```

The TPA-factor decode and Mamba-3 step CUDA extensions are JIT-built (nvcc, for the GPU's architecture) on first use;
the `olala-jit` volume keeps them across container restarts.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)

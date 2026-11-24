<!-- markdownlint-disable MD001 MD041 -->
## Olala fork: environment variables

This branch (`dragon-v0.26`) adds the Olala hybrid model (Mamba-3 MIMO + Differential TPA + latent MoE).
Its kernels and fast paths are controlled by the variables below; every default is the tuned production setting.

| Variable | Default | Effect |
|---|---|---|
| `OLALA_TPA_FACTOR` | `1` | TPA-factorized paged KV cache for the DiffTPA layers (stores the rank-4 factors, 2.5x smaller than dense K/V) with the wgmma decode-attention kernel. `0` = dense paged KV + standard attention backend. Under speculative decoding it is kept with `OLALA_TPA_FACTOR_SPEC` (on by default on SM70 / SM80 / SM120); otherwise a speculative config sets it to `0`. |
| `OLALA_MAMBA3_STEP` | `cuda` | Mamba-3 decode step: `cuda` = persistent CUDA kernel (bf16 and fp32 state); any other value = CuteDSL step (`ops/mamba3/step_cute.py`). |
| `OLALA_GROUPED_PREFILL` | `1` | Group-parallel exact Mamba-3 varlen prefill for long prompts; `0` = single-pass kernel. |
| `OLALA_GEMV` | `1` | Single-token Triton GEMV for decode projections; `0` = regular GEMM. |
| `OLALA_TPA_CONCAT` | `1` | One GEMM over the concatenated DiffTPA input projections; `0` = separate projections. |
| `OLALA_MOE_SMALL` | `1` | Small-batch latent-MoE decode path (gather-GEMV kernels); `0` = generic fused-MoE pipeline. |
| `OLALA_MOE_INT8` | `1` | W8A16 checkpoints: latent-MoE decode through the weight-only small and medium kernels (`latent_moe_int8.py`), per-channel routed-expert prefill through Olala's grouped GEMMs. `0` = per-channel INT8 experts through vLLM's `int8_w8a16` fused MoE at every batch size; group-32 and INT4 affine (`W4A16_AFFINE`) experts keep Olala's grouped GEMMs. |
| `OLALA_INT8_MOE_PREFILL` | `olala` | W8A16 checkpoints, prefill of per-channel INT8 routed experts: `olala` = Olala's grouped GEMMs, `vllm` = vLLM's `int8_w8a16` fused MoE (2 % slower on an RTX 5090). |
| `VLLM_W8A16_INT8_DECODE` | `1` | W8A16 checkpoints: Triton decode kernels (GEMV, skinny GEMM) for the INT8 linears; `0` = prefill kernels at every batch size. Declared in `vllm/envs.py`, like `VLLM_W8A16_INT8_DEQ_SCRATCH_MB` (default `48`, the dequantization buffer of the prefill path). |
| `OLALA_MAMBA3_VERIFY` | `cuda` | Speculative verify of the Mamba-3 layers: `cuda` = one fused launch for every draft position (`csrc/mamba3_verify.cu`, fp32 state, 7A1B shapes at TP=1), bit-identical to chaining the CUDA decode step; any other value = the decode step launched once per position. |
| `OLALA_TPA_FACTOR_SPEC` | on under spec on SM70 / SM80 / SM120, else `0` | F1: the TPA-factorized cache under speculative decoding. Verify rows write their factor rows for every draft position and attend through the factor decode kernel, one single-token row per position, with KV splits sized for the verify requests (KV traffic x (1 + K) on long contexts). The factor blocks are 16 tokens on SM80 / SM120 / SM70 (13 elsewhere). With a speculative config the default is on for compute capability majors 7, 8 and 12 (the config hook sets it and logs it) and off elsewhere. Measured on sm_120 (RTX 5090), sm_86 (RTX 3080, native) and SM70 (V100 through volta-forever, which reports 8.0 or 7.0); sm_80 (A100) and sm_89 (Ada) share the SM80-family kernels but were not measured. `0` = the dense paged KV cache under spec (the factor cache is turned off, and an explicit `OLALA_TPA_FACTOR=1` is refused); `1` = F1 on any GPU. Runs without a speculative config are unchanged. |
| `OLALA_TPA_SPEC_FUSED` | `1` | Speculative verify of the DiffTPA layers: fused K/V reconstruction, token shift, q/k norms and scaling (`tpa_decode_qkv_spec`); `0` = eager per-op path. |
| `OLALA_DSPINE_FUSED`, `OLALA_DSPINE_GEMV`, `OLALA_DSPINE_CUDA_NORM` | `0` | Opt-in fast path of a DSpine drafter (compiled injection and transition scoring, `2` also compiles the draft layers, `3` adds hand-fused draft kernels for steps of at most `OLALA_DSPINE_FUSED_MAX_ROWS` rows (64); skinny GEMV; CUDA RMSNorm). `3` is the recommended setting. See [DSpine](docs/features/speculative_decoding/dspine.md). |
| `OLALA_DSPINE_SHARED_HEAD` | `0` | A DSpine drafter uses the target's LM head (INT8 on W8A16 checkpoints) instead of its own copy, which is not loaded. |
| `OLALA_DSPINE_INT8` | on (SM70, SM80 family, SM120) | A DSpine drafter's decoder linears and context projection in INT8 W8A16: per-channel RTN at load time, through the target's W8A16 linear method. `0` keeps them BF16; `1` forces INT8 on other GPUs. See [DSpine](docs/features/speculative_decoding/dspine.md). |
| `OLALA_SPLIT_PREFILL` | `0` | Scheduler, while running decodes and prefill work are both pending: alternate prefill-only and decode-only steps, so that decode steps replay the FULL CUDA graph. `OLALA_SPLIT_PREFILL_MAX_DECODES` (`0` = no limit) splits only up to that many running decodes; `OLALA_SPLIT_PREFILL_DECODE_STEPS` (`1`) and `OLALA_SPLIT_PREFILL_BACKOFF` (`4`) tune the alternation. |
| `OLALA_PREFILL_NOSYNC` | `0` | Prefill metadata of the Mamba-3, DiffTPA and factor-cache builders from the CPU, without a device-to-host copy; `OLALA_PREFILL_NOSYNC_CHECK=1` compares with the device values (debug). |
| `OLALA_TS_FOLD` | `0` | DiffTPA token-shift buffers stored in the page of the preceding Mamba-3 layer (no token-shift KV groups). |
| `OLALA_SPEC_LIGHT` | `0` | Speculative decoding: one Mamba-3 page per request with `1 + K` light columns instead of `1 + K` state pages. Needs the fused verify kernel (`OLALA_MAMBA3_VERIFY=cuda`, SM80 or later, 7A1B shapes at TP=1, fp32 state); ignored elsewhere. |
| `OLALA_VOLTA_SPEC_LIGHT` | `0` | SM70 (V100): `1` takes the light columns of `OLALA_SPEC_LIGHT=1` (the Volta verify edition commits them); unset, SM70 keeps column slots. The capacity / long-context setting: on a V100 (Q8, K3) about 30 % more KV capacity, +13 % at c = 128 with no preemptions, -3.5 % at c = 1 (the committed drafts are replayed position by position). |
| `VLLM_KV_CACHE_GROUP_SIZE` | unset | Overrides the hybrid allocator's group size (`7` with `OLALA_TS_FOLD`). |
| `OLALA_TPA_MIXED_FUSED` | `0` | Speculative steps that mix prefills and verifies: the verify rows keep the fused DiffTPA path (`tpa_decode_qkv_spec`). |
| `OLALA_STEP_TRACE` | unset | Development: per-step JSONL trace of Model Runner V2 at that path (phase timings, CUDA-graph modes); `OLALA_STEP_TRACE_SYNCS=1` also counts host syncs. |
| `OLALA_TPA_BUILD_DIR` | `~/.cache/olala_tpa_factor` | JIT build directory of the CUDA extensions (TPA decode attention, Mamba-3 step). |
| `OLALA_JIT_CUDA_HOME` | auto | CUDA toolkit used to JIT-build those extensions. Auto = torch's `CUDA_HOME` if its nvcc major matches `torch.version.cuda`, else the newest matching `/usr/local/cuda-<major>.*` (e.g. a cu128 torch on a host whose default toolkit is CUDA 13). The extensions are built for the running GPU (SM80 or later, see GPU support below); if the build fails, vLLM warns and falls back to dense KV + CuteDSL step. |
| `OLALA_VOLTA` | auto | Volta (V100) detection, see GPU support below: auto = a native sm_70 GPU, or a process running through volta-forever (`VOLTA_FOREVER` set, or its `libcuda.so.1` proxy loaded); `1` / `0` forces the SM70 family on / off (`0` = the dispatch of the capability the process is told, e.g. SM80 through volta-forever). |
| `OLALA_VOLTA_TC` | `int8,bf16` on SM70 | SM70 family: the fp16 tensor-core edition (`olala/volta_tc.py`). The W8A16 linears (`int8`) and the bf16 model's dense projections (`bf16`) run on cuBLAS's fp16 HMMA GEMMs from 9 activation rows (33 for linears whose rows need 2-3 exact fp16 terms) and in prefill, with exactly scaled operands; fp16 weight copies cost about 2 GiB. Unset or `1` = those two kinds; `0` = off (the FMA and emulated paths, their outputs bit for bit); a comma list picks kinds, `moe` adds the routed experts above 32 tokens (not in the default: on chat_math it moved one confident prompt position per quantized checkpoint). Ignored on other GPUs; `VF_SAFE=1` turns it off, all kinds, whatever this variable says. |
| `VF_SAFE` | unset | volta-forever's conservative switch, read by this fork too (once per process): only the exact value `1` turns it on (`0`, empty, `true`... do not). It means no path whose results leave the conservative reference class. On the SM70 family the fp16 tensor-core edition (`OLALA_VOLTA_TC`) is then off: HMMA truncates its fp32 sums, a systematic bias toward zero. The fp32-FMA kernels stay on: the Volta editions that return the stock values bit for bit (the prefill kernels), and those that accumulate in fp32 and only reorder a sum, without a bias (the decode GEMVs, the few-row GEMV `OLALA_GEMV_ROWS_CUDA`), which strictly are not bit-identical to stock either. |
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
- the router of 33-2048-token batches runs fp32 FMA chains (`router_chain_volta.cu`) that return cuBLAS's logits
  bit for bit (cuBLAS runs this bf16 GEMM as fp32 SIMT chains on a V100, with few CTAs); `OLALA_ROUTER_CHAIN=0`
  keeps cuBLAS;
- the Mamba-3 decode step runs its scalar-FMA update (`-DSTEP_VOLTA`); with an fp32 state, the state goes straight
  to registers, all loads in flight at once, three CTAs per SM (`mamba3_step_volta_kernel`, the 1-stage kernel's
  bits; `OLALA_STEP_PIPE=0` keeps that one);
- the TPA-factor decode attention has an FMA kernel (`tpa_factor_decode_volta.cu`), with one-tile splits and a
  4-head split merge;
- the speculative verify of the Mamba-3 layers runs that step's FMA arithmetic at every draft position
  (`mamba3_verify.cu`, `-DVERIFY_VOLTA`), so a verified position is bit for bit the chained `STEP_VOLTA` step;
- the latent-MoE small path covers up to 32 tokens; above, the medium path's projections take the same GEMV
  dispatch and its grouped GEMMs one 16x64x32 single-stage tile;
- the Mamba-3 prefill's chunk-state and output-pass kernels and the DiffTPA prefill attention have FMA editions
  (`mamba3_chunk_state_volta.cu`, `mamba3_scan_prefill_volta.cu`, `tpa_prefill_attn_volta.cu`) that return the SM80
  kernels' and FlashAttention-2's values, bit for bit through volta-forever (`OLALA_VOLTA_PREFILL=0` keeps those);
- unquantized routed experts above the medium path (prefill chunks) take 16-row single-stage tiles in vLLM's fused
  MoE kernel (same bits as its default tiles);
- the Triton kernels that the first requests would compile (Mamba-3 and TPA-factor prefill, LM head, logprobs) are
  compiled at start for every specialization a batch can hit (`vllm/model_executor/warmup/olala_triton_warmup.py`).

On a V100-SXM2-32GB through volta-forever this takes the bf16 model from 55 to 174 tok/s of decode at one
request, from 12.6 to 72 tok/s per request at eight, and its prefill (480-token prompts) from ~1590 to ~2700
tok/s; `OLALA_VOLTA=0` gives the SM80 dispatch back, token for token.

From 9 rows, the projections then leave those paths for Volta's fp16 tensor cores (`OLALA_VOLTA_TC`, on by
default): an fp16 copy of each weight (INT8 codes exactly; bf16 rows scaled by a power of two, with a sparse
term for the few elements that do not fit), each activation row split into one to three exactly scaled fp16
terms, cuBLAS 12.9's native HMMA GEMM with fp32 sums, and a per-row FMA fallback for the rare rows the terms
do not cover. Outputs stay at the CPU-bf16 level of the checkpoints. Through volta-forever on a
V100-SXM2-32GB, decode per request at 32 / 128 requests and the TTFT of a 480-token prompt: bf16 20.7 -> 31.3
/ 6.9 -> 10.7 tok/s, 0.177 -> 0.110 s; Q8 24.2 -> 32.5 / 7.8 -> 12.0, 0.192 -> 0.106 s; Q4A2 22.2 -> 29.0 /
7.5 -> 11.1, 0.196 -> 0.115 s; 1-8 rows unchanged; 1.8-2.8 GiB less KV cache (the fp16 weight copies take
about 2 GiB). `OLALA_VOLTA_TC=0` gives the previous outputs back, bit for bit. With `VF_SAFE=1`
(volta-forever's conservative switch) this edition, with its truncation bias, is off; the fp32-FMA kernels
that only reorder a sum stay on, though strictly they are not bit-identical to stock either.

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

### Speculative decoding

Olala verifies drafts with a column-slot protocol: each request keeps `1 + K` Mamba-3 state and DiffTPA
token-shift columns, and the next step resumes from the column of the last accepted token, so a rejection needs
no state copy. The validated configuration uses Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`; candidate-head
drafters such as DSpine select it automatically). Olala exposes auxiliary hidden states (`SupportsEagle3`) to
EAGLE-3, DFlash and [DSpine](docs/features/speculative_decoding/dspine.md) drafters; by default the outputs of
layers 1, 9, 17, 25 and 33.

Requirements: the factorized TPA cache with F1 (`OLALA_TPA_FACTOR_SPEC`, on by default on SM70 / SM80 / SM120)
or the dense paged KV cache (`OLALA_TPA_FACTOR=0`, the default elsewhere), `OLALA_REPLAY_SSM` unset or `0`, fp32
Mamba-3 state (the default). With a DSpine drafter, e.g. on an
RTX 5090 with a Q8 target:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 OLALA_REPLAY_SSM=0 \
OLALA_DSPINE_FUSED=2 OLALA_DSPINE_GEMV=1 OLALA_DSPINE_CUDA_NORM=1 \
vllm serve <target> --trust-remote-code --mamba-ssm-cache-dtype float32 --max-num-seqs 2 \
    --speculative-config '{"method": "dflash", "model": "<drafter>", "num_speculative_tokens": 3,
                           "attention_backend": "TRITON_ATTN"}'
```

Speculation pays at low concurrency only. The spec layout holds about four times fewer tokens than direct
serving with ReplaySSM and the factorized TPA cache (fp32 state times `1 + K` columns, dense TPA KV, drafter KV),
and steps that mix prefills with verifies run outside the FULL CUDA graphs. Measured with a v1 DSpine drafter,
`K = 3`, against direct serving with ReplaySSM and the factorized cache (Q8 target, RTX 5090): x1.20 greedy and
x1.11 sampled output tokens/s at one request; slower from 8 concurrent requests under continuous arrivals. Use
dynamic speculative decoding (`num_speculative_tokens_per_batch_size`, `0` above the break-even batch size) or
route batched traffic to a direct server.

Five opt-in flags remove most of that gap: `OLALA_SPLIT_PREFILL=1 OLALA_SPLIT_PREFILL_MAX_DECODES=16` (decode
steps stay in the FULL graphs), `OLALA_PREFILL_NOSYNC=1`, `OLALA_TS_FOLD=1 OLALA_SPEC_LIGHT=1
VLLM_KV_CACHE_GROUP_SIZE=7` (x1.55-1.74 KV capacity, so `--max-num-seqs 128`), `OLALA_TPA_MIXED_FUSED=1` and
`OLALA_DSPINE_SHARED_HEAD=1`. On an RTX 5090 (heretic-v3-Q8 target, v3 DSpine drafter, `K = 5`, 512 output
tokens, continuous arrivals), spec then beats direct serving with ReplaySSM and the factorized cache at every
concurrency from 1 to 128: x1.4-1.6 at one request, x1.07-1.20 at 8, x1.9-2.3 at 128 output tokens/s.

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

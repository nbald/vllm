# DSpine

DSpine is a parallel drafter built on [DFlash](README.md): a few Qwen3 decoder
layers draft a whole block in one forward, conditioned on auxiliary hidden
states of the target model. After every draft layer, an adjacent causal
injection writes a gated message from position `t - 1` into position `t`
(position 1 receives the anchor token's code). The candidates are the top-k
tokens of the drafter's head; their transitions are rescored with an exact fp32
refinement of the last write over a learned transfer codebook, and the DFlash2
candidate sampler walks those scores. Model Runner V2 is selected
automatically.

## Usage

```bash
vllm serve /path/to/target \
    --speculative-config '{"method": "dflash", "model": "/path/to/dspine", "num_speculative_tokens": 3}'
```

- `1 + num_speculative_tokens` must not exceed the checkpoint's `block_size`.
  The draft block is causal, so a shorter block is an exact prefix of the
  trained one.
- Dynamic speculative decoding (`num_speculative_tokens_per_batch_size`) is
  supported. The drafter always drafts its full block and the runner truncates
  it to the scheduled length; a length of 0 skips the draft forward. A shorter
  non-zero length therefore saves verify work only: on a V100, `K = 1` in a
  `K = 2` server drafted at the `K = 2` width (draft 6.5 ms against 2.1 ms for a
  static `K = 1` at 4 requests).
- An attention backend for the drafter alone can be set with
  `"attention_backend"` in the speculative config. Prefer it over the global
  `--attention-backend`, which also changes the target's kernels.
- Watermarking is not supported, as for the other candidate-head drafters.
- With an Olala target, the TPA-factorized KV cache (`OLALA_TPA_FACTOR`, on by
  default) keeps its verify path, F1 (`OLALA_TPA_FACTOR_SPEC`), on by default
  for compute capability majors 7, 8 and 12, with a log line. It was measured
  on sm_120 (RTX 5090), sm_86 (RTX 3080) and SM70 (V100 through volta-forever);
  sm_80 (A100) and sm_89 (Ada) share the SM80-family kernels but were not
  measured. Its accuracy is that of the
  factor cache (the verify adds no loss of its own) and it doubles the KV
  capacity. Elsewhere, or with `OLALA_TPA_FACTOR_SPEC=0`, a speculative config
  turns the factor cache off, with a log line, and uses the dense paged KV
  cache; `OLALA_TPA_FACTOR=1` set explicitly is then refused at startup.

## Checkpoint format

`config.json` is a Qwen3 config (`model_type: "qwen3"`) with
`architectures: ["DSpineDraftModel"]` and:

| Key | Meaning |
|---|---|
| `vocab_size`, `draft_vocab_size` | Both equal to the target's vocabulary size (required). |
| `block_size` | Draft block length, anchor included. |
| `dflash_config.mask_token_id` | Token fed at the masked draft positions. |
| `dflash_config.target_layer_ids` | Target layers whose outputs feed the drafter (0-based; vLLM captures aux index `id + 1`). |
| `dflash_config.block_size`, `causal: true`, `sample_from_anchor: false` | Block geometry. |
| `dflash_config.transfer_dim`, `message_dim` | Injection dimensions (`transfer_dim <= hidden_size`). |
| `dflash_config.transition_top_k` | Candidates per position. |
| `dflash_config.transfer_space_eps` | Injection epsilon (default `1e-6`). |
| `has_own_lm_head` | Optional. `true`: the drafter's own `lm_head.weight` is loaded instead of sharing the target's head, e.g. a BF16 head for a quantized target. |
| `layer_types`, `sliding_window`, `use_sliding_window` | Optional. `sliding_attention` on every layer with a window `W` limits the drafter's attention over its context K/V to `q - k < W` (block keys are always visible), so the drafter KV cache is a sliding-window cache. The target keeps its full context. |

Tensors, BF16, in the HF names of the DFlash loader:

| Tensor | Shape |
|---|---|
| `fc.weight` | `(hidden, n_aux * target_hidden)` |
| `hidden_norm.weight`, `norm.weight` | `(hidden,)` |
| `layers.<i>.*` | Qwen3 decoder layer |
| `injection.readouts.<i>.weight` | `(transfer_dim, hidden)`, one per layer |
| `injection.writes.<i>.weight` | `(hidden, message_dim)`, one per layer |
| `injection.message.weight`, `injection.receiver_gate.weight`, `injection.predecessor_gate.weight` | `(message_dim, transfer_dim)` |
| `injection.gate_bias` | `(message_dim,)` |
| `transfer_codebook` | `(vocab, transfer_dim)` |
| `lm_head.weight` | `(vocab, hidden)`, only with `has_own_lm_head` |

The input embedding is always the target's. A trained mask embedding can ship
as `mask_embedding.pt` (`{"mask_token_id": int, "embedding": (hidden,)}`), as
for DFlash.

Checkpoints in the original DSpine layout also load: a
`dflash_config.selected_prefix_code_bridge` section is mapped to the keys above,
and the `selected_prefix_bridge.*` tensors (`codebook`, `read_weight`,
`early_read_weights`, `feature_write_weights`, `gate`, `predecessor_gate`,
`prev`) are mapped to the `injection.*` names. Only independent causal writes at
every layer with a serial exact last-write refinement and a zero terminal
projection are supported; other settings are rejected at load time.

## Fast draft path

The eager draft step is several hundred small kernels. Three opt-in flags,
read at import, make it cheaper; they leave the target untouched and keep the
target's output distribution (every draft token is verified):

| Variable | Effect |
|---|---|
| `OLALA_DSPINE_FUSED=1` | The injection after every layer and the transition scoring each run as one `torch.compile`'d function. Single rank only for the transition scoring. |
| `OLALA_DSPINE_FUSED=2` | Also compiles every draft decoder layer. |
| `OLALA_DSPINE_FUSED=3` | Level 2 plus hand-fused Triton kernels (`qwen3_dspine_kernels.py`) for draft steps of at most `OLALA_DSPINE_FUSED_MAX_ROWS` rows (64; larger steps keep level 2): q/k norm + rotary, context K/V store, inputs, injection, transition scoring. Single rank only for the transition scoring. |
| `OLALA_DSPINE_GEMV=1` | The drafter's BF16 linears and head use the skinny GEMV of the Olala kernels (`olala_linear`), up to 64 rows. |
| `OLALA_DSPINE_CUDA_NORM=1` | The drafter's RMSNorms use vLLM's CUDA kernels instead of the native implementation. |

The fused path is not bit-identical to the eager one (bf16 rounding of
different GEMM splits); accepted lengths matched the eager drafter in the
validation runs. On an RTX 5090 with a 5-layer drafter and `K = 3`, all three
flags with `"attention_backend": "TRITON_ATTN"` take a captured draft step from
1.61 ms to 0.77 ms at one request.

Level 3 is the recommended setting. The draft step is about 180 small kernels at level 2. On GPUs with a high
per-kernel cost their count sets the draft time: on a V100 through volta-forever, 1.6 µs of graph launch per kernel.
Level 3 replaces the chains with single launches:
- an RTX 5090 at `K = 5` goes from 183 to 115 kernels per draft step, and from 0.64 to 0.52 ms at one request;
- a V100 at `K = 3` goes from 1.71 to 1.28 ms per draft step.

The kernels use plain loads, reductions and FMAs (tl.dot only in the transition scoring, with an FMA path on the
SM70 family), so they run on SM70, the SM80 family and SM120. They are not bit-identical to level 2.

Measured against level 2, served greedy and sampled acceptance holds: within 0.4 % on an RTX 5090, 1-16 requests,
three seeds. Output tokens/s at one to two requests: ×1.01-1.03 on a V100, an RTX 3080 and an RTX 5090. Elsewhere
the ratio is within noise.

On the SM70 family the injection keeps its explicit update kernel. Its delta form, combined with the fused q/k path,
lowered acceptance there: the drafter's GEMVs run on the fp16 tensor-core path from 9 rows.
`OLALA_DSPINE_FUSED_OFF=qk,ctx,prep,inject,delta,transitions,compile` turns groups off, for attribution.

`OLALA_DSPINE_SHARED_HEAD=1` makes a drafter that ships its own head (`has_own_lm_head`) use the target's
instead: the drafter's `lm_head.weight` is not loaded (0.47 GB for a 151,936 x 1,536 BF16 head), the draft
logits go through the target's head (INT8 on W8A16 checkpoints) and the exact transition rescoring reads
dequantized rows. Use it when the drafter's head is a copy of the target's: on the Q8 target the served
acceptance changed by -0.01 % offline.

`OLALA_DSPINE_INT8` quantizes the drafter's own decoder linears (qkv, o, gate_up, down) and its context projection
`fc` at load time: symmetric RTN of the BF16 weights, one fp32 scale per output channel. They run through the
target's W8A16 linear method, which takes its fp16 tensor-core copy on the V100 family, and the fused context K/V
projection uses the same INT8 rows. The injection and transition weights, the codebook and the norms stay BF16.

On the Q8 target with the v3r2 drafter, offline acceptance on the 2,000 held-out answers is unchanged: τ@5 3.696
greedy and 3.625 sampled. Served greedy against the BF16 drafter (`K = 5`, `K = 3` on a V100):
- **one request:** +1.5 % on an RTX 5090 (draft step 0.71 → 0.63 ms) and +2.6 % on a V100 (2.04 → 1.68 ms);
- **8 to 128 requests:** within ±1 %, apart from two points.
  - c = 16 on the 5090: +7.8 %.
  - c = 32 on an RTX 3080: −1.6 / −1.8 % over two rounds. The time to first token there drops by 22 %.

It is on by default on the GPUs measured (SM70, the SM80 family, SM120) despite the RTX 3080 point at c = 32.
`OLALA_DSPINE_INT8=0` keeps the BF16 linears, and `1` forces INT8 on other GPUs.

## Serving on a V100

On a V100 (SM70 through volta-forever, the Q8 Olala-v1.0-mini-7A1B target and the Télégraphe drafter), the
speculative server takes up to 16 concurrent streams. **Above 16 streams, route the traffic to a direct server**
(the same target without a speculative config). This is the role of `MAX_SEQS` in the RTX 5090 setting.
Up to 16 streams, the speculative server uses:
- `num_speculative_tokens = 2`, with `num_speculative_tokens_per_batch_size` `[[1, 2, 2], [3, 6, 0], [7, 16, 2]]`:
  K = 2 at 1-2 and 7-16 requests, no drafts at 3-6;
- `--max-num-seqs 16`;
- the INT8 drafter (default), `OLALA_DSPINE_FUSED=3`, and F1 (default under spec on SM70);
- `OLALA_VOLTA_MOE_ROWS=1`, the multi-row MoE, as the V100 spec option: +6-7 % at one request, neutral elsewhere.

The decode CUDA graphs need every request count's size, n × (1 + K). The Olala capture list adds them for up to
32 requests (`OLALA_SPEC_GRAPHS_MAX_REQS`). Without them, K = 2 at 4 requests pads its 12 tokens to the 18-token
graph: 26.4 ms per step against 20.0 ms.

```bash
# Through volta-forever 0.1.5 (https://github.com/nbald/volta-forever: install.sh, then its vLLM guide for the
# CUDA 13 toolkit and the persistent caches). VF_PROXY_CACHE is a directory you own, with mode 0700.
VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1 VF_JIT_LINEINFO=1 VF_PROXY_CACHE=$HOME/.cache/vf-proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  OLALA_MAMBA3_VERIFY=cuda OLALA_SPLIT_PREFILL=1 OLALA_SPLIT_PREFILL_MAX_DECODES=16 OLALA_PREFILL_NOSYNC=1 \
  OLALA_TS_FOLD=1 OLALA_SPEC_LIGHT=1 VLLM_KV_CACHE_GROUP_SIZE=7 OLALA_TPA_MIXED_FUSED=1 \
  OLALA_TPA_FACTOR=1 OLALA_TPA_FACTOR_SPEC=1 \
  OLALA_DSPINE_SHARED_HEAD=1 OLALA_DSPINE_FUSED=3 OLALA_DSPINE_GEMV=1 OLALA_DSPINE_CUDA_NORM=1 \
  OLALA_VOLTA_MOE_ROWS=1 \
  vf python3 -m vllm.entrypoints.cli.main serve /path/to/Olala-v1.0-mini-7A1B-Q8 \
    --trust-remote-code --served-model-name olala --reasoning-parser olala \
    --max-model-len 65536 --max-num-seqs 16 \
    --speculative-config '{"method": "dflash", "model": "/path/to/Olala-v1.0-mini-7A1B-Telegraphe",
      "num_speculative_tokens": 2, "attention_backend": "TRITON_ATTN",
      "num_speculative_tokens_per_batch_size": [[1, 2, 2], [3, 6, 0], [7, 16, 2]]}'
```

Output tokens/s against the direct server (max-num-seqs 128), same run, 512-token outputs, one round, through the
volta-forever 0.1.5 release:

| c | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| sampled | ×1.207 | ×1.042 | ×0.997 | ×1.062 | ×1.240 |
| greedy | ×1.274 | ×1.091 | ×1.001 | ×1.123 | ×1.315 |

Why this schedule (measured through volta-forever 0.1.3 with its later fixes):
- **Static K = 2:** ×0.91 sampled / ×0.97 greedy at c = 4, even with the exact graph sizes. Verifying 12 tokens costs
  more on this MoE target than the drafts return.
- **c = 32:** every static K loses (×0.92-0.94 sampled).
- **c = 128:** under KV pressure the running batch stays at 25-64 requests, the range where c = 32 needs K = 0.
  No batch-size schedule serves both: a K = 0 range there gave ×0.96-0.98 at c = 128.
- **KV at 64k-token requests:** the speculative layout holds 11.1×, against 16.4× for direct.

## Limitations

- Model Runner V2 only (selected automatically).
- The fused transition scoring runs on a single rank; with tensor parallelism
  the eager scoring is used.
- No watermarking.

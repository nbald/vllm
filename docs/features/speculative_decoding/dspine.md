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
  it to the scheduled length; a length of 0 skips the draft forward.
- An attention backend for the drafter alone can be set with
  `"attention_backend"` in the speculative config. Prefer it over the global
  `--attention-backend`, which also changes the target's kernels.
- Watermarking is not supported, as for the other candidate-head drafters.
- With an Olala target, the TPA-factorized KV cache (`OLALA_TPA_FACTOR`, on by
  default) has no multi-token verify path. A speculative config turns it off,
  with a log line, and uses the dense paged KV cache; `OLALA_TPA_FACTOR=1` set
  explicitly is refused at startup.

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
| `OLALA_DSPINE_GEMV=1` | The drafter's BF16 linears and head use the skinny GEMV of the Olala kernels (`olala_linear`), up to 64 rows. |
| `OLALA_DSPINE_CUDA_NORM=1` | The drafter's RMSNorms use vLLM's CUDA kernels instead of the native implementation. |

The fused path is not bit-identical to the eager one (bf16 rounding of
different GEMM splits); accepted lengths matched the eager drafter in the
validation runs. On an RTX 5090 with a 5-layer drafter and `K = 3`, all three
flags with `"attention_backend": "TRITON_ATTN"` take a captured draft step from
1.61 ms to 0.77 ms at one request.

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

## Limitations

- Model Runner V2 only (selected automatically).
- The fused transition scoring runs on a single rank; with tensor parallelism
  the eager scoring is used.
- No watermarking.

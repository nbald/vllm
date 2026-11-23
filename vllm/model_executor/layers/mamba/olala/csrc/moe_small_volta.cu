// Weight-only latent MoE for 3-32 tokens on Volta (V100), natively or through volta-forever: CUDA editions of the
// Triton Volta small path (latent_moe_int8.olala_moe_small_int8_volta: _volta_small_up_kernel, _i8_small_down_kernel,
// _volta_small_exit_kernel) that return its bits. Every output is the same set of fp32 chains as in the Triton tile
// (one chain per k position c of the tile's k extent BK: an fmaf chain over the k steps k = s BK + c, from +0),
// combined by the same tree (the Triton layout's in-thread pairwise tree, lane butterfly and cross-warp order), with
// the same epilogue roundings; the routing reproduces _topk_slot instruction for instruction. Only the work
// decomposition differs:
//   route   top-k of every token once (not once per program), and the rows routed to each expert, deterministic;
//   up      the shared expert's up projection over k-major activations (16 tokens x 4 rows per thread), and the routed
//           experts by expert: each expert's weights are read once for all the rows routed to it (16-byte loads);
//   down    the same expert grouping;
//   exit    the routed latent formed once (latent_kernel), fc2_latent + shared down with 16 tokens x 4 rows per thread.
// Triton tiles by tokens rounded up to a power of two (mb): routed up [128, 32] (INT8; INT4 affine: [64, 32] position
// pairs), routed down [256, 8] (INT4: [128, 8]), shared up and exit [mb, BN, 64] for mb 4 and 8 (k over 32 lanes x 2
// warps: butterfly, then G0 + G1: NG = 2) and [mb, BN, 32] for mb 16 and 32 (NG = 1). The routed latent's sum is
// ((+0 + p0) + p1) + ... + p5, times the routed scale, rounded to bf16.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int NEXP = 256;    // routed experts
constexpr int N1 = 768;      // routed expert intermediate width (w13 rows, w2's K)
constexpr int KTOP = 6;      // top-k of the model (the routed latent sums K partial rows)
constexpr int D2 = 384;      // routed latent width (fc2_latent's K)
constexpr int SW = 1536;     // shared intermediate width (shared down's K)
constexpr int HID = 1536;    // hidden width (exit rows)

__device__ __forceinline__ float f_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_fma(float a, float b, float c) {
  float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
// int8 byte i of a word as fp32 (exact): the byte, sign-extended
__device__ __forceinline__ float i8_at(uint32_t w, int i) { return (float)(int)(int8_t)((w >> (8 * i)) & 0xffu); }

// The routed latent element bf16(scale * sum_k p_k) for k = 0..KTOP-1 (ZERO_FIRST: +0 + p0 + p1 ...).
template <bool ZERO_FIRST>
__device__ __forceinline__ float latent_elem(const float* __restrict__ part, int m, int k, float scale) {
  const float* p = part + (size_t)m * KTOP * D2 + k;
  float r = ZERO_FIRST ? f_add(__ldg(p), 0.0f) : __ldg(p);
#pragma unroll
  for (int j = 1; j < KTOP; ++j) r = f_add(r, __ldg(p + (size_t)j * D2));
  return bf16_round(f_mul(scale, r));
}

// ---------------------------------------------------------------------------------------------------------------
// exit, variant B (mb 8, 16, 32): lane = chain c mod 32 (as the Triton tile), NG = BK / 32 chain groups in registers,
// RM tokens x RN rows per thread. The activations come k-major (prep_kernel: rT[k][m], sT[k][m], MBP tokens per k),
// so a lane loads its RM tokens' values for one k as 16-byte vectors. Tree: a reduce-scatter butterfly over the 32
// lanes (the same pairs as the Triton all-reduce butterfly: levels xor 16, 8, 4, 2, 1), then G0 + G1 for NG = 2.

template <int R>
__device__ __forceinline__ void load_tokens(const __nv_bfloat16* p, float* a) {
  // R (8 or 16) consecutive bf16 of one k: 16-byte loads
#pragma unroll
  for (int j = 0; j < R / 8; ++j) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p) + j);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int t = 0; t < 4; ++t) { a[8 * j + 2 * t] = bf16_lo(w[t]); a[8 * j + 2 * t + 1] = bf16_hi(w[t]); }
  }
}

// One part (fc2 or shared down): acc[NG][RM * RN] over KD (row-major int8 weights w[n][k], activations aT[k][MBP]).
template <int MBP, int NG, int RM, int RN, int KD>
__device__ __forceinline__ void part_b(const int8_t* __restrict__ w, const __nv_bfloat16* __restrict__ aT, int m0,
                                       const int* n, int lane, float (&acc)[NG][RM * RN]) {
#pragma unroll
  for (int g = 0; g < NG; ++g)
#pragma unroll
    for (int v = 0; v < RM * RN; ++v) acc[g][v] = 0.0f;
#pragma unroll 4
  for (int k0 = 0; k0 < KD; k0 += 32 * NG) {
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      const int k = k0 + 32 * g + lane;
      float wf[RN];
#pragma unroll
      for (int j = 0; j < RN; ++j) wf[j] = (float)__ldg(w + (size_t)n[j] * KD + k);
      float a[RM];
      load_tokens<RM>(aT + (size_t)k * MBP + m0, a);
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) acc[g][i * RN + j] = f_fma(wf[j], a[i], acc[g][i * RN + j]);
    }
  }
}

// Reduce-scatter butterfly over the 32 lanes of V = RM * RN values (V a multiple of 32): returns V / 32 totals; total
// t of lane l is value index (l << log2(V/32)) + t.
template <int V>
__device__ __forceinline__ void reduce_scatter(float (&v)[V], int lane) {
  int half = V / 2;
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1, half >>= 1) {
    const bool up = lane & d;
#pragma unroll
    for (int t = 0; t < V; ++t) {
      if (t < half) {
        const float keep = up ? v[t + half] : v[t], send = up ? v[t] : v[t + half];
        v[t] = f_add(keep, __shfl_xor_sync(0xffffffffu, send, d));
      }
    }
  }
}

template <int MBP, int NG, int RM, int RN>
__global__ void __launch_bounds__(128)
exit_b_kernel(const __nv_bfloat16* __restrict__ rT, const int8_t* __restrict__ fw, const float* __restrict__ fs,
              const __nv_bfloat16* __restrict__ sT, const int8_t* __restrict__ dw, const float* __restrict__ ds,
              __nv_bfloat16* __restrict__ out, int M) {
  constexpr int MG = MBP / RM;            // token groups (warps along m)
  constexpr int NGW = 4 / MG;             // warps along n
  constexpr int V = RM * RN;
  static_assert(V % 32 == 0 && MG * NGW == 4, "tile");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int mg = warp % MG, ng = warp / MG;
  const int m0 = mg * RM;
  int n[RN];
#pragma unroll
  for (int j = 0; j < RN; ++j) n[j] = (blockIdx.x * NGW + ng) * RN + j;
  constexpr int PER = V / 32;
  float acc[NG][V], tf[PER], td[PER];
  part_b<MBP, NG, RM, RN, D2>(fw, rT, m0, n, lane, acc);
#pragma unroll
  for (int g = 0; g < NG; ++g) reduce_scatter<V>(acc[g], lane);
#pragma unroll
  for (int t = 0; t < PER; ++t) tf[t] = NG == 2 ? f_add(acc[0][t], acc[1][t]) : acc[0][t];
  part_b<MBP, NG, RM, RN, SW>(dw, sT, m0, n, lane, acc);
#pragma unroll
  for (int g = 0; g < NG; ++g) reduce_scatter<V>(acc[g], lane);
#pragma unroll
  for (int t = 0; t < PER; ++t) td[t] = NG == 2 ? f_add(acc[0][t], acc[1][t]) : acc[0][t];
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int vi = lane * PER + t, i = vi / RN, j = vi % RN, m = m0 + i;
    if (m < M) {
      const int nn = (blockIdx.x * NGW + ng) * RN + j;   // = n[j], without a dynamically indexed array
      out[(size_t)m * HID + nn] = __float2bfloat16_rn(f_fma(tf[t], __ldg(fs + nn), f_mul(td[t], __ldg(ds + nn))));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Routing (latent_moe_int8._topk_slot): sc = 1 / (1 + 2^(lg * -log2 e)) (ex2.approx, div.full), sb = sc + bias,
// NaN -> -inf; six rounds of max (ties: the lowest expert, among all with sb == max), w = sc of the winner, tot =
// ((0 + w0) + w1) + ..., tw = w / tot (div.full). CTA 0 routes the M tokens (a warp per token, lane l holding
// experts 8l..8l+7) and lists, for each expert in increasing order, the rows m K + k routed to it in increasing order
// (deterministic). The other CTAs write hT[k][m] = hidden[m][k] (k-major, MBP tokens, zero beyond M) for the shared
// expert's up projection.
__device__ __forceinline__ float f_ex2(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

struct RouteMeta {        // int32 [1 + 3 * NEXP + 32 * KTOP]
  int nslots;             // distinct experts
  int slot_e[NEXP];       // expert of slot j (increasing)
  int slot_start[NEXP];   // first index of slot j in rows[]
  int slot_cnt[NEXP];     // rows routed to slot j
  int rows[32 * KTOP];    // rows m K + k, by slot, increasing within a slot
};

template <int MBP>
__global__ void __launch_bounds__(256)
route_kernel(const float* __restrict__ logits, const float* __restrict__ logits_lo, long long sl,
             const float* __restrict__ bias, const __nv_bfloat16* __restrict__ hidden, long long shd,
             int* __restrict__ ids, float* __restrict__ tw, RouteMeta* __restrict__ meta,
             __nv_bfloat16* __restrict__ hT, int M) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (blockIdx.x > 0) {   // hT[k][m]
    for (int e = (blockIdx.x - 1) * 256 + tid; e < HID * MBP; e += (gridDim.x - 1) * 256) {
      const int k = e / MBP, m = e % MBP;
      hT[e] = m < M ? hidden[(size_t)m * shd + k] : __float2bfloat16_rn(0.0f);
    }
    return;
  }
  __shared__ int sids[32 * KTOP];
  __shared__ int scnt[NEXP], sstart[NEXP], sslot[NEXP];
  float bs[8];
  {
    const float4 b0 = __ldg(reinterpret_cast<const float4*>(bias + 8 * lane));
    const float4 b1 = __ldg(reinterpret_cast<const float4*>(bias + 8 * lane) + 1);
    bs[0] = b0.x; bs[1] = b0.y; bs[2] = b0.z; bs[3] = b0.w; bs[4] = b1.x; bs[5] = b1.y; bs[6] = b1.z; bs[7] = b1.w;
  }
  for (int m = warp; m < M; m += 8) {
    float lg[8];
    {
      const float4* p = reinterpret_cast<const float4*>(logits + (size_t)m * sl + 8 * lane);
      const float4 a0 = __ldg(p), a1 = __ldg(p + 1);
      lg[0] = a0.x; lg[1] = a0.y; lg[2] = a0.z; lg[3] = a0.w; lg[4] = a1.x; lg[5] = a1.y; lg[6] = a1.z; lg[7] = a1.w;
      if (logits_lo != nullptr) {
        const float4* q = reinterpret_cast<const float4*>(logits_lo + (size_t)m * sl + 8 * lane);
        const float4 c0 = __ldg(q), c1 = __ldg(q + 1);
        const float lo[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) lg[j] = f_add(lg[j], lo[j]);
      }
    }
    float sc[8], sb[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sc[j] = f_div_full(1.0f, f_add(f_ex2(f_mul(lg[j], __uint_as_float(0xBFB8AA3Bu))), 1.0f));
      const float v = f_add(sc[j], bs[j]);
      sb[j] = v != v ? -INFINITY : v;
    }
    float tot = 0.0f, wk[KTOP];
    int ek[KTOP];
#pragma unroll
    for (int kk = 0; kk < KTOP; ++kk) {
      float mx = sb[0];
#pragma unroll
      for (int j = 1; j < 8; ++j) mx = fmaxf(mx, sb[j]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      int idx = NEXP;
#pragma unroll
      for (int j = 7; j >= 0; --j) if (sb[j] == mx) idx = 8 * lane + j;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffffu, idx, o));
      idx = min(idx, NEXP - 1);
      float w = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) if (8 * lane + j == idx) w = sc[j];
      w = __shfl_sync(0xffffffffu, w, idx >> 3);
      tot = f_add(tot, w);
      wk[kk] = w;
      ek[kk] = idx;
#pragma unroll
      for (int j = 0; j < 8; ++j) if (8 * lane + j == idx) sb[j] = -INFINITY;
    }
    if (lane < KTOP) {
      float wsel = wk[0];
      int esel = ek[0];
#pragma unroll
      for (int kk = 1; kk < KTOP; ++kk) if (lane == kk) { wsel = wk[kk]; esel = ek[kk]; }
      ids[m * KTOP + lane] = esel;
      tw[m * KTOP + lane] = f_div_full(wsel, tot);
      sids[m * KTOP + lane] = esel;
    }
  }
  __syncthreads();
  // per expert (thread e): its rows in increasing order, then slots by a scan over the experts
  const int MK = M * KTOP;
  const int e = tid;
  int cnt = 0;
  for (int r = 0; r < MK; ++r) cnt += sids[r] == e;
  scnt[e] = cnt;
  __syncthreads();
  if (tid == 0) {   // scan (256 entries: one thread is enough next to the routing)
    int start = 0, slot = 0;
    for (int x = 0; x < NEXP; ++x) {
      sstart[x] = start;
      sslot[x] = scnt[x] > 0 ? slot : -1;
      if (scnt[x] > 0) {
        meta->slot_e[slot] = x;
        meta->slot_start[slot] = start;
        meta->slot_cnt[slot] = scnt[x];
        ++slot;
      }
      start += scnt[x];
    }
    meta->nslots = slot;
  }
  __syncthreads();
  if (cnt > 0) {
    int at = sstart[e];
    for (int r = 0; r < MK; ++r)
      if (sids[r] == e) meta->rows[at++] = r;
  }
}

// relu^2 as the Triton epilogue: bf16(max(bf16(y), 0)^2)
__device__ __forceinline__ __nv_bfloat16 relu2_bf16(float y) {
  const float h = fmaxf(bf16_round(y), 0.0f);
  return __float2bfloat16_rn(f_mul(h, h));
}

// 16 int8 of a uint4 as fp32
__device__ __forceinline__ void i8x16(const uint4 u, float* w) {
  const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = i8_at(ww[i >> 2], i & 3);
}

// pairwise tree over 16 consecutive chain values (Triton's in-thread reduction): (0+1), (2+3), ... then pairs
__device__ __forceinline__ float tree16(float* v) {
#pragma unroll
  for (int d = 1; d < 16; d <<= 1)
#pragma unroll
    for (int i = 0; i < 16; i += 2 * d) v[i] = f_add(v[i], v[i + d]);
  return v[0];
}

// ---------------------------------------------------------------------------------------------------------------
// Up: CTAs [0, NSH): the shared expert's up projection for all tokens (the exit's tile code over hT: Triton tile
// [MB, SH_BN, 32], lane = k mod 32, butterfly), relu^2, written k-major (sT[n][m]) for the exit. The other CTAs:
// routed (expert slot, 32 output rows): Triton tile [128, 32] (UP_BK, UP_BN), a thread = 16 consecutive k x 2 rows,
// 8 threads per row (lanes xor 4, 2, 1); the expert's weights are loaded once for every row routed to it.
template <int MBP>
__global__ void __launch_bounds__(128)
up_kernel(const __nv_bfloat16* __restrict__ hT, const int8_t* __restrict__ suw, const float* __restrict__ sus,
          __nv_bfloat16* __restrict__ sT, const float* __restrict__ x, long long sx, const int8_t* __restrict__ w1,
          const float* __restrict__ w1s, const RouteMeta* __restrict__ meta, __nv_bfloat16* __restrict__ h, int M,
          int nsh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if ((int)blockIdx.x < nsh) {
    // Triton shared tiles: 3-8 tokens [4 | 8, 16 | 8, 64] (k over 32 lanes x 2 warps: NG = 2), 9-32 tokens
    // [16 | 32, 8 | 4, 32] (NG = 1)
    constexpr int NG = MBP == 8 ? 2 : 1, RM = MBP == 8 ? 8 : 16, RN = 4, MG = MBP / RM, NGW = 4 / MG;
    constexpr int V = RM * RN, PER = V / 32;
    const int mg = warp % MG, ng = warp / MG, m0 = mg * RM;
    int n[RN];
#pragma unroll
    for (int j = 0; j < RN; ++j) n[j] = (blockIdx.x * NGW + ng) * RN + j;
    float acc[NG][V];
    part_b<MBP, NG, RM, RN, HID>(suw, hT, m0, n, lane, acc);
#pragma unroll
    for (int g = 0; g < NG; ++g) reduce_scatter<V>(acc[g], lane);
#pragma unroll
    for (int t = 0; t < PER; ++t) {
      const int vi = lane * PER + t, i = vi / RN, j = vi % RN, m = m0 + i;
      const int nn = (blockIdx.x * NGW + ng) * RN + j;
      const float tot = NG == 2 ? f_add(acc[0][t], acc[NG - 1][t]) : acc[0][t];
      sT[(size_t)nn * MBP + m] = m < M ? relu2_bf16(f_mul(tot, __ldg(sus + nn))) : __float2bfloat16_rn(0.0f);
    }
    return;
  }
  constexpr int NBU = N1 / 32;
  const int b = blockIdx.x - nsh, slot = b / NBU, nb = b % NBU;
  if (slot >= meta->nslots) return;
  const int e = meta->slot_e[slot], start = meta->slot_start[slot], cnt = meta->slot_cnt[slot];
  const int q = lane & 7, nsub = lane >> 3;
  const int n0 = nb * 32 + (warp * 4 + nsub) * 2;
  const int8_t* we = w1 + (size_t)e * N1 * D2;
  uint4 wr[2][3];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int st = 0; st < 3; ++st)
      wr[j][st] = __ldg(reinterpret_cast<const uint4*>(we + (size_t)(n0 + j) * D2 + st * 128 + 16 * q));
  const float s0 = __ldg(w1s + (size_t)e * N1 + n0), s1 = __ldg(w1s + (size_t)e * N1 + n0 + 1);
  for (int c = 0; c < cnt; ++c) {
    const int row = meta->rows[start + c], m = row / KTOP;
    float acc[2][16];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[j][i] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      float a[16];
      const float4* p = reinterpret_cast<const float4*>(x + (size_t)m * sx + st * 128 + 16 * q);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const float4 v = __ldg(p + t);
        a[4 * t] = bf16_round(v.x); a[4 * t + 1] = bf16_round(v.y); a[4 * t + 2] = bf16_round(v.z); a[4 * t + 3] = bf16_round(v.w);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float w[16];
        i8x16(wr[j][st], w);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[j][i] = f_fma(a[i], w[i], acc[j][i]);
      }
    }
    float y[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float v = tree16(acc[j]);
      v = f_add(v, __shfl_xor_sync(0xffffffffu, v, 4));
      v = f_add(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = f_add(v, __shfl_xor_sync(0xffffffffu, v, 1));
      y[j] = v;
    }
    if (q == 0) {
      h[(size_t)row * N1 + n0] = relu2_bf16(f_mul(y[0], s0));
      h[(size_t)row * N1 + n0 + 1] = relu2_bf16(f_mul(y[1], s1));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Down: (expert slot, 16 output rows): Triton tile [256, 8] (DOWN_BK, DOWN_BN), a thread = 16 consecutive k x 2 rows,
// 16 threads per row (lanes xor 8, 4, 2, 1); part[row, n] = (T * scale[n]) * tw[row].
__global__ void __launch_bounds__(128)
down_kernel(const __nv_bfloat16* __restrict__ h, const int8_t* __restrict__ w2, const float* __restrict__ w2s,
            const float* __restrict__ tw, const RouteMeta* __restrict__ meta, float* __restrict__ part) {
  constexpr int NBD = D2 / 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int slot = blockIdx.x / NBD, nb = blockIdx.x % NBD;
  if (slot >= meta->nslots) return;
  const int e = meta->slot_e[slot], start = meta->slot_start[slot], cnt = meta->slot_cnt[slot];
  const int q = lane & 15, nsub = lane >> 4;
  const int n0 = nb * 16 + (warp * 2 + nsub) * 2;
  const int8_t* we = w2 + (size_t)e * D2 * N1;
  uint4 wr[2][3];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int st = 0; st < 3; ++st)
      wr[j][st] = __ldg(reinterpret_cast<const uint4*>(we + (size_t)(n0 + j) * N1 + st * 256 + 16 * q));
  const float s0 = __ldg(w2s + (size_t)e * D2 + n0), s1 = __ldg(w2s + (size_t)e * D2 + n0 + 1);
  for (int c = 0; c < cnt; ++c) {
    const int row = meta->rows[start + c];
    float acc[2][16];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[j][i] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      const uint4* p = reinterpret_cast<const uint4*>(h + (size_t)row * N1 + st * 256 + 16 * q);
      const uint4 u0 = __ldg(p), u1 = __ldg(p + 1);
      const uint32_t aw[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
      float a[16];
#pragma unroll
      for (int t = 0; t < 8; ++t) { a[2 * t] = bf16_lo(aw[t]); a[2 * t + 1] = bf16_hi(aw[t]); }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float w[16];
        i8x16(wr[j][st], w);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[j][i] = f_fma(a[i], w[i], acc[j][i]);
      }
    }
    float y[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float v = tree16(acc[j]);
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) v = f_add(v, __shfl_xor_sync(0xffffffffu, v, o));
      y[j] = v;
    }
    if (q == 0) {
      const float t = __ldg(tw + row);
      part[(size_t)row * D2 + n0] = f_mul(f_mul(y[0], s0), t);
      part[(size_t)row * D2 + n0 + 1] = f_mul(f_mul(y[1], s1), t);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// INT4 affine routed experts (Q4A2: MLX q4 layout, group 32, int32 (bf16 scale | bf16 bias << 16) per group), the
// Triton GEMV's FMT_INT4 arithmetic (latent_moe_int8._wq_gemv_x, _i4_tiles): position j of a step holds the even and
// odd k (2j, 2j + 1); w = bf16(fma(q, s, b)) for both; t = fma(a_lo, w_lo, a_hi * w_hi); acc_j = acc_j + t (from +0);
// no row scale. Up: Triton tile [64, 32] (16 positions per thread, 4 lanes: xor 2, 1). Down: [128, 8] (8 positions per
// thread, 16 lanes: xor 8, 4, 2, 1).
__device__ __forceinline__ float i4_w(uint32_t byte, bool hi, float s, float b) {
  const uint32_t q = hi ? (byte >> 4) & 15u : byte & 15u;
  return bf16_round(f_fma((float)(int)q, s, b));
}

// P positions (2P k) of one row: acc[j] += t_j for packed bytes pk (P bytes), group (s, b), activations lo / hi.
template <int P>
__device__ __forceinline__ void i4_step(const uint32_t* pk, uint32_t sb, const float* alo, const float* ahi, float* acc) {
  const float s = __uint_as_float(sb << 16), b = __uint_as_float(sb & 0xffff0000u);
#pragma unroll
  for (int j = 0; j < P; ++j) {
    const uint32_t byte = (pk[j >> 2] >> (8 * (j & 3))) & 0xffu;
    const float wl = i4_w(byte, false, s, b), wh = i4_w(byte, true, s, b);
    acc[j] = f_add(acc[j], f_fma(alo[j], wl, f_mul(ahi[j], wh)));
  }
}

template <int P>
__device__ __forceinline__ float treeP(float* v) {
#pragma unroll
  for (int d = 1; d < P; d <<= 1)
#pragma unroll
    for (int i = 0; i < P; i += 2 * d) v[i] = f_add(v[i], v[i + d]);
  return v[0];
}

// routed up, INT4: (expert slot, 32 output rows); thread = 1 row x 16 positions (32 k), 4 threads per row
__global__ void __launch_bounds__(128)
up_i4_kernel(const float* __restrict__ x, long long sx, const uint8_t* __restrict__ w1, const int* __restrict__ w1sb,
             const RouteMeta* __restrict__ meta, __nv_bfloat16* __restrict__ h, int nsh) {
  constexpr int NBU = N1 / 32, KB = D2 / 2, G = D2 / 32;   // bytes per row, groups per row
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int b = blockIdx.x - nsh, slot = b / NBU, nb = b % NBU;
  if (slot >= meta->nslots) return;
  const int e = meta->slot_e[slot], start = meta->slot_start[slot], cnt = meta->slot_cnt[slot];
  const int q = lane & 3, n = nb * 32 + warp * 8 + (lane >> 2);
  const uint8_t* we = w1 + ((size_t)e * N1 + n) * KB;
  const int* se = w1sb + ((size_t)e * N1 + n) * G;
  uint32_t pk[3][4], sb[3];
#pragma unroll
  for (int st = 0; st < 3; ++st) {   // step st: k = st * 128 + 32 q .. + 31 -> bytes st * 64 + 16 q .. + 15, group st * 4 + q
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(we + st * 64 + 16 * q));
    pk[st][0] = u.x; pk[st][1] = u.y; pk[st][2] = u.z; pk[st][3] = u.w;
    sb[st] = (uint32_t)__ldg(se + st * 4 + q);
  }
  for (int c = 0; c < cnt; ++c) {
    const int row = meta->rows[start + c], m = row / KTOP;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      float alo[16], ahi[16];
      const float4* p = reinterpret_cast<const float4*>(x + (size_t)m * sx + st * 128 + 32 * q);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const float4 v = __ldg(p + t);
        alo[2 * t] = bf16_round(v.x); ahi[2 * t] = bf16_round(v.y);
        alo[2 * t + 1] = bf16_round(v.z); ahi[2 * t + 1] = bf16_round(v.w);
      }
      i4_step<16>(pk[st], sb[st], alo, ahi, acc);
    }
    float v = treeP<16>(acc);
    v = f_add(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = f_add(v, __shfl_xor_sync(0xffffffffu, v, 1));
    if (q == 0) h[(size_t)row * N1 + n] = relu2_bf16(v);
  }
}

// routed down, INT4: (expert slot, 16 output rows); thread = 2 rows x 8 positions (16 k), 16 threads per row
__global__ void __launch_bounds__(128)
down_i4_kernel(const __nv_bfloat16* __restrict__ h, const uint8_t* __restrict__ w2, const int* __restrict__ w2sb,
               const float* __restrict__ tw, const RouteMeta* __restrict__ meta, float* __restrict__ part) {
  constexpr int NBD = D2 / 16, KB = N1 / 2, G = N1 / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int slot = blockIdx.x / NBD, nb = blockIdx.x % NBD;
  if (slot >= meta->nslots) return;
  const int e = meta->slot_e[slot], start = meta->slot_start[slot], cnt = meta->slot_cnt[slot];
  const int q = lane & 15, n0 = nb * 16 + (warp * 2 + (lane >> 4)) * 2;
  uint32_t pk[2][3][2], sb[2][3];
#pragma unroll
  for (int jn = 0; jn < 2; ++jn) {
    const uint8_t* we = w2 + ((size_t)e * D2 + n0 + jn) * KB;
    const int* se = w2sb + ((size_t)e * D2 + n0 + jn) * G;
#pragma unroll
    for (int st = 0; st < 3; ++st) {   // k = st * 256 + 16 q .. + 15 -> bytes st * 128 + 8 q .. + 7, group (st * 256 + 16 q) / 32
      const uint2 u = __ldg(reinterpret_cast<const uint2*>(we + st * 128 + 8 * q));
      pk[jn][st][0] = u.x; pk[jn][st][1] = u.y;
      sb[jn][st] = (uint32_t)__ldg(se + (st * 256 + 16 * q) / 32);
    }
  }
  for (int c = 0; c < cnt; ++c) {
    const int row = meta->rows[start + c];
    float acc[2][8];
#pragma unroll
    for (int jn = 0; jn < 2; ++jn)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[jn][j] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      // positions 8q..8q+7: k = 16 q + 2j (lo) and + 1 (hi), i.e. the two bf16 halves of 32-bit word j of the 16 k
      const uint4* hp = reinterpret_cast<const uint4*>(h + (size_t)row * N1 + st * 256 + 16 * q);
      const uint4 u = __ldg(hp), u2 = __ldg(hp + 1);
      const uint32_t aw[8] = {u.x, u.y, u.z, u.w, u2.x, u2.y, u2.z, u2.w};
      float alo[8], ahi[8];
#pragma unroll
      for (int t = 0; t < 8; ++t) { alo[t] = bf16_lo(aw[t]); ahi[t] = bf16_hi(aw[t]); }
#pragma unroll
      for (int jn = 0; jn < 2; ++jn) i4_step<8>(pk[jn][st], sb[jn][st], alo, ahi, acc[jn]);
    }
    float y[2];
#pragma unroll
    for (int jn = 0; jn < 2; ++jn) {
      float v = treeP<8>(acc[jn]);
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) v = f_add(v, __shfl_xor_sync(0xffffffffu, v, o));
      y[jn] = v;
    }
    if (q == 0) {
      const float t = __ldg(tw + row);
      part[(size_t)row * D2 + n0] = f_mul(y[0], t);
      part[(size_t)row * D2 + n0 + 1] = f_mul(y[1], t);
    }
  }
}

// rT[k][m] = routed latent (bf16), m < MBP (zero beyond M)
template <int MBP>
__global__ void __launch_bounds__(256)
latent_kernel(const float* __restrict__ part, __nv_bfloat16* __restrict__ rT, int M, float scale) {
  const int e = blockIdx.x * 256 + threadIdx.x;
  if (e < D2 * MBP) {
    const int k = e / MBP, m = e % MBP;
    rT[e] = __float2bfloat16_rn(m < M ? latent_elem<true>(part, m, k, scale) : 0.0f);
  }
}

}  // namespace

// The small path for 3..32 tokens (INT8 per-row dense weights; INT8 per-row or INT4 affine routed experts): route, up
// (shared + routed), down, latent, exit; the intermediates are the caller's (for tests). x: fp32 [M, >= D2] (row stride sx), logits / logits_lo: fp32
// [M, NEXP] (row stride sl; logits_lo may be undefined), hidden: bf16 [M, HID] (row stride shd).
void small_launch(torch::Tensor hidden, torch::Tensor x, torch::Tensor logits, c10::optional<torch::Tensor> logits_lo,
                  torch::Tensor bias, torch::Tensor w1, torch::Tensor w1s, torch::Tensor w2, torch::Tensor w2s,
                  torch::Tensor suw, torch::Tensor sus, torch::Tensor sdw, torch::Tensor sds, torch::Tensor fw,
                  torch::Tensor fs, torch::Tensor ids, torch::Tensor tw, torch::Tensor meta, torch::Tensor hT,
                  torch::Tensor h, torch::Tensor part, torch::Tensor sT, torch::Tensor rT, torch::Tensor out,
                  int64_t M, double scale) {
  TORCH_CHECK(M >= 3 && M <= 32);
  const int m = (int)M, mbp = m <= 8 ? 8 : m <= 16 ? 16 : 32;
  // routed experts: INT8 per row ([E, N, K] int8, fp32 [E, N]) or INT4 affine ([E, N, K / 2] uint8, int32 [E, N, K / 32])
  const bool int4 = w1.scalar_type() == torch::kUInt8;
  TORCH_CHECK(w1.size(0) == NEXP && w1.size(1) == N1 && w1.size(2) == (int4 ? D2 / 2 : D2) && w2.size(1) == D2 &&
              w2.size(2) == (int4 ? N1 / 2 : N1));
  TORCH_CHECK(!int4 || (w2.scalar_type() == torch::kUInt8 && w1s.scalar_type() == torch::kInt32 && w2s.scalar_type() == torch::kInt32 &&
                        w1s.size(2) == D2 / 32 && w2s.size(2) == N1 / 32));
  TORCH_CHECK(meta.numel() * 4 >= (int64_t)sizeof(RouteMeta) && hT.numel() >= HID * mbp && sT.numel() >= SW * mbp &&
              rT.numel() >= D2 * mbp);
  TORCH_CHECK(x.stride(1) == 1 && logits.stride(1) == 1 && hidden.stride(1) == 1 && x.stride(0) % 4 == 0 &&
              logits.stride(0) % 4 == 0);
  auto st = c10::cuda::getCurrentCUDAStream();
  const float* lo = nullptr;
  if (logits_lo.has_value() && logits_lo->defined()) {
    TORCH_CHECK(logits_lo->stride(0) == logits.stride(0) && logits_lo->stride(1) == 1);
    lo = logits_lo->data_ptr<float>();
  }
  auto bf = [](torch::Tensor t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
  auto* rm = reinterpret_cast<RouteMeta*>(meta.data_ptr<int>());
  const int maxe = std::min(NEXP, m * KTOP);
  const int tcta = 1 + 12;   // routing CTA + 12 for hT
  const int nsh = mbp == 32 ? SW / 8 : SW / 16;
  const float sc = (float)scale;
  const float* lg = logits.data_ptr<float>();
  const float* bi = bias.data_ptr<float>();
  const int8_t* f = fw.data_ptr<int8_t>();
  const int8_t* d = sdw.data_ptr<int8_t>();
#define OLALA_SMALL_PATH(MBP)                                                                                          \
  route_kernel<MBP><<<tcta, 256, 0, st>>>(lg, lo, logits.stride(0), bi, bf(hidden), hidden.stride(0), ids.data_ptr<int>(), \
                                          tw.data_ptr<float>(), rm, bf(hT), m);                                         \
  if (int4) {                                                                                                       \
    up_kernel<MBP><<<nsh, 128, 0, st>>>(bf(hT), suw.data_ptr<int8_t>(), sus.data_ptr<float>(), bf(sT), nullptr, 0,       \
                                        nullptr, nullptr, rm, bf(h), m, nsh);                                             \
    up_i4_kernel<<<maxe * (N1 / 32), 128, 0, st>>>(x.data_ptr<float>(), x.stride(0), w1.data_ptr<uint8_t>(),          \
                                                   w1s.data_ptr<int>(), rm, bf(h), 0);                                  \
    down_i4_kernel<<<maxe * (D2 / 16), 128, 0, st>>>(bf(h), w2.data_ptr<uint8_t>(), w2s.data_ptr<int>(),              \
                                                     tw.data_ptr<float>(), rm, part.data_ptr<float>());                  \
  } else {                                                                                                          \
    up_kernel<MBP><<<nsh + maxe * (N1 / 32), 128, 0, st>>>(bf(hT), suw.data_ptr<int8_t>(), sus.data_ptr<float>(), bf(sT), \
        x.data_ptr<float>(), x.stride(0), w1.data_ptr<int8_t>(), w1s.data_ptr<float>(), rm, bf(h), m, nsh);               \
    down_kernel<<<maxe * (D2 / 16), 128, 0, st>>>(bf(h), w2.data_ptr<int8_t>(), w2s.data_ptr<float>(),                 \
                                                  tw.data_ptr<float>(), rm, part.data_ptr<float>());                    \
  }                                                                                                                 \
  latent_kernel<MBP><<<(D2 * MBP + 255) / 256, 256, 0, st>>>(part.data_ptr<float>(), bf(rT), m, sc);
  if (mbp == 8) {
    OLALA_SMALL_PATH(8)
    exit_b_kernel<8, 2, 8, 4><<<HID / 16, 128, 0, st>>>(bf(rT), f, fs.data_ptr<float>(), bf(sT), d, sds.data_ptr<float>(),
                                                       bf(out), m);
  } else if (mbp == 16) {
    OLALA_SMALL_PATH(16)
    exit_b_kernel<16, 1, 16, 4><<<HID / 16, 128, 0, st>>>(bf(rT), f, fs.data_ptr<float>(), bf(sT), d, sds.data_ptr<float>(),
                                                         bf(out), m);
  } else {
    OLALA_SMALL_PATH(32)
    exit_b_kernel<32, 1, 16, 4><<<HID / 8, 128, 0, st>>>(bf(rT), f, fs.data_ptr<float>(), bf(sT), d, sds.data_ptr<float>(),
                                                        bf(out), m);
  }
#undef OLALA_SMALL_PATH
}

int64_t route_meta_ints() { return (sizeof(RouteMeta) + 3) / 4; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("small", &small_launch);
  m.def("route_meta_ints", &route_meta_ints);
}

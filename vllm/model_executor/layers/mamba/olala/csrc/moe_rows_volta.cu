// Weight-only latent MoE of Olala for 2..32 decode rows on Volta (V100), natively or through volta-forever, in three
// launches per layer (volta_cuda.moe_rows; the small path takes four):
//   entry    [fc1_latent ; shared up] INT8 rows and the router (bf16 hi + lo rows, summed exactly in fp32, two warps
//            per expert over halves of K) for all rows in one few-row GEMV launch. The router CTAs come first in the
//            grid and take a ticket: the last of them routes the rows (top-6 of sigmoid + correction bias with
//            latent_moe_int8._topk_slot's tie rule, one packed-key warp max per pick) and lists the pairs of every
//            selected expert in a fixed order while the INT8 CTAs are still streaming. The latent is stored once as
//            bf16 (the small path rounds it to bf16 on load).
//   experts  each selected expert in SL slices of its intermediate width: a CTA streams its slice of the up weights and
//            the matching columns of the down weights once for all the rows routed to the expert (every load issued
//            first), relu^2 in between, and writes one fp32 partial per (slice, pair). The shared expert's down
//            projection runs in the same launch on its own CTAs. One ticket per row: the CTA completing a row's
//            KTOP x SL partials sums them in a fixed order (pair slot, then slice) into the routed latent (bf16, routed
//            scale applied).
//   exit     fc2_latent over the routed latent + the shared expert's output, bf16.
// The few-row GEMVs run branch-free over the MB rows (rows >= M read row M - 1; their sums are dropped at the store),
// so the per-row FMA chains interleave. Routed experts INT8 per row (Q8) or INT4 affine (Q4A2, dequantized and rounded
// to bf16 with integer ops). Summation orders differ from the small path (rounding-level outputs); every result is
// deterministic (fixed orders; tickets only choose which CTA performs a sum). The counters are reset by the CTA that
// completes them. ROWS_CHECK=1 builds bounds-check every data-dependent
// index (a violation is recorded in the counters' err words and the index replaced by 0). `which` (1 entry, 2 experts,
// 4 exit) and the flags (2 / 4: timing variants without tickets / routing) serve the development measurements.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int NEXP = 256;   // routed experts
constexpr int N1 = 768;     // routed intermediate width
constexpr int KTOP = 6;     // experts per row
constexpr int D1 = 384;     // latent width (fc1 rows, routed experts' K, fc2's K)
constexpr int SW = 1536;    // shared intermediate width
constexpr int HID = 1536;   // hidden width
constexpr int MAXM = 32;    // rows
constexpr int SL = 8;       // slices of an expert's intermediate width
constexpr int NS = N1 / SL; // 96 intermediate rows per slice

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
// byte i of (codes ^ 0x80808080) as 2^23 + q + 128, minus that offset: q exactly (gemv_rows_volta.cu's i8f)
__device__ __forceinline__ float i8f(uint32_t biased, int i) {
  return __uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7540u | i)) - 8388736.0f;
}
__device__ __forceinline__ void i8x16(const uint4 u, float* w) {
  const uint32_t ww[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = i8f(ww[i >> 2], i & 3);
}
__device__ __forceinline__ uint4 ld_stream(const void* p) { return __ldcs(reinterpret_cast<const uint4*>(p)); }
// the raw words through an empty asm: their conversion is then redone where it is used (per routed pair) instead of
// being hoisted out of the pair loop into hundreds of live registers
__device__ __forceinline__ uint4 opaque(uint4 v) {
  asm volatile("" : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w));
  return v;
}
__device__ __forceinline__ float relu2_bf16(float y) {  // bf16(max(bf16(y), 0)^2), the small path's epilogue
  const float h = fmaxf(bf16_round(y), 0.0f);
  return bf16_round(h * h);
}
__device__ __forceinline__ float f_ex2(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_div_full(float a, float b) {
  float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
// 8 bf16 of a uint4 as fp32
__device__ __forceinline__ void bf16x8(const uint4 u, float* a) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int t = 0; t < 4; ++t) { a[2 * t] = bf16_lo(w[t]); a[2 * t + 1] = bf16_hi(w[t]); }
}

struct Meta {                 // int32 words
  int nslots;                 // selected experts
  int pad[3];
  int4 slot[NEXP];            // slot j: {expert (increasing in j), first index in rows[], pairs, 0}
  int rows[MAXM * KTOP];      // pairs m * KTOP + k by slot, increasing within a slot
  float tw[MAXM * KTOP];      // normalized routing weight of pair m * KTOP + k
};

struct Counters {             // zeroed once; every counter is reset by the CTA that completes it
  unsigned int entry;
  unsigned int tok[MAXM];
  unsigned int err[4];        // ROWS_CHECK builds: first violation code, its index, violation count (host-reset)
};

#ifndef ROWS_CHECK
#define ROWS_CHECK 0
#endif
// A data-dependent index i that must lie in [0, n): in ROWS_CHECK builds a violation is recorded in ctr->err (the
// first code and value, a count) and the index replaced by 0, so it never reaches memory; otherwise i unchanged.
__device__ __forceinline__ int chk(Counters* ctr, int code, int i, int n) {
  if (ROWS_CHECK && (unsigned)i >= (unsigned)n) {
    if (atomicCAS(&ctr->err[0], 0u, (unsigned)code) == 0u) ctr->err[1] = (unsigned)i;
    atomicAdd(&ctr->err[2], 1u);
    return 0;
  }
  return i;
}

template <int MB>
__device__ __forceinline__ void x_step16(const __nv_bfloat16* __restrict__ x, long long ldx, int M, int k,
                                         float (&xv)[16], int m) {
  // 16 consecutive bf16 of row min(m, M - 1) at k (two 16-byte loads): no branch, the caller drops rows >= M
  const int mm = m < M ? m : M - 1;
  const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)mm * ldx + k);
  bf16x8(__ldg(p), xv);
  bf16x8(__ldg(p + 1), xv + 8);
}

template <int NV>
__device__ __forceinline__ void warp_allreduce(float (&v)[NV]) {
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    float a = v[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    v[i] = a;
  }
}

// -------------------------------------------------------------------------------------------------------------
// entry: 256-thread CTAs. [0, E_NRC) the router (a warp per expert: hi and lo rows, w = hi + lo exact in fp32), then
// [E_NRC, E_NRC + E_NIC) the INT8 rows (fc1_latent 384, shared up 1536; a warp per 2 rows). Lanes over K (16 k per
// lane and step for INT8, 8 for bf16), all weight vectors of a warp loaded first, activations through L1.
constexpr int E_THREADS = 256;
constexpr int E_WARPS = E_THREADS / 32;
constexpr int E_RW = 2;                              // router warps per expert (K halves)
constexpr int E_NRC = NEXP * E_RW / E_WARPS;         // router CTAs (64)
constexpr int E_NIC = (D1 + SW) / (2 * E_WARPS);     // INT8 CTAs (120)

// The key of expert idx with corrected score v: the largest key is the largest v, then the smallest index (the base's
// "v == max, smallest index" rule); NaN ranks as -inf; -0 as +0 (equal in the base's comparison). 0 = removed.
__device__ __forceinline__ unsigned long long route_key(float v, int idx) {
  unsigned u = __float_as_uint(v != v ? -INFINITY : v);
  if (u == 0x80000000u) u = 0u;
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(NEXP - 1 - idx);
}
__device__ __forceinline__ unsigned long long umax64(unsigned long long a, unsigned long long b) {
  return a > b ? a : b;
}

template <int MB>
__device__ __forceinline__ void route_rows(const float* __restrict__ lg, const float* __restrict__ bias, int M,
                                           Meta* __restrict__ meta, Counters* __restrict__ ctr,
                                           unsigned char* sid8, int* scnt, int* sstart) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float bs[8];
  {
    const float4 b0 = __ldg(reinterpret_cast<const float4*>(bias + 8 * lane));
    const float4 b1 = __ldg(reinterpret_cast<const float4*>(bias + 8 * lane) + 1);
    bs[0] = b0.x; bs[1] = b0.y; bs[2] = b0.z; bs[3] = b0.w; bs[4] = b1.x; bs[5] = b1.y; bs[6] = b1.z; bs[7] = b1.w;
  }
  for (int m = warp; m < M; m += E_WARPS) {
    float sc[8];
    unsigned long long key[8];
    {
      const float4* p = reinterpret_cast<const float4*>(lg + (size_t)m * NEXP + 8 * lane);
      const float4 a0 = __ldcg(p), a1 = __ldcg(p + 1);
      const float l8[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sc[j] = f_div_full(1.0f, f_ex2(l8[j] * __uint_as_float(0xBFB8AA3Bu)) + 1.0f);
        key[j] = route_key(sc[j] + bs[j], 8 * lane + j);
      }
    }
    unsigned long long best = key[0];
#pragma unroll
    for (int j = 1; j < 8; ++j) best = umax64(best, key[j]);
    float tot = 0.0f, wk[KTOP];
    int ek[KTOP];
#pragma unroll
    for (int kk = 0; kk < KTOP; ++kk) {
      unsigned long long w = best;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) w = umax64(w, __shfl_xor_sync(0xffffffffu, w, o));
      const int idx = NEXP - 1 - (int)(unsigned)(w & 0xffffffffu);
      const int owner = idx >> 3, jj = idx & 7;
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) if (j == jj) s = sc[j];
      s = __shfl_sync(0xffffffffu, s, owner);
      tot += s;
      wk[kk] = s;
      ek[kk] = idx;
      if (lane == owner) {
#pragma unroll
        for (int j = 0; j < 8; ++j) if (j == jj) key[j] = 0ull;
        best = key[0];
#pragma unroll
        for (int j = 1; j < 8; ++j) best = umax64(best, key[j]);
      }
    }
    if (lane < KTOP) {
      float wsel = wk[0];
      int esel = ek[0];
#pragma unroll
      for (int kk = 1; kk < KTOP; ++kk) if (lane == kk) { wsel = wk[kk]; esel = ek[kk]; }
      meta->tw[m * KTOP + lane] = f_div_full(wsel, tot);
      sid8[m * KTOP + lane] = (unsigned char)esel;
    }
  }
  __syncthreads();
  // the pairs of every expert in increasing order: counts (4 pairs per word compare), a warp scan, the lists
  const int MK = M * KTOP, nw = (MK + 3) >> 2;
  const unsigned tail = (MK & 3) ? (1u << (8 * (MK & 3))) - 1u : 0xffffffffu;
  const unsigned* w8 = reinterpret_cast<const unsigned*>(sid8);
  for (int e = tid; e < NEXP; e += E_THREADS) {
    const unsigned rep = 0x01010101u * (unsigned)e;
    int c = 0;
    for (int w = 0; w < nw; ++w) {
      unsigned eq = __vcmpeq4(w8[w], rep);
      if (w == nw - 1) eq &= tail;
      c += __popc(eq) >> 3;
    }
    scnt[e] = c;
  }
  __syncthreads();
  if (warp == 0) {
    int c8[8], cs = 0, fs = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { c8[j] = scnt[8 * lane + j]; cs += c8[j]; fs += c8[j] > 0; }
    int ci = cs, fi = fs;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, ci, o), b = __shfl_up_sync(0xffffffffu, fi, o);
      if (lane >= o) { ci += a; fi += b; }
    }
    int start = ci - cs, slot = fi - fs;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sstart[8 * lane + j] = start;
      if (c8[j] > 0) meta->slot[chk(ctr, 8, slot++, NEXP)] = make_int4(8 * lane + j, start, c8[j], 0);
      start += c8[j];
    }
    if (lane == 31) meta->nslots = fi;
  }
  __syncthreads();
  for (int e = tid; e < NEXP; e += E_THREADS) {
    if (scnt[e] == 0) continue;
    const unsigned rep = 0x01010101u * (unsigned)e;
    int at = sstart[e];
    for (int w = 0; w < nw; ++w) {
      unsigned eq = __vcmpeq4(w8[w], rep);
      if (w == nw - 1) eq &= tail;
      while (eq) {
        const int b = __ffs(eq) - 1;   // bit 8 i + 7 .. 8 i set for byte i
        meta->rows[chk(ctr, 1, at++, MK)] = 4 * w + (b >> 3);
        eq &= ~(0xffu << (b & ~7));
      }
    }
  }
}

template <int MB>
__global__ void __launch_bounds__(E_THREADS)
entry_kernel(const __nv_bfloat16* __restrict__ x, long long ldx, int M,
             const __nv_bfloat16* __restrict__ rhl,   // router [2 * NEXP, HID] bf16: hi rows, then lo rows
             const int8_t* __restrict__ f1w, const float* __restrict__ f1s,
             const int8_t* __restrict__ suw, const float* __restrict__ sus,
             const float* __restrict__ bias,
             __nv_bfloat16* __restrict__ xin,          // [MB, D1] bf16 latent (fc1)
             __nv_bfloat16* __restrict__ s,            // [MB, SW] bf16 relu^2(shared up)
             float* __restrict__ lg,                   // [MB, NEXP] fp32 router logits
             Meta* __restrict__ meta, Counters* __restrict__ ctr, int flags) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if ((int)blockIdx.x >= E_NRC) {
    // INT8 rows r0, r0 + 1 of [fc1 (384) ; shared up (1536)]: 3 steps of 16 k per lane (512 k)
    const int r0 = ((blockIdx.x - E_NRC) * E_WARPS + warp) * 2;
    const bool fc1 = r0 < D1;
    const int8_t* w0 = fc1 ? f1w + (size_t)r0 * HID : suw + (size_t)(r0 - D1) * HID;
    uint4 u[2][3];
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      u[0][st] = ld_stream(w0 + st * 512 + 16 * lane);
      u[1][st] = ld_stream(w0 + HID + st * 512 + 16 * lane);
    }
    float acc[2][MB];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      float w[2][16];
      i8x16(u[0][st], w[0]);
      i8x16(u[1][st], w[1]);
      const int k = st * 512 + 16 * lane;
#pragma unroll
      for (int m = 0; m < MB; ++m) {
        float xv[16];
        x_step16<MB>(x, ldx, M, k, xv, m);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          acc[0][m] = fmaf(w[0][i], xv[i], acc[0][m]);
          acc[1][m] = fmaf(w[1][i], xv[i], acc[1][m]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) warp_allreduce<MB>(acc[r]);
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int m = 0; m < MB; ++m) {
        if (lane == (r * MB + m) % 32 && m < M) {
          const int n = r0 + r;
          if (fc1) xin[(size_t)m * D1 + n] = __float2bfloat16_rn(acc[r][m] * __ldg(f1s + n));
          else s[(size_t)m * SW + (n - D1)] = __float2bfloat16_rn(relu2_bf16(acc[r][m] * __ldg(sus + (n - D1))));
        }
      }
    return;
  }
  // router: expert e = hi row e + lo row e over two warps (K halves of 768: 3 steps of 8 bf16 per lane), the halves
  // added through shared memory in a fixed order
  {
    constexpr int EPC = E_WARPS / E_RW;                // experts per CTA
    const int el = warp / E_RW, kh = warp % E_RW, e = blockIdx.x * EPC + el;
    const __nv_bfloat16* wh = rhl + (size_t)e * HID + kh * (HID / E_RW);
    const __nv_bfloat16* wl = rhl + (size_t)(NEXP + e) * HID + kh * (HID / E_RW);
    uint4 uh[3], ul[3];
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      uh[st] = ld_stream(wh + st * 256 + 8 * lane);
      ul[st] = ld_stream(wl + st * 256 + 8 * lane);
    }
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      const uint32_t h4[4] = {uh[st].x, uh[st].y, uh[st].z, uh[st].w}, l4[4] = {ul[st].x, ul[st].y, ul[st].z, ul[st].w};
      float w[8];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        w[2 * t] = bf16_lo(h4[t]) + bf16_lo(l4[t]);       // exact: 16 significant bits
        w[2 * t + 1] = bf16_hi(h4[t]) + bf16_hi(l4[t]);
      }
      const int k = kh * (HID / E_RW) + st * 256 + 8 * lane;
#pragma unroll
      for (int m = 0; m < MB; ++m) {
        float a[8];
        bf16x8(__ldg(reinterpret_cast<const uint4*>(x + (size_t)(m < M ? m : M - 1) * ldx + k)), a);
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[m] = fmaf(w[t], a[t], acc[m]);
      }
    }
    warp_allreduce<MB>(acc);
    __shared__ float halves[E_WARPS][MB];
#pragma unroll
    for (int m = 0; m < MB; ++m)
      if (lane == m % 32) halves[warp][m] = acc[m];
    __syncthreads();
    for (int i = threadIdx.x; i < EPC * MB; i += E_THREADS) {
      const int ee = i / MB, m = i % MB;
      if (m < M) lg[(size_t)m * NEXP + blockIdx.x * EPC + ee] = halves[ee * E_RW][m] + halves[ee * E_RW + 1][m];
    }
  }
  if (flags & 4) return;   // timing variant: no routing
  // ticket among the router CTAs: the last one routes (the INT8 CTAs may still be streaming)
  __shared__ bool last;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(&ctr->entry, 1u) == E_NRC - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  __shared__ __align__(16) unsigned char sid8[MAXM * KTOP];
  __shared__ int scnt[NEXP], sstart[NEXP];
  route_rows<MB>(lg, bias, M, meta, ctr, sid8, scnt, sstart);
  if (threadIdx.x == 0) ctr->entry = 0u;
}

// -------------------------------------------------------------------------------------------------------------
// experts: CTAs [0, NSDC) the shared down projection (8 warps x 2 rows), then (slot j, slice sl) CTAs.
constexpr int X_THREADS = 256;
constexpr int X_NSDC = SW / 16;   // shared-down CTAs: 16 rows each

// Routed slice of an INT8 per-row expert: hs[c][n] = relu2(s1[n] (W1[n] . x[m_c])) for the slice's NS rows, then
// part[sl][pair_c][d] = (W2[d][slice] . hs[c]) s2[d] tw[pair_c]. Up: rows p * 32 + (tid >> 3), k chunk q = tid & 7
// (k = st * 128 + 16 q, 8 threads per row); down: rows d = (tid >> 1) + 128 i, columns 48 * (tid & 1) .. + 47 of the
// slice (2 threads per row). All 18 weight vectors are loaded before any arithmetic.
template <int MB>
__device__ __forceinline__ void routed_i8(const __nv_bfloat16* __restrict__ xin, const int8_t* __restrict__ w1,
                                          const float* __restrict__ w1s, const int8_t* __restrict__ w2,
                                          const float* __restrict__ w2s, const Meta* __restrict__ meta, int e,
                                          int start, int cnt, int sl, float* __restrict__ part, float (*hs)[NS],
                                          Counters* ctr, int MK) {
  const int tid = threadIdx.x, q = tid & 7, rr = tid >> 3, hh = tid & 1;
  const int8_t* we1 = w1 + (size_t)e * N1 * D1;
  const int8_t* we2 = w2 + (size_t)e * D1 * N1;
  uint4 u1[3][3], u2[3][3];
#pragma unroll
  for (int p = 0; p < 3; ++p)
#pragma unroll
    for (int st = 0; st < 3; ++st)
      u1[p][st] = ld_stream(we1 + (size_t)(sl * NS + p * 32 + rr) * D1 + st * 128 + 16 * q);
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int c = 0; c < 3; ++c)
      u2[i][c] = ld_stream(we2 + (size_t)((tid >> 1) + 128 * i) * N1 + sl * NS + 48 * hh + 16 * c);
  for (int c = 0; c < cnt; ++c) {
    const int m = chk(ctr, 2, meta->rows[start + c], MK) / KTOP;
    float acc[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      float a[16];
      const uint4* px = reinterpret_cast<const uint4*>(xin + (size_t)m * D1 + st * 128 + 16 * q);
      bf16x8(__ldcg(px), a);
      bf16x8(__ldcg(px + 1), a + 8);
#pragma unroll
      for (int p = 0; p < 3; ++p) {
        float w[16];
        i8x16(opaque(u1[p][st]), w);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[p] = fmaf(w[i], a[i], acc[p]);
      }
    }
#pragma unroll
    for (int p = 0; p < 3; ++p) {
      float v = acc[p];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (q == 0) hs[c][p * 32 + rr] = relu2_bf16(v * __ldg(w1s + (size_t)e * N1 + sl * NS + p * 32 + rr));
    }
  }
  __syncthreads();
  for (int c = 0; c < cnt; ++c) {
    const int pr = chk(ctr, 3, meta->rows[start + c], MK);
    const float t = meta->tw[pr];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      float acc = 0.0f;
#pragma unroll
      for (int cc = 0; cc < 3; ++cc) {
        float w[16];
        i8x16(opaque(u2[i][cc]), w);
#pragma unroll
        for (int z = 0; z < 16; ++z) acc = fmaf(w[z], hs[c][48 * hh + 16 * cc + z], acc);
      }
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      if (hh == 0) {
        const int d = (tid >> 1) + 128 * i;
        part[((size_t)sl * MB * KTOP + pr) * D1 + d] = acc * __ldg(w2s + (size_t)e * D1 + d) * t;
      }
    }
  }
}

// INT4 affine (Q4A2: MLX q4, group 32; byte j of a group holds k = 2j (low nibble) and 2j + 1; int32 (bf16 scale |
// bf16 bias << 16) per group; w = bf16(q s + b); no row scale). Up: rows p * 64 + (tid >> 2) (p = 0, 1; the second
// pass 32 rows), groups st * 4 + (tid & 3) (4 threads per row); down: rows d = tid + 256 i (i = 0, 1; the second 128
// rows), the slice's 3 groups (48 bytes) per thread.
// bf16 round to nearest even of a finite fp32 with integer ops (the bf16 conversion is an sm_80 instruction, which
// volta-forever lowers to a longer sequence)
__device__ __forceinline__ float bf16_rne(float v) {
  const uint32_t u = __float_as_uint(v);
  return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}
__device__ __forceinline__ void i4x32(uint4 u, uint32_t sb, float* w) {
  // the 32 weights of a group in k order
  const float sc = __uint_as_float(sb << 16), bi = __uint_as_float(sb & 0xffff0000u);
  const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t byte = (ww[j >> 2] >> (8 * (j & 3))) & 0xffu;
    w[2 * j] = bf16_rne(fmaf((float)(byte & 15u), sc, bi));
    w[2 * j + 1] = bf16_rne(fmaf((float)(byte >> 4), sc, bi));
  }
}

template <int MB>
__device__ __forceinline__ void routed_i4(const __nv_bfloat16* __restrict__ xin, const uint8_t* __restrict__ w1,
                                          const int* __restrict__ w1sb, const uint8_t* __restrict__ w2,
                                          const int* __restrict__ w2sb, const Meta* __restrict__ meta, int e,
                                          int start, int cnt, int sl, float* __restrict__ part, float (*hs)[NS],
                                          Counters* ctr, int MK) {
  constexpr int KB1 = D1 / 2, G1 = D1 / 32, KB2 = N1 / 2, G2 = N1 / 32;
  const int tid = threadIdx.x, q = tid & 3, rr = tid >> 2;
  const bool up2 = tid < 128;     // second up pass: rows 64..95
  const bool dn2 = tid < 128;     // second down rows: 256..383
  uint4 u1[2][3], u2[2][3];
  uint32_t s1[2][3], s2[2][3];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int n = sl * NS + p * 64 + rr;
    if (p == 0 || up2) {
      const uint8_t* we = w1 + ((size_t)e * N1 + n) * KB1;
      const int* se = w1sb + ((size_t)e * N1 + n) * G1;
#pragma unroll
      for (int st = 0; st < 3; ++st) {
        u1[p][st] = ld_stream(we + st * 64 + 16 * q);
        s1[p][st] = (uint32_t)__ldg(se + st * 4 + q);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int d = tid + 256 * i;
    if (i == 0 || dn2) {
      const uint8_t* we = w2 + ((size_t)e * D1 + d) * KB2 + sl * (NS / 2);
      const int* se = w2sb + ((size_t)e * D1 + d) * G2 + sl * (NS / 32);
#pragma unroll
      for (int g = 0; g < 3; ++g) {
        u2[i][g] = ld_stream(we + 16 * g);
        s2[i][g] = (uint32_t)__ldg(se + g);
      }
    }
  }
  for (int c = 0; c < cnt; ++c) {
    const int m = chk(ctr, 2, meta->rows[start + c], MK) / KTOP;
    float acc[2] = {0.0f, 0.0f};
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      float a[32];
      const uint4* px = reinterpret_cast<const uint4*>(xin + (size_t)m * D1 + st * 128 + 32 * q);
#pragma unroll
      for (int t = 0; t < 4; ++t) bf16x8(__ldcg(px + t), a + 8 * t);
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        if (p == 0 || up2) {
          float w[32];
          i4x32(opaque(u1[p][st]), s1[p][st], w);
#pragma unroll
          for (int i = 0; i < 32; ++i) acc[p] = fmaf(w[i], a[i], acc[p]);
        }
      }
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      float v = acc[p];
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (q == 0 && (p == 0 || up2)) hs[c][p * 64 + rr] = relu2_bf16(v);
    }
  }
  __syncthreads();
  for (int c = 0; c < cnt; ++c) {
    const int pr = chk(ctr, 3, meta->rows[start + c], MK);
    const float t = meta->tw[pr];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (i == 0 || dn2) {
        float acc = 0.0f;
#pragma unroll
        for (int g = 0; g < 3; ++g) {
          float w[32];
          i4x32(opaque(u2[i][g]), s2[i][g], w);
#pragma unroll
          for (int z = 0; z < 32; ++z) acc = fmaf(w[z], hs[c][32 * g + z], acc);
        }
        part[((size_t)sl * MB * KTOP + pr) * D1 + tid + 256 * i] = acc * t;
      }
    }
  }
}

template <int MB, bool I4>
__global__ void __launch_bounds__(X_THREADS)
experts_kernel(const __nv_bfloat16* __restrict__ xin, const __nv_bfloat16* __restrict__ s,
               const void* __restrict__ w1, const void* __restrict__ w1s,
               const void* __restrict__ w2, const void* __restrict__ w2s,
               const int8_t* __restrict__ sdw, const float* __restrict__ sds,
               const Meta* __restrict__ meta, Counters* __restrict__ ctr,
               float* __restrict__ part,      // [SL, MB * KTOP, D1] slice partials (routing weight applied)
               __nv_bfloat16* __restrict__ lat, // [MB, D1] routed latent (scaled, bf16)
               float* __restrict__ sd,        // [MB, HID] shared down (fp32)
               int M, float scale, int flags) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if ((int)blockIdx.x < X_NSDC) {
    // shared down rows r0, r0 + 1 over K = SW (3 steps of 16 k per lane)
    const int r0 = (blockIdx.x * 8 + warp) * 2;
    const int8_t* w0 = sdw + (size_t)r0 * SW;
    uint4 u[2][3];
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      u[0][st] = ld_stream(w0 + st * 512 + 16 * lane);
      u[1][st] = ld_stream(w0 + SW + st * 512 + 16 * lane);
    }
    float acc[2][MB];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
      float w[2][16];
      i8x16(u[0][st], w[0]);
      i8x16(u[1][st], w[1]);
      const int k = st * 512 + 16 * lane;
#pragma unroll
      for (int m = 0; m < MB; ++m) {
        float xv[16];
        x_step16<MB>(s, SW, M, k, xv, m);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          acc[0][m] = fmaf(w[0][i], xv[i], acc[0][m]);
          acc[1][m] = fmaf(w[1][i], xv[i], acc[1][m]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) warp_allreduce<MB>(acc[r]);
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int m = 0; m < MB; ++m)
        if (lane == (r * MB + m) % 32 && m < M) sd[(size_t)m * HID + r0 + r] = acc[r][m] * __ldg(sds + r0 + r);
    return;
  }
  const int b = blockIdx.x - X_NSDC, j = b / SL, sl = b % SL;
  if (j >= meta->nslots) return;
  const int MK = M * KTOP;
  const int4 rec = meta->slot[j];
  const int e = chk(ctr, 4, rec.x, NEXP), start = chk(ctr, 5, rec.y, MK);
  int cnt = chk(ctr, 6, rec.z, MAXM + 1);
  if (ROWS_CHECK && start + cnt > MK) cnt = chk(ctr, 7, start + cnt, MK + 1);
  __shared__ float hs[MAXM][NS];
  if constexpr (I4)
    routed_i4<MB>(xin, static_cast<const uint8_t*>(w1), static_cast<const int*>(w1s), static_cast<const uint8_t*>(w2),
                  static_cast<const int*>(w2s), meta, e, start, cnt, sl, part, hs, ctr, MK);
  else
    routed_i8<MB>(xin, static_cast<const int8_t*>(w1), static_cast<const float*>(w1s), static_cast<const int8_t*>(w2),
                  static_cast<const float*>(w2s), meta, e, start, cnt, sl, part, hs, ctr, MK);
  if (flags & 2) return;   // timing variant: no tickets
  // ticket per row: the CTA completing a row's KTOP x SL partials forms its routed latent
  __shared__ int done[MAXM];
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    bool any = false;
    for (int c = 0; c < cnt; ++c) {
      const int m = chk(ctr, 9, meta->rows[start + c] / KTOP, M);
      done[c] = atomicAdd(&ctr->tok[m], 1u) == KTOP * SL - 1 ? m : -1;
      any |= done[c] >= 0;
    }
    if (any) __threadfence();
  }
  __syncthreads();
  for (int c = 0; c < cnt; ++c) {
    const int m = done[c];
    if (m < 0) continue;
    for (int d = tid; d < D1; d += X_THREADS) {
      float v = 0.0f;
#pragma unroll
      for (int k = 0; k < KTOP; ++k)
#pragma unroll
        for (int z = 0; z < SL; ++z) v += __ldcg(part + ((size_t)z * MB * KTOP + m * KTOP + k) * D1 + d);
      lat[(size_t)m * D1 + d] = __float2bfloat16_rn(v * scale);
    }
    if (tid == 0) ctr->tok[m] = 0u;
  }
}

// -------------------------------------------------------------------------------------------------------------
// exit: out[m, n] = bf16(fc2[n] . lat[m] * fs[n] + sd[m, n]); 8 warps x 2 rows per CTA, K = 384 = 3 x 32 lanes x 4
constexpr int Z_CTAS = HID / 16;

template <int MB>
__global__ void __launch_bounds__(256)
exit_kernel(const __nv_bfloat16* __restrict__ lat, const int8_t* __restrict__ fw, const float* __restrict__ fs,
            const float* __restrict__ sd, __nv_bfloat16* __restrict__ out, long long ldo, int M) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int r0 = (blockIdx.x * 8 + warp) * 2;

  uint32_t u[2][3];
#pragma unroll
  for (int st = 0; st < 3; ++st) {
    u[0][st] = __ldcs(reinterpret_cast<const unsigned int*>(fw + (size_t)r0 * D1 + st * 128 + 4 * lane));
    u[1][st] = __ldcs(reinterpret_cast<const unsigned int*>(fw + (size_t)(r0 + 1) * D1 + st * 128 + 4 * lane));
  }
  // the (row, token) outputs t = r MB + m this lane stores (t = lane, lane + 32), their shared-down values first
  constexpr int NT = (2 * MB + 31) / 32;
  float sdv[NT], fsv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int t = lane + 32 * i, r = t / MB, m = t % MB;
    const bool ok = t < 2 * MB && m < M;
    sdv[i] = ok ? __ldcg(sd + (size_t)m * HID + r0 + r) : 0.0f;
    fsv[i] = ok ? __ldg(fs + r0 + r) : 0.0f;
  }
  __shared__ __align__(16) __nv_bfloat16 ls[MB * D1];
  for (int i = threadIdx.x; i < MB * D1 / 8; i += 256) {
    const int m = (i * 8) / D1;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (m < M) v = __ldcg(reinterpret_cast<const uint4*>(lat) + i);
    reinterpret_cast<uint4*>(ls)[i] = v;
  }
  __syncthreads();
  float acc[2][MB];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
#pragma unroll
  for (int st = 0; st < 3; ++st) {
    float w[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t b = u[r][st] ^ 0x80808080u;
#pragma unroll
      for (int i = 0; i < 4; ++i) w[r][i] = i8f(b, i);
    }
    const int k = st * 128 + 4 * lane;
#pragma unroll
    for (int m = 0; m < MB; ++m) {
      const uint2 a = *reinterpret_cast<const uint2*>(ls + m * D1 + k);
      const float xv[4] = {bf16_lo(a.x), bf16_hi(a.x), bf16_lo(a.y), bf16_hi(a.y)};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[0][m] = fmaf(w[0][i], xv[i], acc[0][m]);
        acc[1][m] = fmaf(w[1][i], xv[i], acc[1][m]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) warp_allreduce<MB>(acc[r]);
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int t = lane + 32 * i, r = t / MB, m = t % MB;
    float v = 0.0f;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
#pragma unroll
      for (int mm = 0; mm < MB; ++mm)
        if (rr * MB + mm == t) v = acc[rr][mm];
    if (t < 2 * MB && m < M) out[(size_t)m * ldo + r0 + r] = __float2bfloat16_rn(v * fsv[i] + sdv[i]);
  }
}

// workspace (fp32 words): xin (bf16) [MB, D1] | lg [MB, NEXP] | sd [MB, HID] | part [SL, MB*K, D1] |
// s (bf16) [MB, SW] | lat (bf16) [MB, D1] | meta
struct Ws {
  __nv_bfloat16* xin; float* lg; float* sd; float* part; __nv_bfloat16* s; __nv_bfloat16* lat; Meta* meta;
};
Ws carve(float* p, int mb) {
  Ws w;
  w.xin = reinterpret_cast<__nv_bfloat16*>(p); p += mb * D1 / 2;
  w.lg = p; p += mb * NEXP;
  w.sd = p; p += mb * HID;
  w.part = p; p += SL * mb * KTOP * D1;
  w.s = reinterpret_cast<__nv_bfloat16*>(p); p += mb * SW / 2;
  w.lat = reinterpret_cast<__nv_bfloat16*>(p); p += mb * D1 / 2;
  w.meta = reinterpret_cast<Meta*>(p);
  return w;
}

template <int MB>
void launch(const torch::Tensor& x, const torch::Tensor& rhl, const torch::Tensor& f1w, const torch::Tensor& f1s,
            const torch::Tensor& suw, const torch::Tensor& sus, const torch::Tensor& bias, const torch::Tensor& w1,
            const torch::Tensor& w1s, const torch::Tensor& w2, const torch::Tensor& w2s, const torch::Tensor& sdw,
            const torch::Tensor& sds, const torch::Tensor& fw, const torch::Tensor& fs, const torch::Tensor& ws,
            const torch::Tensor& ctr, const torch::Tensor& out, int M, float scale, int which, int flags) {
  auto st = c10::cuda::getCurrentCUDAStream();
  const Ws w = carve(ws.data_ptr<float>(), MB);
  auto* c = reinterpret_cast<Counters*>(ctr.data_ptr<int>());
  auto bf = [](const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  if (which & 1) entry_kernel<MB><<<E_NRC + E_NIC, E_THREADS, 0, st>>>(bf(x), x.stride(0), M, bf(rhl),
      f1w.data_ptr<int8_t>(), f1s.data_ptr<float>(), suw.data_ptr<int8_t>(), sus.data_ptr<float>(),
      bias.data_ptr<float>(), w.xin, w.s, w.lg, w.meta, c, flags);
  const int maxe = std::min(NEXP, M * KTOP);
  const bool i4 = w1.scalar_type() == torch::kUInt8;
  auto* ek = i4 ? experts_kernel<MB, true> : experts_kernel<MB, false>;
  if (which & 2) ek<<<X_NSDC + maxe * SL, X_THREADS, 0, st>>>(w.xin, w.s, w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(),
      w2s.data_ptr(), sdw.data_ptr<int8_t>(), sds.data_ptr<float>(), w.meta, c, w.part, w.lat, w.sd, M, scale, flags);
  if (which & 4) exit_kernel<MB><<<Z_CTAS, 256, 0, st>>>(w.lat, fw.data_ptr<int8_t>(), fs.data_ptr<float>(), w.sd,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), M);
}

}  // namespace

int64_t workspace_floats(int64_t mb) {
  return mb * D1 / 2 + mb * NEXP + mb * HID + SL * mb * KTOP * D1 + mb * SW / 2 + mb * D1 / 2 + (sizeof(Meta) + 3) / 4;
}
int64_t counter_ints() { return (sizeof(Counters) + 3) / 4; }
int64_t err_offset() { return offsetof(Counters, err) / 4; }
int64_t check_build() { return ROWS_CHECK; }
// word offsets (from the workspace base, for MB) of lg and meta, and of meta's fields: nslots, slot (int4), rows, tw
std::vector<int64_t> layout(int64_t mb) {
  const int64_t lg = mb * D1 / 2, meta = workspace_floats(mb) - (sizeof(Meta) + 3) / 4;
  return {lg, meta, (int64_t)offsetof(Meta, nslots) / 4, (int64_t)offsetof(Meta, slot) / 4,
          (int64_t)offsetof(Meta, rows) / 4, (int64_t)offsetof(Meta, tw) / 4};
}

template <typename K>
std::tuple<std::string, int64_t, int64_t, int64_t, int64_t> info_of(const char* name, K kernel, int threads) {
  cudaFuncAttributes a{};
  TORCH_CHECK(cudaFuncGetAttributes(&a, kernel) == cudaSuccess, name);
  int occ = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, threads, 0) == cudaSuccess, name);
  return {name, a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, occ};
}

std::vector<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t>> kernel_info() {
  std::vector<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t>> v;
#define OLALA_INFO(MB)                                                                     \
  v.push_back(info_of("entry" #MB, entry_kernel<MB>, E_THREADS));                          \
  v.push_back(info_of("experts_i8_" #MB, experts_kernel<MB, false>, X_THREADS));           \
  v.push_back(info_of("experts_i4_" #MB, experts_kernel<MB, true>, X_THREADS));            \
  v.push_back(info_of("exit" #MB, exit_kernel<MB>, 256));
  OLALA_INFO(2) OLALA_INFO(4) OLALA_INFO(8) OLALA_INFO(16) OLALA_INFO(32)
#undef OLALA_INFO
  return v;
}

void moe_rows(torch::Tensor x, torch::Tensor rhl, torch::Tensor f1w, torch::Tensor f1s, torch::Tensor suw,
              torch::Tensor sus, torch::Tensor bias, torch::Tensor w1, torch::Tensor w1s, torch::Tensor w2,
              torch::Tensor w2s, torch::Tensor sdw, torch::Tensor sds, torch::Tensor fw, torch::Tensor fs,
              torch::Tensor ws, torch::Tensor ctr, torch::Tensor out, int64_t M, double scale, int64_t which,
              int64_t flags) {
  TORCH_CHECK(M >= 1 && M <= MAXM);
  const int m = (int)M;
  const float sc = (float)scale;
#define OLALA_ROWS(MB) \
  launch<MB>(x, rhl, f1w, f1s, suw, sus, bias, w1, w1s, w2, w2s, sdw, sds, fw, fs, ws, ctr, out, m, sc, (int)which, \
             (int)flags)
  if (m <= 2) OLALA_ROWS(2);
  else if (m <= 4) OLALA_ROWS(4);
  else if (m <= 8) OLALA_ROWS(8);
  else if (m <= 16) OLALA_ROWS(16);
  else OLALA_ROWS(32);
#undef OLALA_ROWS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("moe_rows", &moe_rows);
  mod.def("workspace_floats", &workspace_floats);
  mod.def("counter_ints", &counter_ints);
  mod.def("err_offset", &err_offset);
  mod.def("check_build", &check_build);
  mod.def("layout", &layout);
  mod.def("kernel_info", &kernel_info);
}

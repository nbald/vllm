// Olala Mamba-3 MIMO prefill: chunk-local states, Volta (V100) edition: plain fp32 FMA, no tensor cores, no
// ldmatrix. Same arguments, math and bf16 roundings as mamba3_chunk_state_sm80.cu:
//   Sloc[c, h][n][p] = sum_j khat_j[n] * PsiV_j[p] * ts_j * exp(dtot[c, h] - dacs_j)     (fp32, (NC, H, N, P))
// over the rows j = token * R + rank of chunk c. Built JIT by mimo_ssd.py on the SM70 family (see
// mamba3_scan_prefill_volta.cu for why: volta-forever emulates bf16 mma.sync with scalar code).
//
// The operands are rounded to bf16 exactly as in the sm80 kernel and kept in shared memory as fp32; each state
// element is one thread's chain of fmaf over the rows in increasing order, which is how volta-forever evaluates the
// sm80 kernel's m16n8k16 chain: under volta-forever both kernels return the same bits.
//
// One CTA (8 warps, ~28 KB of shared memory: several CTAs per SM) per (chunk, head), 8-token sub-blocks (32 rows).
// Thread tid prepares rank (tid >> 3) & 3 of token tid >> 5: the khat columns 32 g + 4 j .. + 3 (g = 0..3, j = tid & 7:
// every thread rotates 4 of the (n, n + 64) pairs) and the PsiV * w columns 8 j .. + 7, computing the sm80 kernel's
// bf16 instructions (which volta-forever runs as sequences of 10-35 instructions) directly (see bf16r); it
// accumulates n rows 8 (tid >> 4) .. + 7 x p cols 4 (tid & 15) .. + 3 of the state.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32;
constexpr int CC = 8, ROWS = CC * R;  // 32 rows per sub-block
constexpr int THREADS = 256;
constexpr int LDA = N + 4;  // fp32 khat rows: 4-word bank shift per row

struct Smem {
  float a[ROWS * LDA];  // khat [row][n]
  float b[ROWS * P];    // PsiV * w [row][p]
  float kb[R * N];      // bias rounded to bf16
  float psi[R * P];
};

struct Raw {     // one thread's global inputs of a sub-block
  uint2 k[4];    // rank (tid >> 3) & 3, columns 32 g + 4 j .. 32 g + 4 j + 3
  float4 th;     // thl of columns 4 j .. 4 j + 3
  uint4 v;       // V[token][8 j .. 8 j + 7]
  float ts, dacs;
};

// 8 bf16 of a strided row: one 16-byte load when aligned, else the two aligned 16-byte pieces around it and a
// funnel shift (the in_proj output interleaves heads with an odd stride; the alignment is uniform per CTA)
__device__ __forceinline__ uint4 ld8(const __nv_bfloat16* __restrict__ p, bool vec) {
  if (vec) return __ldg(reinterpret_cast<const uint4*>(p));
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const unsigned off = (unsigned)(a & 15);  // even: 0 .. 14 bytes
  const uint4* base = reinterpret_cast<const uint4*>(a - off);
  const uint4 u0 = __ldg(base);
  if (off == 0) return u0;
  const uint4 u1 = __ldg(base + 1);
  const unsigned w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
  const unsigned wi = off >> 2, sh = (off & 3) * 8;
  unsigned o[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const unsigned lo = wi == 0 ? w[e] : wi == 1 ? w[e + 1] : wi == 2 ? w[e + 2] : w[e + 3];
    const unsigned hi = wi == 0 ? w[e + 1] : wi == 1 ? w[e + 2] : wi == 2 ? w[e + 3] : w[e + 4];
    o[e] = __funnelshift_r(lo, hi, sh);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// The sm80 kernel's bf16 roundings, computed in fp32 (as in mamba3_scan_prefill_volta.cu). bf16r(x): x rounded to
// bf16 (to nearest, ties to even; NaN to 0x7fff), kept as fp32: cvt.rn.bf16.f32 as volta-forever runs it and the
// hardware does. A bf16 sum (__hadd2) is the fp32 sum rounded once to bf16 (fp32 has the 2 x 8 + 2 bits that make the
// double rounding innocuous), computed without flushing subnormals as the bf16 units do.
__device__ __forceinline__ float bf16r(float x) {
  const unsigned b = __float_as_uint(x);
  const unsigned r = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
  return __uint_as_float((b & 0x7fffffffu) > 0x7f800000u ? 0x7fff0000u : r);
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float bf_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ void load_raw(Raw& rw, int tid, int t0, int nv, int h, int T, int H, long ldv, long lhv,
                                         bool vec,
                                         const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
                                         const float* __restrict__ thl, const float* __restrict__ dacs,
                                         const float* __restrict__ ts) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7;
  const bool ok = tok < nv;
  const long t = t0 + tok;
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  const __nv_bfloat16* kr = K + (t * R + r) * N + 4 * j;
#pragma unroll
  for (int g = 0; g < 4; ++g) rw.k[g] = ok ? __ldg(reinterpret_cast<const uint2*>(kr + 32 * g)) : make_uint2(0u, 0u);
  rw.th = ok ? __ldg(reinterpret_cast<const float4*>(thl + (t * H + h) * NQ + 4 * j)) : make_float4(0.f, 0.f, 0.f, 0.f);
  rw.v = ok ? ld8(V + t * ldv + h * lhv + 8 * j, vec) : z;
  rw.ts = ok ? __ldg(ts + (long)h * T + t) : 0.f;
  rw.dacs = ok ? __ldg(dacs + (long)h * T + t) : 0.f;
}

// the sm80 kernel's operand preparation, value for value (bias in bf16, rotation in fp32, PsiV * w in fp32 rounded once
// to bf16); fp32 row-major destinations
__device__ __forceinline__ void store_operands(Smem& S, const Raw& rw, int tid, float dl) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7, row = tok * R + r;
  {
    float kv[4][4];  // column 32 g + 4 j + e
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const float4 bk = *reinterpret_cast<const float4*>(&S.kb[r * N + 32 * g + 4 * j]);
      const float bka[4] = {bk.x, bk.y, bk.z, bk.w};
      const unsigned wk[2] = {rw.k[g].x, rw.k[g].y};
#pragma unroll
      for (int e = 0; e < 4; ++e) kv[g][e] = bf16r(add_rn((e & 1) ? bf_hi(wk[e >> 1]) : bf_lo(wk[e >> 1]), bka[e]));
    }
    const float th[4] = {rw.th.x, rw.th.y, rw.th.z, rw.th.w};
    // rotation of the pairs (4 j + e, 64 + 4 j + e) in fp32, with the products the sm80 kernel's machine code fuses
    // (explicit forms, so that the compiler cannot pick others)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float cs, sn;
      const float a = __fmaf_rn(floorf(th[e] * 0.15915494309189535f), -6.283185307179586f, th[e]);
      __sincosf(a, &sn, &cs);
      const float y0 = kv[0][e], y1 = kv[2][e];
      kv[0][e] = bf16r(__fmaf_rn(cs, y0, -__fmul_rn(sn, y1)));
      kv[2][e] = bf16r(__fmaf_rn(cs, y1, __fmul_rn(sn, y0)));
    }
#pragma unroll
    for (int g = 0; g < 4; ++g)
      *reinterpret_cast<float4*>(&S.a[row * LDA + 32 * g + 4 * j]) = make_float4(kv[g][0], kv[g][1], kv[g][2], kv[g][3]);
  }
  {
    const float w = rw.ts * __expf(dl - rw.dacs);  // ts = 0 past the chunk end
    const float4* pp = reinterpret_cast<const float4*>(&S.psi[r * P + 8 * j]);
    const float4 pa = pp[0], pb = pp[1];
    const float ps[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
    const unsigned wv[4] = {rw.v.x, rw.v.y, rw.v.z, rw.v.w};
    float pv[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      pv[2 * e] = bf16r(bf_lo(wv[e]) * ps[2 * e] * w);
      pv[2 * e + 1] = bf16r(bf_hi(wv[e]) * ps[2 * e + 1] * w);
    }
    float4* dst = reinterpret_cast<float4*>(&S.b[row * P + 8 * j]);
    dst[0] = make_float4(pv[0], pv[1], pv[2], pv[3]);
    dst[1] = make_float4(pv[4], pv[5], pv[6], pv[7]);
  }
}

__global__ void __launch_bounds__(THREADS, 2) chunk_state_volta_kernel(
    const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, const float* __restrict__ kbias,
    const float* __restrict__ psi, const float* __restrict__ thl, const float* __restrict__ dacs,
    const float* __restrict__ ts, const float* __restrict__ dtot, const int* __restrict__ ch_start,
    const int* __restrict__ ch_len, float* __restrict__ sloc, int T, int H, long ldv, long lhv) {
  __shared__ __align__(16) Smem S;
  const int tid = threadIdx.x, item = blockIdx.x, c = item / H, h = item % H;
  const int cs = ch_start[c], cl = ch_len[c], nsb = (cl + CC - 1) / CC;
  const float dl = dtot[item];
  const bool vec = ((ldv | lhv) & 7) == 0;
  for (int i = tid; i < R * N; i += THREADS) S.kb[i] = bf16r(kbias[(long)h * R * N + i]);
  for (int i = tid; i < R * P; i += THREADS) S.psi[i] = psi[(long)h * R * P + i];
  Raw rw;
  load_raw(rw, tid, cs, min(CC, cl), h, T, H, ldv, lhv, vec, K, V, thl, dacs, ts);

  const int n0 = 8 * (tid >> 4), p0 = 4 * (tid & 15);
  float acc[8][4];
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = 0.f;

  for (int it = 0; it < nsb; ++it) {
    __syncthreads();
    store_operands(S, rw, tid, dl);
    __syncthreads();
    if (it + 1 < nsb)
      load_raw(rw, tid, cs + (it + 1) * CC, min(CC, cl - (it + 1) * CC), h, T, H, ldv, lhv, vec, K, V, thl, dacs, ts);
#pragma unroll 4
    for (int jr = 0; jr < ROWS; ++jr) {
      const float4 ka = *reinterpret_cast<const float4*>(&S.a[jr * LDA + n0]);
      const float4 kc = *reinterpret_cast<const float4*>(&S.a[jr * LDA + n0 + 4]);
      const float4 pp = *reinterpret_cast<const float4*>(&S.b[jr * P + p0]);
      const float kk[8] = {ka.x, ka.y, ka.z, ka.w, kc.x, kc.y, kc.z, kc.w};
      const float pb[4] = {pp.x, pp.y, pp.z, pp.w};
#pragma unroll
      for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(kk[a], pb[b], acc[a][b]);
    }
  }
  float* dst = sloc + (long)item * N * P;
#pragma unroll
  for (int a = 0; a < 8; ++a)
    *reinterpret_cast<float4*>(&dst[(n0 + a) * P + p0]) = make_float4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]);
}
}  // namespace

void mamba3_chunk_state_volta(torch::Tensor K, torch::Tensor V, torch::Tensor kbias, torch::Tensor psi,
                              torch::Tensor thl, torch::Tensor dacs, torch::Tensor ts, torch::Tensor dtot,
                              torch::Tensor ch_start, torch::Tensor ch_len, torch::Tensor sloc, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && K.size(-1) == N && K.size(-2) == R && thl.size(-1) == NQ);
  TORCH_CHECK(K.scalar_type() == at::kBFloat16 && V.scalar_type() == at::kBFloat16);
  TORCH_CHECK(K.is_contiguous() && V.stride(2) == 1 && thl.is_contiguous() && sloc.is_contiguous());
  TORCH_CHECK(kbias.is_contiguous() && psi.is_contiguous() && dacs.is_contiguous() && ts.is_contiguous());
  if (NC * H == 0) return;
  chunk_state_volta_kernel<<<(int)(NC * H), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
      kbias.data_ptr<float>(), psi.data_ptr<float>(), thl.data_ptr<float>(), dacs.data_ptr<float>(),
      ts.data_ptr<float>(), dtot.data_ptr<float>(), ch_start.data_ptr<int>(), ch_len.data_ptr<int>(),
      sloc.data_ptr<float>(), T, H, V.stride(0), V.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("chunk_state", &mamba3_chunk_state_volta); }

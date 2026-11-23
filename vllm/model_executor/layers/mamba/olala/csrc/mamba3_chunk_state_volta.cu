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
// One CTA (8 warps, ~27 KB of shared memory: several CTAs per SM) per (chunk, head), 8-token sub-blocks (32 rows).
// Thread tid prepares rank (tid >> 3) & 3 of token tid >> 5: 16 khat columns (8 j, 64 + 8 j; j = tid & 7) and 8
// PsiV * w columns (8 j); it accumulates n rows 8 (tid >> 4) .. + 7 x p cols 4 (tid & 15) .. + 3 of the state.
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
  __nv_bfloat16 kb[R * N];
  float psi[R * P];
};

union U8 {
  uint4 u;
  __nv_bfloat162 h2[4];
};

struct Raw {     // one thread's global inputs of a sub-block
  uint4 k[2];    // rank (tid >> 3) & 3, columns 8 j and 64 + 8 j
  float4 th[2];  // thl of columns 8 j .. 8 j + 7 (rotated groups, j < 4)
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

__device__ __forceinline__ void st8f(float* dst, const U8& u) {
  float2 f[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) f[e] = __bfloat1622float2(u.h2[e]);
  reinterpret_cast<float4*>(dst)[0] = make_float4(f[0].x, f[0].y, f[1].x, f[1].y);
  reinterpret_cast<float4*>(dst)[1] = make_float4(f[2].x, f[2].y, f[3].x, f[3].y);
}

__device__ __forceinline__ void load_raw(Raw& rw, int tid, int t0, int nv, int h, int T, int H, long ldv, long lhv,
                                         bool vec,
                                         const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
                                         const float* __restrict__ thl, const float* __restrict__ dacs,
                                         const float* __restrict__ ts) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7;
  const bool ok = tok < nv;
  const long t = t0 + tok;
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  const __nv_bfloat16* kr = K + (t * R + r) * N + 8 * j;
  rw.k[0] = ok ? __ldg(reinterpret_cast<const uint4*>(kr)) : z;
  rw.k[1] = ok ? __ldg(reinterpret_cast<const uint4*>(kr + 64)) : z;
  if (j < 4) {
    const float4* tp = reinterpret_cast<const float4*>(thl + (t * H + h) * NQ + 8 * j);
    const float4 zf = make_float4(0.f, 0.f, 0.f, 0.f);
    rw.th[0] = ok ? __ldg(tp) : zf;
    rw.th[1] = ok ? __ldg(tp + 1) : zf;
  }
  rw.v = ok ? ld8(V + t * ldv + h * lhv + 8 * j, vec) : z;
  rw.ts = ok ? __ldg(ts + (long)h * T + t) : 0.f;
  rw.dacs = ok ? __ldg(dacs + (long)h * T + t) : 0.f;
}

// the sm80 kernel's operand preparation, value for value (bias in bf16, rotation in fp32, PsiV * w in fp32
// rounded once to bf16); one rank per thread instead of two, fp32 row-major destinations
__device__ __forceinline__ void store_operands(Smem& S, const Raw& rw, int tid, float dl) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7, row = tok * R + r;
  const bool rot = j < 4;
  float cs[8], sn[8];
  if (rot) {
    const float th[8] = {rw.th[0].x, rw.th[0].y, rw.th[0].z, rw.th[0].w,
                         rw.th[1].x, rw.th[1].y, rw.th[1].z, rw.th[1].w};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float a = th[e] - 6.283185307179586f * floorf(th[e] * 0.15915494309189535f);
      __sincosf(a, &sn[e], &cs[e]);
    }
  }
  {
    U8 k0, k1, b0, b1;
    k0.u = rw.k[0];
    k1.u = rw.k[1];
    b0.u = *reinterpret_cast<const uint4*>(&S.kb[r * N + 8 * j]);
    b1.u = *reinterpret_cast<const uint4*>(&S.kb[r * N + 64 + 8 * j]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      k0.h2[e] = __hadd2(k0.h2[e], b0.h2[e]);
      k1.h2[e] = __hadd2(k1.h2[e], b1.h2[e]);
    }
    if (rot) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 y0 = __bfloat1622float2(k0.h2[e]), y1 = __bfloat1622float2(k1.h2[e]);
        const float ca = cs[2 * e], cb = cs[2 * e + 1], sa = sn[2 * e], sb = sn[2 * e + 1];
        k0.h2[e] = __floats2bfloat162_rn(ca * y0.x - sa * y1.x, cb * y0.y - sb * y1.y);
        k1.h2[e] = __floats2bfloat162_rn(sa * y0.x + ca * y1.x, sb * y0.y + cb * y1.y);
      }
    }
    st8f(&S.a[row * LDA + 8 * j], k0);
    st8f(&S.a[row * LDA + 64 + 8 * j], k1);
  }
  {
    const float w = rw.ts * __expf(dl - rw.dacs);  // ts = 0 past the chunk end
    const float4* pp = reinterpret_cast<const float4*>(&S.psi[r * P + 8 * j]);
    const float4 pa = pp[0], pb = pp[1];
    const float ps[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
    U8 v;
    v.u = rw.v;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 x = __bfloat1622float2(v.h2[e]);
      v.h2[e] = __floats2bfloat162_rn(x.x * ps[2 * e] * w, x.y * ps[2 * e + 1] * w);
    }
    st8f(&S.b[row * P + 8 * j], v);
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
  for (int i = tid; i < R * N; i += THREADS) S.kb[i] = __float2bfloat16(kbias[(long)h * R * N + i]);
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

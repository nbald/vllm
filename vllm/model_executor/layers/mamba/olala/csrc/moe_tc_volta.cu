// Routed experts of the weight-only latent MoE on Volta's fp16 tensor cores (V100, natively or through
// volta-forever, which passes the fp16 mma.m8n8k4 through to Volta's HMMA unchanged).
//
// Grouped GEMM over moe_align_block_size's sorted entries (fused_moe's layout): one CTA per row block of
// BM = 8 * MT entries of one expert and BN = 128 output features; the rows' terms go through shared memory in
// slabs of 128 k, the weights stream from DRAM, the next 32 k in flight during the current products. The experts stay INT8 (one fp32 scale per
// output row) or INT4 affine (codes 0..15, bf16 scale and bias per group of 32 along K) in memory; a lane
// converts its weight row to fp16 in registers (exact: both code ranges are fp16 integers). The activation
// rows come split into T exact fp16 terms (row r = sum_t h[t][r] * 2^-sh[t][r], volta_tc._tc_split_kernel),
// so every product is exact and the sums are the tensor cores' fp32 sums:
//   INT8:  y[r, n] = scale[n] * sum_t 2^-sh[t][r] * sum_k h_t[r, k] q[n, k]
//   INT4:  y[r, n] = sum_t 2^-sh[t][r] * sum_g s_g[n] * (sum_{k in g} h_t[r, k] q[n, k]) + sum_g b_g[n] X_g[r]
// with X_g[r] = sum_{k in g} x[r, k] (fp32, volta_tc._xsum_kernel), i.e. x . w for w = s_g q + b_g, without the bf16
// rounding of the dequantized weight the emulated kernel applies. Rows that the T terms do not cover take an
// fp32 FMA fallback in the same CTA (the same decomposition), so each row's path depends on that row only.
// Up: out = bf16(relu(bf16(y))^2) [entry, n] (as _wq_moe_sorted_kernel); down: out = y * w_route[entry] (fp32).
//
// mma.m8n8k4.row.col.f32 fragments (one independent 8x8x4 product per octet q: lanes 4q..4q+3, 16+4q..16+4q+3;
// t = (lane & 3) + 4 * (lane >> 4 & 1)): A row t (k 0..3, two half2), B column t (k 0..3, two half2), C value v
// at row (lane & 1) + 4 * (lane >> 4 & 1) + 2 * (v >> 1 & 1), column 2 * (lane >> 1 & 1) + (v & 1) + 4 * (v >> 2).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int THREADS = 128, BN = 128, KC = 32, KS = 128, KSP = KS + 8, MAXG = 32;

__device__ __forceinline__ void mma884(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t hsub2u(uint32_t v, float c) {
  __half2 h = *reinterpret_cast<__half2*>(&v);
  h = __hsub2(h, __float2half2_rn(c));
  return *reinterpret_cast<uint32_t*>(&h);
}

// 4 INT8 codes -> 2 half2 (k order), exact: byte q + 128 under 0x64 is the half 1024 + q + 128.
__device__ __forceinline__ void i8_h2(uint32_t q, uint32_t& lo, uint32_t& hi) {
  const uint32_t b = q ^ 0x80808080u;
  lo = hsub2u(__byte_perm(b, 0x64646464u, 0x4140), 1152.f);
  hi = hsub2u(__byte_perm(b, 0x64646464u, 0x4342), 1152.f);
}

// 8 INT4 codes (low nibble = even k) -> 4 half2 (k order), exact: 0x64 over the code is 1024 + q.
__device__ __forceinline__ void i4_h2(uint32_t p, uint32_t (&h)[4]) {
  const uint32_t ev = p & 0x0F0F0F0Fu, od = (p >> 4) & 0x0F0F0F0Fu;
  const uint32_t t0 = __byte_perm(ev, od, 0x5140), t1 = __byte_perm(ev, od, 0x7362);
  h[0] = hsub2u(__byte_perm(t0, 0x64646464u, 0x4140), 1024.f);
  h[1] = hsub2u(__byte_perm(t0, 0x64646464u, 0x4342), 1024.f);
  h[2] = hsub2u(__byte_perm(t1, 0x64646464u, 0x4140), 1024.f);
  h[3] = hsub2u(__byte_perm(t1, 0x64646464u, 0x4342), 1024.f);
}

__device__ __forceinline__ float bf16_lo(uint32_t sb) { return __uint_as_float(sb << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t sb) { return __uint_as_float(sb & 0xFFFF0000u); }

__device__ __forceinline__ float pow2neg(int s) { return __int_as_float((127 - s) << 23); }

// FMT 1: INT8 [E, N, K], fp32 scale [E, N]; FMT 3: INT4 [E, N, K / 2], int32 [E, N, K / 32] (bf16 s | b << 16)
template <int FMT, int T, bool UP, int MT>
__global__ void __launch_bounds__(THREADS) moe_tc_kernel(
    const __half* __restrict__ h,         // [T, R, K] exact fp16 terms of the activation rows
    const int* __restrict__ sh,           // [T, R] term shifts
    const int* __restrict__ ok,           // [R] the terms cover the row
    const __nv_bfloat16* __restrict__ a,  // [R, K] the activation rows (fallback)
    const float* __restrict__ xs,         // INT4: [R, K / 32] the rows' group sums X_g
    const uint8_t* __restrict__ w, const void* __restrict__ ws,
    void* __restrict__ out,               // up: bf16 [num_valid, N]; down: fp32 [num_valid, N]
    const float* __restrict__ tw,         // [num_valid] routing weights (down)
    const int* __restrict__ sorted, const int* __restrict__ eids, const int* __restrict__ ntp,
    int num_valid, int R, int N, int K, int topk) {
  constexpr int BM = 8 * MT;
  // the BM rows' terms, one slab of KS k at a time; row stride KSP: the 8 rows of an octet hit 8 bank groups
  __shared__ __align__(16) __half As[T][BM][KSP];
  __shared__ float Xs[BM][MAXG];  // INT4: the rows' group sums
  __shared__ int ent[BM], arow[BM], okr[BM], shr[T][BM];
  __shared__ int any_bad;

  const int pid_m = blockIdx.x, n0 = blockIdx.y * BN;
  if (pid_m * BM >= __ldg(ntp)) return;
  const int e = __ldg(eids + pid_m);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = K / 32;
  if (tid == 0) any_bad = 0;
  __syncthreads();
  if (tid < BM) {
    const int en = __ldg(sorted + pid_m * BM + tid);
    const bool valid = en < num_valid;
    const int r = valid ? (UP ? en / topk : en) : -1;
    ent[tid] = valid ? en : -1;
    arow[tid] = r;
    const int o = valid ? __ldg(ok + r) : 1;
    okr[tid] = o;
#pragma unroll
    for (int t = 0; t < T; ++t) shr[t][tid] = valid ? __ldg(sh + t * R + r) : 0;
    if (!o) atomicOr(&any_bad, 1);
  }
  __syncthreads();
  if constexpr (FMT == 3) {
    for (int i = tid; i < BM * G; i += THREADS) {
      const int m = i / G, g = i % G, r = arow[m];
      Xs[m][g] = r >= 0 ? __ldg(xs + (size_t)r * G + g) : 0.f;
    }
  }

  const int q = (lane >> 2) & 3, t8 = (lane & 3) + 4 * ((lane >> 4) & 1);
  const int nb = n0 + warp * 32 + q * 8;     // first column of the octet
  const int ncol = nb + t8;                  // this lane's B column (weight row)
  const int crow = (lane & 1) + 4 * ((lane >> 4) & 1), ccol = 2 * ((lane >> 1) & 1);
  float acc[T][MT][8];
  float accb[MT][8];
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int s = 0; s < MT; ++s)
#pragma unroll
      for (int v = 0; v < 8; ++v) acc[t][s][v] = 0.f;
#pragma unroll
  for (int s = 0; s < MT; ++s)
#pragma unroll
    for (int v = 0; v < 8; ++v) accb[s][v] = 0.f;

  const size_t wrow = (size_t)e * N + ncol;
  // this lane's weight row, KC codes per step: INT8 32 bytes (two uint4), INT4 16 bytes (one uint4); the next
  // step's load is in flight while this one's products run
  constexpr int WV = FMT == 1 ? 2 : 1;
  const uint4* wp = FMT == 1 ? reinterpret_cast<const uint4*>(w + wrow * K)
                             : reinterpret_cast<const uint4*>(w + wrow * (K / 2));
  uint4 wc[WV], wn[WV];
#pragma unroll
  for (int i = 0; i < WV; ++i) wc[i] = __ldg(wp + i);
  const uint32_t* sbp = FMT == 3 ? reinterpret_cast<const uint32_t*>(ws) + ((size_t)e * N + nb) * G : nullptr;
  for (int kc = 0; kc < K; kc += KC) {
    const int step = kc / KC;
    if (kc % KS == 0) {  // stage the next slab of the rows' terms (16-byte chunks; padding rows: zeros)
      __syncthreads();
      constexpr int CPR = KS / 8;
      for (int i = tid; i < T * BM * CPR; i += THREADS) {
        const int t = i / (BM * CPR), m = (i / CPR) % BM, c = i % CPR;
        const int r = arow[m];
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r >= 0) v = __ldg(reinterpret_cast<const uint4*>(h + ((size_t)t * R + r) * K + kc) + c);
        *reinterpret_cast<uint4*>(&As[t][m][c * 8]) = v;
      }
      __syncthreads();
    }
    const int ko = kc % KS;
    if (kc + KC < K) {
#pragma unroll
      for (int i = 0; i < WV; ++i) wn[i] = __ldg(wp + (step + 1) * WV + i);
    }
    float sc[4], bi[4];
    if constexpr (FMT == 3) {  // group g = step: s_g, b_g of this lane's four C columns
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t sb = __ldg(sbp + (size_t)(ccol + (i & 1) + 4 * (i >> 1)) * G + step);
        sc[i] = bf16_lo(sb);
        bi[i] = bf16_hi(sb);
      }
    }
    uint32_t bh[16];  // KC k of this lane's weight row as 16 half2
    if constexpr (FMT == 1) {
      const uint32_t u[8] = {wc[0].x, wc[0].y, wc[0].z, wc[0].w, wc[WV - 1].x, wc[WV - 1].y, wc[WV - 1].z,
                             wc[WV - 1].w};
#pragma unroll
      for (int j = 0; j < 8; ++j) i8_h2(u[j], bh[2 * j], bh[2 * j + 1]);
    } else {
      const uint32_t u[4] = {wc[0].x, wc[0].y, wc[0].z, wc[0].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t hh[4];
        i4_h2(u[j], hh);
#pragma unroll
        for (int i = 0; i < 4; ++i) bh[4 * j + i] = hh[i];
      }
    }
    float grp[T][MT][8];  // INT4: this group's partial sums
    if constexpr (FMT == 3) {
#pragma unroll
      for (int t = 0; t < T; ++t)
#pragma unroll
        for (int s = 0; s < MT; ++s)
#pragma unroll
          for (int v = 0; v < 8; ++v) grp[t][s][v] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < KC / 4; ++j) {
#pragma unroll
      for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int s = 0; s < MT; ++s) {
          const uint2 av = *reinterpret_cast<const uint2*>(&As[t][s * 8 + t8][ko + 4 * j]);
          if constexpr (FMT == 1)
            mma884(acc[t][s], av.x, av.y, bh[2 * j], bh[2 * j + 1]);
          else
            mma884(grp[t][s], av.x, av.y, bh[2 * j], bh[2 * j + 1]);
        }
      }
    }
    if constexpr (FMT == 3) {  // acc += s_g[n] * partial sums; bias b_g[n] * X_g[row]
#pragma unroll
      for (int s = 0; s < MT; ++s) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int i = (v & 1) + 2 * (v >> 2);
#pragma unroll
          for (int t = 0; t < T; ++t) acc[t][s][v] = fmaf(sc[i], grp[t][s][v], acc[t][s][v]);
          const int m = s * 8 + crow + 2 * ((v >> 1) & 1);
          accb[s][v] = fmaf(bi[i], Xs[m][step], accb[s][v]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < WV; ++i) wc[i] = wn[i];
  }

  // epilogue: rows the terms cover
  float scl[4] = {1.f, 1.f, 1.f, 1.f};
  if constexpr (FMT == 1) {
    const float* sp = reinterpret_cast<const float*>(ws) + (size_t)e * N + nb;
#pragma unroll
    for (int i = 0; i < 4; ++i) scl[i] = __ldg(sp + ccol + (i & 1) + 4 * (i >> 1));
  }
#pragma unroll
  for (int s = 0; s < MT; ++s) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int m = s * 8 + crow + 2 * ((v >> 1) & 1);
      const int en = ent[m];
      if (en < 0 || !okr[m]) continue;
      float y = 0.f;
#pragma unroll
      for (int t = 0; t < T; ++t) y += acc[t][s][v] * pow2neg(shr[t][m]);
      const int i = (v & 1) + 2 * (v >> 2);
      if constexpr (FMT == 1) y *= scl[i];
      else y += accb[s][v];
      const int n = nb + ccol + (v & 1) + 4 * (v >> 2);
      if constexpr (UP) {
        float hv = __bfloat162float(__float2bfloat16_rn(y));
        hv = fmaxf(hv, 0.f);
        reinterpret_cast<__nv_bfloat16*>(out)[(size_t)en * N + n] = __float2bfloat16_rn(hv * hv);
      } else {
        reinterpret_cast<float*>(out)[(size_t)en * N + n] = y * __ldg(tw + en);
      }
    }
  }
  if (!any_bad) return;
  // fallback: rows the terms do not cover, fp32 FMA (thread = column), same decomposition
  const int n = n0 + tid;
  for (int m = 0; m < BM; ++m) {
    const int en = ent[m];
    if (en < 0 || okr[m]) continue;
    const __nv_bfloat16* x = a + (size_t)arow[m] * K;
    float y = 0.f;
    if constexpr (FMT == 1) {
      const int8_t* wq = reinterpret_cast<const int8_t*>(w) + ((size_t)e * N + n) * K;
      for (int k = 0; k < K; ++k) y = fmaf(__bfloat162float(x[k]), (float)wq[k], y);
      y *= __ldg(reinterpret_cast<const float*>(ws) + (size_t)e * N + n);
    } else {
      const uint8_t* wq = w + ((size_t)e * N + n) * (K / 2);
      const uint32_t* sbp = reinterpret_cast<const uint32_t*>(ws) + ((size_t)e * N + n) * G;
      for (int g = 0; g < G; ++g) {
        float p = 0.f;
        for (int k = 0; k < 32; ++k) {
          const uint8_t b = wq[(g * 32 + k) >> 1];
          const float qv = (float)((k & 1) ? (b >> 4) : (b & 15));
          p = fmaf(__bfloat162float(x[g * 32 + k]), qv, p);
        }
        const uint32_t sb = __ldg(sbp + g);
        y = fmaf(bf16_lo(sb), p, y);
        y = fmaf(bf16_hi(sb), Xs[m][g], y);
      }
    }
    if constexpr (UP) {
      float hv = __bfloat162float(__float2bfloat16_rn(y));
      hv = fmaxf(hv, 0.f);
      reinterpret_cast<__nv_bfloat16*>(out)[(size_t)en * N + n] = __float2bfloat16_rn(hv * hv);
    } else {
      reinterpret_cast<float*>(out)[(size_t)en * N + n] = y * __ldg(tw + en);
    }
  }
}

// One warp, two probes of the m8n8k4 fragments: D = A B with A[m][0] = m (probe 0) or 1 (probe 1), B[0][n] =
// 1 or n, every other element 0: out[probe][lane][v] = the row (probe 0) / column (probe 1) of value v.
__global__ void probe_kernel(float* out) {
  const int lane = threadIdx.x;
  const int t8 = (lane & 3) + 4 * ((lane >> 4) & 1);
  for (int p = 0; p < 2; ++p) {
    __half2 a0 = __halves2half2(__float2half(p == 0 ? (float)t8 : 1.f), __float2half(0.f));
    __half2 b0 = __halves2half2(__float2half(p == 0 ? 1.f : (float)t8), __float2half(0.f));
    __half2 z = __float2half2_rn(0.f);
    float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    mma884(c, *reinterpret_cast<uint32_t*>(&a0), *reinterpret_cast<uint32_t*>(&z),
           *reinterpret_cast<uint32_t*>(&b0), *reinterpret_cast<uint32_t*>(&z));
    for (int v = 0; v < 8; ++v) out[(p * 32 + lane) * 8 + v] = c[v];
  }
}

template <int FMT, int T, bool UP, int MT>
void launch(const torch::Tensor& h, const torch::Tensor& sh, const torch::Tensor& ok, const torch::Tensor& a,
            const torch::Tensor& xs, const torch::Tensor& w, const torch::Tensor& ws, torch::Tensor& out, const torch::Tensor& tw,
            const torch::Tensor& sorted, const torch::Tensor& eids, const torch::Tensor& ntp, int num_valid,
            int N, int K, int topk) {
  constexpr int BM = 8 * MT;
  const int R = a.size(0);
  const int EM = sorted.size(0);
  dim3 grid((EM + BM - 1) / BM, N / BN);
  moe_tc_kernel<FMT, T, UP, MT><<<grid, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(h.data_ptr()), sh.data_ptr<int>(), ok.data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), FMT == 3 ? xs.data_ptr<float>() : nullptr,
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      ws.data_ptr(), out.data_ptr(), UP ? nullptr : tw.data_ptr<float>(), sorted.data_ptr<int>(),
      eids.data_ptr<int>(), ntp.data_ptr<int>(), num_valid, R, N, K, topk);
}

}  // namespace

// fmt 1 (INT8 per row) or 3 (INT4 affine g32); terms 1..2; mt 1..2 (BM = 8 * mt, the sorted layout's block)
void moe_tc(torch::Tensor h, torch::Tensor sh, torch::Tensor ok, torch::Tensor a, torch::Tensor xs, torch::Tensor w,
            torch::Tensor ws, torch::Tensor out, torch::Tensor tw, torch::Tensor sorted, torch::Tensor eids,
            torch::Tensor ntp, int64_t num_valid, int64_t N, int64_t K, int64_t topk, int64_t fmt, int64_t terms,
            bool up, int64_t mt) {
  TORCH_CHECK(K % KS == 0 && K / 32 <= MAXG && N % BN == 0, "moe_tc: K % 128, K <= 1024, N % 128");
#define MOE_TC_CASE(F, TT, U, M)                                                                         \
  if (fmt == F && terms == TT && up == U && mt == M)                                                     \
    return launch<F, TT, U, M>(h, sh, ok, a, xs, w, ws, out, tw, sorted, eids, ntp, num_valid, N, K, topk);
#define MOE_TC_FMT(F) \
  MOE_TC_CASE(F, 1, true, 1) MOE_TC_CASE(F, 1, false, 1) MOE_TC_CASE(F, 2, true, 1) MOE_TC_CASE(F, 2, false, 1) \
  MOE_TC_CASE(F, 1, true, 2) MOE_TC_CASE(F, 1, false, 2) MOE_TC_CASE(F, 2, true, 2) MOE_TC_CASE(F, 2, false, 2)
  MOE_TC_FMT(1)
  MOE_TC_FMT(3)
  TORCH_CHECK(false, "moe_tc: unsupported fmt / terms / mt");
}

torch::Tensor probe() {
  auto out = torch::zeros({2, 32, 8}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  probe_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_tc", &moe_tc);
  m.def("probe", &probe);
}

// Few-row GEMV for Volta (V100), natively or through volta-forever: y[m, n] = sum_k w[n, k] x[m, k] for
// 2..8 rows of bf16 activations, with fp32 accumulation and plain FMA (no tensor cores). Weights: INT8 with one
// fp32 scale per row (W8A16 per channel: the scale multiplies the fp32 row sum) or one fp16 / fp32 scale per
// group of 32 (W8A16 per group: the scale multiplies the fp32 weight), bf16, or fp32. Output bf16 or fp32, into
// an (M, N) view of row stride ldy.
//
// Built JIT by decode_gemv.py on Volta. The Triton multi-row GEMV (_gemv_rows_kernel) reads 160-450 GB/s at 8
// rows: one weight row per lane, so every activation element is reloaded for every weight row it meets. Here a
// lane holds NRW weight rows x VEC consecutive k as fp32 (INT8 codes through a byte permute and one add: exact,
// no I2F) and all M activation rows' VEC values, loaded first, so each activation load serves NRW rows and the M
// rows are an exact template parameter (no branches in the loop); the weights stream in 8-16 byte vector loads
// with an evict-first hint (the activations stay in L1), the next step's already in flight. Small CTAs (4 warps)
// and few registers keep 4-6 CTAs per SM to hide the memory latency. A warp covers NRW rows over a K slice; the
// CTA's warps split its rows (RG groups) and K (WK slices, for narrow weights), the K slices meet in shared
// memory. Summation order: per lane sequential over its k, a butterfly over the 32 lanes, then the WK slices.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int THREADS = 128, WARPS = THREADS / 32;
enum Scale { NONE = 0, ROW = 1, GROUP = 2 };

__device__ __forceinline__ float i8f(uint32_t biased, int i) {
  // byte i of (codes ^ 0x80808080), i.e. q + 128, as the low byte of 2^23's mantissa: 2^23 + q + 128 exactly
  return __uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7540u | i)) - 8388736.0f;
}

// VEC weights of one row (4..32 bytes), streamed (evict-first: they are read once).
template <typename WT, int VEC> struct WRaw {
  static constexpr int WORDS = VEC * (int)sizeof(WT) / 4;
  uint32_t u[WORDS];
  __device__ __forceinline__ void load(const WT* p) {
    if constexpr (WORDS == 1) {
      u[0] = __ldcs(reinterpret_cast<const unsigned int*>(p));
    } else if constexpr (WORDS == 2) {
      const uint2 v = __ldcs(reinterpret_cast<const uint2*>(p));
      u[0] = v.x; u[1] = v.y;
    } else {
#pragma unroll
      for (int j = 0; j < WORDS / 4; ++j) {
        const uint4 v = __ldcs(reinterpret_cast<const uint4*>(p) + j);
        u[4 * j] = v.x; u[4 * j + 1] = v.y; u[4 * j + 2] = v.z; u[4 * j + 3] = v.w;
      }
    }
  }
  __device__ __forceinline__ void to_float(float* w) const {
    if constexpr (std::is_same_v<WT, int8_t>) {
#pragma unroll
      for (int j = 0; j < WORDS; ++j) {
        const uint32_t b = u[j] ^ 0x80808080u;
#pragma unroll
        for (int i = 0; i < 4; ++i) w[4 * j + i] = i8f(b, i);
      }
    } else if constexpr (std::is_same_v<WT, __nv_bfloat16>) {
#pragma unroll
      for (int j = 0; j < WORDS; ++j) {
        w[2 * j] = __uint_as_float(u[j] << 16);
        w[2 * j + 1] = __uint_as_float(u[j] & 0xffff0000u);
      }
    } else {
#pragma unroll
      for (int j = 0; j < WORDS; ++j) w[j] = __uint_as_float(u[j]);
    }
  }
};

// VEC bf16 activations (8 or 16 bytes), kept packed until use.
template <int VEC> struct XRaw {
  static constexpr int WORDS = VEC / 2;
  uint32_t u[WORDS];
  __device__ __forceinline__ void load(const __nv_bfloat16* p) {
    if constexpr (WORDS == 4) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
      u[0] = v.x; u[1] = v.y; u[2] = v.z; u[3] = v.w;
    } else {
      const uint2 v = __ldg(reinterpret_cast<const uint2*>(p));
      u[0] = v.x; u[1] = v.y;
    }
  }
  __device__ __forceinline__ void to_float(float* x) const {
#pragma unroll
    for (int j = 0; j < WORDS; ++j) {
      x[2 * j] = __uint_as_float(u[j] << 16);
      x[2 * j + 1] = __uint_as_float(u[j] & 0xffff0000u);
    }
  }
};

template <typename ST> __device__ __forceinline__ float to_f(ST s);
template <> __device__ __forceinline__ float to_f<float>(float s) { return s; }
template <> __device__ __forceinline__ float to_f<__half>(__half s) { return __half2float(s); }

template <typename OT> __device__ __forceinline__ OT from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <int NRW> constexpr int min_blocks() { return NRW <= 2 ? 6 : 4; }

template <typename WT, typename OT, typename ST, int M, int NRW, int VEC, int SCALE>
__global__ void __launch_bounds__(THREADS, min_blocks<NRW>())
gemv_rows_kernel(const __nv_bfloat16* __restrict__ x, long long ldx, const WT* __restrict__ w, long long ldw,
                 const ST* __restrict__ s, long long lds, OT* __restrict__ y, long long ldy, int N, int K, int WK) {
  __shared__ float red[WARPS][NRW][M];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int RG = WARPS / WK, rg = warp / WK, ks = warp % WK;
  const int n0 = (blockIdx.x * RG + rg) * NRW;
  const int kslice = K / WK, k_begin = ks * kslice, k_end = k_begin + kslice;
  float acc[NRW][M];
#pragma unroll
  for (int r = 0; r < NRW; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  // rows past N read row N - 1 (valid memory) and are never stored
  long long wrow[NRW];
#pragma unroll
  for (int r = 0; r < NRW; ++r) wrow[r] = (long long)min(n0 + r, N - 1) * ldw;
  WRaw<WT, VEC> cur[NRW], nxt[NRW];
  int k = k_begin + lane * VEC;
#pragma unroll
  for (int r = 0; r < NRW; ++r) cur[r].load(w + wrow[r] + k);
  for (; k < k_end; k += 32 * VEC) {
    const int kn = k + 32 * VEC < k_end ? k + 32 * VEC : k;  // the last step reloads its own (cached) tile
#pragma unroll
    for (int r = 0; r < NRW; ++r) nxt[r].load(w + wrow[r] + kn);
    XRaw<VEC> xr[M];
#pragma unroll
    for (int m = 0; m < M; ++m) xr[m].load(x + m * ldx + k);
    float wf[NRW][VEC];
#pragma unroll
    for (int r = 0; r < NRW; ++r) {
      cur[r].to_float(wf[r]);
      if constexpr (SCALE == GROUP) {
        const float sg = to_f<ST>(s[(long long)min(n0 + r, N - 1) * lds + (k >> 5)]);
#pragma unroll
        for (int i = 0; i < VEC; ++i) wf[r][i] *= sg;
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float xv[VEC];
      xr[m].to_float(xv);
#pragma unroll
      for (int i = 0; i < VEC; ++i)
#pragma unroll
        for (int r = 0; r < NRW; ++r) acc[r][m] = fmaf(wf[r][i], xv[i], acc[r][m]);
    }
#pragma unroll
    for (int r = 0; r < NRW; ++r) cur[r] = nxt[r];
  }
  // butterfly over the 32 lanes: every lane ends with the warp's sums
#pragma unroll
  for (int r = 0; r < NRW; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }
  if (WK == 1) {
    // lane i stores entry i of the warp's NRW x M sums (NRW * M <= 32)
#pragma unroll
    for (int r = 0; r < NRW; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        if (lane == r * M + m && n0 + r < N) {
          float v = acc[r][m];
          if constexpr (SCALE == ROW) v *= to_f<ST>(s[n0 + r]);
          y[m * ldy + n0 + r] = from_f<OT>(v);
        }
      }
    return;
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < NRW; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) red[warp][r][m] = acc[r][m];
  }
  __syncthreads();
  for (int t = threadIdx.x; t < RG * NRW * M; t += THREADS) {
    const int g = t / (NRW * M), r = (t / M) % NRW, m = t % M;
    const int n = (blockIdx.x * RG + g) * NRW + r;
    if (n < N) {
      float v = 0.f;
      for (int j = 0; j < WK; ++j) v += red[g * WK + j][r][m];
      if constexpr (SCALE == ROW) v *= to_f<ST>(s[n]);
      y[m * ldy + n] = from_f<OT>(v);
    }
  }
}

template <typename WT, typename OT, typename ST, int SCALE, int NRW, int VEC, int M>
void launch_m(const torch::Tensor& x, const torch::Tensor& w, const ST* sp, long long lds, const torch::Tensor& y,
              int wk) {
  const int N = w.size(0), K = x.size(1);
  const int RG = WARPS / wk;
  const dim3 grid((N + RG * NRW - 1) / (RG * NRW));
  gemv_rows_kernel<WT, OT, ST, M, NRW, VEC, SCALE><<<grid, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), reinterpret_cast<const WT*>(w.data_ptr()),
      w.stride(0), sp, lds, reinterpret_cast<OT*>(y.data_ptr()), y.stride(0), N, K, wk);
}

template <typename WT, typename OT, typename ST, int SCALE, int NRW, int VEC>
void launch_vec(const torch::Tensor& x, const torch::Tensor& w, const ST* sp, long long lds, const torch::Tensor& y,
                int wk) {
  switch (x.size(0)) {
    case 2: launch_m<WT, OT, ST, SCALE, NRW, VEC, 2>(x, w, sp, lds, y, wk); break;
    case 3: launch_m<WT, OT, ST, SCALE, NRW, VEC, 3>(x, w, sp, lds, y, wk); break;
    case 4: launch_m<WT, OT, ST, SCALE, NRW, VEC, 4>(x, w, sp, lds, y, wk); break;
    case 5: launch_m<WT, OT, ST, SCALE, NRW, VEC, 5>(x, w, sp, lds, y, wk); break;
    case 6: launch_m<WT, OT, ST, SCALE, NRW, VEC, 6>(x, w, sp, lds, y, wk); break;
    case 7: launch_m<WT, OT, ST, SCALE, NRW, VEC, 7>(x, w, sp, lds, y, wk); break;
    default: launch_m<WT, OT, ST, SCALE, NRW, VEC, 8>(x, w, sp, lds, y, wk); break;
  }
}

template <typename WT, typename OT, typename ST, int SCALE>
void launch_typed(const torch::Tensor& x, const torch::Tensor& w, const ST* sp, long long lds, const torch::Tensor& y,
                  int wk, int vec, int nrw) {
  if constexpr (!std::is_same_v<WT, float> && SCALE != GROUP) {
    if (nrw == 4) {
      if (vec == 8) return launch_vec<WT, OT, ST, SCALE, 4, 8>(x, w, sp, lds, y, wk);
      return launch_vec<WT, OT, ST, SCALE, 4, 4>(x, w, sp, lds, y, wk);
    }
  }
  if (vec == 8) return launch_vec<WT, OT, ST, SCALE, 2, 8>(x, w, sp, lds, y, wk);
  return launch_vec<WT, OT, ST, SCALE, 2, 4>(x, w, sp, lds, y, wk);
}

template <typename WT, typename ST, int SCALE>
void launch_out(const torch::Tensor& x, const torch::Tensor& w, const ST* sp, long long lds, const torch::Tensor& y,
                int wk, int vec, int nrw) {
  if (y.scalar_type() == torch::kFloat32) launch_typed<WT, float, ST, SCALE>(x, w, sp, lds, y, wk, vec, nrw);
  else launch_typed<WT, __nv_bfloat16, ST, SCALE>(x, w, sp, lds, y, wk, vec, nrw);
}
}  // namespace

// y (M, N) view (unit column stride) = x (M, K) bf16 @ deq(w).T for 2 <= M <= 8. scale: none (bf16 / fp32 w),
// fp32 [N] (INT8 per row) or fp16 / fp32 [N, K / 32] (INT8 per group). wk: K slices per CTA (1, 2, 4), vec: k per
// lane per step (4, 8), nrw: weight rows per warp (2, or 4 for INT8 per row / bf16).
void gemv_rows(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> scale, torch::Tensor y, int64_t wk,
               int64_t vec, int64_t nrw) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1);
  TORCH_CHECK(M >= 2 && M <= 8 && w.size(1) == K && y.size(0) == M && y.size(1) == N);
  TORCH_CHECK(wk == 1 || wk == 2 || wk == 4, "WK must divide the 4 warps");
  TORCH_CHECK(vec == 4 || vec == 8, "VEC must be 4 or 8");
  TORCH_CHECK(nrw == 2 || nrw == 4, "NRW must be 2 or 4");
  TORCH_CHECK(K % (wk * 32 * vec) == 0, "K must be a multiple of WK * 32 * VEC");
  TORCH_CHECK(y.scalar_type() == torch::kFloat32 || y.scalar_type() == torch::kBFloat16, "output: bf16 or fp32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.stride(0) * 2) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && (w.stride(0) * w.element_size()) % 16 == 0,
              "x and w rows must be 16-byte aligned");
  if (w.scalar_type() == torch::kInt8) {
    TORCH_CHECK(scale.has_value(), "INT8 weights need a scale");
    const auto& s = *scale;
    if (s.dim() == 1) {
      TORCH_CHECK(s.scalar_type() == torch::kFloat32 && s.size(0) == N && s.is_contiguous());
      launch_out<int8_t, float, ROW>(x, w, s.data_ptr<float>(), 0, y, wk, vec, nrw);
    } else {
      TORCH_CHECK(s.dim() == 2 && s.size(0) == N && s.size(1) * 32 == K && s.stride(1) == 1, "group scales: [N, K/32]");
      if (s.scalar_type() == torch::kFloat16) {
        launch_out<int8_t, __half, GROUP>(x, w, reinterpret_cast<const __half*>(s.data_ptr()), s.stride(0), y, wk,
                                          vec, nrw);
      } else {
        TORCH_CHECK(s.scalar_type() == torch::kFloat32);
        launch_out<int8_t, float, GROUP>(x, w, s.data_ptr<float>(), s.stride(0), y, wk, vec, nrw);
      }
    }
  } else if (w.scalar_type() == torch::kBFloat16) {
    TORCH_CHECK(!scale.has_value());
    launch_out<__nv_bfloat16, float, NONE>(x, w, nullptr, 0, y, wk, vec, nrw);
  } else {
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && !scale.has_value(), "weights: int8, bf16 or fp32");
    launch_out<float, float, NONE>(x, w, nullptr, 0, y, wk, vec, nrw);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv_rows", &gemv_rows); }

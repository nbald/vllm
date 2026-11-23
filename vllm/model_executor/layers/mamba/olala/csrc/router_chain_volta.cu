// Olala latent-MoE router logits above decode sizes, Volta (V100) edition: plain fp32 FMA, no cuBLAS.
//
//   y[m, e] = chain_hi[m, e] + chain_lo[m, e],   chain_X[m, e] = sum_k x[m, k] * X[e, k]   (fp32, k ascending)
//
// with x bf16 [M, K] and the router weight split into bf16 hi / lo halves ([hi ; lo], [2E, K], olala.py's bf16x2
// mode). Elsewhere this is mm(x, [hi ; lo].T, out_dtype=fp32) followed by the add of the two halves. On a V100 cuBLAS
// has no bf16 GEMM kernel of its own: through volta-forever it runs the fp32 SIMT magma_sgemmEx, whose result is
// a sequential fp32 FMA chain over k per output (checked bit for bit on the router's shapes), but with few CTAs
// at decode sizes (~215-250 us per layer from 33 to 512 rows). Here every output is one thread's chain of fmaf over
// k in increasing order from +0, as magma's, and the two chains are added once, as the slice add: the same bits,
// with tiles sized to the batch so that the GPU fills at 33-512 rows.
//
// One CTA per (BM rows, BE experts); thread (ty, tx) owns rows ty + i TY (i < TM) and experts tx + j TX (j < TN),
// a hi and a lo accumulator each. The x and weight tiles go through shared memory as fp32 (exact), double-buffered:
// the next one or two BK-wide tiles are loaded into registers while the current one is consumed (volta-forever runs
// cp.async as a synchronous copy, so the prefetch is plain loads). Built JIT by router_volta.py on the SM70 family.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int KALIGN = 64;  // K must be a multiple of every tile's BK

__device__ __forceinline__ void bf16x8_to_f32(const uint4 v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

// BK: k per tile; LDK: padded fp32 row (consecutive rows in different 16-byte bank groups: conflict-free float4 reads)
template <int BM, int BE, int TM, int TN, int BK>
struct Cfg {
  static constexpr int TY = BM / TM, TX = BE / TN, NT = TX * TY, LDK = BK + 4;
  static constexpr int XCH = BM * BK / 8;       // 16-byte chunks of the x tile
  static constexpr int WCH = 2 * BE * BK / 8;   // 16-byte chunks of the hi + lo weight tile
  static constexpr int CH = (XCH + WCH + NT - 1) / NT;   // chunks per thread
  static_assert(BM % TM == 0 && BE % TN == 0 && NT % 32 == 0 && NT <= 256 && KALIGN % BK == 0, "tile");
  static_assert(2 * (BM + 2 * BE) * LDK * 4 <= 48 * 1024, "static shared memory");
};

template <int BM, int BE, int TM, int TN, int BK, int PF>
__global__ void __launch_bounds__(Cfg<BM, BE, TM, TN, BK>::NT)
router_chain_kernel(const __nv_bfloat16* __restrict__ x, const long long sx, const __nv_bfloat16* __restrict__ w,
                    float* __restrict__ y, const long long sy, const int M, const int K, const int E) {
  using C = Cfg<BM, BE, TM, TN, BK>;
  __shared__ __align__(16) float xs[2][BM][C::LDK];
  __shared__ __align__(16) float ws[2][2 * BE][C::LDK];   // rows 0..BE-1 hi, BE..2BE-1 lo
  const int t = threadIdx.x, tx = t % C::TX, ty = t / C::TX;
  const int m0 = blockIdx.x * BM, e0 = blockIdx.y * BE;

  // global -> registers (bf16, 16-byte chunks; rows past M read as zero) -> shared memory (fp32)
  auto load = [&](uint4 (&st)[C::CH], int k0) {
#pragma unroll
    for (int i = 0; i < C::CH; ++i) {
      const int c = t + i * C::NT;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (c < C::XCH) {
        const int r = c / (BK / 8), q = c % (BK / 8);
        if (m0 + r < M) v = __ldg(reinterpret_cast<const uint4*>(x + (long long)(m0 + r) * sx + k0 + q * 8));
      } else if (c < C::XCH + C::WCH) {
        const int c2 = c - C::XCH, r = c2 / (BK / 8), q = c2 % (BK / 8);
        const int row = r < BE ? e0 + r : E + e0 + (r - BE);
        v = __ldg(reinterpret_cast<const uint4*>(w + (long long)row * K + k0 + q * 8));
      }
      st[i] = v;
    }
  };
  auto store = [&](const uint4 (&st)[C::CH], int b) {
#pragma unroll
    for (int i = 0; i < C::CH; ++i) {
      const int c = t + i * C::NT;
      float f[8];
      bf16x8_to_f32(st[i], f);
      float* dst = nullptr;
      if (c < C::XCH) {
        dst = &xs[b][c / (BK / 8)][(c % (BK / 8)) * 8];
      } else if (c < C::XCH + C::WCH) {
        const int c2 = c - C::XCH;
        dst = &ws[b][c2 / (BK / 8)][(c2 % (BK / 8)) * 8];
      }
      if (dst) {
        *reinterpret_cast<float4*>(dst) = make_float4(f[0], f[1], f[2], f[3]);
        *reinterpret_cast<float4*>(dst + 4) = make_float4(f[4], f[5], f[6], f[7]);
      }
    }
  };

  float ah[TM][TN], al[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) { ah[i][j] = 0.f; al[i][j] = 0.f; }
  // one BK-wide tile of every chain, k ascending
  auto compute = [&](int b) {
#pragma unroll
    for (int kk = 0; kk < BK; kk += 4) {
      float4 xv[TM], hv[TN], lv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) xv[i] = *reinterpret_cast<const float4*>(&xs[b][ty + i * C::TY][kk]);
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        hv[j] = *reinterpret_cast<const float4*>(&ws[b][tx + j * C::TX][kk]);
        lv[j] = *reinterpret_cast<const float4*>(&ws[b][BE + tx + j * C::TX][kk]);
      }
      // k ascending: kk, kk + 1, kk + 2, kk + 3
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          ah[i][j] = __fmaf_rn(xv[i].x, hv[j].x, ah[i][j]); al[i][j] = __fmaf_rn(xv[i].x, lv[j].x, al[i][j]);
        }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          ah[i][j] = __fmaf_rn(xv[i].y, hv[j].y, ah[i][j]); al[i][j] = __fmaf_rn(xv[i].y, lv[j].y, al[i][j]);
        }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          ah[i][j] = __fmaf_rn(xv[i].z, hv[j].z, ah[i][j]); al[i][j] = __fmaf_rn(xv[i].z, lv[j].z, al[i][j]);
        }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          ah[i][j] = __fmaf_rn(xv[i].w, hv[j].w, ah[i][j]); al[i][j] = __fmaf_rn(xv[i].w, lv[j].w, al[i][j]);
        }
    }
  };

  const int nt = K / BK;
  uint4 ra[C::CH];
  if constexpr (PF == 1) {   // one tile in flight while a tile is consumed
    load(ra, 0);
    store(ra, 0);
    __syncthreads();
    for (int kt = 0; kt < nt; ++kt) {
      const int b = kt & 1;
      if (kt + 1 < nt) load(ra, (kt + 1) * BK);
      compute(b);
      if (kt + 1 < nt) store(ra, b ^ 1);
      __syncthreads();
    }
  } else {   // two tiles in flight: when tile kt is consumed, ra holds tile kt + 1 and rb tile kt + 2
    uint4 rb[C::CH];
    load(ra, 0);
    store(ra, 0);
    if (1 < nt) load(ra, BK);
    if (2 < nt) load(rb, 2 * BK);
    __syncthreads();
    for (int kt = 0; kt < nt; kt += 2) {
      compute(0);
      if (kt + 1 < nt) store(ra, 1);
      __syncthreads();
      if (kt + 3 < nt) load(ra, (kt + 3) * BK);
      if (kt + 1 < nt) {
        compute(1);
        if (kt + 2 < nt) store(rb, 0);
        __syncthreads();
        if (kt + 4 < nt) load(rb, (kt + 4) * BK);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = m0 + ty + i * C::TY;
    if (m < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) y[(long long)m * sy + e0 + tx + j * C::TX] = __fadd_rn(ah[i][j], al[i][j]);
    }
  }
}

template <int BM, int BE, int TM, int TN, int BK, int PF = 1>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int E) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(E % BE == 0, "router_chain: E must be a multiple of the expert tile");
  dim3 grid((M + BM - 1) / BM, E / BE);
  router_chain_kernel<BM, BE, TM, TN, BK, PF><<<grid, Cfg<BM, BE, TM, TN, BK>::NT, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), y.data_ptr<float>(), y.stride(0), M, K, E);
}

}  // namespace

// Tile per batch size. V100 through volta-forever, graphed, weights from DRAM, us per layer at E = 256, K = 1536
// (today's mm + add in brackets): 33 rows 34 (286), 64: 32 (241), 128: 39 (239), 256: 57 (235), 512: 92 (236),
// 1024: 164 (248), 2048: 303 (353).
static int pick_tile(int M) {
  if (M <= 64) return 0;
  if (M <= 256) return 1;
  if (M <= 384) return 2;
  if (M <= 512) return 3;
  if (M <= 768) return 4;
  if (M <= 1024) return 3;
  if (M <= 1536) return 4;
  return 5;
}

// y [M, E] fp32 = chain(x, w[:E]) + chain(x, w[E:]); tile -1: chosen by M (else one of pick_tile's, for tests).
void router_chain(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t E, int64_t tile) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kFloat);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2 && w.size(0) == 2 * E && w.size(1) == x.size(1));
  TORCH_CHECK(x.stride(1) == 1 && w.is_contiguous() && y.stride(1) == 1 && y.size(0) == x.size(0) && y.size(1) == E);
  TORCH_CHECK(x.size(1) % KALIGN == 0 && E % 64 == 0 && (x.stride(0) * 2) % 16 == 0
              && (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) == 0 && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) == 0,
              "router_chain: shape / alignment");
  if (x.size(0) == 0) return;
  switch (tile < 0 ? pick_tile((int)x.size(0)) : (int)tile) {
    case 0: launch<16, 8, 1, 1, 32, 2>(x, w, y, (int)E); break;
    case 1: launch<32, 16, 2, 2, 32, 2>(x, w, y, (int)E); break;
    case 2: launch<16, 32, 4, 1, 32, 2>(x, w, y, (int)E); break;
    case 3: launch<64, 32, 4, 2, 16, 2>(x, w, y, (int)E); break;
    case 4: launch<32, 32, 4, 2, 32, 1>(x, w, y, (int)E); break;
    case 5: launch<64, 64, 4, 4, 16, 2>(x, w, y, (int)E); break;
    default: TORCH_CHECK(false, "router_chain: unknown tile");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("router_chain", &router_chain, py::arg("x"), py::arg("w"), py::arg("y"), py::arg("E"), py::arg("tile") = -1); }

// Olala Differential-TPA decode attention over the TPA-factorized paged KV cache, Volta (V100) edition: plain fp32
// FMA, no tensor cores, no cp.async. Same cache layout, launch contract and partial outputs as
// tpa_factor_decode_sm80.cu (the split combine and tpa_factor.py are shared). Built JIT by tpa_factor.py on a V100,
// native or through volta-forever: there sm_70 has no bf16 mma.sync / ldmatrix / cp.async, which volta-forever emulates
// with scalar code (one bf16 m16n8k16 ~ 200 instructions), while this kernel's FMA code runs natively.
//
// One CTA per (request, KV split), one warp per KV group (12 warps; the group's 4 query heads), lane l owns dims
// 4l..4l+3. Per 13-token tile the factor rows (B' bf16 as stored, coefficients as fp32) are staged in shared memory;
// each warp rebuilds its group's K row of every token from the rank-4 factors (k_t = C_t B'_t + D_t B'_{t-1}, 32 FMA
// per lane), scores its 4 heads (16 FMA + a reduce-scatter of 6 shuffles that leaves head l/8's full dot product in
// lanes 8(l/8)..8(l/8)+7), runs the online softmax per head in those lanes, then rebuilds V the same way and
// accumulates o += p v. Rebuilding K/V per group costs 12x128x8 FMA per token and plane, fewer than the qB = Q B'^T
// factor route (48x4x128) of the tensor-core kernels.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {
constexpr int HQ = 48, HKV = 12, R = 4, D = 128, GQ = HQ / HKV;
constexpr int SPLIT_TILE = 13;                 // split granularity of tpa_factor.py (TILE there)
constexpr int T = 13;                          // tokens per inner tile
constexpr int ROWP = 608, OFF_C = 512, NCOEF = 96;
constexpr int THREADS = HKV * 32;
constexpr int BCHUNKS = R * D / 8;             // 64 x 16 B of B' per token and plane
constexpr int CCHUNKS = NCOEF / 8;             // 12 x 16 B of coefficients per token and plane
static_assert(GQ == 4 && D == 128, "the lane / head mapping below assumes 4 heads per group and 128 dims");

struct __align__(16) Smem {
  __nv_bfloat16 bk[(T + 1) * R * D];   // token t0-1+j, rank r -> row j*R + r (j = 0: the halo t0-1)
  __nv_bfloat16 bv[(T + 1) * R * D];
  float ck[T * NCOEF];                 // token t0+j: [C (12 groups x 4) | D (12 x 4)]
  float cv[T * NCOEF];
  long long row[T + 1];                // element offset of token t0-1+j's row in the planes, -1 = zero fill
};

__device__ __forceinline__ void bf16x4(const __nv_bfloat16* p, float* f) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xffff0000u);
  f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xffff0000u);
}

__global__ void __launch_bounds__(THREADS, 1)
tpa_factor_decode_volta_kernel(const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl, long long bstride,
                               const __nv_bfloat16* __restrict__ q, const int* __restrict__ bt, int bt_stride,
                               const int* __restrict__ seqlens, float* __restrict__ part_o, float* __restrict__ part_m,
                               float* __restrict__ part_l, int hqp, float sm_scale, float softcap,
                               const int* __restrict__ tok_per_split_ptr, int BS) {
  extern __shared__ __align__(16) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tok_per_split = *tok_per_split_ptr;
  const float inv_cap = softcap > 0.f ? 1.f / softcap : 0.f;
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, g = tid >> 5, lane = tid & 31, hh = lane >> 3, d0 = lane * 4;
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);
  const size_t base = ((size_t)b * nsplit + sp) * hqp;
  if (stop <= start) {
    for (int i = tid; i < hqp; i += THREADS) { part_m[base + i] = -CUDART_INF_F; part_l[base + i] = 0.f; }
    return;
  }
  const int* btb = bt + (size_t)b * bt_stride;

  float qv[GQ][4];   // this group's 4 query heads, this lane's 4 dims
#pragma unroll
  for (int h = 0; h < GQ; ++h) bf16x4(q + ((size_t)b * HQ + g * GQ + h) * D + d0, qv[h]);
  float o[GQ][4];
#pragma unroll
  for (int h = 0; h < GQ; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) o[h][i] = 0.f;
  float m_run = -CUDART_INF_F, l_run = 0.f;   // head hh of this lane (identical in its 8 lanes)

  for (int t0 = start; t0 < stop; t0 += T) {
    __syncthreads();   // the previous tile's smem is consumed
    if (tid <= T) {    // token t0-1+tid: row offset; tokens before 0 or from `stop` on are zero-filled
      const int t = t0 - 1 + tid;
      S.row[tid] = (t >= 0 && t < stop) ? (long long)btb[t / BS] * bstride + (long long)(t % BS) * ROWP : -1;
    }
    __syncthreads();
    for (int c = tid; c < (T + 1) * 2 * BCHUNKS; c += THREADS) {   // B' rows of both planes, bf16 as stored
      const int j = c / (2 * BCHUNKS), rem = c % (2 * BCHUNKS), pl = rem / BCHUNKS, cc = rem % BCHUNKS;
      const long long row = S.row[j];
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row >= 0) v = __ldg(reinterpret_cast<const uint4*>((pl == 0 ? kpl : vpl) + row + cc * 8));
      *reinterpret_cast<uint4*>((pl == 0 ? S.bk : S.bv) + j * R * D + cc * 8) = v;
    }
    for (int c = tid; c < T * 2 * CCHUNKS; c += THREADS) {          // coefficients of tokens t0.., as fp32
      const int j = c / (2 * CCHUNKS), rem = c % (2 * CCHUNKS), pl = rem / CCHUNKS, cc = rem % CCHUNKS;
      const long long row = S.row[j + 1];
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row >= 0) v = __ldg(reinterpret_cast<const uint4*>((pl == 0 ? kpl : vpl) + row + OFF_C + cc * 8));
      float* dst = (pl == 0 ? S.ck : S.cv) + j * NCOEF + cc * 8;
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) { dst[2 * i] = __uint_as_float(w[i] << 16); dst[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
    }
    __syncthreads();

    // ---- scores: k_t = C_t B'_t + D_t B'_{t-1} for this warp's group, s[h][t] = q_h . k_t
    float sc[T];
    {
      float bp[R][4], bc[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) bf16x4(S.bk + r * D + d0, bp[r]);
#pragma unroll
      for (int j = 1; j <= T; ++j) {
#pragma unroll
        for (int r = 0; r < R; ++r) bf16x4(S.bk + (j * R + r) * D + d0, bc[r]);
        const float4 cf = *reinterpret_cast<const float4*>(S.ck + (j - 1) * NCOEF + g * R);
        const float4 df = *reinterpret_cast<const float4*>(S.ck + (j - 1) * NCOEF + HQ + g * R);
        const float cr[R] = {cf.x, cf.y, cf.z, cf.w}, dr[R] = {df.x, df.y, df.z, df.w};
        float k[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float acc = 0.f;
#pragma unroll
          for (int r = 0; r < R; ++r) acc = fmaf(cr[r], bc[r][i], fmaf(dr[r], bp[r][i], acc));
          k[i] = acc;
        }
        float p[GQ];
#pragma unroll
        for (int h = 0; h < GQ; ++h) p[h] = qv[h][0] * k[0] + qv[h][1] * k[1] + qv[h][2] * k[2] + qv[h][3] * k[3];
        // reduce-scatter over the 32 lanes: lanes 8h..8h+7 end with head h's full dot product
        const bool hi4 = lane & 16, hi3 = lane & 8;
        float a0 = hi4 ? p[2] : p[0], a1 = hi4 ? p[3] : p[1];
        const float s0 = hi4 ? p[0] : p[2], s1 = hi4 ? p[1] : p[3];
        a0 += __shfl_xor_sync(0xffffffffu, s0, 16);
        a1 += __shfl_xor_sync(0xffffffffu, s1, 16);
        float a = hi3 ? a1 : a0;
        a += __shfl_xor_sync(0xffffffffu, hi3 ? a0 : a1, 8);
        a += __shfl_xor_sync(0xffffffffu, a, 4);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        sc[j - 1] = a;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) bp[r][i] = bc[r][i];
      }
    }
    // ---- online softmax of head hh over this tile (the 8 lanes of a head compute it alike)
    float mx = m_run;
#pragma unroll
    for (int j = 0; j < T; ++j) {
      float s = sc[j] * sm_scale;
      if (softcap > 0.f) s = softcap * tanhf(s * inv_cap);
      s = (t0 + j < stop) ? s : -CUDART_INF_F;
      sc[j] = s;
      mx = fmaxf(mx, s);
    }
    const float alpha = (m_run == -CUDART_INF_F) ? 0.f : __expf(m_run - mx);
    float lsum = 0.f;
#pragma unroll
    for (int j = 0; j < T; ++j) {
      const float pj = (sc[j] == -CUDART_INF_F) ? 0.f : __expf(sc[j] - mx);
      sc[j] = pj;
      lsum += pj;
    }
    l_run = l_run * alpha + lsum;
    m_run = mx;
#pragma unroll
    for (int h = 0; h < GQ; ++h) {
      const float al = __shfl_sync(0xffffffffu, alpha, h * 8);
#pragma unroll
      for (int i = 0; i < 4; ++i) o[h][i] *= al;
    }
    // ---- o += p_t v_t with v_t = Cv_t B'v_t + Dv_t B'v_{t-1}
    {
      float bp[R][4], bc[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) bf16x4(S.bv + r * D + d0, bp[r]);
#pragma unroll
      for (int j = 1; j <= T; ++j) {
#pragma unroll
        for (int r = 0; r < R; ++r) bf16x4(S.bv + (j * R + r) * D + d0, bc[r]);
        const float4 cf = *reinterpret_cast<const float4*>(S.cv + (j - 1) * NCOEF + g * R);
        const float4 df = *reinterpret_cast<const float4*>(S.cv + (j - 1) * NCOEF + HQ + g * R);
        const float cr[R] = {cf.x, cf.y, cf.z, cf.w}, dr[R] = {df.x, df.y, df.z, df.w};
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float acc = 0.f;
#pragma unroll
          for (int r = 0; r < R; ++r) acc = fmaf(cr[r], bc[r][i], fmaf(dr[r], bp[r][i], acc));
          v[i] = acc;
        }
#pragma unroll
        for (int h = 0; h < GQ; ++h) {
          const float ph = __shfl_sync(0xffffffffu, sc[j - 1], h * 8);
#pragma unroll
          for (int i = 0; i < 4; ++i) o[h][i] = fmaf(ph, v[i], o[h][i]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) bp[r][i] = bc[r][i];
      }
    }
  }
  // ---- partials: unnormalized o, running max and sum per head (as the sm80 kernel)
#pragma unroll
  for (int h = 0; h < GQ; ++h)
    *reinterpret_cast<float4*>(part_o + (base + g * GQ + h) * D + d0) = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
  if ((lane & 7) == 0) { part_m[base + g * GQ + hh] = m_run; part_l[base + g * GQ + hh] = l_run; }
}
}  // namespace

void tpa_factor_decode_volta_launch(torch::Tensor q, torch::Tensor kplane, torch::Tensor vplane, torch::Tensor block_table,
                                    torch::Tensor seq_lens, torch::Tensor tok_per_split, torch::Tensor part_o, torch::Tensor part_m,
                                    torch::Tensor part_l, double sm_scale, double softcap, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1), hqp = part_o.size(2);
  TORCH_CHECK(kplane.dim() == 3 && kplane.size(2) == ROWP && kplane.stride(2) == 1 && kplane.stride(1) == ROWP && q.is_contiguous());
  TORCH_CHECK(vplane.stride(0) == kplane.stride(0) && vplane.stride(1) == ROWP && kplane.size(1) == block_size && block_size % SPLIT_TILE == 0);
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 && tok_per_split.dtype() == torch::kInt32);
  TORCH_CHECK(hqp >= HQ && part_o.size(3) == D && part_o.is_contiguous() && part_m.is_contiguous() && part_l.is_contiguous());
  TORCH_CHECK((kplane.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(kplane.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(vplane.data_ptr()) % 16 == 0, "factor planes must be 16B aligned");
  const int smem = (int)sizeof(Smem);
  tpa_factor_decode_volta_kernel<<<dim3(B, SPLIT), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(kplane.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(vplane.data_ptr()),
      (long long)kplane.stride(0), reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), block_table.data_ptr<int>(),
      (int)block_table.stride(0), seq_lens.data_ptr<int>(), part_o.data_ptr<float>(), part_m.data_ptr<float>(),
      part_l.data_ptr<float>(), hqp, (float)sm_scale, (float)softcap, tok_per_split.data_ptr<int>(), (int)block_size);
}
int64_t tpa_factor_volta_smem_bytes() { return sizeof(Smem); }
int64_t tpa_factor_volta_occupancy() {
  int nb = -1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, tpa_factor_decode_volta_kernel, THREADS, (int)sizeof(Smem));
  return nb;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &tpa_factor_decode_volta_launch); m.def("smem_bytes", &tpa_factor_volta_smem_bytes);
  m.def("occupancy", &tpa_factor_volta_occupancy); m.attr("TILE") = SPLIT_TILE; m.attr("INNER_TILE") = T;
}

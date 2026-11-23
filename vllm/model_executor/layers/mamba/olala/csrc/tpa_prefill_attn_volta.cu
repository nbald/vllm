// Olala Differential-TPA prefill attention, Volta (V100) edition: causal varlen attention over the dense K / V that
// tpa_factor.reconstruct_dense_kv rebuilds, head dim 128, 4 query heads per KV head, optional tanh softcap; plain fp32
// FMA, no tensor cores. Built JIT by tpa_prefill.py on the SM70 family, where it replaces FlashAttention-2's
// flash_fwd_kernel<128, 128, 64, 4> (its bf16 mma.sync / ldmatrix / cp.async are emulated there by volta-forever
// with scalar code, ~200 instructions per m16n8k16, while this kernel's FMA code runs natively).
//
// It computes FlashAttention-2's values, operation for operation:
//   - s = q . k as one chain of fmaf over d = 0..127, from 0 (volta-forever evaluates FA2's bf16 m16n8k16 chain so);
//   - with a softcap: s = tanh.approx(s * (scale / softcap)), and the exponentials take softcap * log2(e), as FA2's
//     host code sets them up; keys past the row's causal limit or the sequence end: -inf;
//   - keys in blocks of 64, from the last block a row reaches down to block 0 (FA2's order; a block that is
//     entirely masked for a row is a no-op in FA2 and here alike), online softmax per block: running max m,
//     rescale exp2((m_old - m) * scale_log2) of the output and of the row sum, p = exp2(fma(s, scale_log2, -m *
//     scale_log2)); the row sum kept as FA2's four per-thread partials (keys 8 i + 2 q, 8 i + 2 q + 1, in that
//     order, partial q = 0..3), added (q0 + q2) + (q1 + q3) at the end;
//   - o = o * rescale, then one chain of fmaf over the block's 64 keys of bf16(p) * v; out = bf16(o * (1 / sum)).
// So under volta-forever it returns FA2's bits.
//
// One CTA (8 warps, ~82 KB of shared memory) per (16 query rows, KV head, sequence): the 64 rows (row, query head of
// the group) share each K / V block. Per block: S = Q K^T (thread: 4 rows x 4 keys), the softmax statistics (thread:
// one row x one of FA2's four column classes), O += P V (thread: 4 rows x 8 dims). K and V are staged through one
// fp32 buffer (K^T, then V), each block's global loads issued one phase ahead into registers.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int D = 128, GQ = 4, BM = 16, HR = BM * GQ, KB = 64;  // HR: rows of a CTA (query row x head)
constexpr int THREADS = 256;
constexpr int LDP = KB + 4;      // fp32 score rows
constexpr int LDV = D + 4;       // fp32 V rows

struct Smem {
  float qs[D * HR];              // Q^T [d][hr], hr = row * GQ + head
  float kv[KB * LDV];            // K^T [d][key] (ld KB), then V [key][d] (ld LDV)
  float ps[HR * LDP];            // scores, then bf16(p) [hr][key]
  float rscale[HR];              // this block's rescale of each row's output
  float rinv[HR];                // 1 / row sum
};

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void bf16x8(const uint4& u, float* f) {
  const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    f[2 * e] = __uint_as_float(w[e] << 16);
    f[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
  }
}

template <bool SOFTCAP>
__global__ void __launch_bounds__(THREADS, 1) attn_fwd_volta_kernel(
    const __nv_bfloat16* __restrict__ q, long sq_t, long sq_h,    // (Tq, HQ, D)
    const __nv_bfloat16* __restrict__ k, long sk_t, long sk_h,    // (Tk, HKV, D)
    const __nv_bfloat16* __restrict__ v, long sv_t, long sv_h,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    __nv_bfloat16* __restrict__ out, long so_t, long so_h,        // (Tq, HQ, D)
    float qk_mul, float scale_log2) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, g = blockIdx.x, b = blockIdx.z;
  const int mb = gridDim.y - 1 - blockIdx.y;  // the longest rows first
  const int q_beg = cu_q[b], tq = cu_q[b + 1] - q_beg, k_beg = cu_k[b], tk = cu_k[b + 1] - k_beg;
  const int m0 = mb * BM;
  if (m0 >= tq) return;
  const int off = tk - tq;                                   // causal limit of row r: key <= m0 + r + off
  const int last_row = min(m0 + BM, tq) - 1;
  const int kmax = min(tk, last_row + off + 1);              // keys this CTA can see
  const int nblk = (kmax + KB - 1) / KB;

  // ---- Q^T, fp32 (rows past the sequence: 0)
  for (int c = tid; c < HR * (D / 8); c += THREADS) {
    const int hr = c & (HR - 1), dc = c / HR, row = hr / GQ, hd = hr % GQ;
    float f[8];
    uint4 u = make_uint4(0u, 0u, 0u, 0u);
    if (m0 + row < tq)
      u = __ldg(reinterpret_cast<const uint4*>(q + (long)(q_beg + m0 + row) * sq_t + (long)(g * GQ + hd) * sq_h + 8 * dc));
    bf16x8(u, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) S.qs[(8 * dc + e) * HR + hr] = f[e];
  }
  const __nv_bfloat16* kg = k + (long)k_beg * sk_t + (long)g * sk_h;
  const __nv_bfloat16* vg = v + (long)k_beg * sv_t + (long)g * sv_h;
  // K block loads: key c & 63, 8-dim chunk c >> 6 (transposed stores without bank conflicts);
  // V block loads: key c >> 4, chunk c & 15 (coalesced)
  uint4 pre[4];
  auto load_k = [&](int blk) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + THREADS * i, key = blk * KB + (c & 63), dc = c >> 6;
      pre[i] = key < tk ? __ldg(reinterpret_cast<const uint4*>(kg + (long)key * sk_t + 8 * dc)) : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto load_v = [&](int blk) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + THREADS * i, key = blk * KB + (c >> 4), dc = c & 15;
      pre[i] = key < tk ? __ldg(reinterpret_cast<const uint4*>(vg + (long)key * sv_t + 8 * dc)) : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto store_k = [&]() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + THREADS * i, key = c & 63, dc = c >> 6;
      float f[8];
      bf16x8(pre[i], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) S.kv[(8 * dc + e) * KB + key] = f[e];
    }
  };
  auto store_v = [&]() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + THREADS * i, key = c >> 4, dc = c & 15;
      float f[8];
      bf16x8(pre[i], f);
      float4* dst = reinterpret_cast<float4*>(&S.kv[key * LDV + 8 * dc]);
      dst[0] = make_float4(f[0], f[1], f[2], f[3]);
      dst[1] = make_float4(f[4], f[5], f[6], f[7]);
    }
  };

  // S phase: rows 4 hs .. 4 hs + 3 (query row hs, its 4 heads), keys 4 ks .. 4 ks + 3
  const int hs = tid >> 4, ks = tid & 15;
  const int lim_s = m0 + hs + off;                           // causal limit of this thread's query row
  // statistics: row hr_st, column class cq (keys 8 i + 2 cq, 8 i + 2 cq + 1)
  const int hr_st = tid >> 2, cq = tid & 3;
  float m_run = -INFINITY, rsum = 0.f;
  // PV phase: rows 4 hp .. 4 hp + 3, dims 4 dp .. 4 dp + 3 and 64 + 4 dp .. 64 + 4 dp + 3
  const int hp = tid >> 4, dp = tid & 15;
  float acc[4][8];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[a][e] = 0.f;

  load_k(nblk - 1);
  for (int blk = nblk - 1; blk >= 0; --blk) {
    __syncthreads();  // previous PV phase done with kv / ps
    store_k();
    __syncthreads();
    load_v(blk);
    // ---- S = Q K^T
    float s[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int c = 0; c < 4; ++c) s[a][c] = 0.f;
#pragma unroll 8
    for (int d = 0; d < D; ++d) {
      const float4 qa = *reinterpret_cast<const float4*>(&S.qs[d * HR + 4 * hs]);
      const float4 kk = *reinterpret_cast<const float4*>(&S.kv[d * KB + 4 * ks]);
      const float qv[4] = {qa.x, qa.y, qa.z, qa.w}, kv4[4] = {kk.x, kk.y, kk.z, kk.w};
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int c = 0; c < 4; ++c) s[a][c] = fmaf(qv[a], kv4[c], s[a][c]);
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int key = blk * KB + 4 * ks + c;
      const bool masked = key > lim_s || key >= tk;
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        float x = s[a][c];
        if (SOFTCAP) x = tanh_approx(x * qk_mul);
        s[a][c] = masked ? -INFINITY : x;
      }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a)
      *reinterpret_cast<float4*>(&S.ps[(4 * hs + a) * LDP + 4 * ks]) = make_float4(s[a][0], s[a][1], s[a][2], s[a][3]);
    __syncthreads();  // scores complete; K^T no longer read
    store_v();
    // ---- statistics of row hr_st, class cq: running max, rescale, p = exp2(...), FA2's partial sum
    {
      float x[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 t = *reinterpret_cast<const float2*>(&S.ps[hr_st * LDP + 8 * i + 2 * cq]);
        x[2 * i] = t.x;
        x[2 * i + 1] = t.y;
      }
      float mx = m_run;
#pragma unroll
      for (int i = 0; i < 16; ++i) mx = fmaxf(mx, x[i]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      const float cur = mx == -INFINITY ? 0.f : mx;
      const float sc = exp2f((m_run - cur) * scale_log2);
      const float msc = mx == -INFINITY ? 0.f : mx * scale_log2;
      float p[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) p[i] = exp2f(__fmaf_rn(x[i], scale_log2, -msc));
      rsum = __fmaf_rn(rsum, sc, p[0]);
#pragma unroll
      for (int i = 1; i < 16; ++i) rsum = __fadd_rn(rsum, p[i]);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        *reinterpret_cast<float2*>(&S.ps[hr_st * LDP + 8 * i + 2 * cq]) = make_float2(rbf(p[2 * i]), rbf(p[2 * i + 1]));
      m_run = mx;
      if (cq == 0) S.rscale[hr_st] = sc;
    }
    __syncthreads();  // p, rescale and V ready
    if (blk > 0) load_k(blk - 1);
    // ---- O = O * rescale + P V
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const float r = S.rscale[4 * hp + a];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[a][e] *= r;
    }
#pragma unroll 2
    for (int kk = 0; kk < KB; kk += 4) {
      float4 pr[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) pr[a] = *reinterpret_cast<const float4*>(&S.ps[(4 * hp + a) * LDP + kk]);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float4 v0 = *reinterpret_cast<const float4*>(&S.kv[(kk + u) * LDV + 4 * dp]);
        const float4 v1 = *reinterpret_cast<const float4*>(&S.kv[(kk + u) * LDV + 64 + 4 * dp]);
        const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          const float pa = u == 0 ? pr[a].x : u == 1 ? pr[a].y : u == 2 ? pr[a].z : pr[a].w;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[a][e] = fmaf(pa, vv[e], acc[a][e]);
        }
      }
    }
  }
  // ---- row sums: FA2's quad all-reduce of the four partials, then 1 / sum
  {
    float sum = rsum;
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
    if (cq == 0) S.rinv[hr_st] = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
  }
  __syncthreads();
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int hr = 4 * hp + a, row = hr / GQ, hd = hr % GQ;
    if (m0 + row >= tq) continue;
    const float r = S.rinv[hr];
    __nv_bfloat16* o = out + (long)(q_beg + m0 + row) * so_t + (long)(g * GQ + hd) * so_h;
    __nv_bfloat162 w0[2], w1[2];
    w0[0] = __floats2bfloat162_rn(acc[a][0] * r, acc[a][1] * r);
    w0[1] = __floats2bfloat162_rn(acc[a][2] * r, acc[a][3] * r);
    w1[0] = __floats2bfloat162_rn(acc[a][4] * r, acc[a][5] * r);
    w1[1] = __floats2bfloat162_rn(acc[a][6] * r, acc[a][7] * r);
    *reinterpret_cast<uint2*>(o + 4 * dp) = *reinterpret_cast<const uint2*>(w0);
    *reinterpret_cast<uint2*>(o + 64 + 4 * dp) = *reinterpret_cast<const uint2*>(w1);
  }
}
}  // namespace

// out = causal attention of (q, k, v), FlashAttention-2's varlen_fwd values (see above). scale / softcap: as given to
// FA2 (softcap 0: none); the derived constants are computed as FA2's host code computes them.
void tpa_prefill_attn_volta(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cu_q, torch::Tensor cu_k,
                            torch::Tensor out, int64_t max_seqlen_q, double softmax_scale, double softcap) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && out.sizes() == q.sizes());
  TORCH_CHECK(q.size(2) == D && k.size(2) == D && v.size(2) == D, "head dim 128 only");
  const int hq = q.size(1), hkv = k.size(1);
  TORCH_CHECK(v.size(1) == hkv && hq == GQ * hkv, "4 query heads per KV head only");
  for (const auto& t : {q, k, v, out}) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.stride(2) == 1);
    TORCH_CHECK(t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(cu_q.scalar_type() == at::kInt && cu_k.scalar_type() == at::kInt && cu_q.numel() == cu_k.numel());
  const int nseq = cu_q.numel() - 1;
  if (nseq <= 0 || max_seqlen_q <= 0 || q.size(0) == 0) return;
  // FA2 (flash_api.cpp, set_params_fprop): float arguments; softcap folds the scale into the tanh argument
  const float scale_f = (float)softmax_scale, cap_f = (float)softcap;
  const bool cap = cap_f > 0.f;
  const float qk_mul = cap ? scale_f / cap_f : 0.f;
  const float scale_log2 = cap ? (float)(cap_f * M_LOG2E) : (float)(scale_f * M_LOG2E);
  const int smem = (int)sizeof(Smem);
  static bool attr[2] = {false, false};
  auto kern = cap ? attn_fwd_volta_kernel<true> : attn_fwd_volta_kernel<false>;
  if (!attr[cap]) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    attr[cap] = true;
  }
  const dim3 grid(hkv, (unsigned)((max_seqlen_q + BM - 1) / BM), nseq);
  kern<<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), q.stride(0), q.stride(1),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), k.stride(0), k.stride(1),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), v.stride(0), v.stride(1),
      cu_q.data_ptr<int>(), cu_k.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), qk_mul, scale_log2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_fwd", &tpa_prefill_attn_volta); }

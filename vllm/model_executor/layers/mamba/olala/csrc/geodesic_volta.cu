// Geodesic residual update (models/olala.py: _geodesic_norm_kernel) for Volta (V100), natively or through
// volta-forever: h = x cos(theta) + (g - c x) (R sin(theta) / |g - c x|), with c = x.g / |x|^2, R = |x|,
// theta = min((min(|g - c x| / R, clamp) * scale + bias) * inv_depth, clamp), one row of D bf16 per CTA.
//
// The Triton kernel runs one 64-thread CTA per row; through volta-forever it takes ~9 us per call at any row count
// (latency, not bandwidth: 6 KB per row). This edition returns the Triton kernel's bits: the same element-to-lane
// assignment for the three row reductions (Triton thread t holds 4 reps of 8 elements, r * 512 + 8 t + i), the same
// in-thread tree (fma pairs, then a pairwise tree over the pairs and the reps), the same butterfly (xor 16 .. 1) and
// cross-warp sum, and the same instructions as the Triton kernel's PTX for every rounding step (div.full.f32,
// sqrt.approx.ftz.f32, libdevice's FTZ sinf / cosf fast path, cvt.rn.bf16x2.f32), each pinned with inline PTX so
// that no contraction differs. Its CTA has 64 x REP threads: thread u handles rep u / 64 of Triton thread u % 64
// (one 16-byte load of x and of g per thread instead of four), the reps meet in shared memory.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float f_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_fma(float a, float b, float c) {
  float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float f_fma_ftz(float a, float b, float c) {
  float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float f_mul_ftz(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_sqrt_approx(float a) { float r; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_max(float a, float b) { float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_min(float a, float b) { float r; asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ uint32_t bf16x2_rn(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

// libdevice __nv_sinf / __nv_cosf with __CUDA_FTZ = 1 (Triton's enable_reflect_ftz), fast path (|a| < 105615):
// the instruction sequence of the Triton kernel's PTX. The slow path (Payne-Hanek, |a| >= 105615 or inf) cannot
// be reached here (theta <= clamp, and theta >= min(bias, clamp * scale + bias) * inv_depth); it falls back to
// sinf / cosf.
__device__ __forceinline__ void trig_reduce(float a, int& q, float& r) {
  const float t = f_mul(a, __uint_as_float(0x3F22F983u));
  asm("cvt.rni.ftz.s32.f32 %0, %1;" : "=r"(q) : "f"(t));
  float qf; asm("cvt.rn.f32.s32 %0, %1;" : "=f"(qf) : "r"(q));
  r = f_fma_ftz(qf, __uint_as_float(0xBFC90FDAu), a);
  r = f_fma_ftz(qf, __uint_as_float(0xB3A22168u), r);
  r = f_fma_ftz(qf, __uint_as_float(0xA7C234C5u), r);
}
// sin(r + q pi/2) as libdevice's kernel: the sine polynomial for even q, the cosine one for odd q, sign by q & 2
__device__ __forceinline__ float trig_poly(int q, float r) {
  const float r2 = f_mul_ftz(r, r);
  const bool odd = q & 1;
  const float base = odd ? 1.0f : r;
  const float t = f_fma_ftz(r2, base, 0.0f);
  float c1, c2, c3;
  if (odd) {
    c1 = f_fma_ftz(__uint_as_float(0x37CBAC00u), r2, __uint_as_float(0xBAB607EDu));
    c2 = __uint_as_float(0x3D2AAABBu);
    c3 = __uint_as_float(0xBEFFFFFFu);
  } else {
    c1 = __uint_as_float(0xB94D4153u);
    c2 = __uint_as_float(0x3C0885E4u);
    c3 = __uint_as_float(0xBE2AAAA8u);
  }
  float p = f_fma_ftz(c1, r2, c2);
  p = f_fma_ftz(p, r2, c3);
  p = f_fma_ftz(p, t, base);
  if (q & 2) p = f_fma_ftz(p, -1.0f, 0.0f);
  return p;
}
__device__ __forceinline__ void sincos_ftz(float a, float& s, float& c) {
  if (fabsf(a) < 105615.0f || a != a) {
    int q; float r;
    trig_reduce(a, q, r);
    s = trig_poly(q, r);
    c = trig_poly(q + 1, r);
  } else {
    s = sinf(a);
    c = cosf(a);
  }
}

__device__ __forceinline__ void unpack8(const uint4 u, float* f) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { f[2 * i] = __uint_as_float(w[i] << 16); f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
}

// The rep tree of the Triton kernel for 8 elements: fma pairs (2q, 2q+1), then ((p0 + p1) + (p2 + p3)).
__device__ __forceinline__ float rep_tree(const float* a, const float* b) {
  float p[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) p[q] = f_fma(a[2 * q], b[2 * q], f_mul(a[2 * q + 1], b[2 * q + 1]));
  return f_add(f_add(p[0], p[1]), f_add(p[2], p[3]));
}

__device__ __forceinline__ float warp_bfly(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = f_add(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// REP = reps of 512 elements (4: BLOCK = 2048, D <= 2048); thread u = rep (u / 64) of Triton thread (u % 64).
template <int REP>
__global__ void __launch_bounds__(64 * REP)
geodesic_volta_kernel(const __nv_bfloat16* __restrict__ x, long long sx, const __nv_bfloat16* __restrict__ g,
                      long long sg, __nv_bfloat16* __restrict__ out, long long so, const __nv_bfloat16* __restrict__ scale_p,
                      const __nv_bfloat16* __restrict__ bias_p, float inv_depth, float clamp_val, int D) {
  __shared__ float part[3][REP][64];
  __shared__ float fin[3][2];
  const int row = blockIdx.x, u = threadIdx.x, t = u & 63, rep = u >> 6;
  const int idx = rep * 512 + t * 8;
  // every global load first, in parallel: through the model's weight streams x, scale and bias come from DRAM
  // (the Triton kernel loads scale and bias after the reductions: a second memory round trip)
  const unsigned short sc_raw = __ldg(reinterpret_cast<const unsigned short*>(scale_p));
  const unsigned short bi_raw = __ldg(reinterpret_cast<const unsigned short*>(bias_p));
  float xf[8], gf[8];
  uint4 xu = make_uint4(0u, 0u, 0u, 0u), gu = make_uint4(0u, 0u, 0u, 0u);
  if (idx < D) {  // D % 8 == 0: a rep's 8 elements are all in or all out
    xu = *reinterpret_cast<const uint4*>(x + row * sx + idx);
    gu = *reinterpret_cast<const uint4*>(g + row * sg + idx);
  }
  unpack8(xu, xf);
  unpack8(gu, gf);
  part[0][rep][t] = rep_tree(xf, xf);
  part[1][rep][t] = rep_tree(xf, gf);
  part[2][rep][t] = rep_tree(gf, gf);
  __syncthreads();
  if (u < 64) {  // Triton thread t: ((R0 + R1) + (R2 + R3)), then the warp butterfly
    const int w = u >> 5;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float v = f_add(f_add(part[j][0][t], part[j][1][t]), f_add(part[j][2][t], part[j][3][t]));
      v = warp_bfly(v);
      if ((u & 31) == 0) fin[j][w] = v;
    }
  }
  __syncthreads();
  const float xx = f_add(fin[0][0], fin[0][1]);
  const float xg = f_add(fin[1][0], fin[1][1]);
  const float gg = f_add(fin[2][0], fin[2][1]);
  const float coeff = f_div_full(xg, f_max(xx, __uint_as_float(0x2B8CBCCCu)));
  const float ts = f_fma(xx, f_mul(coeff, coeff), f_fma(-xg, f_add(coeff, coeff), gg));
  const float safe_tangent = f_max(f_sqrt_approx(f_max(ts, 0.0f)), __uint_as_float(0x322BCC77u));
  const float safe_r = f_max(f_sqrt_approx(xx), __uint_as_float(0x358637BDu));
  const float sc = __uint_as_float((uint32_t)sc_raw << 16), bi = __uint_as_float((uint32_t)bi_raw << 16);
  float theta = f_min(f_div_full(safe_tangent, safe_r), clamp_val);
  theta = f_min(f_mul(inv_depth, f_fma(theta, sc, bi)), clamp_val);
  float sn, cs;
  sincos_ftz(theta, sn, cs);
  const float k = f_div_full(f_mul(safe_r, sn), safe_tangent);
  if (idx < D) {
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = f_fma(cs, xf[i], f_mul(f_fma(-coeff, xf[i], gf[i]), k));
    uint4 ov;
    ov.x = bf16x2_rn(o[0], o[1]); ov.y = bf16x2_rn(o[2], o[3]);
    ov.z = bf16x2_rn(o[4], o[5]); ov.w = bf16x2_rn(o[6], o[7]);
    *reinterpret_cast<uint4*>(out + row * so + idx) = ov;
  }
}

}  // namespace

void geodesic_volta(torch::Tensor x, torch::Tensor g, torch::Tensor out, torch::Tensor scale, torch::Tensor bias,
                    double inv_depth, double clamp_val) {
  const int D = x.size(-1);
  const long long rows = x.numel() / D;
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && g.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(scale.dtype() == torch::kBFloat16 && bias.dtype() == torch::kBFloat16);
  TORCH_CHECK(D % 8 == 0 && D <= 2048 && D > 1024, "D in (1024, 2048], a multiple of 8 (Triton BLOCK 2048)");
  TORCH_CHECK(x.stride(-1) == 1 && g.stride(-1) == 1 && out.stride(-1) == 1);
  const long long sx = x.dim() > 1 ? x.stride(-2) : 0, sg = g.dim() > 1 ? g.stride(-2) : 0;
  const long long so = out.dim() > 1 ? out.stride(-2) : 0;
  TORCH_CHECK(sx % 8 == 0 && sg % 8 == 0 && so % 8 == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
  if (rows == 0) return;
  geodesic_volta_kernel<4><<<(unsigned)rows, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), sx, reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()), sg,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), so, reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), (float)inv_depth, (float)clamp_val, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("geodesic", &geodesic_volta); }

// Olala Mamba-3 MIMO decode step (one token per request), persistent CUDA kernel; replaces the CuteDSL step on the
// fused decode path (vllm/model_executor/layers/mamba/olala/mamba3.py). Built JIT by mamba3_step_cuda.py.
// Mamba-3 MIMO decode step, persistent edition: each CTA (128 threads) walks (token, head) items with a 2-stage
// cp.async pipeline (state 16 KB + previous B 4 KB + token rows per item) so the next item's loads overlap the
// current item's preamble, update and store. Warp w owns state rows p in [16w, 16w+16); lane l owns columns
// s in [4l, 4l+4) -> per row a warp touches 256 contiguous bytes of the smem tile (conflict-free), the rank-4
// coefficients B'[r][s], Bstate[r][s], C'[r][s] of the thread's 4 columns live in registers.
// Two state storage dtypes (template F32): bf16 runs the update and the C dot on tensor cores; fp32 keeps the state
// unrounded end to end (fp32 FMA update and C dot over the same fragment layout) — memory-bound either way.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

// STEP_VOLTA (mamba3_step_cuda.py on a V100, natively or through volta-forever): sm_70 has no bf16 mma.sync, ldmatrix
// or cp.async, so volta-forever emulates them with scalar code (one bf16 m16n8k16 ~ 200 instructions, cp.async as a
// synchronous copy) while plain fp32 FMA runs natively. The fp32-state step then takes the scalar FMA update + C dot
// (STEP_F32_FMA, same math as the tensor-core split) in the 1-stage kernel instead of the tensor-core pipeline.
#if defined(STEP_VOLTA) && !defined(STEP_F32_FMA)
#define STEP_F32_FMA
#endif

namespace {
constexpr int H = 48, D = 64, S = 128, R = 4, NA = 32;
#ifndef STEP_NST
#define STEP_NST 2
#endif
constexpr int THREADS = 128, NST = STEP_NST;
// fp32 stages are 2x larger: one stage and 4 CTAs/SM beats a 2-stage pipeline at 2 CTAs/SM (2.85 vs 2.53 TB/s, B=64)
template <bool F32> constexpr int kStages = F32 ? 1 : NST;
constexpr int LDS = S + 8;   // padded smem row (bf16 elems) -> conflict-free ldmatrix on the state tile
constexpr int LDB = S + 8;   // padded rows for the B / C operand tiles
constexpr int LDSF = S + 8;  // padded fp32 state row (8-word skew: conflict-free float2 fragment access)
template <bool F32> struct StateTile { __nv_bfloat16 state[D * LDS]; };   // 17 KB (padded rows)
template <> struct StateTile<true> { float state[D * LDSF]; };            // 33 KB
template <bool F32> struct __align__(128) Stage : StateTile<F32> {
  __nv_bfloat16 bst[R * S];       // 1 KB previous B (k_pool)
  __nv_bfloat16 bm[R * S];        // 1 KB token B rows
  __nv_bfloat16 cm[R * S];        // 1 KB token C rows
  __nv_bfloat16 x[D], z[D], xst[D];
  __nv_bfloat16 angp[NA];
  float angst[NA];
};
// Volta builds: the fp32-state step runs the FMA update, which needs neither the zero rows that pad the mma's k nor
// the biases in shared memory (read through L1 instead); the tile then fits two CTAs per SM (45 KB instead of 52 KB).
#ifdef STEP_VOLTA
template <bool F32> constexpr bool kSlim = F32;
#else
template <bool F32> constexpr bool kSlim = false;
#endif
template <bool F32> struct __align__(128) Smem {
  Stage<F32> st[kStages<F32>];
  float bias_k[kSlim<F32> ? 4 : R * S], bias_q[kSlim<F32> ? 4 : R * S];   // per-head (fixed per CTA)
  float xproj[R * D], zproj[R * D], outproj[R * D];
  float sCos[NA], sSin[NA];
  __nv_bfloat16 sBk[kSlim<F32> ? 8 : 16][LDB];   // mma B operand of the update: rows 0..3 B'[r], 4..7 Bstate[r], 8..15 zero
  __nv_bfloat16 sCb[kSlim<F32> ? 4 : 8][LDB];    // rotated C' bf16, rows 4..7 zero: mma B operand for o = h C'^T
  float sTheta[NA];
};
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }

struct Args {
  void* ssm; __nv_bfloat16* kp; __nv_bfloat16* vp; float* angle; int P;
  const float* A; const __nv_bfloat16* Bm; const __nv_bfloat16* Cm; const float* Dp; const __nv_bfloat16* x; const float* dt; const float* trap;
  const float* xproj; const float* zproj; const float* outproj; const __nv_bfloat16* z; const float* bias_q; const float* bias_k;
  const __nv_bfloat16* angp; const int* slots; __nv_bfloat16* y; int B;
  long long bc_bstride, angp_stride;   // batch stride of Bm/Cm (elements), row stride of angle_proj
  long long ssm_sst, kp_sst, vp_sst, ang_sst;   // slot (dim 0) strides of the pools in elements (pages may be padded)
  // raw mode (preamble fused): x/z/dt/A/trap come from the head-interleaved in_proj rows zxdt, Bm/Cm are the raw
  // in_proj_dyn rows, RMS-normed here with gains wb/wc
  int raw; const __nv_bfloat16* zxdt; long long zxdt_stride; const float* dt_bias; float a_floor;
  const float* wb; const float* wc; float eps;
  int pdl;   // launched with programmatic stream serialization: wait for the producer grid before reading its output
};

template <bool F32> __device__ __forceinline__ void issue_item(const Args& a, Stage<F32>& st, int item, int t) {
  const int h = item % H, b = item / H;
  const int slot = a.slots[b]; const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  if constexpr (F32) {   // state 32 KB = 2048 x 16B -> 16 per thread
    const float* src = reinterpret_cast<const float*>(a.ssm) + (size_t)srow * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
    for (int i = 0; i < 16; ++i) { const int c = t + i * THREADS, row = c >> 5, ch = c & 31; cp16(st.state + row * LDSF + ch * 4, src + c * 4); }
  } else {               // state 16 KB = 1024 x 16B -> 8 per thread
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(a.ssm) + (size_t)srow * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
    for (int i = 0; i < 8; ++i) { const int c = t + i * THREADS, row = c >> 4, ch = c & 15; cp16(st.state + row * LDS + ch * 8, src + c * 8); }
  }
  // previous B (k_pool), token B / C rows: 64 x 16B chunks each
  if (t < 64) {
    const int r = t >> 4, c = t & 15;
    cp16(st.bst + r * S + c * 8, a.kp + (size_t)srow * a.kp_sst + ((size_t)r * H + h) * S + c * 8);
    cp16(st.bm + t * 8, a.Bm + (size_t)b * a.bc_bstride + t * 8);
    cp16(st.cm + t * 8, a.Cm + (size_t)b * a.bc_bstride + t * 8);
  } else {
    const int q = t - 64;   // 64 threads: x, z, xst (8 chunks each), angp (4), angst (8 fp32 chunks)
    if (q < 16 && a.raw) {   // head rows of zxdt are 2D+3 elements: not 16B aligned, plain loads
      const __nv_bfloat16* src = a.zxdt + (size_t)b * a.zxdt_stride + (size_t)h * (2 * D + 3) + (q < 8 ? D : 0) + (q & 7) * 8;
      __nv_bfloat16* dst = (q < 8 ? st.x : st.z) + (q & 7) * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) dst[i] = src[i];
    }
    else if (q < 8) cp16(st.x + q * 8, a.x + ((size_t)b * H + h) * D + q * 8);
    else if (q < 16) cp16(st.z + (q - 8) * 8, a.z + ((size_t)b * H + h) * D + (q - 8) * 8);
    else if (q < 24) cp16(st.xst + (q - 16) * 8, a.vp + (size_t)srow * a.vp_sst + (size_t)h * D + (q - 16) * 8);
    else if (q < 28) cp16(st.angp + (q - 24) * 8, a.angp + (size_t)b * a.angp_stride + (q - 24) * 8);
    else if (q < 36) cp16(st.angst + (q - 28) * 4, a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA + (q - 28) * 4);
  }
}

#ifndef STEP_MINB
#define STEP_MINB 4
#endif
template <bool F32> __global__ void __launch_bounds__(THREADS, STEP_MINB) mamba3_step_cuda_kernel(const Args a) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem<F32>& S_ = *reinterpret_cast<Smem<F32>*>(smem_raw);
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int h = blockIdx.x % H, nb_stride = gridDim.x / H;   // grid is a multiple of H
  const int nitems = a.B * H;
  if constexpr (!kSlim<F32>) {
    for (int i = t; i < 8 * LDB; i += THREADS) { S_.sBk[8 + i / LDB][i % LDB] = __float2bfloat16(0.f); if (i < 4 * LDB) S_.sCb[4 + i / LDB][i % LDB] = __float2bfloat16(0.f); }
  }
  // per-head constants once per CTA
  if constexpr (!kSlim<F32>) {
    const int r = t >> 5, c = t & 31;
    cp16(S_.bias_k + r * S + c * 4, a.bias_k + ((size_t)r * H + h) * S + c * 4);
    cp16(S_.bias_q + r * S + c * 4, a.bias_q + ((size_t)r * H + h) * S + c * 4); }
  // bias of (rank r, column j) of this head: shared memory, or global memory through L1 in the slim tile
  auto bias_k = [&](int r, int j) -> float { if constexpr (kSlim<F32>) return __ldg(a.bias_k + ((size_t)r * H + h) * S + j); else return S_.bias_k[r * S + j]; };
  auto bias_q = [&](int r, int j) -> float { if constexpr (kSlim<F32>) return __ldg(a.bias_q + ((size_t)r * H + h) * S + j); else return S_.bias_q[r * S + j]; };
  if (t < 64) { const int r = t >> 4, c = t & 15; cp16(S_.xproj + r * D + c * 4, a.xproj + ((size_t)r * H + h) * D + c * 4); cp16(S_.outproj + r * D + c * 4, a.outproj + ((size_t)r * H + h) * D + c * 4); }
  else { const int q = t - 64, r = q >> 4, c = q & 15; cp16(S_.zproj + r * D + c * 4, a.zproj + ((size_t)r * H + h) * D + c * 4); }
  // the per-head constants above are weights; everything below may be the previous kernel's output
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (a.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // PDL: sm_90+ only (pdl is 0 below)
#endif
  int b0 = blockIdx.x / H;
  auto item_of = [&](int k) { return (b0 + k * nb_stride) * H + h; };   // token index b = b0 + k*nb_stride
  const int nk = (a.B - b0 + nb_stride - 1) / nb_stride;                 // items of this CTA
  if (b0 >= a.B) return;
#pragma unroll
  for (int s = 0; s < kStages<F32> - 1; ++s) { if (s < nk) issue_item(a, S_.st[s], item_of(s), t); cp_commit(); }
  for (int k = 0; k < nk; ++k) {
    const int item = item_of(k);
    { const int pf = k + kStages<F32> - 1; if (pf < nk) issue_item(a, S_.st[pf % kStages<F32>], item_of(pf), t); cp_commit(); }
    const int b = item / H;
    const int slot = a.slots[b]; const bool valid = slot >= 0 && slot < a.P;
    float dt, A, trap;
    if (a.raw) {   // _decode_preamble_kernel: softplus (identity above 20), clamp, sigmoid
      const __nv_bfloat16* hp = a.zxdt + (size_t)b * a.zxdt_stride + (size_t)h * (2 * D + 3) + 2 * D;
      const float v = bf2f(hp[0]) + a.dt_bias[h], av = bf2f(hp[1]), tv = bf2f(hp[2]);
      dt = v > 20.f ? v : logf(1.f + expf(v));
      A = fminf(-(av > 20.f ? av : logf(1.f + expf(av))), -a.a_floor);
      trap = 1.f / (1.f + expf(-tv));
    } else { dt = a.dt[b * H + h]; A = a.A[b * H + h]; trap = a.trap[b * H + h]; }
    const float Dh = a.Dp[h];
    cp_wait<kStages<F32> - 1>();
    __syncthreads();
    Stage<F32>& st = S_.st[k % kStages<F32>];
    if (a.raw) {   // _bc_norm_kernel: per (B/C, r) row RMS norm + gain, bf16-rounded; warp w owns rows 2w, 2w+1
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int row = warp * 2 + rr;   // 0..3 B rows, 4..7 C rows
        __nv_bfloat16* v = (row < R ? st.bm : st.cm) + (row & 3) * S + lane * 4;
        const float* w = (row < R ? a.wb : a.wc) + lane * 4;
        float f[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { f[i] = bf2f(v[i]); ss += f[i] * f[i]; }
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = 1.f / sqrtf(ss / S + a.eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = __float2bfloat16(f[i] * inv * w[i]);
      }
    }
    const float alpha = __expf(A * dt), gamma = trap * dt, beta = (1.f - trap) * dt * alpha;
    // ---- preamble: angles, bias + rotary on B/C (bf16-rounded)
    // carried phase wrapped to [0, 2*pi) as in prefill: unwrapped it grows with the context and loses fp32 precision
    if (t < NA) { float th = st.angst[t] + tanhf(bf2f(st.angp[t])) * dt * 3.14159265358979323846f; th -= 6.283185307179586f * floorf(th / 6.283185307179586f); S_.sTheta[t] = th; sincosf(th, &S_.sSin[t], &S_.sCos[t]); }
    __syncthreads();
#ifndef V_NOPRE
    for (int i = t; i < R * (S / 2); i += THREADS) {
      const int r = i / (S / 2), j = i % (S / 2);
      float blo = bf2f(st.bm[r * S + j]) + bias_k(r, j), bhi = bf2f(st.bm[r * S + j + 64]) + bias_k(r, j + 64);
      float clo = bf2f(st.cm[r * S + j]) + bias_q(r, j), chi = bf2f(st.cm[r * S + j + 64]) + bias_q(r, j + 64);
      if (j < NA) {
        const float c = S_.sCos[j], s_ = S_.sSin[j];
        const float b0 = blo * c - bhi * s_, b1 = blo * s_ + bhi * c, c0 = clo * c - chi * s_, c1 = clo * s_ + chi * c;
        blo = b0; bhi = b1; clo = c0; chi = c1;
      }
      const __nv_bfloat16 z0 = __float2bfloat16(0.f);
      S_.sBk[r][j] = __float2bfloat16(blo); S_.sBk[r][j + 64] = __float2bfloat16(bhi);
      S_.sBk[4 + r][j] = st.bst[r * S + j]; S_.sBk[4 + r][j + 64] = st.bst[r * S + j + 64];
      S_.sCb[r][j] = __float2bfloat16(clo); S_.sCb[r][j + 64] = __float2bfloat16(chi);

    }
#endif
    __syncthreads();
    const int qd = lane & 3, p0 = warp * 16 + (lane >> 2);
    if constexpr (F32) {
#ifndef STEP_F32_FMA
      // ---- tensor-core fp32 step. h_new = alpha h + A Bk with the fp32 accumulator seeded by alpha h (the state never
      // leaves fp32) and A (k < 4: gamma x xproj, 4 <= k < 8: beta xstate xproj) split into three bf16 terms
      // hi + mid + lo (24 mantissa bits, as fp32); Bk is bf16 already, so every product is exact in fp32. Then
      // o = h_new C'^T with h_new split the same way (the m16n8 accumulators of two n8 tiles are the A fragment of a
      // k16 step). ~fp32-FMA accuracy at a fraction of the issue slots of the scalar loop (which was issue-bound).
      uint32_t ah[3][4];
      {
        float av[2][2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int p = p0 + e * 8;
          const float xp = bf2f(st.x[p]), xs = bf2f(st.xst[p]);
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const int k = 2 * qd + jj, r = k & 3;
            const float xpj = S_.xproj[r * D + p];
            av[e][jj] = (k < 4) ? gamma * xp * xpj : beta * xs * xpj;
          }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float h0 = rbf(av[e][0]), h1 = rbf(av[e][1]);
          const float m0 = rbf(av[e][0] - h0), m1 = rbf(av[e][1] - h1);
          ah[0][e] = pack_bf16(h0, h1); ah[1][e] = pack_bf16(m0, m1);
          ah[2][e] = pack_bf16(av[e][0] - h0 - m0, av[e][1] - h1 - m1);
        }
#pragma unroll
        for (int q = 0; q < 3; ++q) { ah[q][2] = 0u; ah[q][3] = 0u; }
      }
      float acc[16][4];
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {
        const int n = jt * 8 + 2 * qd;
        const float2 h0 = *reinterpret_cast<const float2*>(st.state + p0 * LDSF + n);
        const float2 h1 = *reinterpret_cast<const float2*>(st.state + (p0 + 8) * LDSF + n);
        acc[jt][0] = alpha * h0.x; acc[jt][1] = alpha * h0.y; acc[jt][2] = alpha * h1.x; acc[jt][3] = alpha * h1.y;
      }
      const int mi = lane >> 3, li = lane & 7;
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        uint32_t bfr[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[(mi & 1) * 8 + li][jp * 16 + (mi >> 1) * 8])));
#pragma unroll
        for (int q = 2; q >= 0; --q) { mma16816(acc[2 * jp], ah[q], bfr); mma16816(acc[2 * jp + 1], ah[q], bfr + 2); }
      }
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {   // h_new back into the tile for the coalesced pool store
        const int n = jt * 8 + 2 * qd;
        *reinterpret_cast<float2*>(st.state + p0 * LDSF + n) = make_float2(acc[jt][0], acc[jt][1]);
        *reinterpret_cast<float2*>(st.state + (p0 + 8) * LDSF + n) = make_float2(acc[jt][2], acc[jt][3]);
      }
      float oc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        uint32_t bfr[4];
        ldmatrix_x4(bfr, &S_.sCb[li][c * 16 + (mi & 1) * 8]);
        const float* v[4] = {&acc[2 * c][0], &acc[2 * c][2], &acc[2 * c + 1][0], &acc[2 * c + 1][2]};
        uint32_t a3[3][4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float x0 = v[q][0], x1 = v[q][1];
          const float h0 = rbf(x0), h1 = rbf(x1), m0 = rbf(x0 - h0), m1 = rbf(x1 - h1);
          a3[0][q] = pack_bf16(h0, h1); a3[1][q] = pack_bf16(m0, m1); a3[2][q] = pack_bf16(x0 - h0 - m0, x1 - h1 - m1);
        }
#pragma unroll
        for (int q = 2; q >= 0; --q) mma16816(oc, a3[q], bfr);
      }
      // oc: rows p0 / p0 + 8, n = 2qd + {0,1} = r on lanes qd < 2 (rows 4..7 of sCb are zero)
      float ys[2] = {0.f, 0.f};
      if (qd < 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int p = p0 + e * 8;
          const float xp = bf2f(st.x[p]), zp = bf2f(st.z[p]);
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int r = 2 * qd + j;
            float orr = oc[e * 2 + j] + Dh * xp * S_.xproj[r * D + p];
            const float zv = zp * S_.zproj[r * D + p];
            orr *= __fdividef(zv, 1.f + __expf(-zv));
            ys[e] += orr * S_.outproj[r * D + p];
          }
        }
      }
      ys[0] += __shfl_xor_sync(0xffffffffu, ys[0], 1); ys[1] += __shfl_xor_sync(0xffffffffu, ys[1], 1);
      if (qd == 0) { a.y[((size_t)b * H + h) * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f); a.y[((size_t)b * H + h) * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f); }
#else
      // ---- h_new = alpha h + sum_k A[p][k] Bk[k][s] in fp32 over the mma fragment layout (rows p0, p0+8;
      // columns jt*8 + 2qd + {0,1}); then o[p][r] = sum_s h_new[p][s] C'[r][s], reduced over the 4 qd lanes
      float av[2][8];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int p = p0 + e * 8;
        const float xp = bf2f(st.x[p]), xs = bf2f(st.xst[p]);
#pragma unroll
        for (int r = 0; r < R; ++r) { const float xpj = S_.xproj[r * D + p]; av[e][r] = gamma * xp * xpj; av[e][4 + r] = beta * xs * xpj; }
      }
      float o[2][R];
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int r = 0; r < R; ++r) o[e][r] = 0.f;
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {
        const int n = jt * 8 + 2 * qd;
        float2 bk[8], cr[R];
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) bk[kk] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&S_.sBk[kk][n]));
#pragma unroll
        for (int r = 0; r < R; ++r) cr[r] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&S_.sCb[r][n]));
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float2* hp = reinterpret_cast<float2*>(st.state + (p0 + e * 8) * LDSF + n);
          float2 hv = *hp;
          hv.x *= alpha; hv.y *= alpha;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) { hv.x = fmaf(av[e][kk], bk[kk].x, hv.x); hv.y = fmaf(av[e][kk], bk[kk].y, hv.y); }
          *hp = hv;
#pragma unroll
          for (int r = 0; r < R; ++r) o[e][r] = fmaf(hv.x, cr[r].x, fmaf(hv.y, cr[r].y, o[e][r]));
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int r = 0; r < R; ++r) { o[e][r] += __shfl_xor_sync(0xffffffffu, o[e][r], 1); o[e][r] += __shfl_xor_sync(0xffffffffu, o[e][r], 2); }
      float ys[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {   // lane qd takes r = qd
        const int p = p0 + e * 8;
        const float xp = bf2f(st.x[p]), zp = bf2f(st.z[p]);
        float orr = o[e][0];
#pragma unroll
        for (int r = 1; r < R; ++r) orr = (qd == r) ? o[e][r] : orr;
        orr += Dh * xp * S_.xproj[qd * D + p];
        const float zv = zp * S_.zproj[qd * D + p];
        orr *= __fdividef(zv, 1.f + __expf(-zv));
        ys[e] = orr * S_.outproj[qd * D + p];
        ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 1); ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 2);
      }
      if (qd == 0) { a.y[((size_t)b * H + h) * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f); a.y[((size_t)b * H + h) * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f); }
#endif
    } else {
    // ---- state update on tensor cores: h_new = alpha h + A Bk, A[p][k] = (k<4: gamma x[p] xproj[k][p]; 4<=k<8: beta xstate[p] xproj[k-4][p])
    uint32_t af[4];
    {
      float av[2][2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int p = p0 + e * 8;
        const float xp = bf2f(st.x[p]), xs = bf2f(st.xst[p]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int k = 2 * qd + jj;
          const int r = k & 3;
          const float xpj = S_.xproj[r * D + p];
          av[e][jj] = (k < 4) ? gamma * xp * xpj : beta * xs * xpj;
        }
      }
      af[0] = pack_bf16(av[0][0], av[0][1]); af[1] = pack_bf16(av[1][0], av[1][1]); af[2] = 0u; af[3] = 0u;
    }
    float acc[16][4];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int n = jt * 8 + 2 * qd;
      float2 h0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.state + p0 * LDS + n));
      float2 h1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.state + (p0 + 8) * LDS + n));
      acc[jt][0] = alpha * h0.x; acc[jt][1] = alpha * h0.y; acc[jt][2] = alpha * h1.x; acc[jt][3] = alpha * h1.y;
    }
    {
      const int mi = lane >> 3, li = lane & 7;
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {   // two n8 tiles per ldmatrix.x4.trans: (k0-7,n8), (k8-15,n8), (k0-7,n8+1), (k8-15,n8+1)
        uint32_t bfr[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[(mi & 1) * 8 + li][jp * 16 + (mi >> 1) * 8])));
        mma16816(acc[2 * jp], af, bfr);
        mma16816(acc[2 * jp + 1], af, bfr + 2);
      }
    }
    // write h_new (bf16) back into the tile: it is the pool value and the A operand of the C dot
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int n = jt * 8 + 2 * qd;
      *reinterpret_cast<__nv_bfloat162*>(st.state + p0 * LDS + n) = __floats2bfloat162_rn(acc[jt][0], acc[jt][1]);
      *reinterpret_cast<__nv_bfloat162*>(st.state + (p0 + 8) * LDS + n) = __floats2bfloat162_rn(acc[jt][2], acc[jt][3]);
    }
    __syncwarp();
    // ---- o[p][r] = sum_s h_new[p][s] C'[r][s] on tensor cores: A = this warp's 16 state rows (bf16, smem), B = sCb rows
    {
      float oc[4] = {0.f, 0.f, 0.f, 0.f};
      const int mi = lane >> 3, li = lane & 7;
#pragma unroll
      for (int kk = 0; kk < S; kk += 16) {
        uint32_t af[4], bfr[4];
        ldmatrix_x4(af, st.state + (warp * 16 + (mi & 1) * 8 + li) * LDS + kk + (mi >> 1) * 8);
        // B operand (n = r rows 0..7, k = s): x4 covers k halves kk..kk+7 and kk+8..kk+15 for the 8 n rows (+ next k16 unused half)
        ldmatrix_x4(bfr, &S_.sCb[(mi & 1) * 0 + li][kk + (mi & 1) * 8 + (mi >> 1) * 0]);
        mma16816(oc, af, bfr);
      }
      // oc: (row p0 = warp*16 + lane/4, n = 2(lane%4) + {0,1}) and (p0 + 8, same n); n = r for lane%4 < 2, zero otherwise
      float ys[2] = {0.f, 0.f};
      if (qd < 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {          // e: row p0 (c0,c1) or p0+8 (c2,c3)
          const int p = p0 + e * 8;
          const float xp = bf2f(st.x[p]), zp = bf2f(st.z[p]);
#pragma unroll
          for (int j = 0; j < 2; ++j) {        // n = 2qd + j = r
            const int r = 2 * qd + j;
            float orr = oc[e * 2 + j] + Dh * xp * S_.xproj[r * D + p];
            const float zv = zp * S_.zproj[r * D + p];
            orr *= __fdividef(zv, 1.f + __expf(-zv));
            ys[e] += orr * S_.outproj[r * D + p];
          }
        }
      }
      ys[0] += __shfl_xor_sync(0xffffffffu, ys[0], 1); ys[1] += __shfl_xor_sync(0xffffffffu, ys[1], 1);
      if (qd == 0) { a.y[((size_t)b * H + h) * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f); a.y[((size_t)b * H + h) * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f); }
    }
    }
    __syncthreads();   // state tile fully updated in smem
#ifdef V_NOSTORE
    if (false) {
#else
    if (valid) {
#endif
      // coalesced store of the state tile (16B per thread per chunk)
      if constexpr (F32) {
        float* dst = reinterpret_cast<float*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
        for (int i = 0; i < 16; ++i) { const int c = t + i * THREADS, row = c >> 5, ch = c & 31; *reinterpret_cast<uint4*>(dst + c * 4) = *reinterpret_cast<const uint4*>(st.state + row * LDSF + ch * 4); }
      } else {
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
        for (int i = 0; i < 8; ++i) { const int c = t + i * THREADS, row = c >> 4, ch = c & 15; *reinterpret_cast<uint4*>(dst + c * 8) = *reinterpret_cast<const uint4*>(st.state + row * LDS + ch * 8); }
      }
      if (t < 64) { const int r = t >> 4, c = t & 15; *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) = *reinterpret_cast<const uint4*>(&S_.sBk[r][c * 8]); }
      if (t < D) a.vp[(size_t)slot * a.vp_sst + (size_t)h * D + t] = st.x[t];
      if (t < NA) a.angle[(size_t)slot * a.ang_sst + (size_t)h * NA + t] = S_.sTheta[t];
    }
    __syncthreads();   // stage reusable by the prefetch of item + 2*grid
  }
}
// ---------------------------------------------------------------------------------------------------------------------
// fp32-state step, pipelined edition. The 1-stage kernel above serializes load -> compute -> store per CTA, so the
// DRAM pipe idles while a CTA computes (and the CTA idles while its 32 KB tile streams in): measured ~3.6 TB/s of
// reads alone. Here the state tile is consumed into the mma accumulators (seeded by alpha h) right after it lands,
// which frees the smem tile at once: the next item's state is prefetched while this item computes, the small token
// rows are double-buffered, and h_new goes straight from the accumulators to the pool (full 32 B sectors per row
// segment), with no smem round trip and no store loop. Same math, same order: bit-identical to the kernel above.
constexpr int LDP = S + 8;   // padded fp32 tile row (conflict-free float2 fragment reads); 4 CTAs/SM still fit
struct __align__(128) SmallStage {
  __nv_bfloat16 bst[R * S];   // previous B' (k_pool)
  __nv_bfloat16 bm[R * S];    // token B rows
  __nv_bfloat16 cm[R * S];    // token C rows
  __nv_bfloat16 x[D], z[D], xst[D];
  __nv_bfloat16 angp[NA];
  float angst[NA];
};
struct __align__(128) SmemP {
  float state[D * LDP];
  SmallStage sm[2];
  float bias_k[R * S], bias_q[R * S];
  float xproj[R * D], zproj[R * D], outproj[R * D];
  float sCos[NA], sSin[NA];
  __nv_bfloat16 sBk[16][LDB];
  __nv_bfloat16 sCb[8][LDB];
  float sTheta[NA];
};

// x = hi + mid + lo (bf16 each), bit-identical to the rbf-based split of the kernel above, two values per cvt
__device__ __forceinline__ void split3_x2(float x0, float x1, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
  hi = pack_bf16(x0, x1);
  const float r0 = x0 - __uint_as_float(hi << 16), r1 = x1 - __uint_as_float(hi & 0xffff0000u);
  mid = pack_bf16(r0, r1);
  lo = pack_bf16(r0 - __uint_as_float(mid << 16), r1 - __uint_as_float(mid & 0xffff0000u));
}

__device__ __forceinline__ void pipe_issue(const Args& a, SmemP& S_, SmallStage& st, int item, int t) {
  const int h = item % H, b = item / H;
  const int slot = a.slots[b]; const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  const float* src = reinterpret_cast<const float*>(a.ssm) + (size_t)srow * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
  for (int i = 0; i < 16; ++i) { const int c = t + i * THREADS, row = c >> 5, ch = c & 31; cp16(S_.state + row * LDP + ch * 4, src + c * 4); }
  if (t < 64) {
    const int r = t >> 4, c = t & 15;
    cp16(st.bst + r * S + c * 8, a.kp + (size_t)srow * a.kp_sst + ((size_t)r * H + h) * S + c * 8);
    cp16(st.bm + t * 8, a.Bm + (size_t)b * a.bc_bstride + t * 8);
    cp16(st.cm + t * 8, a.Cm + (size_t)b * a.bc_bstride + t * 8);
  } else {
    const int q = t - 64;
    if (q < 8) cp16(st.x + q * 8, a.x + ((size_t)b * H + h) * D + q * 8);
    else if (q < 16) cp16(st.z + (q - 8) * 8, a.z + ((size_t)b * H + h) * D + (q - 8) * 8);
    else if (q < 24) cp16(st.xst + (q - 16) * 8, a.vp + (size_t)srow * a.vp_sst + (size_t)h * D + (q - 16) * 8);
    else if (q < 28) cp16(st.angp + (q - 24) * 8, a.angp + (size_t)b * a.angp_stride + (q - 24) * 8);
    else if (q < 36) cp16(st.angst + (q - 28) * 4, a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA + (q - 28) * 4);
  }
}

__global__ void __launch_bounds__(THREADS, 4) mamba3_step_f32_pipe_kernel(const Args a) {
  extern __shared__ __align__(128) char smem_raw[];
  SmemP& S_ = *reinterpret_cast<SmemP*>(smem_raw);
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int h = blockIdx.x % H, nb_stride = gridDim.x / H;
  for (int i = t; i < 8 * LDB; i += THREADS) { S_.sBk[8 + i / LDB][i % LDB] = __float2bfloat16(0.f); if (i < 4 * LDB) S_.sCb[4 + i / LDB][i % LDB] = __float2bfloat16(0.f); }
  { const int r = t >> 5, c = t & 31;
    cp16(S_.bias_k + r * S + c * 4, a.bias_k + ((size_t)r * H + h) * S + c * 4);
    cp16(S_.bias_q + r * S + c * 4, a.bias_q + ((size_t)r * H + h) * S + c * 4); }
  if (t < 64) { const int r = t >> 4, c = t & 15; cp16(S_.xproj + r * D + c * 4, a.xproj + ((size_t)r * H + h) * D + c * 4); cp16(S_.outproj + r * D + c * 4, a.outproj + ((size_t)r * H + h) * D + c * 4); }
  else { const int q = t - 64, r = q >> 4, c = q & 15; cp16(S_.zproj + r * D + c * 4, a.zproj + ((size_t)r * H + h) * D + c * 4); }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (a.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // PDL: sm_90+ only (pdl is 0 below)
#endif
  const int b0 = blockIdx.x / H;
  auto item_of = [&](int k) { return (b0 + k * nb_stride) * H + h; };
  const int nk = (a.B - b0 + nb_stride - 1) / nb_stride;
  if (b0 >= a.B) return;
  pipe_issue(a, S_, S_.sm[0], item_of(0), t); cp_commit();
  const int qd = lane & 3, p0 = warp * 16 + (lane >> 2);
  for (int k = 0; k < nk; ++k) {
    const int item = item_of(k), b = item / H;
    const int slot = a.slots[b]; const bool valid = slot >= 0 && slot < a.P;
    const float dt = a.dt[b * H + h], A = a.A[b * H + h], trap = a.trap[b * H + h];
    const float Dh = a.Dp[h];
    const float alpha = __expf(A * dt), gamma = trap * dt, beta = (1.f - trap) * dt * alpha;
    cp_wait<0>();
    __syncthreads();
    SmallStage& st = S_.sm[k & 1];
    // consume the state tile into the accumulators (h_new = alpha h + ...), then hand the tile to the prefetch
    float acc[16][4];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int n = jt * 8 + 2 * qd;
      const float2 h0 = *reinterpret_cast<const float2*>(S_.state + p0 * LDP + n);
      const float2 h1 = *reinterpret_cast<const float2*>(S_.state + (p0 + 8) * LDP + n);
      acc[jt][0] = alpha * h0.x; acc[jt][1] = alpha * h0.y; acc[jt][2] = alpha * h1.x; acc[jt][3] = alpha * h1.y;
    }
    if (t < NA) { float th = st.angst[t] + tanhf(bf2f(st.angp[t])) * dt * 3.14159265358979323846f; th -= 6.283185307179586f * floorf(th / 6.283185307179586f); S_.sTheta[t] = th; sincosf(th, &S_.sSin[t], &S_.sCos[t]); }
    __syncthreads();   // tile consumed by every warp; theta ready
    if (k + 1 < nk) pipe_issue(a, S_, S_.sm[(k + 1) & 1], item_of(k + 1), t);
    cp_commit();
    for (int i = t; i < R * (S / 2); i += THREADS) {
      const int r = i / (S / 2), j = i % (S / 2);
      float blo = bf2f(st.bm[r * S + j]) + S_.bias_k[r * S + j], bhi = bf2f(st.bm[r * S + j + 64]) + S_.bias_k[r * S + j + 64];
      float clo = bf2f(st.cm[r * S + j]) + S_.bias_q[r * S + j], chi = bf2f(st.cm[r * S + j + 64]) + S_.bias_q[r * S + j + 64];
      if (j < NA) {
        const float c = S_.sCos[j], s_ = S_.sSin[j];
        const float b0_ = blo * c - bhi * s_, b1 = blo * s_ + bhi * c, c0 = clo * c - chi * s_, c1 = clo * s_ + chi * c;
        blo = b0_; bhi = b1; clo = c0; chi = c1;
      }
      S_.sBk[r][j] = __float2bfloat16(blo); S_.sBk[r][j + 64] = __float2bfloat16(bhi);
      S_.sBk[4 + r][j] = st.bst[r * S + j]; S_.sBk[4 + r][j + 64] = st.bst[r * S + j + 64];
      S_.sCb[r][j] = __float2bfloat16(clo); S_.sCb[r][j + 64] = __float2bfloat16(chi);
    }
    __syncthreads();
    uint32_t ah[3][4];
    {
      float av[2][2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int p = p0 + e * 8;
        const float xp = bf2f(st.x[p]), xs = bf2f(st.xst[p]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int kk = 2 * qd + jj, r = kk & 3;
          const float xpj = S_.xproj[r * D + p];
          av[e][jj] = (kk < 4) ? gamma * xp * xpj : beta * xs * xpj;
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) split3_x2(av[e][0], av[e][1], ah[0][e], ah[1][e], ah[2][e]);
#pragma unroll
      for (int q = 0; q < 3; ++q) { ah[q][2] = 0u; ah[q][3] = 0u; }
    }
    const int mi = lane >> 3, li = lane & 7;
#pragma unroll
    for (int jp = 0; jp < 8; ++jp) {
      uint32_t bfr[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[(mi & 1) * 8 + li][jp * 16 + (mi >> 1) * 8])));
#pragma unroll
      for (int q = 2; q >= 0; --q) { mma16816(acc[2 * jp], ah[q], bfr); mma16816(acc[2 * jp + 1], ah[q], bfr + 2); }
    }
    if (valid) {   // h_new straight to the pool: per warp instruction 8 rows x 32 contiguous bytes
      float* dst = reinterpret_cast<float*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {
        const int n = jt * 8 + 2 * qd;
        *reinterpret_cast<float2*>(dst + p0 * S + n) = make_float2(acc[jt][0], acc[jt][1]);
        *reinterpret_cast<float2*>(dst + (p0 + 8) * S + n) = make_float2(acc[jt][2], acc[jt][3]);
      }
    }
    float oc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint32_t bfr[4];
      ldmatrix_x4(bfr, &S_.sCb[li][c * 16 + (mi & 1) * 8]);
      const float* v[4] = {&acc[2 * c][0], &acc[2 * c][2], &acc[2 * c + 1][0], &acc[2 * c + 1][2]};
      uint32_t a3[3][4];
#pragma unroll
      for (int q = 0; q < 4; ++q) split3_x2(v[q][0], v[q][1], a3[0][q], a3[1][q], a3[2][q]);
#pragma unroll
      for (int q = 2; q >= 0; --q) mma16816(oc, a3[q], bfr);
    }
    float ys[2] = {0.f, 0.f};
    if (qd < 2) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int p = p0 + e * 8;
        const float xp = bf2f(st.x[p]), zp = bf2f(st.z[p]);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int r = 2 * qd + j;
          float orr = oc[e * 2 + j] + Dh * xp * S_.xproj[r * D + p];
          const float zv = zp * S_.zproj[r * D + p];
          orr *= __fdividef(zv, 1.f + __expf(-zv));
          ys[e] += orr * S_.outproj[r * D + p];
        }
      }
    }
    ys[0] += __shfl_xor_sync(0xffffffffu, ys[0], 1); ys[1] += __shfl_xor_sync(0xffffffffu, ys[1], 1);
    if (qd == 0) { a.y[((size_t)b * H + h) * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f); a.y[((size_t)b * H + h) * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f); }
    if (valid) {
      if (t < 64) { const int r = t >> 4, c = t & 15; *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) = *reinterpret_cast<const uint4*>(&S_.sBk[r][c * 8]); }
      if (t < D) a.vp[(size_t)slot * a.vp_sst + (size_t)h * D + t] = st.x[t];
      if (t < NA) a.angle[(size_t)slot * a.ang_sst + (size_t)h * NA + t] = S_.sTheta[t];
    }
    __syncthreads();   // sBk / sCb / sTheta / this small stage are reused by the next item
  }
}

#ifdef STEP_VOLTA
// ---------------------------------------------------------------------------------------------------------------------
// fp32-state step, Volta edition (STEP_VOLTA builds). sm_70 has no cp.async: through volta-forever each cp.async of the
// 1-stage kernel is a synchronous load + shared store, so a CTA meets every item's 32 KB state and token rows one copy
// after the other, then computes while the DRAM pipe idles (213 us per layer at 32 sequences against a ~116 us floor).
// Here the state never goes through shared memory: each thread loads the 64 state elements it updates (rows p0 and
// p0 + 8, columns jt * 8 + 2 qd + {0, 1}: the 1-stage kernel's FMA fragment layout) straight into registers, all in
// flight at once with the item's token / pool rows and head constants (registers, then a shared stage) and scalars,
// and stores h_new from the same registers (per warp instruction 8 rows x 32 contiguous bytes). The small tile fits
// several CTAs per SM, whose loads overlap one another's computation (a one-item register prefetch at fewer CTAs per
// SM measured slower). Items (token, head) are dealt to the CTAs round-robin, so any grid balances. The arithmetic is
// the 1-stage kernel's STEP_F32_FMA code, operation for operation and in the same order, on the same values: the
// outputs, the state and the k / v / angle pools are bit-identical.
struct __align__(16) VStage {
  __nv_bfloat16 bst[R * S];   // previous B' (k_pool)
  __nv_bfloat16 bm[R * S];    // token B rows
  __nv_bfloat16 cm[R * S];    // token C rows
  __nv_bfloat16 x[D], z[D], xst[D];
  __nv_bfloat16 angp[NA];
  float angst[NA];
  float xproj[R * D], zproj[R * D], outproj[R * D];   // the item's head
};
struct __align__(16) VSmem {
  VStage st;
  float sCos[NA], sSin[NA], sTheta[NA];
  __nv_bfloat16 sBk[8][LDB];   // rows 0..3 B'[r] rotated (bf16), 4..7 Bstate[r]
  __nv_bfloat16 sCb[4][LDB];   // rotated C' (bf16)
};

// One item's global loads into registers: the thread's state fragment, its share of the token / pool rows (the
// 1-stage kernel's issue_item chunks: threads 0..63 the k_pool / B / C chunks, 64..99 x, z, xst, angp, angst) and of
// the head constants (threads 0..63 xproj + outproj, 64..127 zproj), and the item's scalars.
struct VLoad {
  float2 h[2][16];
  uint4 sm[3];
  float4 c[2];
  float dt, A, trap;
};
__device__ __forceinline__ void v_issue(const Args& a, const int item, const int slot, const int t, const int p0,
                                        const int qd, VLoad& L) {
  const int h = item % H, b = item / H;
  const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  const float* src = reinterpret_cast<const float*>(a.ssm) + (size_t)srow * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) L.h[e][jt] = *reinterpret_cast<const float2*>(src + (p0 + e * 8) * S + jt * 8 + 2 * qd);
  if (t < 64) {
    const int r = t >> 4, c = t & 15;
    L.sm[0] = *reinterpret_cast<const uint4*>(a.kp + (size_t)srow * a.kp_sst + ((size_t)r * H + h) * S + c * 8);
    L.sm[1] = *reinterpret_cast<const uint4*>(a.Bm + (size_t)b * a.bc_bstride + t * 8);
    L.sm[2] = *reinterpret_cast<const uint4*>(a.Cm + (size_t)b * a.bc_bstride + t * 8);
    L.c[0] = __ldg(reinterpret_cast<const float4*>(a.xproj + ((size_t)r * H + h) * D + c * 4));
    L.c[1] = __ldg(reinterpret_cast<const float4*>(a.outproj + ((size_t)r * H + h) * D + c * 4));
  } else {
    const int q = t - 64;
    if (q < 8) L.sm[0] = *reinterpret_cast<const uint4*>(a.x + ((size_t)b * H + h) * D + q * 8);
    else if (q < 16) L.sm[0] = *reinterpret_cast<const uint4*>(a.z + ((size_t)b * H + h) * D + (q - 8) * 8);
    else if (q < 24) L.sm[0] = *reinterpret_cast<const uint4*>(a.vp + (size_t)srow * a.vp_sst + (size_t)h * D + (q - 16) * 8);
    else if (q < 28) L.sm[0] = *reinterpret_cast<const uint4*>(a.angp + (size_t)b * a.angp_stride + (q - 24) * 8);
    else if (q < 36) L.sm[0] = *reinterpret_cast<const uint4*>(a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA + (q - 28) * 4);
    const int r = q >> 4, c = q & 15;
    L.c[0] = __ldg(reinterpret_cast<const float4*>(a.zproj + ((size_t)r * H + h) * D + c * 4));
  }
  L.dt = a.dt[b * H + h]; L.A = a.A[b * H + h]; L.trap = a.trap[b * H + h];
}

__device__ __forceinline__ void v_stage(VStage& st, const VLoad& L, const int t) {
  if (t < 64) {
    const int r = t >> 4, c = t & 15;
    *reinterpret_cast<uint4*>(st.bst + r * S + c * 8) = L.sm[0];
    *reinterpret_cast<uint4*>(st.bm + t * 8) = L.sm[1];
    *reinterpret_cast<uint4*>(st.cm + t * 8) = L.sm[2];
    *reinterpret_cast<float4*>(st.xproj + r * D + c * 4) = L.c[0];
    *reinterpret_cast<float4*>(st.outproj + r * D + c * 4) = L.c[1];
  } else {
    const int q = t - 64;
    if (q < 8) *reinterpret_cast<uint4*>(st.x + q * 8) = L.sm[0];
    else if (q < 16) *reinterpret_cast<uint4*>(st.z + (q - 8) * 8) = L.sm[0];
    else if (q < 24) *reinterpret_cast<uint4*>(st.xst + (q - 16) * 8) = L.sm[0];
    else if (q < 28) *reinterpret_cast<uint4*>(st.angp + (q - 24) * 8) = L.sm[0];
    else if (q < 36) *reinterpret_cast<uint4*>(st.angst + (q - 28) * 4) = L.sm[0];
    const int r = q >> 4, c = q & 15;
    *reinterpret_cast<float4*>(st.zproj + r * D + c * 4) = L.c[0];
  }
}

__global__ void __launch_bounds__(THREADS, 3) mamba3_step_volta_kernel(const Args a) {
  __shared__ VSmem S_;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int qd = lane & 3, p0 = warp * 16 + (lane >> 2);
  const int nitems = a.B * H, stride = gridDim.x;   // items b * H + h, dealt round-robin
  int item = blockIdx.x;
  if (item >= nitems) return;
  VStage& st = S_.st;
  VLoad cur;
  int slotc = a.slots[item / H];
  v_issue(a, item, slotc, t, p0, qd, cur);
  v_stage(st, cur, t);
  int slotn = item + stride < nitems ? a.slots[(item + stride) / H] : 0;   // the next item's slot, one item ahead
  __syncthreads();
  for (; item < nitems; item += stride) {
    const int b = item / H, h = item % H;
    auto bias_k = [&](int r, int j) -> float { return __ldg(a.bias_k + ((size_t)r * H + h) * S + j); };
    auto bias_q = [&](int r, int j) -> float { return __ldg(a.bias_q + ((size_t)r * H + h) * S + j); };
    const int slot = slotc; const bool valid = slot >= 0 && slot < a.P;
    const float dt = cur.dt, A = cur.A, trap = cur.trap;
    const float Dh = a.Dp[h];
    const float alpha = __expf(A * dt), gamma = trap * dt, beta = (1.f - trap) * dt * alpha;
    // ---- preamble: angles, bias + rotary on B/C (bf16-rounded)
    if (t < NA) { float th = st.angst[t] + tanhf(bf2f(st.angp[t])) * dt * 3.14159265358979323846f; th -= 6.283185307179586f * floorf(th / 6.283185307179586f); S_.sTheta[t] = th; sincosf(th, &S_.sSin[t], &S_.sCos[t]); }
    __syncthreads();
    for (int i = t; i < R * (S / 2); i += THREADS) {
      const int r = i / (S / 2), j = i % (S / 2);
      float blo = bf2f(st.bm[r * S + j]) + bias_k(r, j), bhi = bf2f(st.bm[r * S + j + 64]) + bias_k(r, j + 64);
      float clo = bf2f(st.cm[r * S + j]) + bias_q(r, j), chi = bf2f(st.cm[r * S + j + 64]) + bias_q(r, j + 64);
      if (j < NA) {
        const float c = S_.sCos[j], s_ = S_.sSin[j];
        const float b0 = blo * c - bhi * s_, b1 = blo * s_ + bhi * c, c0 = clo * c - chi * s_, c1 = clo * s_ + chi * c;
        blo = b0; bhi = b1; clo = c0; chi = c1;
      }
      S_.sBk[r][j] = __float2bfloat16(blo); S_.sBk[r][j + 64] = __float2bfloat16(bhi);
      S_.sBk[4 + r][j] = st.bst[r * S + j]; S_.sBk[4 + r][j + 64] = st.bst[r * S + j + 64];
      S_.sCb[r][j] = __float2bfloat16(clo); S_.sCb[r][j + 64] = __float2bfloat16(chi);
    }
    __syncthreads();
    // ---- h_new = alpha h + sum_k A[p][k] Bk[k][s] in fp32 (rows p0, p0+8; columns jt*8 + 2qd + {0,1}); then
    // o[p][r] = sum_s h_new[p][s] C'[r][s], reduced over the 4 qd lanes (the 1-stage kernel's STEP_F32_FMA code)
    float av[2][8];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int p = p0 + e * 8;
      const float xp = bf2f(st.x[p]), xs = bf2f(st.xst[p]);
#pragma unroll
      for (int r = 0; r < R; ++r) { const float xpj = st.xproj[r * D + p]; av[e][r] = gamma * xp * xpj; av[e][4 + r] = beta * xs * xpj; }
    }
    float o[2][R];
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int r = 0; r < R; ++r) o[e][r] = 0.f;
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int n = jt * 8 + 2 * qd;
      float2 bk[8], cr[R];
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) bk[kk] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&S_.sBk[kk][n]));
#pragma unroll
      for (int r = 0; r < R; ++r) cr[r] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&S_.sCb[r][n]));
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float2 hv = cur.h[e][jt];
        hv.x *= alpha; hv.y *= alpha;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) { hv.x = fmaf(av[e][kk], bk[kk].x, hv.x); hv.y = fmaf(av[e][kk], bk[kk].y, hv.y); }
        cur.h[e][jt] = hv;
#pragma unroll
        for (int r = 0; r < R; ++r) o[e][r] = fmaf(hv.x, cr[r].x, fmaf(hv.y, cr[r].y, o[e][r]));
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int r = 0; r < R; ++r) { o[e][r] += __shfl_xor_sync(0xffffffffu, o[e][r], 1); o[e][r] += __shfl_xor_sync(0xffffffffu, o[e][r], 2); }
    float ys[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {   // lane qd takes r = qd
      const int p = p0 + e * 8;
      const float xp = bf2f(st.x[p]), zp = bf2f(st.z[p]);
      float orr = o[e][0];
#pragma unroll
      for (int r = 1; r < R; ++r) orr = (qd == r) ? o[e][r] : orr;
      orr += Dh * xp * st.xproj[qd * D + p];
      const float zv = zp * st.zproj[qd * D + p];
      orr *= __fdividef(zv, 1.f + __expf(-zv));
      ys[e] = orr * st.outproj[qd * D + p];
      ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 1); ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 2);
    }
    if (qd == 0) { a.y[((size_t)b * H + h) * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f); a.y[((size_t)b * H + h) * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f); }
    if (valid) {   // h_new straight from the registers to the pool, then B' / x / theta
      float* dst = reinterpret_cast<float*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) *reinterpret_cast<float2*>(dst + (p0 + e * 8) * S + jt * 8 + 2 * qd) = cur.h[e][jt];
      if (t < 64) { const int r = t >> 4, c = t & 15; *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) = *reinterpret_cast<const uint4*>(&S_.sBk[r][c * 8]); }
      if (t < D) a.vp[(size_t)slot * a.vp_sst + (size_t)h * D + t] = st.x[t];
      if (t < NA) a.angle[(size_t)slot * a.ang_sst + (size_t)h * NA + t] = S_.sTheta[t];
    }
    __syncthreads();   // sBk / sCb / sTheta and the stage are free for the next item
    if (item + stride < nitems) {
      slotc = slotn;
      if (item + 2 * stride < nitems) slotn = a.slots[(item + 2 * stride) / H];
      v_issue(a, item + stride, slotc, t, p0, qd, cur);
      v_stage(st, cur, t);
      __syncthreads();
    }
  }
}
#endif  // STEP_VOLTA
}  // namespace

void mamba3_step_cuda(torch::Tensor ssm_pool, torch::Tensor k_pool, torch::Tensor v_pool, torch::Tensor angle_pool,
                    torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dp, torch::Tensor x, torch::Tensor dt, torch::Tensor trap,
                    torch::Tensor xproj, torch::Tensor zproj, torch::Tensor outproj, torch::Tensor z, torch::Tensor bias_q, torch::Tensor bias_k,
                    torch::Tensor angle_proj, torch::Tensor slots, torch::Tensor y, int64_t ctas_per_sm,
                    std::optional<torch::Tensor> raw_zxdt, std::optional<torch::Tensor> raw_dt_bias, double a_floor,
                    std::optional<torch::Tensor> raw_wb, std::optional<torch::Tensor> raw_wc, double eps, bool pdl) {
  Args a;
  a.ssm = ssm_pool.data_ptr(); a.kp = reinterpret_cast<__nv_bfloat16*>(k_pool.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr());
  a.angle = angle_pool.data_ptr<float>(); a.P = ssm_pool.size(0); a.A = A.data_ptr<float>(); a.Bm = reinterpret_cast<const __nv_bfloat16*>(Bm.data_ptr()); a.Cm = reinterpret_cast<const __nv_bfloat16*>(Cm.data_ptr());
  a.Dp = Dp.data_ptr<float>(); a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); a.dt = dt.data_ptr<float>(); a.trap = trap.data_ptr<float>();
  a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>(); a.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr());
  a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>(); a.angp = reinterpret_cast<const __nv_bfloat16*>(angle_proj.data_ptr()); a.slots = slots.data_ptr<int>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr()); a.B = x.size(0);
  a.bc_bstride = Bm.stride(0); a.angp_stride = angle_proj.stride(0);
  a.ssm_sst = ssm_pool.stride(0); a.kp_sst = k_pool.stride(0); a.vp_sst = v_pool.stride(0); a.ang_sst = angle_pool.stride(0);
  TORCH_CHECK(ssm_pool.stride(1) == D * S && ssm_pool.stride(2) == S && ssm_pool.stride(3) == 1, "ssm_pool inner dims must be contiguous");
  TORCH_CHECK(k_pool.stride(1) == H * S && k_pool.stride(2) == S && k_pool.stride(3) == 1, "k_pool inner dims must be contiguous");
  TORCH_CHECK(v_pool.stride(1) == D && v_pool.stride(2) == 1 && angle_pool.stride(1) == NA && angle_pool.stride(2) == 1);
  const bool f32 = ssm_pool.dtype() == torch::kFloat32;
  TORCH_CHECK((a.ssm_sst * ssm_pool.element_size()) % 16 == 0 && (a.kp_sst * 2) % 16 == 0 && (a.vp_sst * 2) % 16 == 0 && (a.ang_sst * 4) % 16 == 0, "pool slot strides must be 16B aligned");
  TORCH_CHECK(Bm.stride(1) == S && Bm.stride(2) == 1 && Cm.stride(0) == Bm.stride(0) && Cm.stride(1) == S && angle_proj.stride(1) == 1);
  TORCH_CHECK((Bm.stride(0) * 2) % 16 == 0 && (angle_proj.stride(0) * 2) % 16 == 0, "B/C batch stride and angle_proj row stride must be 16B aligned");
  TORCH_CHECK((f32 || ssm_pool.dtype() == torch::kBFloat16) && k_pool.dtype() == torch::kBFloat16 && v_pool.dtype() == torch::kBFloat16 && angle_pool.dtype() == torch::kFloat32);
  a.pdl = pdl ? 1 : 0;
  a.raw = 0; a.zxdt = nullptr; a.zxdt_stride = 0; a.dt_bias = nullptr; a.a_floor = 0.f; a.wb = a.wc = nullptr; a.eps = 0.f;
  if (raw_zxdt.has_value()) {
    const auto& zx = *raw_zxdt;
    TORCH_CHECK(zx.stride(1) == 1 && raw_dt_bias->dtype() == torch::kFloat32 && raw_wb->dtype() == torch::kFloat32 && raw_wc->dtype() == torch::kFloat32);
    a.raw = 1; a.zxdt = reinterpret_cast<const __nv_bfloat16*>(zx.data_ptr()); a.zxdt_stride = zx.stride(0);
    a.dt_bias = raw_dt_bias->data_ptr<float>(); a.a_floor = (float)a_floor; a.wb = raw_wb->data_ptr<float>(); a.wc = raw_wc->data_ptr<float>(); a.eps = (float)eps;
    a.B = zx.size(0);
  }
  auto launch = [&](auto kernel, int smem) {
    // once per kernel (the pipelined and 1-stage fp32 kernels share F32)
    static const void* done[4] = {}; bool seen = false;
    for (auto& d : done) { if (d == (const void*)kernel) { seen = true; break; } if (!d) { d = (const void*)kernel; break; } }
    if (!seen) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int nsm = 132; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    int fit = 1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, THREADS, smem);
    const int cps = std::max(1, std::min<int>((int)ctas_per_sm, fit));
    const int items = a.B * H; int grid = std::min<int>(items, nsm * cps); grid = std::max(H, (grid / H) * H);
    if (a.pdl) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(grid); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = smem; cfg.stream = c10::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr; cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, a);
    } else {
      kernel<<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
    }
  };
  // fp32 state, preamble done upstream: the pipelined kernel (OLALA_STEP_PIPE=0 keeps the 1-stage one); on Volta its
  // FMA edition mamba3_step_volta_kernel (the tensor-core pipelined kernel is emulated there); else (raw rows, bf16
  // state) the 1-stage kernel, with the FMA update on Volta
#ifdef STEP_VOLTA
  constexpr bool volta = true;
#else
  constexpr bool volta = false;
#endif
  static const bool pipe = [] { const char* e = getenv("OLALA_STEP_PIPE"); return !(e && e[0] == '0'); }();
#if defined(STEP_VOLTA) && !defined(STEP_VOLTA_1STAGE)   // STEP_VOLTA_1STAGE: tests build the 1-stage kernel
  if (f32 && !a.raw && pipe) {   // items dealt round-robin: the grid need not be a multiple of H
    int nsm = 80; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    int fit = 1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, mamba3_step_volta_kernel, THREADS, 0);
    const int cps = std::max(1, std::min<int>((int)ctas_per_sm, fit));
    const int grid = std::max(1, std::min(a.B * H, nsm * cps));
    mamba3_step_volta_kernel<<<grid, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
    return;
  }
#endif
  if (f32 && !a.raw && pipe && !volta) launch(mamba3_step_f32_pipe_kernel, (int)sizeof(SmemP));
  else if (f32) launch(mamba3_step_cuda_kernel<true>, (int)sizeof(Smem<true>));
  else launch(mamba3_step_cuda_kernel<false>, (int)sizeof(Smem<false>));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &mamba3_step_cuda, py::arg("ssm"), py::arg("kp"), py::arg("vp"), py::arg("angle"), py::arg("A"), py::arg("Bm"), py::arg("Cm"), py::arg("Dp"), py::arg("x"), py::arg("dt"), py::arg("trap"), py::arg("xproj"), py::arg("zproj"), py::arg("outproj"), py::arg("z"), py::arg("bias_q"), py::arg("bias_k"), py::arg("angle_proj"), py::arg("slots"), py::arg("y"), py::arg("ctas_per_sm"), py::arg("raw_zxdt") = py::none(), py::arg("raw_dt_bias") = py::none(), py::arg("a_floor") = 0.0, py::arg("raw_wb") = py::none(), py::arg("raw_wc") = py::none(), py::arg("eps") = 0.0, py::arg("pdl") = false); }

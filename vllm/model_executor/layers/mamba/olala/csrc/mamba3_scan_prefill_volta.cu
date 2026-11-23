// Olala Mamba-3 MIMO prefill, output pass, Volta (V100) edition: plain fp32 FMA, no tensor cores, no ldmatrix.
// Same arguments, math and bf16 roundings as mamba3_scan_prefill_sm80.cu (see there for the formulas). Built JIT
// by mimo_ssd.py on the SM70 family: there sm_70 has no bf16 mma.sync / ldmatrix, which volta-forever emulates
// with scalar code (~200 instructions per m16n8k16, 64 of them FMAs), while this kernel's FMA code runs natively.
//
// Every product of the sm80 kernel is kept: the bf16 operands are rounded exactly as there, then stored in shared
// memory as fp32 (a bf16 times a bf16 is exact in fp32), and each output is one thread's chain of fmaf over k in
// increasing k, starting from the same accumulator. That is how volta-forever evaluates an emulated m16n8k16
// (fma.rn.f32, one term after the other, in the order of k), so under volta-forever this kernel returns the sm80
// kernel's bits; on other GPUs (whose tensor cores add differently) it is rounding-level close to it.
//
// One CTA (8 warps, ~92 KB of shared memory) per (128-token chunk, head), 8-token sub-blocks (32 MIMO rows,
// row = token * R + rank). Per sub-block:
//   prep    thread (token tid >> 5, rank (tid >> 3) & 3, column chunk tid & 7), as the sm80 kernel: qhat, khat * ts * e,
//           PsiV, gate, from registers loaded one sub-block ahead;
//   phase 1 warp w, lane l: o rows of token w (its 4 ranks) x p cols 2 l, 2 l + 1 over n = 0..127, and G rows of
//           that token x j col l;
//   phase 2 o += G @ PsiV (j = 0..31), D skip, gate, sum over the ranks in the thread; the state (fp32 registers,
//           thread: n rows 8 (tid >> 4) .. + 7, p cols 4 (tid & 15) .. + 3) S = dec S + k'^T @ PsiV (j = 0..31), and
//           its bf16 copy for the next sub-block's qhat @ S.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32, C = 8, ROWS = C * R;
constexpr int THREADS = 256;
constexpr int LDQ = N + 4;      // fp32 qhat / k' rows: 4-word bank shift per row
constexpr int LDG = ROWS + 4;   // fp32 G rows
constexpr float LOG_EMIN = 80.f;

struct Smem {
  float q[ROWS * LDQ];     // qhat [row][n]
  float k[ROWS * LDQ];     // khat * ts * e [row][n]
  float sb[N * P];         // state, bf16-rounded [n][p]
  float pv[ROWS * P];      // PsiV [row][p]
  float gz[ROWS * P];      // phi_r * silu(z * zeta_r) [row][p]
  float g[ROWS * LDG];     // masked G [row i][row j], bf16-rounded
  float tok[C][4];         // da_cs, da_cs_rev, ts, gamma per token
  __nv_bfloat16 qb[R * N], kb[R * N], psi[R * P];
  float phi[R * P], zeta[R * P];
};

union U8 {
  uint4 u;
  __nv_bfloat162 h2[4];
};

struct Raw {
  uint4 q[2], k[2];  // rank (tid >> 3) & 3 of token tid >> 5, columns 8 j and 64 + 8 j
  float4 th[2];      // thl, columns 8 j .. 8 j + 7 (j < 4)
  uint4 v, z;        // columns 8 j .. 8 j + 7
  float dcs, dcsr, ts, gam;
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

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// the 8 bf16 of u as fp32, into 8 consecutive floats of shared memory
__device__ __forceinline__ void st8f(float* dst, const U8& u) {
  float2 f[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) f[e] = __bfloat1622float2(u.h2[e]);
  reinterpret_cast<float4*>(dst)[0] = make_float4(f[0].x, f[0].y, f[1].x, f[1].y);
  reinterpret_cast<float4*>(dst)[1] = make_float4(f[2].x, f[2].y, f[3].x, f[3].y);
}

// thread: token tid >> 5, rank (tid >> 3) & 3, column chunk j = tid & 7 (qhat / khat: 8 j and 64 + 8 j;
// PsiV / gate: 8 j)
__device__ __forceinline__ void load_raw(Raw& rw, int tid, int t0, int nv, int h, int T, int H, long ldv, long ldz,
                                         long lhv, long lhz, bool vec,
                                         const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                         const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ Z,
                                         const float* __restrict__ thl, const float* __restrict__ dcs8,
                                         const float* __restrict__ dcsr8, const float* __restrict__ ts_g,
                                         const float* __restrict__ gam_g) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7;
  const bool ok = tok < nv;
  const long t = t0 + tok;
  const uint4 z0 = make_uint4(0u, 0u, 0u, 0u);
  const long off = (t * R + r) * N + 8 * j;
  rw.q[0] = ok ? __ldg(reinterpret_cast<const uint4*>(Q + off)) : z0;
  rw.q[1] = ok ? __ldg(reinterpret_cast<const uint4*>(Q + off + 64)) : z0;
  rw.k[0] = ok ? __ldg(reinterpret_cast<const uint4*>(K + off)) : z0;
  rw.k[1] = ok ? __ldg(reinterpret_cast<const uint4*>(K + off + 64)) : z0;
  if (j < 4) {
    const float4* tp = reinterpret_cast<const float4*>(thl + (t * H + h) * NQ + 8 * j);
    const float4 zf = make_float4(0.f, 0.f, 0.f, 0.f);
    rw.th[0] = ok ? __ldg(tp) : zf;
    rw.th[1] = ok ? __ldg(tp + 1) : zf;
  }
  rw.v = ok ? ld8(V + t * ldv + h * lhv + 8 * j, vec) : z0;
  rw.z = ok ? ld8(Z + t * ldz + h * lhz + 8 * j, vec) : z0;
  const long ht = (long)h * T + t;
  rw.dcs = ok ? __ldg(dcs8 + ht) : 0.f;
  rw.dcsr = ok ? __ldg(dcsr8 + ht) : 0.f;
  rw.ts = ok ? __ldg(ts_g + ht) : 0.f;
  rw.gam = ok ? __ldg(gam_g + ht) : 0.f;
}

// the sm80 kernel's operand preparation, value for value; only the destinations differ (fp32, row-major)
__device__ __forceinline__ void store_operands(Smem& S, const Raw& rw, int tid, const float* a0, bool last_tok,
                                               __nv_bfloat16* __restrict__ fk, long fk_rank_stride) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7, row = tok * R + r;
  {
    U8 q0, q1, k0, k1, bq0, bq1, bk0, bk1;
    q0.u = rw.q[0];
    q1.u = rw.q[1];
    k0.u = rw.k[0];
    k1.u = rw.k[1];
    bq0.u = *reinterpret_cast<const uint4*>(&S.qb[r * N + 8 * j]);
    bq1.u = *reinterpret_cast<const uint4*>(&S.qb[r * N + 64 + 8 * j]);
    bk0.u = *reinterpret_cast<const uint4*>(&S.kb[r * N + 8 * j]);
    bk1.u = *reinterpret_cast<const uint4*>(&S.kb[r * N + 64 + 8 * j]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      q0.h2[e] = __hadd2(q0.h2[e], bq0.h2[e]);
      q1.h2[e] = __hadd2(q1.h2[e], bq1.h2[e]);
      k0.h2[e] = __hadd2(k0.h2[e], bk0.h2[e]);
      k1.h2[e] = __hadd2(k1.h2[e], bk1.h2[e]);
    }
    if (j < 4) {
      const float th[8] = {rw.th[0].x, rw.th[0].y, rw.th[0].z, rw.th[0].w,
                           rw.th[1].x, rw.th[1].y, rw.th[1].z, rw.th[1].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float ca, sa, cb, sb;
        float a = th[2 * e] + a0[2 * e], b = th[2 * e + 1] + a0[2 * e + 1];
        a = a - 6.283185307179586f * floorf(a * 0.15915494309189535f);
        b = b - 6.283185307179586f * floorf(b * 0.15915494309189535f);
        __sincosf(a, &sa, &ca);
        __sincosf(b, &sb, &cb);
        const float2 x0 = __bfloat1622float2(q0.h2[e]), x1 = __bfloat1622float2(q1.h2[e]);
        const float2 y0 = __bfloat1622float2(k0.h2[e]), y1 = __bfloat1622float2(k1.h2[e]);
        q0.h2[e] = __floats2bfloat162_rn(ca * x0.x - sa * x1.x, cb * x0.y - sb * x1.y);
        q1.h2[e] = __floats2bfloat162_rn(sa * x0.x + ca * x1.x, sb * x0.y + cb * x1.y);
        k0.h2[e] = __floats2bfloat162_rn(ca * y0.x - sa * y1.x, cb * y0.y - sb * y1.y);
        k1.h2[e] = __floats2bfloat162_rn(sa * y0.x + ca * y1.x, sb * y0.y + cb * y1.y);
      }
    }
    if (last_tok) {
      *reinterpret_cast<uint4*>(fk + r * fk_rank_stride + 8 * j) = k0.u;
      *reinterpret_cast<uint4*>(fk + r * fk_rank_stride + 64 + 8 * j) = k1.u;
    }
    // khat * (ts * e_j) in one bf16 product (ts itself is held in bf16)
    const __nv_bfloat162 tse2 = __float2bfloat162_rn(rbf(rw.ts) * __expf(fmaxf(rw.dcsr, -LOG_EMIN)));
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      k0.h2[e] = __hmul2(k0.h2[e], tse2);
      k1.h2[e] = __hmul2(k1.h2[e], tse2);
    }
    st8f(&S.q[row * LDQ + 8 * j], q0);
    st8f(&S.q[row * LDQ + 64 + 8 * j], q1);
    st8f(&S.k[row * LDQ + 8 * j], k0);
    st8f(&S.k[row * LDQ + 64 + 8 * j], k1);
  }
  {
    const int col = 8 * j;
    U8 va, za, ps, g8;
    va.u = rw.v;
    za.u = rw.z;
    ps.u = *reinterpret_cast<const uint4*>(&S.psi[r * P + col]);
    const float4* zp = reinterpret_cast<const float4*>(&S.zeta[r * P + col]);
    const float4* fp = reinterpret_cast<const float4*>(&S.phi[r * P + col]);
    const float4 za4 = zp[0], zb4 = zp[1], fa4 = fp[0], fb4 = fp[1];
    const float zt[8] = {za4.x, za4.y, za4.z, za4.w, zb4.x, zb4.y, zb4.z, zb4.w};
    const float ft[8] = {fa4.x, fa4.y, fa4.z, fa4.w, fb4.x, fb4.y, fb4.z, fb4.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      va.h2[e] = __hmul2(va.h2[e], ps.h2[e]);
      const float2 z = __bfloat1622float2(za.h2[e]);
      const float g0 = 0.5f * z.x * zt[2 * e], g1 = 0.5f * z.y * zt[2 * e + 1];
      const float s0 = rbf(g0 * tanh_approx(g0) + g0), s1 = rbf(g1 * tanh_approx(g1) + g1);  // silu in bf16
      g8.h2[e] = __floats2bfloat162_rn(ft[2 * e] * s0, ft[2 * e + 1] * s1);
    }
    st8f(&S.pv[row * P + col], va);
    st8f(&S.gz[row * P + col], g8);
  }
  if ((tid & 31) == 0) {
    S.tok[tok][0] = rw.dcs;
    S.tok[tok][1] = rw.dcsr;
    S.tok[tok][2] = rw.ts;
    S.tok[tok][3] = rw.gam;
  }
}

__global__ void __launch_bounds__(THREADS, 1) scan_volta_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,  // (T, R, N)
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ Z,  // (T, H, P), row strides ldv / ldz
    const float* __restrict__ qbias, const float* __restrict__ kbias,          // (H, R, N)
    const float* __restrict__ psi, const float* __restrict__ phi, const float* __restrict__ zeta,  // (H, R, P)
    const float* __restrict__ Dskip,                                           // (H,)
    const float* __restrict__ thl, const float* __restrict__ ain,              // (T, H, NQ), (NC, H, NQ)
    const float* __restrict__ dcs8, const float* __restrict__ dcsr8,           // (H, T), 8-token groups
    const float* __restrict__ ts_g, const float* __restrict__ gam_g,           // (H, T)
    const int* __restrict__ ch_start, const int* __restrict__ ch_len, const int* __restrict__ ch_last,
    const float* __restrict__ sin_t,                                           // (NC, H, P, N)
    __nv_bfloat16* __restrict__ out,                                           // (T, H, P)
    __nv_bfloat16* __restrict__ fin_k,                                         // (NC, R, H, N)
    int T, int H, long ldv, long ldz, long lhv, long lhz) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, item = blockIdx.x, c = item / H, h = item % H;
  const int cs = ch_start[c], cl = ch_len[c], nsb = (cl + C - 1) / C, last_seq = ch_last[c];
  const float dsk = Dskip[h];
  const bool vec = ((ldv | ldz | lhv | lhz) & 7) == 0;
  for (int i = tid; i < R * N; i += THREADS) {
    S.qb[i] = __float2bfloat16(qbias[(long)h * R * N + i]);
    S.kb[i] = __float2bfloat16(kbias[(long)h * R * N + i]);
  }
  for (int i = tid; i < R * P; i += THREADS) {
    const long o = (long)h * R * P + i;
    S.psi[i] = __float2bfloat16(psi[o]);
    S.phi[i] = rbf(phi[o]);
    S.zeta[i] = zeta[o];
  }
  float a0[8];
  {
    const int j = tid & 7;
#pragma unroll
    for (int e = 0; e < 8; ++e) a0[e] = j < 4 ? ain[(long)item * NQ + 8 * j + e] : 0.f;
  }
  const int lane = tid & 31, warp = tid >> 5;
  // phase 1 / 2a / epilogue: token t1 = warp (its qhat and G rows are warp-uniform loads), o cols po, po + 1 = 2 lane,
  // 2 lane + 1 (contiguous state loads); G col jg = lane
  const int t1 = warp, po = 2 * lane, jg = lane, tg = jg >> 2;
  const float* qrow = &S.q[(4 * t1) * LDQ];
  // state tile: n rows n0 .. n0 + 7, p cols p0 .. p0 + 3
  const int n0 = 8 * (tid >> 4), p0 = 4 * (tid & 15);
  float st[8][4];
  {
    const float* sbg = sin_t + (long)item * P * N;
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b) st[a][b] = __ldg(sbg + (long)(p0 + b) * N + n0 + a);
#pragma unroll
    for (int a = 0; a < 8; ++a)
      *reinterpret_cast<float4*>(&S.sb[(n0 + a) * P + p0]) =
          make_float4(rbf(st[a][0]), rbf(st[a][1]), rbf(st[a][2]), rbf(st[a][3]));
  }
  Raw rw;
  load_raw(rw, tid, cs, min(C, cl), h, T, H, ldv, ldz, lhv, lhz, vec, Q, K, V, Z, thl, dcs8, dcsr8, ts_g, gam_g);
  const long fk_rank_stride = (long)H * N;
  __nv_bfloat16* fk = fin_k + ((long)(c * R) * H + h) * N;

  for (int it = 0; it < nsb; ++it) {
    const int t0 = cs + it * C, nv = min(C, cl - it * C);
    __syncthreads();  // previous sub-block done with q / k / pv / gz / g / tok; S.sb written
    store_operands(S, rw, tid, a0, last_seq >= 0 && it + 1 == nsb && (tid >> 5) == nv - 1, fk, fk_rank_stride);
    __syncthreads();
    if (it + 1 < nsb)
      load_raw(rw, tid, t0 + C, min(C, cl - (it + 1) * C), h, T, H, ldv, ldz, lhv, lhz, vec, Q, K, V, Z, thl, dcs8,
               dcsr8, ts_g, gam_g);

    // ---- phase 1: o = qhat @ S (4 ranks of token t1 x cols po, po + 1), G = qhat @ k'^T (same rows x col jg)
    float o[4][2], ga[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) o[r][0] = o[r][1] = ga[r] = 0.f;
#pragma unroll 2
    for (int n = 0; n < N; n += 4) {
      float4 qv[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) qv[r] = *reinterpret_cast<const float4*>(&qrow[r * LDQ + n]);
      const float4 kv = *reinterpret_cast<const float4*>(&S.k[jg * LDQ + n]);
      float2 sv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) sv[u] = *reinterpret_cast<const float2*>(&S.sb[(n + u) * P + po]);
      const float kk[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float a = u == 0 ? qv[r].x : u == 1 ? qv[r].y : u == 2 ? qv[r].z : qv[r].w;
          o[r][0] = fmaf(a, sv[u].x, o[r][0]);
          o[r][1] = fmaf(a, sv[u].y, o[r][1]);
          ga[r] = fmaf(a, kk[u], ga[r]);
        }
      }
    }
    {
      const float el = __expf(S.tok[t1][0]);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        o[r][0] *= el;
        o[r][1] *= el;
      }
      const int ti = t1, tj = tg;
      const float dj = S.tok[tj][0], lej = fmaxf(S.tok[tj][1], -LOG_EMIN);
      const float tsi = ti < nv ? rbf(S.tok[ti][2]) : 0.f;  // the trap scale is held in bf16
      const float dg = (ti < nv && tsi > 0.f) ? S.tok[ti][3] / tsi : 0.f;
      const float cf = ti > tj ? __expf(fminf(S.tok[ti][0] - dj, 0.f) - lej) : (ti == tj ? dg * __expf(-lej) : 0.f);
#pragma unroll
      for (int r = 0; r < 4; ++r) S.g[(4 * t1 + r) * LDG + jg] = rbf(ga[r] * cf);
    }
    __syncthreads();  // G complete; every warp done reading S.sb

    // ---- phase 2: o += G @ PsiV (j = 0..31)
#pragma unroll 2
    for (int jj = 0; jj < ROWS; jj += 4) {
      float4 gv[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) gv[r] = *reinterpret_cast<const float4*>(&S.g[(4 * t1 + r) * LDG + jj]);
      float2 pvv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) pvv[u] = *reinterpret_cast<const float2*>(&S.pv[(jj + u) * P + po]);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float a = u == 0 ? gv[r].x : u == 1 ? gv[r].y : u == 2 ? gv[r].z : gv[r].w;
          o[r][0] = fmaf(a, pvv[u].x, o[r][0]);
          o[r][1] = fmaf(a, pvv[u].y, o[r][1]);
        }
      }
    }
    // state: S = dec S + k'^T @ PsiV
    if (it + 1 < nsb) {
      const float dec = __expf(S.tok[nv - 1][0]);
#pragma unroll
      for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) st[a][b] *= dec;
#pragma unroll 4
      for (int jr = 0; jr < ROWS; ++jr) {
        const float4 ka = *reinterpret_cast<const float4*>(&S.k[jr * LDQ + n0]);
        const float4 kb2 = *reinterpret_cast<const float4*>(&S.k[jr * LDQ + n0 + 4]);
        const float4 pp = *reinterpret_cast<const float4*>(&S.pv[jr * P + p0]);
        const float kk[8] = {ka.x, ka.y, ka.z, ka.w, kb2.x, kb2.y, kb2.z, kb2.w};
        const float pb[4] = {pp.x, pp.y, pp.z, pp.w};
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
          for (int b = 0; b < 4; ++b) st[a][b] = fmaf(kk[a], pb[b], st[a][b]);
      }
    }
    // ---- D skip, gate, sum over the R ranks: the sm80 kernel's roundings, (o + D PsiV) fused, the gate product
    // rounded, then (r0 + r1) + (r2 + r3) as its two shuffle rounds add (explicit forms: the compiler would fuse
    // the gate products into the in-thread sums)
    {
      float v[4][2];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = 4 * t1 + r;
        const float2 pv = *reinterpret_cast<const float2*>(&S.pv[row * P + po]);
        const float2 gz = *reinterpret_cast<const float2*>(&S.gz[row * P + po]);
        v[r][0] = __fmul_rn(__fmaf_rn(pv.x, dsk, o[r][0]), gz.x);
        v[r][1] = __fmul_rn(__fmaf_rn(pv.y, dsk, o[r][1]), gz.y);
      }
      const float s0 = __fadd_rn(__fadd_rn(v[0][0], v[1][0]), __fadd_rn(v[2][0], v[3][0]));
      const float s1 = __fadd_rn(__fadd_rn(v[0][1], v[1][1]), __fadd_rn(v[2][1], v[3][1]));
      if (t1 < nv)
        *reinterpret_cast<__nv_bfloat162*>(out + ((long)(t0 + t1) * H + h) * P + po) = __floats2bfloat162_rn(s0, s1);
    }
    if (it + 1 < nsb) {
#pragma unroll
      for (int a = 0; a < 8; ++a)
        *reinterpret_cast<float4*>(&S.sb[(n0 + a) * P + p0]) =
            make_float4(rbf(st[a][0]), rbf(st[a][1]), rbf(st[a][2]), rbf(st[a][3]));
    }
  }
}
}  // namespace

void mamba3_scan_prefill_volta(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Z,
                               torch::Tensor qbias, torch::Tensor kbias, torch::Tensor psi, torch::Tensor phi,
                               torch::Tensor zeta, torch::Tensor Dskip, torch::Tensor thl, torch::Tensor ain,
                               torch::Tensor dcs8, torch::Tensor dcsr8, torch::Tensor ts, torch::Tensor gam,
                               torch::Tensor ch_start, torch::Tensor ch_len, torch::Tensor ch_last,
                               torch::Tensor sin_t, torch::Tensor out, torch::Tensor fin_k, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && Z.size(2) == P && Q.size(-1) == N && Q.size(-2) == R && thl.size(-1) == NQ);
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && out.is_contiguous() && sin_t.is_contiguous());
  TORCH_CHECK(V.stride(2) == 1 && Z.stride(2) == 1 && V.scalar_type() == at::kBFloat16 && Z.scalar_type() == at::kBFloat16);
  if (NC * H == 0) return;
  const int smem = (int)sizeof(Smem);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(scan_volta_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(scan_volta_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    attr = true;
  }
  scan_volta_kernel<<<(int)(NC * H), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr()),
      qbias.data_ptr<float>(), kbias.data_ptr<float>(), psi.data_ptr<float>(), phi.data_ptr<float>(),
      zeta.data_ptr<float>(), Dskip.data_ptr<float>(), thl.data_ptr<float>(), ain.data_ptr<float>(),
      dcs8.data_ptr<float>(), dcsr8.data_ptr<float>(), ts.data_ptr<float>(), gam.data_ptr<float>(),
      ch_start.data_ptr<int>(), ch_len.data_ptr<int>(), ch_last.data_ptr<int>(), sin_t.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(fin_k.data_ptr()), T, H,
      V.stride(0), Z.stride(0), V.stride(1), Z.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scan", &mamba3_scan_prefill_volta); }

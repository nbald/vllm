// Olala Mamba-3 MIMO speculative verify: every draft position of a request in one launch (fp32 SSM state).
//
// Column-slot protocol (vllm/v1/attention/backends/olala_spec_utils.py): request r owns the state slots
// cols[r, 0..k]; its current state lives in column max(num_accepted[r] - 1, 0). Position t of the verify block reads
// the state of position t - 1 and writes column t, so next step's acceptance selects a column without any copy.
//
// The chained step kernel launched once per position (and the ~18 gather/scatter kernels around each launch) is
// replaced by one persistent launch: a CTA walks (request, head[, row block]) items and, for each, loads its state
// rows once into the mma accumulators, then runs the k + 1 positions back to back with the state kept in registers.
// Per position it writes the column-t state (ssm, rotated B, x, angle) and the output row. The token rows of position
// t + 1 are prefetched (cp.async) while position t computes. The math is the fp32 pipelined step of mamba3_step.cu,
// operation for operation, so the outputs and states are bit-identical to chaining that kernel over the positions.
//
// Light-column layout (LIGHT = true, OLALA_SPEC_LIGHT): one state page per request instead of 1 + k. The page holds
// the fp32 SSM state once and 1 + k "light" columns of the small per-position pieces (angle, rotated B, x and the
// position's (alpha, gamma, beta)). Position 0 (always accepted) writes the SSM state in place and its small pieces
// to column 0; positions t >= 1 write only their small pieces to column t. Next step, the a = num_accepted - 1
// accepted drafts are committed first: the state is advanced from column 0 by the same update (same operands, same
// instruction order) for t = 1..a, so the state entering position 0 is bit-identical to the column-a state of the
// column-slot protocol, and the rejected positions cost nothing. Memory: ~1.82 MB per layer and request at k = 3
// against 4 x 1.63 MB; traffic: one state read + one state write per step against one read + (1 + k) writes.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int H = 48, D = 64, S = 128, R = 4, NA = 32;
constexpr int LDB = S + 8;

struct __align__(16) TokStage {
  __nv_bfloat16 bm[R * S];   // token B rows (pre bias / rotary)
  __nv_bfloat16 cm[R * S];   // token C rows
  __nv_bfloat16 x[D], z[D];
  __nv_bfloat16 angp[NA];
};
struct __align__(128) Smem {
  TokStage tk[2];
  __nv_bfloat16 bst0[R * S];    // previous rotated B of the initial state (k_pool), position 0 only
  __nv_bfloat16 xprev[2][D];    // previous x: [0] from v_pool for position 0, then ping-pong
  float angst0[NA];             // angle state of the initial state, position 0 only
  float bias_k[R * S], bias_q[R * S];
  float xproj[R * D], zproj[R * D], outproj[R * D];
  __nv_bfloat16 sBk[16][LDB];   // rows 0..3 rotated B of this position, 4..7 of the previous one, 8..15 zero
  __nv_bfloat16 sCb[8][LDB];    // rotated C, rows 4..7 zero
  float sCos[NA], sSin[NA];
  float sTheta[NA];             // angle state after this position
  __nv_bfloat16 cx[2][D];       // LIGHT commit: x of position j - 1 / j
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
// x = hi + mid + lo (bf16 each), as in mamba3_step.cu
__device__ __forceinline__ void split3_x2(float x0, float x1, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
  hi = pack_bf16(x0, x1);
  const float r0 = x0 - __uint_as_float(hi << 16), r1 = x1 - __uint_as_float(hi & 0xffff0000u);
  mid = pack_bf16(r0, r1);
  lo = pack_bf16(r0 - __uint_as_float(mid << 16), r1 - __uint_as_float(mid & 0xffff0000u));
}

struct Args {
  float* ssm; __nv_bfloat16* kp; __nv_bfloat16* vp; float* angle; int P;
  long long ssm_sst, kp_sst, vp_sst, ang_sst;   // slot strides of the pools (elements)
  const float* A; const float* dt; const float* trap;          // (N, H)
  const __nv_bfloat16* Bm; const __nv_bfloat16* Cm; long long bc_bstride;   // (N, R, S) views
  const __nv_bfloat16* x; const __nv_bfloat16* z;                // (N, H, D)
  const __nv_bfloat16* angp; long long angp_stride;              // (N, NA) view
  const float* Dp; const float* xproj; const float* zproj; const float* outproj;   // (H), (R, H, D)
  const float* bias_q; const float* bias_k;                      // (R, H, S)
  const int* qsl;          // (lanes + 1) decode-token offsets
  const int* cols; long long cols_stride; int ncols;   // (lanes, ncols) state slot columns
  const int* num_accepted; // (lanes)
  __nv_bfloat16* y; int N; int lanes;
  int pdl;
  // LIGHT: column strides (elements) of the small pools, the (alpha, gamma, beta, -) pool and its strides
  long long kp_cst, vp_cst, ang_cst; float* agb; long long agb_sst, agb_cst; int light_cols;
};

// Token rows of one position: 148 16-byte chunks (B 64, C 64, x 8, z 8, angle_proj 4).
template <int NT> __device__ __forceinline__ void issue_tok(const Args& a, TokStage& st, int tok, int h, int t) {
  for (int c = t; c < 148; c += NT) {
    if (c < 64) cp16(st.bm + c * 8, a.Bm + (size_t)tok * a.bc_bstride + c * 8);
    else if (c < 128) cp16(st.cm + (c - 64) * 8, a.Cm + (size_t)tok * a.bc_bstride + (c - 64) * 8);
    else if (c < 136) cp16(st.x + (c - 128) * 8, a.x + ((size_t)tok * H + h) * D + (c - 128) * 8);
    else if (c < 144) cp16(st.z + (c - 136) * 8, a.z + ((size_t)tok * H + h) * D + (c - 136) * 8);
    else cp16(st.angp + (c - 144) * 8, a.angp + (size_t)tok * a.angp_stride + (c - 144) * 8);
  }
}

// One CTA (4 warps, 16 state rows each) per (request, head) item. Splitting a head's rows across CTAs
// measured no faster at batch 1 (the per-position chain latency dominates, not the state stores) and
// would race on the shared per-position B / x / angle columns.
constexpr int WARPS = 4;
template <bool LIGHT>
__global__ void __launch_bounds__(32 * WARPS, 4) mamba3_verify_f32_kernel(const Args a) {
  constexpr int NT = 32 * WARPS;
  extern __shared__ __align__(128) char smem_raw[];
  Smem& S_ = *reinterpret_cast<Smem*>(smem_raw);
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int h = blockIdx.x % H, nb_stride = gridDim.x / H;
  for (int i = t; i < 8 * LDB; i += NT) { S_.sBk[8 + i / LDB][i % LDB] = __float2bfloat16(0.f); if (i < 4 * LDB) S_.sCb[4 + i / LDB][i % LDB] = __float2bfloat16(0.f); }
  for (int c = t; c < 448; c += NT) {   // per-head constants: bias_k/q 128 chunks each, x/z/out proj 64 each
    if (c < 128) { const int r = c >> 5, q = c & 31; cp16(S_.bias_k + r * S + q * 4, a.bias_k + ((size_t)r * H + h) * S + q * 4); }
    else if (c < 256) { const int r = (c - 128) >> 5, q = (c - 128) & 31; cp16(S_.bias_q + r * S + q * 4, a.bias_q + ((size_t)r * H + h) * S + q * 4); }
    else { const int k = (c - 256) >> 6, r = ((c - 256) & 63) >> 4, q = (c - 256) & 15;
      float* dst = k == 0 ? S_.xproj : (k == 1 ? S_.zproj : S_.outproj);
      const float* src = k == 0 ? a.xproj : (k == 1 ? a.zproj : a.outproj);
      cp16(dst + r * D + q * 4, src + ((size_t)r * H + h) * D + q * 4); }
  }
  cp_commit();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (a.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  const float Dh = a.Dp[h];
  const int qd = lane & 3, p0 = warp * 16 + (lane >> 2), mi = lane >> 3, li = lane & 7;
  for (int r = blockIdx.x / H; r < a.lanes; r += nb_stride) {
    const int start = a.qsl[r];
    const int qlen = min(a.qsl[r + 1] - start, a.ncols);
    if (qlen <= 0) continue;
    // LIGHT: one slot per request, the current small pieces in light column c0 of that slot (the SSM state in
    // the slot itself, advanced over columns 1..c0 below); otherwise column c0 of the column-slot protocol.
    const int c0 = LIGHT ? min(max(a.num_accepted[r] - 1, 0), a.light_cols - 1) : max(a.num_accepted[r] - 1, 0);
    const int in_slot = a.cols[(size_t)r * a.cols_stride + (LIGHT ? 0 : c0)];
    if (in_slot < 0 || in_slot >= a.P) {   // no state to start from: zero outputs, no writes
      for (int i = t; i < qlen * D; i += NT) a.y[((size_t)(start + i / D) * H + h) * D + i % D] = __float2bfloat16(0.f);
      continue;
    }
    const long long kp_in = (long long)in_slot * a.kp_sst + (LIGHT ? (long long)c0 * a.kp_cst : 0);
    const long long vp_in = (long long)in_slot * a.vp_sst + (LIGHT ? (long long)c0 * a.vp_cst : 0);
    const long long ang_in = (long long)in_slot * a.ang_sst + (LIGHT ? (long long)c0 * a.ang_cst : 0);
    // position 0 token rows + the initial state's small pieces, then the state rows straight into registers
    issue_tok<NT>(a, S_.tk[0], start, h, t);
    for (int c = t; c < 80; c += NT) {
      if (c < 64) { const int rr = c >> 4, q = c & 15; cp16(S_.bst0 + rr * S + q * 8, a.kp + kp_in + ((size_t)rr * H + h) * S + q * 8); }
      else if (c < 72) cp16(S_.xprev[0] + (c - 64) * 8, a.vp + vp_in + (size_t)h * D + (c - 64) * 8);
      else cp16(S_.angst0 + (c - 72) * 4, a.angle + ang_in + (size_t)h * NA + (c - 72) * 4);
    }
    cp_commit();
    float acc[16][4];
    {
      const float* src = a.ssm + (size_t)in_slot * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {
        const int n = jt * 8 + 2 * qd;
        const float2 h0 = *reinterpret_cast<const float2*>(src + p0 * S + n);
        const float2 h1 = *reinterpret_cast<const float2*>(src + (p0 + 8) * S + n);
        acc[jt][0] = h0.x; acc[jt][1] = h0.y; acc[jt][2] = h1.x; acc[jt][3] = h1.y;
      }
    }
    if constexpr (LIGHT) {
      // Deferred commit of last step's accepted drafts: position j's update from its light column (rotated B and
      // x of j and j - 1, alpha / gamma / beta of j), the same operands and operation order as the main loop.
      // It reads the per-head constants (xproj): wait for their cp.async group (only the position-0 group may
      // stay in flight); the __syncthreads below publishes them.
      if (c0 > 0) cp_wait<1>();
      for (int j = 1; j <= c0; ++j) {
        const long long kj = (long long)in_slot * a.kp_sst + (long long)j * a.kp_cst;
        const long long kq = kj - a.kp_cst;
        for (int i = t; i < R * S; i += NT) {
          const int rr = i / S, e = i % S;
          S_.sBk[rr][e] = a.kp[kj + ((size_t)rr * H + h) * S + e];
          S_.sBk[4 + rr][e] = a.kp[kq + ((size_t)rr * H + h) * S + e];
        }
        const long long vj = (long long)in_slot * a.vp_sst + (long long)j * a.vp_cst + (long long)h * D;
        for (int i = t; i < D; i += NT) { S_.cx[1][i] = a.vp[vj + i]; S_.cx[0][i] = a.vp[vj - a.vp_cst + i]; }
        const float* g = a.agb + (long long)in_slot * a.agb_sst + (long long)j * a.agb_cst + (long long)h * 4;
        const float alpha = g[0], gamma = g[1], beta = g[2];
        __syncthreads();
        uint32_t ah[3][4];
        {
          float av[2][2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int p = p0 + e * 8;
            const float xp = bf2f(S_.cx[1][p]), xs = bf2f(S_.cx[0][p]);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
              const int kk = 2 * qd + jj, rr = kk & 3;
              const float xpj = S_.xproj[rr * D + p];
              av[e][jj] = (kk < 4) ? gamma * xp * xpj : beta * xs * xpj;
            }
          }
#pragma unroll
          for (int e = 0; e < 2; ++e) split3_x2(av[e][0], av[e][1], ah[0][e], ah[1][e], ah[2][e]);
#pragma unroll
          for (int q = 0; q < 3; ++q) { ah[q][2] = 0u; ah[q][3] = 0u; }
        }
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[jt][e] = alpha * acc[jt][e];
        }
#pragma unroll
        for (int jp = 0; jp < 8; ++jp) {
          uint32_t bfr[4];
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[(mi & 1) * 8 + li][jp * 16 + (mi >> 1) * 8])));
#pragma unroll
          for (int q = 2; q >= 0; --q) { mma16816(acc[2 * jp], ah[q], bfr); mma16816(acc[2 * jp + 1], ah[q], bfr + 2); }
        }
        __syncthreads();   // sBk / cx are reloaded by the next j (and sBk rewritten by position 0)
      }
    }
    // dt / A / trap of position 0; later positions are loaded one position ahead
    float dt_n = a.dt[(size_t)start * H + h], A_n = a.A[(size_t)start * H + h], trap_n = a.trap[(size_t)start * H + h];
    for (int pos = 0; pos < qlen; ++pos) {
      const int tok = start + pos;
      const int out_slot = LIGHT ? in_slot : a.cols[(size_t)r * a.cols_stride + pos];
      const bool wr = out_slot >= 0 && out_slot < a.P && (!LIGHT || pos < a.light_cols);
      const bool more = pos + 1 < qlen;
      if (more) issue_tok<NT>(a, S_.tk[(pos + 1) & 1], tok + 1, h, t);
      cp_commit();
      const float dt = dt_n, A = A_n, trap = trap_n;
      if (more) { dt_n = a.dt[(size_t)(tok + 1) * H + h]; A_n = a.A[(size_t)(tok + 1) * H + h]; trap_n = a.trap[(size_t)(tok + 1) * H + h]; }
      const float alpha = __expf(A * dt), gamma = trap * dt, beta = (1.f - trap) * dt * alpha;
      cp_wait<1>();   // everything but the prefetch of position + 1
      __syncthreads();
      TokStage& st = S_.tk[pos & 1];
      // same expression shapes as mamba3_step.cu (bit-identical angles: FMA contraction follows the code)
      if (t < NA) {
        const float prev = pos == 0 ? S_.angst0[t] : S_.sTheta[t];
        float th = prev + tanhf(bf2f(st.angp[t])) * dt * 3.14159265358979323846f;
        th -= 6.283185307179586f * floorf(th / 6.283185307179586f);
        S_.sTheta[t] = th; sincosf(th, &S_.sSin[t], &S_.sCos[t]);
      }
      __syncthreads();
      for (int i = t; i < R * (S / 2); i += NT) {
        const int rr = i / (S / 2), j = i % (S / 2);
        // the previous position's rotated B moves to rows 4..7 (each element is owned by one thread)
        const __nv_bfloat16 plo = pos == 0 ? S_.bst0[rr * S + j] : S_.sBk[rr][j];
        const __nv_bfloat16 phi = pos == 0 ? S_.bst0[rr * S + j + 64] : S_.sBk[rr][j + 64];
        float blo = bf2f(st.bm[rr * S + j]) + S_.bias_k[rr * S + j], bhi = bf2f(st.bm[rr * S + j + 64]) + S_.bias_k[rr * S + j + 64];
        float clo = bf2f(st.cm[rr * S + j]) + S_.bias_q[rr * S + j], chi = bf2f(st.cm[rr * S + j + 64]) + S_.bias_q[rr * S + j + 64];
        if (j < NA) {
          const float c = S_.sCos[j], s_ = S_.sSin[j];
          const float b0_ = blo * c - bhi * s_, b1 = blo * s_ + bhi * c, c0_ = clo * c - chi * s_, c1 = clo * s_ + chi * c;
          blo = b0_; bhi = b1; clo = c0_; chi = c1;
        }
        S_.sBk[4 + rr][j] = plo; S_.sBk[4 + rr][j + 64] = phi;
        S_.sBk[rr][j] = __float2bfloat16(blo); S_.sBk[rr][j + 64] = __float2bfloat16(bhi);
        S_.sCb[rr][j] = __float2bfloat16(clo); S_.sCb[rr][j + 64] = __float2bfloat16(chi);
      }
      __syncthreads();
      const __nv_bfloat16* xs_buf = S_.xprev[pos & 1];
      uint32_t ah[3][4];
      {
        float av[2][2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int p = p0 + e * 8;
          const float xp = bf2f(st.x[p]), xs = bf2f(xs_buf[p]);
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const int kk = 2 * qd + jj, rr = kk & 3;
            const float xpj = S_.xproj[rr * D + p];
            av[e][jj] = (kk < 4) ? gamma * xp * xpj : beta * xs * xpj;
          }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) split3_x2(av[e][0], av[e][1], ah[0][e], ah[1][e], ah[2][e]);
#pragma unroll
        for (int q = 0; q < 3; ++q) { ah[q][2] = 0u; ah[q][3] = 0u; }
      }
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jt][e] = alpha * acc[jt][e];
      }
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        uint32_t bfr[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[(mi & 1) * 8 + li][jp * 16 + (mi >> 1) * 8])));
#pragma unroll
        for (int q = 2; q >= 0; --q) { mma16816(acc[2 * jp], ah[q], bfr); mma16816(acc[2 * jp + 1], ah[q], bfr + 2); }
      }
      if (wr && (!LIGHT || pos == 0)) {   // LIGHT: positions >= 1 keep only their light column
        float* dst = a.ssm + (size_t)out_slot * a.ssm_sst + ((size_t)h * D) * S;
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
            const int rr = 2 * qd + j;
            float orr = oc[e * 2 + j] + Dh * xp * S_.xproj[rr * D + p];
            const float zv = zp * S_.zproj[rr * D + p];
            orr *= __fdividef(zv, 1.f + __expf(-zv));
            ys[e] += orr * S_.outproj[rr * D + p];
          }
        }
      }
      ys[0] += __shfl_xor_sync(0xffffffffu, ys[0], 1); ys[1] += __shfl_xor_sync(0xffffffffu, ys[1], 1);
      if (qd == 0) { a.y[((size_t)tok * H + h) * D + p0] = __float2bfloat16(ys[0]); a.y[((size_t)tok * H + h) * D + p0 + 8] = __float2bfloat16(ys[1]); }
      for (int i = t; i < D; i += NT) S_.xprev[(pos + 1) & 1][i] = st.x[i];
      if (wr) {
        const long long kq = (long long)out_slot * a.kp_sst + (LIGHT ? (long long)pos * a.kp_cst : 0);
        const long long vq = (long long)out_slot * a.vp_sst + (LIGHT ? (long long)pos * a.vp_cst : 0);
        const long long aq = (long long)out_slot * a.ang_sst + (LIGHT ? (long long)pos * a.ang_cst : 0);
        for (int c = t; c < 64; c += NT) { const int rr = c >> 4, q = c & 15; *reinterpret_cast<uint4*>(a.kp + kq + ((size_t)rr * H + h) * S + q * 8) = *reinterpret_cast<const uint4*>(&S_.sBk[rr][q * 8]); }
        for (int i = t; i < D; i += NT) a.vp[vq + (size_t)h * D + i] = st.x[i];
        if (t < NA) a.angle[aq + (size_t)h * NA + t] = S_.sTheta[t];
        if (LIGHT && t == 0) {
          float* g = a.agb + (long long)out_slot * a.agb_sst + (long long)pos * a.agb_cst + (long long)h * 4;
          g[0] = alpha; g[1] = gamma; g[2] = beta; g[3] = 0.f;
        }
      }
      __syncthreads();   // this stage, sBk / sCb / sTheta and xprev[pos & 1] are reused by the next position
    }
    cp_wait<0>();
  }
  // rows past the last request (CUDA graph padding) get a zero output
  const int tail = a.qsl[a.lanes];
  for (int tok = tail + blockIdx.x / H; tok < a.N; tok += nb_stride)
    for (int i = t; i < D; i += NT) a.y[((size_t)tok * H + h) * D + i] = __float2bfloat16(0.f);
}
}  // namespace

void mamba3_verify_cuda(torch::Tensor ssm_pool, torch::Tensor k_pool, torch::Tensor v_pool, torch::Tensor angle_pool,
                        torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dp, torch::Tensor x, torch::Tensor dt,
                        torch::Tensor trap, torch::Tensor xproj, torch::Tensor zproj, torch::Tensor outproj, torch::Tensor z,
                        torch::Tensor bias_q, torch::Tensor bias_k, torch::Tensor angle_proj, torch::Tensor qsl,
                        torch::Tensor cols, torch::Tensor num_accepted, torch::Tensor y, int64_t ctas_per_sm, bool pdl,
                        c10::optional<torch::Tensor> agb_pool) {
  // LIGHT layout: k_pool (P, C, R, H, S), v_pool (P, C, H, D), angle_pool (P, C, H, NA), agb_pool (P, C, H, 4) fp32,
  // with C = 1 + k light columns per slot and cols (lanes, >= 1) holding the slot in column 0.
  const bool light = agb_pool.has_value();
  const int cd = light ? 1 : 0;   // extra column dim
  TORCH_CHECK(ssm_pool.dtype() == torch::kFloat32, "verify kernel: fp32 SSM state only");
  TORCH_CHECK(k_pool.dtype() == torch::kBFloat16 && v_pool.dtype() == torch::kBFloat16 && angle_pool.dtype() == torch::kFloat32);
  TORCH_CHECK(k_pool.dim() == 4 + cd && v_pool.dim() == 3 + cd && angle_pool.dim() == 3 + cd, "pool ranks");
  TORCH_CHECK(ssm_pool.stride(1) == D * S && ssm_pool.stride(2) == S && ssm_pool.stride(3) == 1, "ssm_pool inner dims must be contiguous");
  TORCH_CHECK(k_pool.stride(1 + cd) == H * S && k_pool.stride(2 + cd) == S && k_pool.stride(3 + cd) == 1, "k_pool inner dims must be contiguous");
  TORCH_CHECK(v_pool.stride(1 + cd) == D && v_pool.stride(2 + cd) == 1 && angle_pool.stride(1 + cd) == NA && angle_pool.stride(2 + cd) == 1);
  if (light) {
    const torch::Tensor& g = *agb_pool;
    TORCH_CHECK(g.dtype() == torch::kFloat32 && g.dim() == 4 && g.size(2) == H && g.size(3) == 4 && g.stride(3) == 1 && g.stride(2) == 4);
    TORCH_CHECK(k_pool.size(1) == v_pool.size(1) && k_pool.size(1) == angle_pool.size(1) && k_pool.size(1) == g.size(1));
    TORCH_CHECK((k_pool.stride(1) * 2) % 16 == 0 && (v_pool.stride(1) * 2) % 16 == 0 && (angle_pool.stride(1) * 4) % 16 == 0, "light column strides must be 16B aligned");
  }
  TORCH_CHECK((ssm_pool.stride(0) * 4) % 16 == 0 && (k_pool.stride(0) * 2) % 16 == 0 && (v_pool.stride(0) * 2) % 16 == 0 && (angle_pool.stride(0) * 4) % 16 == 0, "pool slot strides must be 16B aligned");
  TORCH_CHECK(Bm.stride(1) == S && Bm.stride(2) == 1 && Cm.stride(0) == Bm.stride(0) && Cm.stride(1) == S && Cm.stride(2) == 1);
  TORCH_CHECK((Bm.stride(0) * 2) % 16 == 0 && angle_proj.stride(1) == 1 && (angle_proj.stride(0) * 2) % 16 == 0, "B/C / angle_proj row strides must be 16B aligned");
  TORCH_CHECK(x.is_contiguous() && z.is_contiguous() && y.is_contiguous() && A.is_contiguous() && dt.is_contiguous() && trap.is_contiguous());
  TORCH_CHECK(A.dtype() == torch::kFloat32 && dt.dtype() == torch::kFloat32 && trap.dtype() == torch::kFloat32 && Dp.dtype() == torch::kFloat32);
  TORCH_CHECK(qsl.dtype() == torch::kInt32 && cols.dtype() == torch::kInt32 && num_accepted.dtype() == torch::kInt32);
  TORCH_CHECK(cols.dim() == 2 && cols.stride(1) == 1 && num_accepted.is_contiguous() && qsl.is_contiguous());
  Args a;
  a.ssm = ssm_pool.data_ptr<float>(); a.kp = reinterpret_cast<__nv_bfloat16*>(k_pool.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr());
  a.angle = angle_pool.data_ptr<float>(); a.P = ssm_pool.size(0);
  a.ssm_sst = ssm_pool.stride(0); a.kp_sst = k_pool.stride(0); a.vp_sst = v_pool.stride(0); a.ang_sst = angle_pool.stride(0);
  a.A = A.data_ptr<float>(); a.dt = dt.data_ptr<float>(); a.trap = trap.data_ptr<float>();
  a.Bm = reinterpret_cast<const __nv_bfloat16*>(Bm.data_ptr()); a.Cm = reinterpret_cast<const __nv_bfloat16*>(Cm.data_ptr()); a.bc_bstride = Bm.stride(0);
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); a.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr());
  a.angp = reinterpret_cast<const __nv_bfloat16*>(angle_proj.data_ptr()); a.angp_stride = angle_proj.stride(0);
  a.Dp = Dp.data_ptr<float>(); a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>();
  a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>();
  a.qsl = qsl.data_ptr<int>(); a.cols = cols.data_ptr<int>(); a.cols_stride = cols.stride(0);
  a.ncols = light ? (int)k_pool.size(1) : (int)cols.size(1);
  a.light_cols = light ? (int)k_pool.size(1) : 0;
  a.kp_cst = light ? k_pool.stride(1) : 0; a.vp_cst = light ? v_pool.stride(1) : 0; a.ang_cst = light ? angle_pool.stride(1) : 0;
  a.agb = light ? agb_pool->data_ptr<float>() : nullptr;
  a.agb_sst = light ? agb_pool->stride(0) : 0; a.agb_cst = light ? agb_pool->stride(1) : 0;
  a.num_accepted = num_accepted.data_ptr<int>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr()); a.N = x.size(0); a.lanes = cols.size(0);
  TORCH_CHECK(qsl.size(0) >= a.lanes + 1 && num_accepted.size(0) >= a.lanes);
  a.pdl = pdl ? 1 : 0;
  if (a.N == 0) return;
  const int smem = (int)sizeof(Smem);
  // Launch on x's device; the smem opt-in is a per-device function attribute.
  const c10::cuda::CUDAGuard device_guard(x.device());
  constexpr int kMaxDevices = 64;
  auto kern = light ? mamba3_verify_f32_kernel<true> : mamba3_verify_f32_kernel<false>;
  static bool attr_done[2][kMaxDevices] = {};
  const int dev = x.get_device();
  if (dev >= kMaxDevices || !attr_done[light][dev]) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (dev < kMaxDevices) attr_done[light][dev] = true;
  }
  int nsm = 132; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int fit = 1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kern, 32 * WARPS, smem);
  const int cps = std::max(1, std::min<int>((int)ctas_per_sm, fit));
  const int items = std::max(1, a.lanes) * H;
  int grid = std::min<int>(items, nsm * cps); grid = std::max(H, (grid / H) * H);
  if (a.pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(32 * WARPS); cfg.dynamicSmemBytes = smem; cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, a);
  } else {
    kern<<<grid, 32 * WARPS, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("verify", &mamba3_verify_cuda, py::arg("ssm"), py::arg("kp"), py::arg("vp"), py::arg("angle"), py::arg("A"),
        py::arg("Bm"), py::arg("Cm"), py::arg("Dp"), py::arg("x"), py::arg("dt"), py::arg("trap"), py::arg("xproj"),
        py::arg("zproj"), py::arg("outproj"), py::arg("z"), py::arg("bias_q"), py::arg("bias_k"), py::arg("angle_proj"),
        py::arg("qsl"), py::arg("cols"), py::arg("num_accepted"), py::arg("y"), py::arg("ctas_per_sm"), py::arg("pdl"),
        py::arg("agb") = py::none());
}

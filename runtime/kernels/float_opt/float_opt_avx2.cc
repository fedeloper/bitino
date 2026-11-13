// AVX2 + FMA leaf routines of the float_opt tier (compiled with -mavx2 -mfma;
// interface and linkage rules in float_opt_avx2.h).
//
// Numerics, per op, relative to the reference kernels:
//   GEMM (Conv2D, Dense), depthwise: float accumulation with fused
//     multiply-add (one rounding per tap, as vfmaq on NEON); not bit-exact,
//     within the documented float_opt bound (tests/cpp/test_float_kernels.cc).
//   Add, ChannelAffine, Activation, MaxPool2D: the reference's operations
//     with its comparison semantics, so bit-identical (see Simd8 below).
//   AvgPool2D, GlobalAvgPool: float sums (as the portable path).
#include "kernels/float_opt/float_opt_avx2.h"

#if !defined(__AVX2__) || !defined(__FMA__)
#error "float_opt_avx2.cc must be compiled with -mavx2 -mfma (see CMakeLists.txt)"
#endif

#include <immintrin.h>

#include <cstdint>

#include "kernels/float_opt/float_opt_leaf.h"
#include "kernels/float_opt/transformer_leaf.h"

// Full unrolling of the fixed-trip loops over rows / vectors / accumulator
// sets keeps the accumulator arrays in registers (without it GCC keeps them in
// memory and stores every accumulator on each k step).
#define BF_UNROLL _Pragma("GCC unroll 16")

namespace bitino {
namespace fopt_avx2 {
namespace {

using fopt::GemmBlockArgs;

// 8-lane policy for float_opt_leaf.h. Lane semantics of the comparisons
// (MAXPS/MINPS return the second operand unless the comparison holds):
//   relu(v)          max(v, 0)       = v > 0 ? v : 0          (reference relu, NaN -> 0)
//   clamp(v, lo, hi) min(hi, max(lo, v)) = std::min(std::max(v, lo), hi)
//   max_num(m, x)    max(x, m)       = x > m ? x : m          (std::max(m, x): ignores a NaN x)
//   select_ge0(v, n) v >= 0 (ordered) ? v : n                 (NaN -> n, as the reference)
struct Simd8 {
  using T = __m256;
  static constexpr int W = 8;
  static T load(const float* p) { return _mm256_loadu_ps(p); }
  static void store(float* p, T v) { _mm256_storeu_ps(p, v); }
  static T dup(float s) { return _mm256_set1_ps(s); }
  static T add(T a, T b) { return _mm256_add_ps(a, b); }
  static T mul(T a, T b) { return _mm256_mul_ps(a, b); }
  static T div(T a, T b) { return _mm256_div_ps(a, b); }
  static T madd(T acc, T a, T b) { return _mm256_fmadd_ps(a, b, acc); }
  static T relu(T v) { return _mm256_max_ps(v, _mm256_setzero_ps()); }
  static T clamp(T v, T lo, T hi) { return _mm256_min_ps(hi, _mm256_max_ps(lo, v)); }
  static T select_ge0(T v, T neg) { return _mm256_blendv_ps(neg, v, _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ)); }
  static T max_num(T m, T x) { return _mm256_max_ps(x, m); }
  // v >= 0 (ordered) ? 1 : -1;  v > 0 ? 1 : (v < 0 ? -1 : 0) as 1&gt + (-1)&lt (+0 when neither)
  static T sign(T v) {
    return _mm256_blendv_ps(_mm256_set1_ps(-1.f), _mm256_set1_ps(1.f), _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ));
  }
  static T sub(T a, T b) { return _mm256_sub_ps(a, b); }
  static T fma(T a, T b, T c) { return _mm256_fmadd_ps(a, b, c); }
  static T round(T v) { return _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
  static T pow2i(T n) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
  }
  static T min(T a, T b) { return _mm256_min_ps(a, b); }
  static T max(T a, T b) { return _mm256_max_ps(a, b); }
  static T abs(T v) { return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))); }
  static T select_lt(T a, T b, T x, T y) { return _mm256_blendv_ps(y, x, _mm256_cmp_ps(a, b, _CMP_LT_OQ)); }
  static T select_gt(T a, T b, T x, T y) { return _mm256_blendv_ps(y, x, _mm256_cmp_ps(a, b, _CMP_GT_OQ)); }
  static float hsum(T v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
  }
  static float hmax(T v) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
  }
  static T sign0(T v) {
    const __m256 z = _mm256_setzero_ps();
    return _mm256_add_ps(_mm256_and_ps(_mm256_cmp_ps(v, z, _CMP_GT_OQ), _mm256_set1_ps(1.f)),
                         _mm256_and_ps(_mm256_cmp_ps(v, z, _CMP_LT_OQ), _mm256_set1_ps(-1.f)));
  }
};

inline __m256 act8(int32_t act, __m256 v) { return fopt::leaf_activate<Simd8>(act, v); }

// Lane mask with the first n (0..8) lanes set.
inline __m256i first_lanes(int64_t n) {
  const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(n < 0 ? 0 : n > 8 ? 8 : n)), idx);
}

// Independent accumulator sets over k for short tiles: hides the 4-cycle FMA
// latency (2 FMA/cycle need ~8 chains in flight) when R rows give too few
// accumulators. Sets are summed at the end.
template <int R, int NV>
constexpr int ksets() {
  return 12 / (R * NV) < 2 ? 2 : 12 / (R * NV) > 4 ? 4 : 12 / (R * NV);
}

// C[R x 8*NV] (+)= A[R rows][kc] * Bp[kc][8*NV]; rows of A in place (stride
// lda), B packed with panel width 8*NV.
template <int R, int NV>
void tile(const float* a, int64_t lda, int64_t kc, const float* bp, float* c, int64_t ldc, int64_t nvalid, const float* bias,
          bool last, int32_t act) {
  constexpr int NR = 8 * NV, S = ksets<R, NV>();
  const float* ar[R];
  BF_UNROLL
  for (int i = 0; i < R; ++i) ar[i] = a + i * lda;
  const bool full = nvalid == NR;
  __m256i mask[NV];
  BF_UNROLL
  for (int v = 0; v < NV; ++v) mask[v] = first_lanes(nvalid - 8 * v);
  __m256 acc[S][R][NV];
  BF_UNROLL
  for (int v = 0; v < NV; ++v) {
    BF_UNROLL
    for (int i = 0; i < R; ++i) {
      if (bias) acc[0][i][v] = _mm256_loadu_ps(bias + 8 * v);
      else if (full) acc[0][i][v] = _mm256_loadu_ps(c + i * ldc + 8 * v);
      else acc[0][i][v] = _mm256_maskload_ps(c + i * ldc + 8 * v, mask[v]);
    }
  }
  BF_UNROLL
  for (int s = 1; s < S; ++s)
    BF_UNROLL
    for (int i = 0; i < R; ++i)
      BF_UNROLL
      for (int v = 0; v < NV; ++v) acc[s][i][v] = _mm256_setzero_ps();
  int64_t k = 0;
  for (; k + S <= kc; k += S) {
    BF_UNROLL
    for (int s = 0; s < S; ++s) {
      const float* b = bp + (k + s) * NR;
      __m256 bv[NV];
      BF_UNROLL
      for (int v = 0; v < NV; ++v) bv[v] = _mm256_loadu_ps(b + 8 * v);
      BF_UNROLL
      for (int i = 0; i < R; ++i) {
        const __m256 av = _mm256_broadcast_ss(ar[i] + k + s);
        BF_UNROLL
        for (int v = 0; v < NV; ++v) acc[s][i][v] = _mm256_fmadd_ps(av, bv[v], acc[s][i][v]);
      }
    }
  }
  for (; k < kc; ++k) {
    const float* b = bp + k * NR;
    __m256 bv[NV];
    BF_UNROLL
    for (int v = 0; v < NV; ++v) bv[v] = _mm256_loadu_ps(b + 8 * v);
    BF_UNROLL
    for (int i = 0; i < R; ++i) {
      const __m256 av = _mm256_broadcast_ss(ar[i] + k);
      BF_UNROLL
      for (int v = 0; v < NV; ++v) acc[0][i][v] = _mm256_fmadd_ps(av, bv[v], acc[0][i][v]);
    }
  }
  BF_UNROLL
  for (int s = 1; s < S; ++s)
    BF_UNROLL
    for (int i = 0; i < R; ++i)
      BF_UNROLL
      for (int v = 0; v < NV; ++v) acc[0][i][v] = _mm256_add_ps(acc[0][i][v], acc[s][i][v]);
  BF_UNROLL
  for (int i = 0; i < R; ++i)
    BF_UNROLL
    for (int v = 0; v < NV; ++v) {
      __m256 r = last ? act8(act, acc[0][i][v]) : acc[0][i][v];
      if (full) _mm256_storeu_ps(c + i * ldc + 8 * v, r);
      else _mm256_maskstore_ps(c + i * ldc + 8 * v, mask[v], r);
    }
}

// Rows of one panel, MR at a time (R = remaining rows for the last tile).
template <int NV, int MR>
void panel_rows(const GemmBlockArgs& g, const float* bp, float* cp, int64_t nvalid, const float* bias) {
  const bool last = g.last != 0;
  int64_t m = 0;
  for (; m + MR <= g.rows; m += MR) tile<MR, NV>(g.a + m * g.lda, g.lda, g.kc, bp, cp + m * g.ldc, g.ldc, nvalid, bias, last, g.act);
  const int64_t rem = g.rows - m;
  if (rem == 0) return;
  const float* a = g.a + m * g.lda;
  float* c = cp + m * g.ldc;
#define BF_TILE_CASE(r) \
  case r: tile<(r < MR ? r : 1), NV>(a, g.lda, g.kc, bp, c, g.ldc, nvalid, bias, last, g.act); break;
  switch (rem) {
    BF_TILE_CASE(1)
    BF_TILE_CASE(2)
    BF_TILE_CASE(3)
    BF_TILE_CASE(4)
    BF_TILE_CASE(5)
    BF_TILE_CASE(6)
    BF_TILE_CASE(7)
    BF_TILE_CASE(8)
    BF_TILE_CASE(9)
    BF_TILE_CASE(10)
    BF_TILE_CASE(11)
    default: break;
  }
#undef BF_TILE_CASE
}

template <int NV, int MR>
void gemm_panels(const GemmBlockArgs& g) {
  constexpr int NR = 8 * NV;
  for (int64_t p = g.p0; p < g.p1; ++p) {
    const int64_t left = g.ncols - p * NR;
    panel_rows<NV, MR>(g, g.bp + p * g.panel_stride, g.c + p * NR, left < NR ? left : NR, g.bias ? g.bias + p * NR : nullptr);
  }
}

}  // namespace

void gemm_block(const GemmBlockArgs& g) {
  if (g.nr == 16) gemm_panels<2, 3>(g);
  else gemm_panels<1, 6>(g);
}

void depthwise_rows(const fopt::DwArgs& d, int64_t r0, int64_t r1) { fopt::leaf_depthwise_rows<Simd8>(d, r0, r1); }

void add(const float* a, const float* b, float* y, int64_t i0, int64_t i1, int32_t act) { fopt::leaf_add<Simd8>(a, b, y, i0, i1, act); }

void affine_rows(const float* x, const float* s, const float* b, float* y, int64_t c, int64_t r0, int64_t r1) {
  fopt::leaf_affine_rows<Simd8>(x, s, b, y, c, r0, r1);
}

void activation(int32_t type, const float* x, float* y, int64_t i0, int64_t i1, float lo, float hi, float alpha) {
  fopt::leaf_activation<Simd8>(type, x, y, i0, i1, lo, hi, alpha);
}

void prelu_rows(const float* x, const float* slope, float* y, int64_t c, int64_t r0, int64_t r1) {
  fopt::leaf_prelu_rows<Simd8>(x, slope, y, c, r0, r1);
}

void pool_rows(const fopt::PoolArgs& g, int64_t r0, int64_t r1) { fopt::leaf_pool_rows<Simd8>(g, r0, r1); }

void mul(const float* a, const float* b, float* y, int64_t i0, int64_t i1) { fopt::leaf_mul<Simd8>(a, b, y, i0, i1); }

void mul_rows(const float* x, const float* g, float* y, int64_t c, int64_t r0, int64_t r1) {
  fopt::leaf_mul_rows<Simd8>(x, g, y, c, r0, r1);
}

void sign(const float* x, float* y, int64_t i0, int64_t i1, int32_t zero) { fopt::leaf_sign<Simd8>(x, y, i0, i1, zero); }

void fe_rows(const fopt::FeEvalArgs& e, float* y, int64_t r0, int64_t r1, void* scratch) {
  if (e.chunk == kFeChunk) fopt::leaf_fe_rows<Simd8, kFeChunk / 8>(e, y, r0, r1, scratch);
  else fopt::leaf_fe_rows<Simd8, kFeChunk / 16>(e, y, r0, r1, scratch);
}

void unary(int32_t op, const float* x, float* y, int64_t i0, int64_t i1) { fopt::leaf_unary<Simd8>(op, x, y, i0, i1); }

void softmax_rows(const float* x, float* y, int64_t c, int64_t r0, int64_t r1) { fopt::leaf_softmax_rows<Simd8>(x, y, c, r0, r1); }

void fake_quant(const float* x, const float* s, const float* z, float* y, int64_t len, int64_t inner, int64_t r0, int64_t r1, float pre,
                float lo, float hi) {
  fopt::leaf_fake_quant<Simd8>(x, s, z, y, len, inner, r0, r1, pre, lo, hi);
}

void add_run(const float* a, const float* b, float* y, int64_t n) { fopt::leaf_add_run<Simd8>(a, b, y, n); }

void add_scalar_run(const float* a, float b, float* y, int64_t n) { fopt::leaf_add_scalar_run<Simd8>(a, b, y, n); }

void scale_run(float* y, float s, int64_t n) { fopt::leaf_scale_run<Simd8>(y, s, n); }

// ---- double precision (bit-exact LayerNorm / ReduceMean)
// Four rows are transposed 4 floats at a time so that lane j of a __m256d
// accumulates row j in index order: the same sequence of double additions as
// the reference's scalar loop.
void rowsum_f64(const float* x, int64_t stride, int64_t len, double* out) {
  const float *x0 = x, *x1 = x + stride, *x2 = x + 2 * stride, *x3 = x + 3 * stride;
  __m256d s = _mm256_setzero_pd();
  int64_t i = 0;
  for (; i + 4 <= len; i += 4) {
    __m128 r0 = _mm_loadu_ps(x0 + i), r1 = _mm_loadu_ps(x1 + i), r2 = _mm_loadu_ps(x2 + i), r3 = _mm_loadu_ps(x3 + i);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    s = _mm256_add_pd(s, _mm256_cvtps_pd(r0));
    s = _mm256_add_pd(s, _mm256_cvtps_pd(r1));
    s = _mm256_add_pd(s, _mm256_cvtps_pd(r2));
    s = _mm256_add_pd(s, _mm256_cvtps_pd(r3));
  }
  for (; i < len; ++i) s = _mm256_add_pd(s, _mm256_cvtps_pd(_mm_setr_ps(x0[i], x1[i], x2[i], x3[i])));
  _mm256_storeu_pd(out, s);
}

void rowsqdev_f64(const float* x, int64_t stride, int64_t len, const double* mean, double* out) {
  const float *x0 = x, *x1 = x + stride, *x2 = x + 2 * stride, *x3 = x + 3 * stride;
  const __m256d m = _mm256_loadu_pd(mean);
  __m256d v = _mm256_setzero_pd();
  auto step = [&](__m128 c) {
    const __m256d t = _mm256_sub_pd(_mm256_cvtps_pd(c), m);
    v = _mm256_add_pd(v, _mm256_mul_pd(t, t));
  };
  int64_t i = 0;
  for (; i + 4 <= len; i += 4) {
    __m128 r0 = _mm_loadu_ps(x0 + i), r1 = _mm_loadu_ps(x1 + i), r2 = _mm_loadu_ps(x2 + i), r3 = _mm_loadu_ps(x3 + i);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    step(r0);
    step(r1);
    step(r2);
    step(r3);
  }
  for (; i < len; ++i) step(_mm_setr_ps(x0[i], x1[i], x2[i], x3[i]));
  _mm256_storeu_pd(out, v);
}

void layernorm_row(const float* x, const float* g, const float* b, double mean, double sd, float* y, int64_t len) {
  const __m256d m = _mm256_set1_pd(mean), d = _mm256_set1_pd(sd);
  int64_t i = 0;
  for (; i + 4 <= len; i += 4) {
    __m256d t = _mm256_div_pd(_mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(x + i)), m), d);
    if (g) t = _mm256_mul_pd(t, _mm256_cvtps_pd(_mm_loadu_ps(g + i)));
    if (b) t = _mm256_add_pd(t, _mm256_cvtps_pd(_mm_loadu_ps(b + i)));
    _mm_storeu_ps(y + i, _mm256_cvtpd_ps(t));
  }
  for (; i < len; ++i) {
    double t = (static_cast<double>(x[i]) - mean) / sd;
    if (g) t = t * static_cast<double>(g[i]);
    if (b) t = t + static_cast<double>(b[i]);
    y[i] = static_cast<float>(t);
  }
}

void colmean_f64(const float* x, int64_t len, int64_t inner, double cnt, float* y, int64_t i0, int64_t i1) {
  const __m256d c = _mm256_set1_pd(cnt);
  int64_t i = i0;
  for (; i + 8 <= i1; i += 8) {
    __m256d a0 = _mm256_setzero_pd(), a1 = a0;
    for (int64_t k = 0; k < len; ++k) {
      const float* p = x + k * inner + i;
      a0 = _mm256_add_pd(a0, _mm256_cvtps_pd(_mm_loadu_ps(p)));
      a1 = _mm256_add_pd(a1, _mm256_cvtps_pd(_mm_loadu_ps(p + 4)));
    }
    _mm_storeu_ps(y + i, _mm256_cvtpd_ps(_mm256_div_pd(a0, c)));
    _mm_storeu_ps(y + i + 4, _mm256_cvtpd_ps(_mm256_div_pd(a1, c)));
  }
  for (; i + 4 <= i1; i += 4) {
    __m256d a = _mm256_setzero_pd();
    for (int64_t k = 0; k < len; ++k) a = _mm256_add_pd(a, _mm256_cvtps_pd(_mm_loadu_ps(x + k * inner + i)));
    _mm_storeu_ps(y + i, _mm256_cvtpd_ps(_mm256_div_pd(a, c)));
  }
  for (; i < i1; ++i) {
    double a = 0.0;
    for (int64_t k = 0; k < len; ++k) a += static_cast<double>(x[k * inner + i]);
    y[i] = static_cast<float>(a / cnt);
  }
}

void global_avg_pool(const float* x, float* y, int64_t hw, int64_t c, int64_t t0, int64_t t1) {
  fopt::leaf_global_avg_pool<Simd8>(x, y, hw, c, t0, t1);
}
void gap_add_rows(float* acc, const float* x, int64_t rows, int64_t c) { fopt::leaf_gap_add_rows<Simd8>(acc, x, rows, c); }
void gap_finish(const float* acc, float* y, int64_t c, int64_t hw) { fopt::leaf_gap_finish<Simd8>(acc, y, c, hw); }

}  // namespace fopt_avx2
}  // namespace bitino

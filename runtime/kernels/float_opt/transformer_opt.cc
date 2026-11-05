// float_opt kernels of the transformer / BHViT operators and the remaining
// data-movement ops:
//
//   Activation  gelu, sigmoid, tanh, hardswish, sign, sign_zero ("activation_ext")
//   Softmax     float exp, float sums                       (inexact, bound in tests)
//   LayerNorm   double, vectorized across rows               (bit-exact)
//   ReduceMean  double, vectorized across outputs            (bit-exact)
//   FakeQuant   the reference's float operations             (bit-exact)
//   MatMul      float GEMM (float_opt micro-kernels, B packed per run)  (inexact)
//   AddBroadcast, Transpose, Concat, Gather (f32 table)      (bit-exact data movement / adds)
//
// The arithmetic is in transformer_leaf.h (vector policy, shared by every
// backend) and, for the double-precision parts, in the per-backend helpers
// below (AVX2 ones in float_opt_avx2.cc). This file holds the kernel classes,
// the work split and registration; it is compiled with the baseline flags.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/float_opt/float_opt.h"
#include "kernels/float_opt/float_opt_args.h"
#include "kernels/float_opt/float_opt_leaf.h"
#include "kernels/float_opt/float_simd.h"
#include "kernels/float_opt/transformer_leaf.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"

#if defined(__x86_64__)
#include "kernels/float_opt/float_opt_avx2.h"
#endif

namespace bitino {
namespace {

using namespace fopt;

constexpr int64_t kElemGrain = 16384;  // elements per parallel work item (streaming ops)

template <typename Fn>
void parallel_rows(ThreadPool* pool, int64_t rows, int64_t row_elems, const Fn& fn) {
  const int64_t grain = std::max<int64_t>(1, kElemGrain / std::max<int64_t>(row_elems, 1));
  const int64_t blocks = (rows + grain - 1) / grain;
  pool->parallel_for(blocks, 1, [&](int64_t b0, int64_t b1) { fn(b0 * grain, std::min(rows, b1 * grain)); });
}

// ------------------------------------------------- double-precision helpers
// R rows at a time; every per-row sum in index order, one rounding per op.
struct F64Scalar {
  [[maybe_unused]] static constexpr int R = 1;
  [[maybe_unused]] static constexpr bool kLayerNormRows = false;  // true: layernorm_rows() does whole rows (F64Neon)
  static void rowsum(const float* x, int64_t, int64_t len, double* out) {
    double s = 0.0;
    for (int64_t i = 0; i < len; ++i) s += static_cast<double>(x[i]);
    out[0] = s;
  }
  static void rowsqdev(const float* x, int64_t, int64_t len, const double* mean, double* out) {
    double v = 0.0;
    for (int64_t i = 0; i < len; ++i) {
      const double t = static_cast<double>(x[i]) - mean[0];
      v += t * t;
    }
    out[0] = v;
  }
  static void layernorm_row(const float* x, const float* g, const float* b, double mean, double sd, float* y, int64_t len) {
    for (int64_t i = 0; i < len; ++i) {
      double t = (static_cast<double>(x[i]) - mean) / sd;
      if (g) t = t * static_cast<double>(g[i]);
      if (b) t = t + static_cast<double>(b[i]);
      y[i] = static_cast<float>(t);
    }
  }
  static void colmean(const float* x, int64_t len, int64_t inner, double cnt, float* y, int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      double a = 0.0;
      for (int64_t k = 0; k < len; ++k) a += static_cast<double>(x[k * inner + i]);
      y[i] = static_cast<float>(a / cnt);
    }
  }
};

#if BF_FLOAT_OPT_NEON
// Four rows at a time, two per float64x2_t (rows 0/1 and 2/3: two independent
// add chains): lane j accumulates its row in index order. 4 floats per row and
// step: zip the row pairs, widen with FCVTL / FCVTL2.
struct F64Neon {
  static constexpr int R = 4;
  static constexpr bool kLayerNormRows = true;  // LayerNorm: layernorm_rows (8-row groups); R rows: ReduceMean
  static void rowsum(const float* x, int64_t stride, int64_t len, double* out) {
    const float *x0 = x, *x1 = x + stride, *x2 = x + 2 * stride, *x3 = x + 3 * stride;
    float64x2_t s01 = vdupq_n_f64(0.0), s23 = s01;
    int64_t i = 0;
    for (; i + 4 <= len; i += 4) {
      const float32x4_t a = vld1q_f32(x0 + i), b = vld1q_f32(x1 + i), c = vld1q_f32(x2 + i), d = vld1q_f32(x3 + i);
      const float32x4_t ab0 = vzip1q_f32(a, b), ab1 = vzip2q_f32(a, b), cd0 = vzip1q_f32(c, d), cd1 = vzip2q_f32(c, d);
      s01 = vaddq_f64(s01, vcvt_f64_f32(vget_low_f32(ab0)));
      s23 = vaddq_f64(s23, vcvt_f64_f32(vget_low_f32(cd0)));
      s01 = vaddq_f64(s01, vcvt_high_f64_f32(ab0));
      s23 = vaddq_f64(s23, vcvt_high_f64_f32(cd0));
      s01 = vaddq_f64(s01, vcvt_f64_f32(vget_low_f32(ab1)));
      s23 = vaddq_f64(s23, vcvt_f64_f32(vget_low_f32(cd1)));
      s01 = vaddq_f64(s01, vcvt_high_f64_f32(ab1));
      s23 = vaddq_f64(s23, vcvt_high_f64_f32(cd1));
    }
    for (; i < len; ++i) {
      const float p01[2] = {x0[i], x1[i]}, p23[2] = {x2[i], x3[i]};
      s01 = vaddq_f64(s01, vcvt_f64_f32(vld1_f32(p01)));
      s23 = vaddq_f64(s23, vcvt_f64_f32(vld1_f32(p23)));
    }
    vst1q_f64(out, s01);
    vst1q_f64(out + 2, s23);
  }
  static void rowsqdev(const float* x, int64_t stride, int64_t len, const double* mean, double* out) {
    const float *x0 = x, *x1 = x + stride, *x2 = x + 2 * stride, *x3 = x + 3 * stride;
    const float64x2_t m01 = vld1q_f64(mean), m23 = vld1q_f64(mean + 2);
    float64x2_t v01 = vdupq_n_f64(0.0), v23 = v01;
    auto acc = [](float64x2_t v, float64x2_t xv, float64x2_t m) {
      const float64x2_t t = vsubq_f64(xv, m);
      return vaddq_f64(v, vmulq_f64(t, t));
    };
    int64_t i = 0;
    for (; i + 4 <= len; i += 4) {
      const float32x4_t a = vld1q_f32(x0 + i), b = vld1q_f32(x1 + i), c = vld1q_f32(x2 + i), d = vld1q_f32(x3 + i);
      const float32x4_t ab0 = vzip1q_f32(a, b), ab1 = vzip2q_f32(a, b), cd0 = vzip1q_f32(c, d), cd1 = vzip2q_f32(c, d);
      v01 = acc(v01, vcvt_f64_f32(vget_low_f32(ab0)), m01);
      v23 = acc(v23, vcvt_f64_f32(vget_low_f32(cd0)), m23);
      v01 = acc(v01, vcvt_high_f64_f32(ab0), m01);
      v23 = acc(v23, vcvt_high_f64_f32(cd0), m23);
      v01 = acc(v01, vcvt_f64_f32(vget_low_f32(ab1)), m01);
      v23 = acc(v23, vcvt_f64_f32(vget_low_f32(cd1)), m23);
      v01 = acc(v01, vcvt_high_f64_f32(ab1), m01);
      v23 = acc(v23, vcvt_high_f64_f32(cd1), m23);
    }
    for (; i < len; ++i) {
      const float p01[2] = {x0[i], x1[i]}, p23[2] = {x2[i], x3[i]};
      v01 = acc(v01, vcvt_f64_f32(vld1_f32(p01)), m01);
      v23 = acc(v23, vcvt_f64_f32(vld1_f32(p23)), m23);
    }
    vst1q_f64(out, v01);
    vst1q_f64(out + 2, v23);
  }
  static void layernorm_row(const float* x, const float* g, const float* b, double mean, double sd, float* y, int64_t len) {
    const float64x2_t m = vdupq_n_f64(mean), d = vdupq_n_f64(sd);
    int64_t i = 0;
    for (; i + 2 <= len; i += 2) {
      float64x2_t t = vdivq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(x + i)), m), d);
      if (g) t = vmulq_f64(t, vcvt_f64_f32(vld1_f32(g + i)));
      if (b) t = vaddq_f64(t, vcvt_f64_f32(vld1_f32(b + i)));
      vst1_f32(y + i, vcvt_f32_f64(t));
    }
    if (i < len) F64Scalar::layernorm_row(x + i, g ? g + i : nullptr, b ? b + i : nullptr, mean, sd, y + i, len - i);
  }
  // LayerNorm, rows [r0, r1) of d floats. Row sums and squared deviations run
  // NP float64x2_t chains (2 * NP rows) side by side, each lane summing one row in
  // index order (4 columns per step from two 128-bit loads + zips); several
  // independent chains hide the FADD latency of the sequential sums. Then each row
  // is normalized with gamma / beta pre-widened to double (g64 / b64; null if
  // absent), 4 elements per step. Same operations and order as F64Scalar.
  template <int NP>
  static void rowsum_pairs(const float* x, int64_t stride, int64_t len, double* out) {
    float64x2_t s[NP];
    for (int p = 0; p < NP; ++p) s[p] = vdupq_n_f64(0.0);
    int64_t i = 0;
    for (; i + 4 <= len; i += 4)
      for (int p = 0; p < NP; ++p) {
        const float32x4_t a = vld1q_f32(x + 2 * p * stride + i), b = vld1q_f32(x + (2 * p + 1) * stride + i);
        const float32x4_t lo = vzip1q_f32(a, b), hi = vzip2q_f32(a, b);  // columns i, i+1 | i+2, i+3
        s[p] = vaddq_f64(s[p], vcvt_f64_f32(vget_low_f32(lo)));
        s[p] = vaddq_f64(s[p], vcvt_high_f64_f32(lo));
        s[p] = vaddq_f64(s[p], vcvt_f64_f32(vget_low_f32(hi)));
        s[p] = vaddq_f64(s[p], vcvt_high_f64_f32(hi));
      }
    for (; i < len; ++i)
      for (int p = 0; p < NP; ++p) {
        const float pair[2] = {x[2 * p * stride + i], x[(2 * p + 1) * stride + i]};
        s[p] = vaddq_f64(s[p], vcvt_f64_f32(vld1_f32(pair)));
      }
    for (int p = 0; p < NP; ++p) vst1q_f64(out + 2 * p, s[p]);
  }
  template <int NP>
  static void rowsqdev_pairs(const float* x, int64_t stride, int64_t len, const double* mean, double* out) {
    float64x2_t v[NP], m[NP];
    for (int p = 0; p < NP; ++p) {
      v[p] = vdupq_n_f64(0.0);
      m[p] = vld1q_f64(mean + 2 * p);
    }
    int64_t i = 0;
    for (; i + 4 <= len; i += 4)
      for (int p = 0; p < NP; ++p) {
        const float32x4_t a = vld1q_f32(x + 2 * p * stride + i), b = vld1q_f32(x + (2 * p + 1) * stride + i);
        const float32x4_t lo = vzip1q_f32(a, b), hi = vzip2q_f32(a, b);
        const float64x2_t t0 = vsubq_f64(vcvt_f64_f32(vget_low_f32(lo)), m[p]);
        const float64x2_t t1 = vsubq_f64(vcvt_high_f64_f32(lo), m[p]);
        const float64x2_t t2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(hi)), m[p]);
        const float64x2_t t3 = vsubq_f64(vcvt_high_f64_f32(hi), m[p]);
        v[p] = vaddq_f64(v[p], vmulq_f64(t0, t0));
        v[p] = vaddq_f64(v[p], vmulq_f64(t1, t1));
        v[p] = vaddq_f64(v[p], vmulq_f64(t2, t2));
        v[p] = vaddq_f64(v[p], vmulq_f64(t3, t3));
      }
    for (; i < len; ++i)
      for (int p = 0; p < NP; ++p) {
        const float pair[2] = {x[2 * p * stride + i], x[(2 * p + 1) * stride + i]};
        const float64x2_t t = vsubq_f64(vcvt_f64_f32(vld1_f32(pair)), m[p]);
        v[p] = vaddq_f64(v[p], vmulq_f64(t, t));
      }
    for (int p = 0; p < NP; ++p) vst1q_f64(out + 2 * p, v[p]);
  }
  template <bool G, bool B>
  static void normalize_row(const float* x, const double* g, const double* b, double mean, double sd, float* y, int64_t len) {
    const float64x2_t m = vdupq_n_f64(mean), dv = vdupq_n_f64(sd);
    int64_t i = 0;
    for (; i + 4 <= len; i += 4) {
      const float32x4_t xv = vld1q_f32(x + i);
      float64x2_t t0 = vdivq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(xv)), m), dv);
      float64x2_t t1 = vdivq_f64(vsubq_f64(vcvt_high_f64_f32(xv), m), dv);
      if (G) {
        t0 = vmulq_f64(t0, vld1q_f64(g + i));
        t1 = vmulq_f64(t1, vld1q_f64(g + i + 2));
      }
      if (B) {
        t0 = vaddq_f64(t0, vld1q_f64(b + i));
        t1 = vaddq_f64(t1, vld1q_f64(b + i + 2));
      }
      vst1q_f32(y + i, vcvt_high_f32_f64(vcvt_f32_f64(t0), t1));
    }
    for (; i < len; ++i) {
      double t = (static_cast<double>(x[i]) - mean) / sd;
      if (G) t = t * g[i];
      if (B) t = t + b[i];
      y[i] = static_cast<float>(t);
    }
  }
  static void normalize_any(const float* x, const double* g, const double* b, double mean, double sd, float* y, int64_t len) {
    if (g && b) normalize_row<true, true>(x, g, b, mean, sd, y, len);
    else if (g) normalize_row<true, false>(x, g, b, mean, sd, y, len);
    else if (b) normalize_row<false, true>(x, g, b, mean, sd, y, len);
    else normalize_row<false, false>(x, g, b, mean, sd, y, len);
  }
  template <int NP>
  static void layernorm_group(const float* x, const double* g64, const double* b64, int64_t d, double eps, float* y) {
    constexpr int R2 = 2 * NP;
    const double dd = static_cast<double>(d);
    double mean[R2], sd[R2], tmp[R2];
    rowsum_pairs<NP>(x, d, d, tmp);
    for (int j = 0; j < R2; ++j) mean[j] = tmp[j] / dd;
    rowsqdev_pairs<NP>(x, d, d, mean, tmp);
    for (int j = 0; j < R2; ++j) sd[j] = std::sqrt(tmp[j] / dd + eps);
    for (int j = 0; j < R2; ++j) normalize_any(x + j * d, g64, b64, mean[j], sd[j], y + j * d, d);
  }
  static void layernorm_rows(const float* x, const double* g64, const double* b64, int64_t d, double eps, float* y, int64_t r0, int64_t r1) {
    int64_t r = r0;
    for (; r + 8 <= r1; r += 8) layernorm_group<4>(x + r * d, g64, b64, d, eps, y + r * d);
    for (; r + 4 <= r1; r += 4) layernorm_group<2>(x + r * d, g64, b64, d, eps, y + r * d);
    for (; r + 2 <= r1; r += 2) layernorm_group<1>(x + r * d, g64, b64, d, eps, y + r * d);
    for (; r < r1; ++r) {
      const double dd = static_cast<double>(d);
      double tmp, mean;
      F64Scalar::rowsum(x + r * d, d, d, &tmp);
      mean = tmp / dd;
      F64Scalar::rowsqdev(x + r * d, d, d, &mean, &tmp);
      normalize_any(x + r * d, g64, b64, mean, std::sqrt(tmp / dd + eps), y + r * d, d);
    }
  }
  static void colmean(const float* x, int64_t len, int64_t inner, double cnt, float* y, int64_t i0, int64_t i1) {
    const float64x2_t c = vdupq_n_f64(cnt);
    int64_t i = i0;
    for (; i + 4 <= i1; i += 4) {
      float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0;
      for (int64_t k = 0; k < len; ++k) {
        const float32x4_t v = vld1q_f32(x + k * inner + i);
        a0 = vaddq_f64(a0, vcvt_f64_f32(vget_low_f32(v)));
        a1 = vaddq_f64(a1, vcvt_high_f64_f32(v));
      }
      vst1q_f32(y + i, vcombine_f32(vcvt_f32_f64(vdivq_f64(a0, c)), vcvt_f32_f64(vdivq_f64(a1, c))));
    }
    F64Scalar::colmean(x, len, inner, cnt, y, i, i1);
  }
};
using F64Simd4 = F64Neon;
#else
using F64Simd4 = F64Scalar;
#endif

#if BF_FLOAT_OPT_NEON
// ----------------------------------------------------------------- GELU (NEON)
// Same lane results as leaf_unary<Simd4>(kUnGelu) (vgelu is a lane-wise pure
// function), with less work:
//  * vectors without a lane |clamp(x)| < 1/sqrt(2) skip gelu_phi_small (the
//    select would discard it);
//  * "dead" lanes x <= kGeluCutoff (and NaN) give x * 0 whatever phi is. In blocks
//    of 256 elements with at least 5/16 dead lanes, the live lanes are compacted
//    into a buffer (TBL with a 16-entry pattern table per 4-lane mask), evaluated
//    there, and expanded back (inverse TBL, BSL with x * 0 for the dead lanes).
struct GeluLut {
  uint8_t compact[16][16];  // live lanes of mask m moved to the front (rest 0xFF -> 0)
  uint8_t expand[16][16];   // lane j <- compacted position of lane j (dead: 0xFF)
  uint8_t count[16];        // live lanes in m
  constexpr GeluLut() : compact(), expand(), count() {
    for (int m = 0; m < 16; ++m) {
      for (int q = 0; q < 16; ++q) compact[m][q] = expand[m][q] = 0xFF;
      int p = 0;
      for (int j = 0; j < 4; ++j)
        if ((m >> j) & 1) {
          for (int b = 0; b < 4; ++b) {
            compact[m][4 * p + b] = static_cast<uint8_t>(4 * j + b);
            expand[m][4 * j + b] = static_cast<uint8_t>(4 * p + b);
          }
          ++p;
        }
      count[m] = static_cast<uint8_t>(p);
    }
  }
};
constexpr GeluLut kGeluLut{};

// 4-lane mask -> bits (lane j -> bit j); w = {1, 2, 4, 8}
inline uint32_t lane_bits4(uint32x4_t m, uint32x4_t w) { return vaddvq_u32(vandq_u32(m, w)); }

inline float32x4_t gelu_vec_neon(float32x4_t x) {
  const float32x4_t xc = gelu_clamp<Simd4>(x);
  const float32x4_t ax = vabsq_f32(xc);
  float32x4_t phi = gelu_phi_large<Simd4>(xc, ax);
  const uint32x4_t small = vcltq_f32(ax, vdupq_n_f32(kGeluSmall));
  if (vmaxvq_u32(small) != 0u) phi = vbslq_f32(small, gelu_phi_small<Simd4>(xc), phi);
  return gelu_out<Simd4>(x, phi);
}

inline void gelu_dense_neon(const float* x, float* y, int64_t n) {
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, gelu_vec_neon(vld1q_f32(x + i)));
  if (i < n) {
    float buf[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; i + j < n; ++j) buf[j] = x[i + j];
    vst1q_f32(buf, gelu_vec_neon(vld1q_f32(buf)));
    for (int j = 0; i + j < n; ++j) y[i + j] = buf[j];
  }
}

void gelu_neon(const float* x, float* y, int64_t n) {
  constexpr int64_t kBlk = 256;  // multiple of 16
  alignas(16) float buf[kBlk + 4];
  const float32x4_t cut = vdupq_n_f32(kGeluCutoff), zero = vdupq_n_f32(0.f);
  const uint32x4_t w = {1u, 2u, 4u, 8u};
  const GeluLut& L = kGeluLut;
  int64_t i = 0;
  for (; i + kBlk <= n; i += kBlk) {
    const float* xb = x + i;
    float* yb = y + i;
    // live = x > cutoff (the last select of gelu_out); compaction pays off below ~70% live
    uint32x4_t l0 = vdupq_n_u32(0u), l1 = l0;
    for (int64_t j = 0; j < kBlk; j += 8) {
      l0 = vsubq_u32(l0, vcgtq_f32(vld1q_f32(xb + j), cut));
      l1 = vsubq_u32(l1, vcgtq_f32(vld1q_f32(xb + j + 4), cut));
    }
    const uint32_t live = vaddvq_u32(vaddq_u32(l0, l1));
    if (live * 16 > static_cast<uint32_t>(kBlk) * 11) {
      gelu_dense_neon(xb, yb, kBlk);
      continue;
    }
    float* out = buf;
    for (int64_t j = 0; j < kBlk; j += 8) {
      const float32x4_t v0 = vld1q_f32(xb + j), v1 = vld1q_f32(xb + j + 4);
      const uint32_t m0 = lane_bits4(vcgtq_f32(v0, cut), w), m1 = lane_bits4(vcgtq_f32(v1, cut), w);
      vst1q_f32(out, vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(v0), vld1q_u8(L.compact[m0]))));
      out += L.count[m0];
      vst1q_f32(out, vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(v1), vld1q_u8(L.compact[m1]))));
      out += L.count[m1];
    }
    vst1q_f32(out, zero);  // defined padding lanes for the last vector
    const int64_t nlive = out - buf;
    for (int64_t q = 0; q < nlive; q += 4) vst1q_f32(buf + q, gelu_vec_neon(vld1q_f32(buf + q)));
    const float* r = buf;
    for (int64_t j = 0; j < kBlk; j += 8) {
      const float32x4_t v0 = vld1q_f32(xb + j), v1 = vld1q_f32(xb + j + 4);
      const uint32x4_t lm0 = vcgtq_f32(v0, cut), lm1 = vcgtq_f32(v1, cut);
      const uint32_t m0 = lane_bits4(lm0, w), m1 = lane_bits4(lm1, w);
      const uint8x16_t r0 = vqtbl1q_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(r)), vld1q_u8(L.expand[m0]));
      r += L.count[m0];
      const uint8x16_t r1 = vqtbl1q_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(r)), vld1q_u8(L.expand[m1]));
      r += L.count[m1];
      vst1q_f32(yb + j, vbslq_f32(lm0, vreinterpretq_f32_u8(r0), vmulq_f32(v0, zero)));
      vst1q_f32(yb + j + 4, vbslq_f32(lm1, vreinterpretq_f32_u8(r1), vmulq_f32(v1, zero)));
    }
  }
  gelu_dense_neon(x + i, y + i, n - i);
}
#endif

// --------------------------------------------------------------- backends
struct TrSimd4 {
  using F64 = F64Simd4;
  static void unary(int32_t op, const float* x, float* y, int64_t i0, int64_t i1) {
#if BF_FLOAT_OPT_NEON
    if (op == kUnGelu) return gelu_neon(x + i0, y + i0, i1 - i0);
#endif
    leaf_unary<Simd4>(op, x, y, i0, i1);
  }
  static void sign(const float* x, float* y, int64_t i0, int64_t i1, int32_t zero) { leaf_sign<Simd4>(x, y, i0, i1, zero); }
  static void softmax_rows(const float* x, float* y, int64_t c, int64_t r0, int64_t r1) { leaf_softmax_rows<Simd4>(x, y, c, r0, r1); }
  static void fake_quant(const float* x, const float* s, const float* z, float* y, int64_t len, int64_t inner, int64_t r0, int64_t r1,
                         float pre, float lo, float hi) {
    leaf_fake_quant<Simd4>(x, s, z, y, len, inner, r0, r1, pre, lo, hi);
  }
  static void add_run(const float* a, const float* b, float* y, int64_t n) { leaf_add_run<Simd4>(a, b, y, n); }
  static void add_scalar_run(const float* a, float b, float* y, int64_t n) { leaf_add_scalar_run<Simd4>(a, b, y, n); }
  static void scale_run(float* y, float s, int64_t n) { leaf_scale_run<Simd4>(y, s, n); }
  static int64_t nr_for(int64_t) { return 8; }
  static void gemm(const GemmBlockArgs& g) { float_opt_gemm_block_simd4(g); }
};

#if defined(__x86_64__)
struct F64Avx2 {
  static constexpr int R = fopt_avx2::kRowsF64;
  static constexpr bool kLayerNormRows = false;
  static void rowsum(const float* x, int64_t stride, int64_t len, double* out) { fopt_avx2::rowsum_f64(x, stride, len, out); }
  static void rowsqdev(const float* x, int64_t stride, int64_t len, const double* mean, double* out) {
    fopt_avx2::rowsqdev_f64(x, stride, len, mean, out);
  }
  static void layernorm_row(const float* x, const float* g, const float* b, double mean, double sd, float* y, int64_t len) {
    fopt_avx2::layernorm_row(x, g, b, mean, sd, y, len);
  }
  static void colmean(const float* x, int64_t len, int64_t inner, double cnt, float* y, int64_t i0, int64_t i1) {
    fopt_avx2::colmean_f64(x, len, inner, cnt, y, i0, i1);
  }
};

struct TrAvx2 {
  using F64 = F64Avx2;
  static void unary(int32_t op, const float* x, float* y, int64_t i0, int64_t i1) { fopt_avx2::unary(op, x, y, i0, i1); }
  static void sign(const float* x, float* y, int64_t i0, int64_t i1, int32_t zero) { fopt_avx2::sign(x, y, i0, i1, zero); }
  static void softmax_rows(const float* x, float* y, int64_t c, int64_t r0, int64_t r1) { fopt_avx2::softmax_rows(x, y, c, r0, r1); }
  static void fake_quant(const float* x, const float* s, const float* z, float* y, int64_t len, int64_t inner, int64_t r0, int64_t r1,
                         float pre, float lo, float hi) {
    fopt_avx2::fake_quant(x, s, z, y, len, inner, r0, r1, pre, lo, hi);
  }
  static void add_run(const float* a, const float* b, float* y, int64_t n) { fopt_avx2::add_run(a, b, y, n); }
  static void add_scalar_run(const float* a, float b, float* y, int64_t n) { fopt_avx2::add_scalar_run(a, b, y, n); }
  static void scale_run(float* y, float s, int64_t n) { fopt_avx2::scale_run(y, s, n); }
  // as Avx2Gemm in conv_gemm.cc
  static int64_t nr_for(int64_t n) {
    const int64_t pad16 = round_up(n, 16) - n, pad8 = round_up(n, 8) - n;
    return pad8 < pad16 && pad16 * 8 > round_up(n, 16) ? 8 : 16;
  }
  static void gemm(const GemmBlockArgs& g) { fopt_avx2::gemm_block(g); }
};
#endif

// ------------------------------------------------------------- Activation
// gelu / sigmoid / tanh / hardswish (vector math) and sign / sign_zero.
bool parse_ext_activation(const std::string& t, int32_t* op) {
  if (t == "gelu") *op = kUnGelu;
  else if (t == "sigmoid") *op = kUnSigmoid;
  else if (t == "tanh") *op = kUnTanh;
  else if (t == "hardswish") *op = kUnHardswish;
  else if (t == "sign") *op = 100;
  else if (t == "sign_zero") *op = 101;
  else return false;
  return true;
}

bool activation_ext_supported(const KernelContext& ctx) {
  int32_t op;
  return parse_ext_activation(ctx.attrs().get_string("type", ""), &op);
}

template <class B>
class ActivationExtOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { parse_ext_activation(ctx.attrs().get_string("type", ""), &op_); }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.in(0).size();
    // transcendental ops cost ~10-30x a streaming op per element: smaller items
    const int64_t grain = op_ >= 100 ? kElemGrain : kElemGrain / 8;
    const int64_t blocks = (n + grain - 1) / grain;
    const int32_t op = op_;
    ctx.pool->parallel_for(blocks, 1, [&](int64_t b0, int64_t b1) {
      const int64_t i0 = b0 * grain, i1 = std::min(n, b1 * grain);
      if (op >= 100) B::sign(x, y, i0, i1, op - 100);
      else B::unary(op, x, y, i0, i1);
    });
  }

 private:
  int32_t op_ = kUnGelu;
};

// ---------------------------------------------------------------- Softmax
template <class B>
class SoftmaxOpt : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t c = x.dim(-1), rows = x.size() / c;
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    parallel_rows(ctx.pool, rows, c * 8, [&](int64_t r0, int64_t r1) { B::softmax_rows(xp, y, c, r0, r1); });
  }
};

// -------------------------------------------------------------- LayerNorm
template <class B>
class LayerNormOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    eps_ = ctx.attrs().get_number("eps", 1e-5);
    if (B::F64::kLayerNormRows) {
      const size_t d = static_cast<size_t>(ctx.in(0).dim(-1));
      g64_.assign(ctx.has_in(1) ? d : 0, 0.0);
      b64_.assign(ctx.has_in(2) ? d : 0, 0.0);
    }
  }
  void run(const KernelContext& ctx) override {
    using F = typename B::F64;
    constexpr int R = F::R;
    const Tensor& x = ctx.in(0);
    const int64_t d = x.dim(-1), rows = x.size() / d;
    const float* g = opt_f32(ctx, 1);
    const float* b = opt_f32(ctx, 2);
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    const double eps = eps_, dd = static_cast<double>(d);
    if constexpr (F::kLayerNormRows) {  // gamma / beta widened once per run (exact)
      for (int64_t i = 0; g && i < d; ++i) g64_[i] = static_cast<double>(g[i]);
      for (int64_t i = 0; b && i < d; ++i) b64_[i] = static_cast<double>(b[i]);
      const double* g64 = g ? g64_.data() : nullptr;
      const double* b64 = b ? b64_.data() : nullptr;
      const int64_t groups8 = (rows + 7) / 8;  // work items of 8 rows (the widest chain group)
      parallel_rows(ctx.pool, groups8, 8 * d * 4, [&](int64_t g0, int64_t g1) {
        F::layernorm_rows(xp, g64, b64, d, eps, y, g0 * 8, std::min(rows, g1 * 8));
      });
      return;
    }
    const int64_t groups = (rows + R - 1) / R;
    parallel_rows(ctx.pool, groups, R * d * 4, [&](int64_t g0, int64_t g1) {
      for (int64_t gi = g0; gi < g1; ++gi) {
        const int64_t r0 = gi * R, nr = std::min<int64_t>(R, rows - r0);
        double mean[R], sd[R], tmp[R];
        if (nr == R) {
          F::rowsum(xp + r0 * d, d, d, tmp);
          for (int j = 0; j < R; ++j) mean[j] = tmp[j] / dd;
          F::rowsqdev(xp + r0 * d, d, d, mean, tmp);
          for (int j = 0; j < R; ++j) sd[j] = std::sqrt(tmp[j] / dd + eps);
        } else {
          for (int64_t j = 0; j < nr; ++j) {
            F64Scalar::rowsum(xp + (r0 + j) * d, d, d, tmp);
            mean[j] = tmp[0] / dd;
            F64Scalar::rowsqdev(xp + (r0 + j) * d, d, d, &mean[j], tmp);
            sd[j] = std::sqrt(tmp[0] / dd + eps);
          }
        }
        for (int64_t j = 0; j < nr; ++j) F::layernorm_row(xp + (r0 + j) * d, g, b, mean[j], sd[j], y + (r0 + j) * d, d);
      }
    });
  }

 private:
  double eps_ = 1e-5;
  std::vector<double> g64_, b64_;  // kLayerNormRows backends
};

// ------------------------------------------------------------- ReduceMean
struct AxisSplit {
  int64_t outer = 1, len = 1, inner = 1;
};

AxisSplit split_axis(const Shape& s, int64_t axis) {
  if (axis < 0) axis += static_cast<int64_t>(s.size());
  AxisSplit a;
  for (int64_t d = 0; d < axis; ++d) a.outer *= s[d];
  a.len = s[axis];
  for (size_t d = axis + 1; d < s.size(); ++d) a.inner *= s[d];
  return a;
}

template <class B>
class ReduceMeanOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { sp_ = split_axis(ctx.in(0).shape(), ctx.attrs().get_int("axis", -1)); }
  void run(const KernelContext& ctx) override {
    using F = typename B::F64;
    constexpr int R = F::R;
    const AxisSplit sp = sp_;
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const double cnt = static_cast<double>(sp.len);
    if (sp.inner == 1) {  // each output sums `len` contiguous floats
      const int64_t groups = (sp.outer + R - 1) / R;
      parallel_rows(ctx.pool, groups, R * sp.len, [&](int64_t g0, int64_t g1) {
        double s[R];
        for (int64_t gi = g0; gi < g1; ++gi) {
          const int64_t o0 = gi * R, no = std::min<int64_t>(R, sp.outer - o0);
          if (no == R) {
            F::rowsum(x + o0 * sp.len, sp.len, sp.len, s);
            for (int j = 0; j < R; ++j) y[o0 + j] = static_cast<float>(s[j] / cnt);
          } else {
            for (int64_t j = 0; j < no; ++j) {
              F64Scalar::rowsum(x + (o0 + j) * sp.len, sp.len, sp.len, s);
              y[o0 + j] = static_cast<float>(s[0] / cnt);
            }
          }
        }
      });
      return;
    }
    parallel_rows(ctx.pool, sp.outer, sp.len * sp.inner, [&](int64_t o0, int64_t o1) {
      for (int64_t o = o0; o < o1; ++o) F::colmean(x + o * sp.len * sp.inner, sp.len, sp.inner, cnt, y + o * sp.inner, 0, sp.inner);
    });
  }

 private:
  AxisSplit sp_;
};

// -------------------------------------------------------------- FakeQuant
template <class B>
class FakeQuantOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    sp_ = split_axis(ctx.in(0).shape(), ctx.attrs().get_int("axis", -1));
    pre_ = static_cast<float>(ctx.attrs().get_number("pre_scale", 1.0));
    qmin_ = static_cast<float>(ctx.attrs().get_int("qmin", 0));
    qmax_ = static_cast<float>(ctx.attrs().get_int("qmax", 1));
  }
  void run(const KernelContext& ctx) override {
    const AxisSplit sp = sp_;
    const float* x = ctx.in(0).data<float>();
    const float* s = ctx.in(1).data<float>();
    const float* z = ctx.in(2).data<float>();
    float* y = ctx.out(0).data<float>();
    const float pre = pre_, lo = qmin_, hi = qmax_;
    if (sp.inner == 1) {
      parallel_rows(ctx.pool, sp.outer, sp.len, [&](int64_t r0, int64_t r1) { B::fake_quant(x, s, z, y, sp.len, 1, r0, r1, pre, lo, hi); });
    } else {
      parallel_rows(ctx.pool, sp.outer * sp.len, sp.inner,
                    [&](int64_t r0, int64_t r1) { B::fake_quant(x, s, z, y, sp.len, sp.inner, r0, r1, pre, lo, hi); });
    }
  }

 private:
  AxisSplit sp_;
  float pre_ = 1.f, qmin_ = 0.f, qmax_ = 1.f;
};

// ----------------------------------------------------------- AddBroadcast
// y = a + b with b right-aligned and broadcast. The longest suffix of
// dimensions in which b is not broadcast is one contiguous run of b; if b is
// broadcast over the whole last dimension instead, each run adds one scalar.
template <class B>
class AddBroadcastOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Shape& as = ctx.in(0).shape();
    const Shape& bs = ctx.in(1).shape();
    const size_t r = as.size(), off = r - bs.size();
    dims_ = as;
    bstride_.assign(r, 0);
    std::vector<int64_t> bdim(r, 1);
    int64_t st = 1;
    for (size_t d = r; d-- > 0;) {
      bdim[d] = d >= off ? bs[d - off] : 1;
      bstride_[d] = bdim[d] == 1 ? 0 : st;
      st *= bdim[d];
    }
    // suffix [split_, r) where b matches a (dense run) ...
    split_ = static_cast<int>(r);
    while (split_ > 0 && bdim[split_ - 1] == as[split_ - 1]) --split_;
    scalar_ = false;
    if (split_ == static_cast<int>(r)) {  // ... or where b is broadcast (scalar per run)
      while (split_ > 0 && bdim[split_ - 1] == 1) --split_;
      scalar_ = true;
    }
    run_ = 1;
    for (size_t d = split_; d < r; ++d) run_ *= as[d];
  }
  void run(const KernelContext& ctx) override {
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.in(0).size(), run = std::max<int64_t>(run_, 1), outer = run_ > 0 ? n / run : 0;
    parallel_rows(ctx.pool, outer, run, [&](int64_t o0, int64_t o1) {
      for (int64_t o = o0; o < o1; ++o) {
        int64_t rem = o, boff = 0;
        for (int d = split_ - 1; d >= 0; --d) {
          boff += (rem % dims_[d]) * bstride_[d];
          rem /= dims_[d];
        }
        if (scalar_) B::add_scalar_run(a + o * run, b[boff], y + o * run, run);
        else B::add_run(a + o * run, b + boff, y + o * run, run);
      }
    });
  }

 private:
  Shape dims_;
  std::vector<int64_t> bstride_;
  int split_ = 0;
  bool scalar_ = false;
  int64_t run_ = 1;
};

// -------------------------------------------------------------- Transpose
// Copies runs of the longest output suffix that is contiguous in the input
// (e.g. the head dim of [0,2,1,3]); the remaining output index is advanced
// incrementally. Pure data movement, any backend.
class TransposeOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const std::vector<int64_t> perm = ctx.attrs().get_ints("perm", {});
    const Shape& xs = ctx.in(0).shape();
    const size_t r = xs.size();
    std::vector<int64_t> xstride(r, 1);
    for (size_t d = r - 1; d-- > 0;) xstride[d] = xstride[d + 1] * xs[d + 1];
    od_.resize(r);
    st_.resize(r);
    for (size_t i = 0; i < r; ++i) {
      od_[i] = xs[perm[i]];
      st_[i] = xstride[perm[i]];
    }
    split_ = static_cast<int>(r);
    run_ = 1;
    while (split_ > 0 && st_[split_ - 1] == run_) {
      run_ *= od_[split_ - 1];
      --split_;
    }
  }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.in(0).size(), run = run_, outer = run > 0 ? n / run : 0;
    const int split = split_;
    parallel_rows(ctx.pool, outer, run, [&](int64_t o0, int64_t o1) {
      int64_t idx[8] = {0}, src = 0, rem = o0;
      for (int d = split - 1; d >= 0; --d) {
        idx[d] = rem % od_[d];
        rem /= od_[d];
        src += idx[d] * st_[d];
      }
      for (int64_t o = o0; o < o1; ++o) {
        if (run == 1) y[o] = x[src];
        else std::memcpy(y + o * run, x + src, static_cast<size_t>(run) * sizeof(float));
        for (int d = split - 1; d >= 0; --d) {
          if (++idx[d] < od_[d]) {
            src += st_[d];
            break;
          }
          src -= st_[d] * (od_[d] - 1);
          idx[d] = 0;
        }
      }
    });
  }

 private:
  std::vector<int64_t> od_, st_;
  int split_ = 0;
  int64_t run_ = 1;
};

bool transpose_supported(const KernelContext& ctx) { return ctx.in(0).dtype() == DType::kF32 && ctx.in(0).rank() <= 8; }

// ----------------------------------------------------------------- Concat
// Work items are (outer index, input) pairs, each one memcpy.
class ConcatOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Shape& os = ctx.out(0).shape();
    int64_t axis = ctx.attrs().get_int("axis", -1);
    if (axis < 0) axis += static_cast<int64_t>(os.size());
    outer_ = 1;
    int64_t inner = 1;
    for (int64_t d = 0; d < axis; ++d) outer_ *= os[d];
    for (size_t d = axis + 1; d < os.size(); ++d) inner *= os[d];
    out_stride_ = os[axis] * inner;
    len_.clear();
    off_.clear();
    int64_t off = 0;
    for (size_t i = 0; i < ctx.inputs.size(); ++i) {
      len_.push_back(ctx.in(i).dim(static_cast<int>(axis)) * inner);
      off_.push_back(off);
      off += len_.back();
    }
  }
  void run(const KernelContext& ctx) override {
    float* y = ctx.out(0).data<float>();
    const int64_t ni = static_cast<int64_t>(len_.size());
    std::vector<const float*>& src = src_;
    src.resize(len_.size());
    for (int64_t i = 0; i < ni; ++i) src[i] = ctx.in(i).data<float>();
    const int64_t avg = std::max<int64_t>(1, out_stride_ / std::max<int64_t>(ni, 1));
    parallel_rows(ctx.pool, outer_ * ni, avg, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t o = t / ni, i = t % ni;
        std::memcpy(y + o * out_stride_ + off_[i], src[i] + o * len_[i], static_cast<size_t>(len_[i]) * sizeof(float));
      }
    });
  }

 private:
  int64_t outer_ = 1, out_stride_ = 0;
  std::vector<int64_t> len_, off_;
  std::vector<const float*> src_;
};

// ----------------------------------------------------------------- Gather
// f32 tables only (row copies); out-of-range ids give zero rows.
class GatherOpt : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& table = ctx.in(0);
    const int64_t v = table.dim(0), d = table.dim(1), n = ctx.in(1).size();
    const int32_t* ids = ctx.in(1).data<int32_t>();
    const float* t = table.data<float>();
    float* y = ctx.out(0).data<float>();
    parallel_rows(ctx.pool, n, d, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        const int64_t r = ids[i];
        if (r < 0 || r >= v) std::memset(y + i * d, 0, static_cast<size_t>(d) * sizeof(float));
        else std::memcpy(y + i * d, t + r * d, static_cast<size_t>(d) * sizeof(float));
      }
    });
  }
};

bool gather_supported(const KernelContext& ctx) { return ctx.in(0).dtype() == DType::kF32 && ctx.in(0).rank() == 2; }

// ----------------------------------------------------------------- MatMul
// Float GEMM per batch entry with the float_opt micro-kernels: work items are
// (batch, block of mb rows); each packs its batch's B into panels of nr
// columns ([K][nr], zero padded) in thread scratch unless the previous item
// of that thread already did, then runs one GEMM block per K block. A rows
// are read in place. Float accumulation with FMA (as Conv2D/Dense float_opt),
// then y *= scale when scale != 1.
template <class B>
class MatMulOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    nr_ = B::nr_for(g_.n);
    panels_ = (g_.n + nr_ - 1) / nr_;
    kc_ = g_.k <= 512 ? g_.k : round_up((g_.k + (g_.k + 255) / 256 - 1) / ((g_.k + 255) / 256), 4);
    mb_ = 96;
    mblocks_ = (g_.m + mb_ - 1) / mb_;
    zero_bias_.assign(static_cast<size_t>(panels_ * nr_), 0.f);
    scratch_.init(ctx.pool->num_threads(), static_cast<size_t>(panels_ * nr_ * g_.k));
    const int64_t macs = g_.batch * g_.m * g_.n * g_.k;
    grain_ = parallel_grain(g_.batch * mblocks_, macs, int64_t(1) << 19);
  }
  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(g.batch * mblocks_, grain_, [&](int64_t t0, int64_t t1) {
      ScratchLease lease(scratch_);
      float* bp = lease.data();
      int64_t packed = -1;
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t bt = t / mblocks_, m0 = (t % mblocks_) * mb_, rows = std::min(mb_, g.m - m0);
        if (packed != bt) {
          pack_b(b + bt * g.k * g.n, bp);
          packed = bt;
        }
        GemmBlockArgs args;
        args.lda = g.k;
        args.rows = rows;
        args.panel_stride = g.k * nr_;
        args.p0 = 0;
        args.p1 = panels_;
        args.nr = nr_;
        args.c = y + (bt * g.m + m0) * g.n;
        args.ldc = g.n;
        args.ncols = g.n;
        args.act = kActNone;
        for (int64_t k0 = 0; k0 < g.k; k0 += kc_) {
          const int64_t kc = std::min(kc_, g.k - k0);
          args.a = a + (bt * g.m + m0) * g.k + k0;
          args.kc = kc;
          args.bp = bp + k0 * nr_;
          args.bias = k0 == 0 ? zero_bias_.data() : nullptr;
          args.last = k0 + kc >= g.k;
          B::gemm(args);
        }
        if (scale_ != 1.f) B::scale_run(y + (bt * g.m + m0) * g.n, scale_, rows * g.n);
      }
    });
  }

 private:
  // B of one batch entry -> [panels][K][nr].
  void pack_b(const float* bb, float* dst) const {
    const MatMulGeom& g = g_;
    for (int64_t p = 0; p < panels_; ++p) {
      float* d = dst + p * g.k * nr_;
      const int64_t j0 = p * nr_, nv = std::min(nr_, g.n - j0);
      if (!g.transpose_b) {
        for (int64_t k = 0; k < g.k; ++k) {
          std::memcpy(d + k * nr_, bb + k * g.n + j0, static_cast<size_t>(nv) * sizeof(float));
          for (int64_t j = nv; j < nr_; ++j) d[k * nr_ + j] = 0.f;
        }
      } else {
        for (int64_t k = 0; k < g.k; ++k) {
          for (int64_t j = 0; j < nv; ++j) d[k * nr_ + j] = bb[(j0 + j) * g.k + k];
          for (int64_t j = nv; j < nr_; ++j) d[k * nr_ + j] = 0.f;
        }
      }
    }
  }

  MatMulGeom g_;
  float scale_ = 1.f;
  int64_t nr_ = 8, panels_ = 1, kc_ = 0, mb_ = 96, mblocks_ = 1, grain_ = 1;
  std::vector<float> zero_bias_;
  ScratchSlots scratch_;
};

bool matmul_supported(const KernelContext& ctx) {
  return ctx.in(0).dtype() == DType::kF32 && ctx.in(1).dtype() == DType::kF32 && ctx.in(0).dim(-1) > 0;
}

template <class B>
void register_backend(OpRegistry& reg, const char* suffix, int priority, uint32_t features) {
  auto add = [&](const char* op, const char* base, bool (*supports)(const KernelContext&), std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, std::string(base) + suffix, "float_opt", priority, features, supports, create});
  };
  add("Activation", "activation_ext.float_opt", activation_ext_supported, make_kernel<ActivationExtOpt<B>>);
  add("Softmax", "softmax.float_opt", nullptr, make_kernel<SoftmaxOpt<B>>);
  add("LayerNorm", "layernorm.float_opt", nullptr, make_kernel<LayerNormOpt<B>>);
  add("ReduceMean", "reduce_mean.float_opt", nullptr, make_kernel<ReduceMeanOpt<B>>);
  add("FakeQuant", "fake_quant.float_opt", nullptr, make_kernel<FakeQuantOpt<B>>);
  add("AddBroadcast", "add_broadcast.float_opt", nullptr, make_kernel<AddBroadcastOpt<B>>);
  add("MatMul", "matmul.float_opt", matmul_supported, make_kernel<MatMulOpt<B>>);
}

}  // namespace

void float_opt_gelu_simd4(const float* x, float* y, int64_t n, bool fast) {
  if (fast) TrSimd4::unary(kUnGelu, x, y, 0, n);
  else leaf_unary<Simd4>(kUnGelu, x, y, 0, n);
}

void register_float_opt_transformer_kernels(OpRegistry& reg) {
  register_backend<TrSimd4>(reg, "", 40, kFloatOptFeatures);
  reg.add_kernel({"Transpose", "transpose.float_opt", "float_opt", 40, kFloatOptFeatures, transpose_supported, make_kernel<TransposeOpt>});
  reg.add_kernel({"Concat", "concat.float_opt", "float_opt", 40, kFloatOptFeatures, nullptr, make_kernel<ConcatOpt>});
  reg.add_kernel({"Gather", "gather.float_opt", "float_opt", 40, kFloatOptFeatures, gather_supported, make_kernel<GatherOpt>});
#if defined(__x86_64__)
  register_backend<TrAvx2>(reg, "_avx2", kFloatOptAvx2Priority, kFloatOptAvx2Features);
#endif
}

}  // namespace bitino

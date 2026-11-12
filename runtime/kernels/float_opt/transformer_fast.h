// Inexact float32 leaves of the transformer / BHViT operators for the fused
// stages ("*.float_fast" kernels, tier float_opt; transformer_opt.cc). The
// bit-exact stage `neon` never selects float_opt, and the exact float_opt
// kernels (double LayerNorm / ReduceMean, the erfc GELU) stay registered and
// can be forced by name.
//
// Written once against the Simd4 policy (float_simd.h): NEON on AArch64, the
// plain 4-float struct elsewhere. Every leaf uses only lane-wise IEEE
// operations (add, sub, mul, fma, div, min/max, round, compares) in a fixed
// order, plus the AArch64-only pairwise adds of reduce_mean_small, which the
// portable path reproduces operation by operation; so the NEON and the
// portable results are bit-identical, and the error bounds measured on the
// host (tests/cpp/test_transformer_opt.cc) hold for the phone as well.
// Included only by transformer_opt.cc (baseline flags), not by the AVX2
// translation unit.
//
// Numerics (measured bounds against the exact kernels, see the tests and
// docs/numerical-correctness.md):
//   LayerNorm:  float sums in 16 lanes (4 vectors, each lane summing every
//               16th element in index order), mean refined by a second
//               float pass over the deviations (corrected two-pass:
//               mean = m1 + S1/d, var = S2/d - (S1/d)^2; the per-row scalars in
//               float, 4 rows per vector); then y = ((x - mh) - ml) * r [* g] [+ b]
//               with the mean as an unevaluated float sum mh + ml and
//               r = 1/sqrt(var + eps): one multiply instead of a double divide
//               per element.
//   ReduceMean: float sums (len 4 / 8: pairwise trees; otherwise as the
//               LayerNorm row sum), times 1/len when len is a power of two
//               (exact), else one float division per output.
//   Softmax:    exp(x - max) with only the lower clamp (x - max <= 0), one
//               reciprocal per row and a multiply per element.
//   GELU:       x * Phi(x), Phi from erfc as the exact kernel's gelu_phi_large
//               for every lane (no small-|x| branch, no division:
//               t = 1/(1 + z/2) by a float reciprocal estimate refined with two
//               Newton steps), cut-offs as the exact kernel.
#pragma once

#include <cstdint>

#include "kernels/float_opt/float_simd.h"
#include "kernels/float_opt/transformer_leaf.h"

namespace bitino {
namespace fopt {
namespace {

// -------------------------------------------------------------- row sums
// Lane-parallel float sums: 4 accumulators of W lanes, lane j of accumulator a adding x[4W k + aW + j] in
// index order (a W-vector tail goes to a0, the last partial vector, padded, to a1), combined as
// (a0 + a1) + (a2 + a3). The horizontal sum of that vector is the row sum.
template <class V>
inline typename V::T fast_rowsum_vec(const float* x, int64_t n) {
  using T = typename V::T;
  constexpr int W = V::W;
  T a0 = V::dup(0.f), a1 = a0, a2 = a0, a3 = a0;
  int64_t i = 0;
  for (; i + 4 * W <= n; i += 4 * W) {
    a0 = V::add(a0, V::load(x + i));
    a1 = V::add(a1, V::load(x + i + W));
    a2 = V::add(a2, V::load(x + i + 2 * W));
    a3 = V::add(a3, V::load(x + i + 3 * W));
  }
  for (; i + W <= n; i += W) a0 = V::add(a0, V::load(x + i));
  if (i < n) {
    float buf[W];
    for (int j = 0; j < W; ++j) buf[j] = i + j < n ? x[i + j] : 0.f;
    a1 = V::add(a1, V::load(buf));
  }
  return V::add(V::add(a0, a1), V::add(a2, a3));
}

template <class V>
inline float fast_rowsum(const float* x, int64_t n) {
  return V::hsum(fast_rowsum_vec<V>(x, n));
}

// Sums of (x - m) and (x - m)^2 over x[0, n) as lane vectors (layout of fast_rowsum_vec; squares accumulated
// with FMA). Tail lanes are padded with m (deviation 0).
template <class V>
inline void fast_rowdev_vec(const float* x, int64_t n, float m, typename V::T* s1, typename V::T* s2) {
  using T = typename V::T;
  constexpr int W = V::W;
  const T mv = V::dup(m);
  T d0 = V::dup(0.f), d1 = d0, d2 = d0, d3 = d0, q0 = d0, q1 = d0, q2 = d0, q3 = d0;
  int64_t i = 0;
  for (; i + 4 * W <= n; i += 4 * W) {
    const T t0 = V::sub(V::load(x + i), mv), t1 = V::sub(V::load(x + i + W), mv);
    const T t2 = V::sub(V::load(x + i + 2 * W), mv), t3 = V::sub(V::load(x + i + 3 * W), mv);
    d0 = V::add(d0, t0);
    d1 = V::add(d1, t1);
    d2 = V::add(d2, t2);
    d3 = V::add(d3, t3);
    q0 = V::fma(t0, t0, q0);
    q1 = V::fma(t1, t1, q1);
    q2 = V::fma(t2, t2, q2);
    q3 = V::fma(t3, t3, q3);
  }
  for (; i + W <= n; i += W) {
    const T t = V::sub(V::load(x + i), mv);
    d0 = V::add(d0, t);
    q0 = V::fma(t, t, q0);
  }
  if (i < n) {
    float buf[W];
    for (int j = 0; j < W; ++j) buf[j] = i + j < n ? x[i + j] : m;
    const T t = V::sub(V::load(buf), mv);
    d1 = V::add(d1, t);
    q1 = V::fma(t, t, q1);
  }
  *s1 = V::add(V::add(d0, d1), V::add(d2, d3));
  *s2 = V::add(V::add(q0, q1), V::add(q2, q3));
}

// [hsum(a), hsum(b), hsum(c), hsum(d)], each (v0 + v1) + (v2 + v3) as Simd4::hsum (FADDP tree on AArch64).
inline f32x4 hsum4(f32x4 a, f32x4 b, f32x4 c, f32x4 d) {
#if defined(__aarch64__)
  return vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d));
#elif BF_FLOAT_OPT_NEON32
  const float32x4_t r = {hsum(a), hsum(b), hsum(c), hsum(d)};
  return r;
#else
  return {{hsum(a), hsum(b), hsum(c), hsum(d)}};
#endif
}

inline f32x4 sqrt4(f32x4 v) {
#if defined(__aarch64__)
  return vsqrtq_f32(v);
#elif BF_FLOAT_OPT_NEON32
  const float32x4_t r = {__builtin_sqrtf(v[0]), __builtin_sqrtf(v[1]), __builtin_sqrtf(v[2]), __builtin_sqrtf(v[3])};
  return r;
#else
  return {{__builtin_sqrtf(v.v[0]), __builtin_sqrtf(v.v[1]), __builtin_sqrtf(v.v[2]), __builtin_sqrtf(v.v[3])}};
#endif
}

template <class V, bool G, bool B>
inline void fast_ln_normalize(const float* x, const float* g, const float* b, float mhs, float mls, float rs, float* y, int64_t d) {
  using T = typename V::T;
  constexpr int W = V::W;
  const T mh = V::dup(mhs), ml = V::dup(mls), r = V::dup(rs);
  auto one = [&](T xv, T gv, T bv) {
    T t = V::mul(V::sub(V::sub(xv, mh), ml), r);
    if (G && B) return V::fma(t, gv, bv);
    if (G) return V::mul(t, gv);
    if (B) return V::add(t, bv);
    return t;
  };
  int64_t i = 0;
  for (; i + 2 * W <= d; i += 2 * W) {
    const T y0 = one(V::load(x + i), G ? V::load(g + i) : r, B ? V::load(b + i) : r);
    const T y1 = one(V::load(x + i + W), G ? V::load(g + i + W) : r, B ? V::load(b + i + W) : r);
    V::store(y + i, y0);
    V::store(y + i + W, y1);
  }
  for (; i + W <= d; i += W) V::store(y + i, one(V::load(x + i), G ? V::load(g + i) : r, B ? V::load(b + i) : r));
  if (i < d) {
    float bx[W], bg[W], bb[W];
    for (int j = 0; j < W; ++j) {
      const bool in = i + j < d;
      bx[j] = in ? x[i + j] : 0.f;
      bg[j] = G && in ? g[i + j] : 0.f;
      bb[j] = B && in ? b[i + j] : 0.f;
    }
    V::store(bx, one(V::load(bx), V::load(bg), V::load(bb)));
    for (int j = 0; j < W; ++j)  // fixed trip count (a variable-length copy becomes a memcpy call)
      if (i + j < d) y[i + j] = bx[j];
  }
}

// LayerNorm over rows [r0, r1) of d floats (g, b: per-column gamma / beta or null). Rows go in groups of 4
// (the last group repeats its last row) so that the per-row scalars are computed for 4 rows at once, lane j
// for row j; every row's result depends only on its own data. Per row:
//   m1 = sum(x) / d;  s1 = sum(x - m1), s2 = sum((x - m1)^2);  c = s1 / d
//   mh + ml = m1 + c (TwoSum);  var = max(s2 / d - c * c, 0);  r = 1 / sqrt(var + eps)
//   y = ((x - mh) - ml) * r  [* g]  [+ b]  (one FMA with both)
template <class V, bool G, bool B>
void fast_layernorm_rows_t(const float* x, const float* g, const float* b, int64_t d, float eps, float* y, int64_t r0, int64_t r1) {
  using T = f32x4;
  const T dv = dup(static_cast<float>(d)), ev = dup(eps), zero = dup(0.f);
  for (int64_t r = r0; r < r1; r += 4) {
    const float* xr[4];
    for (int j = 0; j < 4; ++j) xr[j] = x + (r + j < r1 ? r + j : r1 - 1) * d;
    const T m1 = div(hsum4(fast_rowsum_vec<V>(xr[0], d), fast_rowsum_vec<V>(xr[1], d), fast_rowsum_vec<V>(xr[2], d),
                           fast_rowsum_vec<V>(xr[3], d)),
                     dv);
    alignas(16) float m1s[4];
    store(m1s, m1);
    T d1[4], d2[4];
    for (int j = 0; j < 4; ++j) fast_rowdev_vec<V>(xr[j], d, m1s[j], &d1[j], &d2[j]);
    const T c = div(hsum4(d1[0], d1[1], d1[2], d1[3]), dv);
    const T q = div(hsum4(d2[0], d2[1], d2[2], d2[3]), dv);
    const T mh = add(m1, c), bb = sub(mh, m1);
    const T ml = add(sub(m1, sub(mh, bb)), sub(c, bb));  // TwoSum error of m1 + c
    const T var = vmax(sub(q, mul(c, c)), zero);
    const T rr = div(dup(1.f), sqrt4(add(var, ev)));
    alignas(16) float mhs[4], mls[4], rs[4];
    store(mhs, mh);
    store(mls, ml);
    store(rs, rr);
    for (int j = 0; j < 4 && r + j < r1; ++j) fast_ln_normalize<V, G, B>(xr[j], g, b, mhs[j], mls[j], rs[j], y + (r + j) * d, d);
  }
}

template <class V>
void fast_layernorm_rows(const float* x, const float* g, const float* b, int64_t d, double eps, float* y, int64_t r0, int64_t r1) {
  const float e = static_cast<float>(eps);
  if (g && b) fast_layernorm_rows_t<V, true, true>(x, g, b, d, e, y, r0, r1);
  else if (g) fast_layernorm_rows_t<V, true, false>(x, g, b, d, e, y, r0, r1);
  else if (b) fast_layernorm_rows_t<V, false, true>(x, g, b, d, e, y, r0, r1);
  else fast_layernorm_rows_t<V, false, false>(x, g, b, d, e, y, r0, r1);
}

// ------------------------------------------------------------ ReduceMean
// Means of outputs [o0, o1) over `len` contiguous floats each (inner == 1).
// len 4 / 8: four outputs per step with pairwise adds, per output
//   len 4: (x0 + x1) + (x2 + x3)
//   len 8: ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))
// (FADDP trees on AArch64; the same additions in the same order elsewhere).
// Other lengths: fast_rowsum. The sum is scaled by inv = 1/len when len is a
// power of two (exact), else divided by len.
#if defined(__aarch64__)
inline float32x4_t reduce4_len4(const float* x) {
  const float32x4_t p01 = vpaddq_f32(vld1q_f32(x), vld1q_f32(x + 4)), p23 = vpaddq_f32(vld1q_f32(x + 8), vld1q_f32(x + 12));
  return vpaddq_f32(p01, p23);
}
inline float32x4_t reduce4_len8(const float* x) {
  const float32x4_t a = vpaddq_f32(vld1q_f32(x), vld1q_f32(x + 4)), b = vpaddq_f32(vld1q_f32(x + 8), vld1q_f32(x + 12));
  const float32x4_t c = vpaddq_f32(vld1q_f32(x + 16), vld1q_f32(x + 20)), d = vpaddq_f32(vld1q_f32(x + 24), vld1q_f32(x + 28));
  return vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d));
}
#endif

inline float sum_len4(const float* x) { return (x[0] + x[1]) + (x[2] + x[3]); }
inline float sum_len8(const float* x) { return ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7])); }

inline bool is_pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

template <class V>
void fast_reduce_mean_rows(const float* x, int64_t len, float* y, int64_t o0, int64_t o1) {
  const float cnt = static_cast<float>(len);
  const bool p2 = is_pow2(len);
  const float inv = 1.f / cnt;  // exact for powers of two
  int64_t o = o0;
  if (len == 4 || len == 8) {
#if defined(__aarch64__)
    const float32x4_t iv = vdupq_n_f32(inv);
    if (len == 8)
      for (; o + 4 <= o1; o += 4) vst1q_f32(y + o, vmulq_f32(reduce4_len8(x + o * 8), iv));
    else
      for (; o + 4 <= o1; o += 4) vst1q_f32(y + o, vmulq_f32(reduce4_len4(x + o * 4), iv));
#endif
    for (; o < o1; ++o) y[o] = (len == 8 ? sum_len8(x + o * 8) : sum_len4(x + o * 4)) * inv;
    return;
  }
  for (; o < o1; ++o) {
    const float s = fast_rowsum<V>(x + o * len, len);
    y[o] = p2 ? s * inv : s / cnt;
  }
}

// --------------------------------------------------------------- Softmax
// 2^n from z = fma(x, log2(e), 0x1.8p23) (n = z - 0x1.8p23 is in the low mantissa bits of z, |n| <= 126):
// bits(z) << 23 + bits(1.0f).
inline f32x4 exp_scale(f32x4 z) {
#if defined(__aarch64__) || BF_FLOAT_OPT_NEON32
  return vreinterpretq_f32_s32(vaddq_s32(vshlq_n_s32(vreinterpretq_s32_f32(z), 23), vdupq_n_s32(0x3f800000)));
#else
  f32x4 r;
  for (int i = 0; i < 4; ++i) {
    uint32_t u;
    __builtin_memcpy(&u, &z.v[i], 4);
    u = (u << 23) + 0x3f800000u;
    __builtin_memcpy(&r.v[i], &u, 4);
  }
  return r;
#endif
}

// exp(x) for x <= 0 (NaN propagates), x clamped below at -87.33654 (2^-126 instead of a subnormal / 0):
// n = round(x / ln2) by the shift trick, r = x - n ln2 (Cephes two-part ln2), exp(x) = 2^n (1 + r + r^2 P(r))
// with the Cephes expf polynomial P (as vexp; pairs evaluated Estrin-style), the last step as one FMA.
inline f32x4 vexp_le0(f32x4 x) {
  x = vmax(x, dup(-87.33654f));
  const f32x4 shift = dup(12582912.0f);  // 0x1.8p23
  const f32x4 z = fma_(x, dup(1.44269504088896341f), shift);
  const f32x4 n = sub(z, shift);
  f32x4 r = fma_(n, dup(-0.693359375f), x);
  r = fma_(n, dup(2.12194440e-4f), r);
  const f32x4 r2 = mul(r, r);
  const f32x4 p01 = fma_(dup(1.9875691500e-4f), r, dup(1.3981999507e-3f));
  const f32x4 p23 = fma_(dup(8.3334519073e-3f), r, dup(4.1665795894e-2f));
  const f32x4 p45 = fma_(dup(1.6666665459e-1f), r, dup(5.0000001201e-1f));
  f32x4 p = fma_(p01, r2, p23);
  p = fma_(p, r2, p45);
  p = fma_(p, r2, r);  // e^r - 1
  const f32x4 sc = exp_scale(z);
  return fma_(p, sc, sc);
}

// Softmax over rows [r0, r1) of c floats: y = exp(x - max) * (1 / sum).
template <class V, int U = 2>
void fast_softmax_rows(const float* x, float* y, int64_t c, int64_t r0, int64_t r1) {
  static_assert(U == 2 || U == 4, "2 or 4 vectors per step");
  using T = typename V::T;
  constexpr int W = V::W;
  for (int64_t r = r0; r < r1; ++r) {
    const float* xr = x + r * c;
    float* yr = y + r * c;
    float m = xr[0];
    int64_t i = 0;
    if (c >= 2 * W) {
      T m0 = V::load(xr), m1 = V::load(xr + W);
      for (i = 2 * W; i + 2 * W <= c; i += 2 * W) {
        m0 = V::max(m0, V::load(xr + i));
        m1 = V::max(m1, V::load(xr + i + W));
      }
      m = V::hmax(V::max(m0, m1));
    }
    for (; i < c; ++i) m = xr[i] > m ? xr[i] : m;
    const T mv = V::dup(m);
    T s0 = V::dup(0.f), s1 = s0, s2 = s0, s3 = s0;
    i = 0;
    for (; i + U * W <= c; i += U * W) {
      T e[U];
      for (int u = 0; u < U; ++u) e[u] = vexp_le0(V::sub(V::load(xr + i + u * W), mv));
      for (int u = 0; u < U; ++u) V::store(yr + i + u * W, e[u]);
      s0 = V::add(s0, e[0]);
      s1 = V::add(s1, e[1]);
      if constexpr (U == 4) {
        s2 = V::add(s2, e[2]);
        s3 = V::add(s3, e[3]);
      }
    }
    for (; i + W <= c; i += W) {
      const T e = vexp_le0(V::sub(V::load(xr + i), mv));
      V::store(yr + i, e);
      s0 = V::add(s0, e);
    }
    if (i < c) {
      float buf[W];
      for (int j = 0; j < W; ++j) buf[j] = i + j < c ? xr[i + j] : -1e30f;  // exp -> 2^-126, masked below
      const T e = vexp_le0(V::sub(V::load(buf), mv));
      V::store(buf, e);
      float tb[W];
      for (int j = 0; j < W; ++j) tb[j] = i + j < c ? buf[j] : 0.f;
      s1 = V::add(s1, V::load(tb));
      for (int j = 0; j < W; ++j)
        if (i + j < c) yr[i + j] = buf[j];
    }
    const float s = V::hsum(V::add(V::add(s0, s1), V::add(s2, s3)));
    const T inv = V::dup(1.f / s);
    i = 0;
    for (; i + 2 * W <= c; i += 2 * W) {
      V::store(yr + i, V::mul(V::load(yr + i), inv));
      V::store(yr + i + W, V::mul(V::load(yr + i + W), inv));
    }
    for (; i + W <= c; i += W) V::store(yr + i, V::mul(V::load(yr + i), inv));
    if (i < c) {
      float buf[W];
      for (int j = 0; j < W; ++j) buf[j] = i + j < c ? yr[i + j] : 0.f;
      V::store(buf, V::mul(V::load(buf), inv));
      for (int j = 0; j < W; ++j)
        if (i + j < c) yr[i + j] = buf[j];
    }
  }
}

// ------------------------------------------------------------------ GELU
// x * Phi(x) with Phi from erfc for every lane (gelu_phi_large of transformer_leaf.h: no small-|x| branch, whose
// Taylor series only tightens the error near 0), cut-offs and overrides of gelu_out (y = x for x > 10, x * 0 for
// x <= kGeluCutoff, so sign(y) = sign(x) elsewhere, as the exact kernel). U vectors per step (independent
// chains for the long FMA / divide latencies; 4 measured fastest on the Cortex-X3).
template <class V>
inline typename V::T vgelu_fast(typename V::T x) {
  const typename V::T xc = gelu_clamp<V>(x);
  return gelu_out<V>(x, gelu_phi_large<V>(xc, V::abs(xc)));
}

template <class V, int U = 4>
void fast_gelu(const float* x, float* y, int64_t i0, int64_t i1) {
  constexpr int W = V::W;
  int64_t i = i0;
  for (; i + U * W <= i1; i += U * W) {
    typename V::T v[U];
    for (int u = 0; u < U; ++u) v[u] = vgelu_fast<V>(V::load(x + i + u * W));
    for (int u = 0; u < U; ++u) V::store(y + i + u * W, v[u]);
  }
  for (; i + W <= i1; i += W) V::store(y + i, vgelu_fast<V>(V::load(x + i)));
  if (i < i1) {
    float buf[W];
    for (int j = 0; j < W; ++j) buf[j] = i + j < i1 ? x[i + j] : 0.f;
    V::store(buf, vgelu_fast<V>(V::load(buf)));
    for (int j = 0; j < W; ++j)
      if (i + j < i1) y[i + j] = buf[j];
  }
}

}  // namespace
}  // namespace fopt
}  // namespace bitino

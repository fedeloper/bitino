// float_opt leaf routines of the transformer / BHViT operators, written once
// against the vector policy of float_opt_leaf.h and instantiated with Simd4
// (NEON / portable, baseline flags) and Simd8 (AVX2 + FMA, float_opt_avx2.cc).
// Same linkage rule as float_opt_leaf.h: internal linkage only, no inline
// function with external linkage (no std::), tails go through the vector code
// on a padded copy so there is no separate scalar math.
//
// Extra policy functions used here (scalar semantics):
//   sub(a,b) fma(a,b,c) = a*b+c with one rounding   round(v) = nearbyint (ties to even)
//   pow2i(n) = 2^n for integer-valued n in [-126, 127]   min max abs
//   select_lt(a,b,x,y) = a < b ? x : y   select_gt(a,b,x,y) = a > b ? x : y
//   hsum(v) hmax(v): horizontal sum / max of the lanes
//
// Numerics (kernel error bounds, measured in tests/cpp/test_transformer_opt.cc):
//   vexp:     Cephes expf reduction + degree-6 polynomial, about 1 ulp; inputs
//             are clamped to [-87.33654, 88.0], so exp(x) for x < -87.34 is
//             2^-126 instead of a subnormal/0.
//   GELU:     x * Phi(x); for |x| < 1/sqrt(2) Phi = (1 + erf(x/sqrt(2)))/2 with the
//             Taylor series of erf; otherwise Phi from erfc with exp(-x^2/2) on the exact
//             split x^2 = q + ql (FMA), so there is no cancellation for x < 0;
//             erfc(z) = t * exp(-z^2 + P(t)), t = 1/(1 + z/2) (Numerical Recipes
//             erfcc, fractional error < 1.2e-7). The double reference has its own
//             cancellation error for x < -5.5 (1 + erf rounds to multiples of
//             2^-53) and is exactly -0 for x <= kGeluCutoff; the kernel returns
//             x * 0 (= -0) there as well, so sign(gelu) always equals the
//             reference's (BiBERT binarizes GELU outputs with sign_zero).
//             Max error: |y - ref| <= 6e-7 |ref| + 2^-52 |x| (measured worst
//             4.4e-7 relative, at x = -3.81: the erfc fit plus float rounding of
//             its polynomial).
//   sigmoid:  1 / (1 + vexp(-x)); tanh: odd polynomial for |x| < 0.625 (Cephes
//             tanhf), else 1 - 2 / (vexp(2|x|) + 1) with the sign of x. Max
//             error 4e-7 |ref| + 1e-37 (measured 0.47 / 0.30 of that).
//   hardswish, sign, sign_zero, FakeQuant: the reference's float operations,
//             bit-exact.
//   Softmax:  vexp(x - max) (x - max in float, as the reference), float sums
//             in W lanes, one IEEE division per element; |y - ref| <= 2e-6 |ref|
//             + 1e-12 (measured 0.14 of that).
#pragma once

#include <cstdint>

namespace bitino {
namespace fopt {
namespace {

// Largest float x for which the double reference GELU x*0.5*(1+erf(x/sqrt(2)))
// is exactly -0 (1 + erf(x/sqrt(2)) == 0 in double, glibc and bionic libm).
constexpr float kGeluCutoff = -8.37438965f;

// exp(x) with x clamped to [-87.33654, 88.0].
template <class V>
inline typename V::T vexp(typename V::T x) {
  using T = typename V::T;
  x = V::min(V::max(x, V::dup(-87.33654f)), V::dup(88.0f));
  const T n = V::round(V::mul(x, V::dup(1.44269504088896341f)));
  T r = V::fma(n, V::dup(-0.693359375f), x);
  r = V::fma(n, V::dup(2.12194440e-4f), r);
  T p = V::dup(1.9875691500e-4f);
  p = V::fma(p, r, V::dup(1.3981999507e-3f));
  p = V::fma(p, r, V::dup(8.3334519073e-3f));
  p = V::fma(p, r, V::dup(4.1665795894e-2f));
  p = V::fma(p, r, V::dup(1.6666665459e-1f));
  p = V::fma(p, r, V::dup(5.0000001201e-1f));
  p = V::fma(p, V::mul(r, r), r);
  p = V::add(p, V::dup(1.f));
  return V::mul(p, V::pow2i(n));
}

// exp(a + c) where a is exact and possibly large (|a| <= 87) and c is small:
// the range reduction subtracts n*ln2 from a exactly (n*0.693359375 has at
// most 17 significant bits) before c is added.
// The lower clamp (-80) keeps t * exp(..) / 2 in GELU a normal float, so no
// subnormal arithmetic (microcode assists on x86) happens even for discarded lanes.
template <class V>
inline typename V::T vexp_split(typename V::T a, typename V::T c) {
  using T = typename V::T;
  const T s = V::min(V::max(V::add(a, c), V::dup(-80.0f)), V::dup(88.0f));
  const T n = V::round(V::mul(s, V::dup(1.44269504088896341f)));
  T r = V::fma(n, V::dup(-0.693359375f), a);
  r = V::add(r, c);
  r = V::fma(n, V::dup(2.12194440e-4f), r);
  // keep the polynomial in its accurate range when a + c was clamped
  r = V::min(V::max(r, V::dup(-0.75f)), V::dup(0.75f));
  T p = V::dup(1.9875691500e-4f);
  p = V::fma(p, r, V::dup(1.3981999507e-3f));
  p = V::fma(p, r, V::dup(8.3334519073e-3f));
  p = V::fma(p, r, V::dup(4.1665795894e-2f));
  p = V::fma(p, r, V::dup(1.6666665459e-1f));
  p = V::fma(p, r, V::dup(5.0000001201e-1f));
  p = V::fma(p, V::mul(r, r), r);
  p = V::add(p, V::dup(1.f));
  return V::mul(p, V::pow2i(n));
}

// GELU (erf form) on one vector, see the header comment. Split into parts so the
// NEON driver (gelu_neon in transformer_opt.cc) can skip the small-|x| branch for
// vectors without such lanes; vgelu composes them unchanged (lane-wise pure ops).
//
// |x| > 10 is overridden in gelu_out (y = x, or -0 under the cutoff): evaluating
// the formulas on x clamped to [-20, 20] keeps every intermediate finite and
// normal (huge |x| would give inf / NaN / subnormal lanes).
template <class V>
inline typename V::T gelu_clamp(typename V::T x) {
  return V::min(V::max(x, V::dup(-20.f)), V::dup(20.f));
}

// Phi(xc) from erfc (|x| >= 1/sqrt(2) lanes); ax = |xc|.
template <class V>
inline typename V::T gelu_phi_large(typename V::T xc, typename V::T ax) {
  using T = typename V::T;
  const T q = V::mul(xc, xc);
  const T ql = V::fma(xc, xc, V::sub(V::dup(0.f), q));  // xc*xc - q exactly
  const T a = V::mul(q, V::dup(-0.5f));
  const T z = V::mul(ax, V::dup(0.70710678118654752f));
  const T t = V::div(V::dup(1.f), V::fma(z, V::dup(0.5f), V::dup(1.f)));
  T p = V::dup(0.17087277f);
  p = V::fma(p, t, V::dup(-0.82215223f));
  p = V::fma(p, t, V::dup(1.48851587f));
  p = V::fma(p, t, V::dup(-1.13520398f));
  p = V::fma(p, t, V::dup(0.27886807f));
  p = V::fma(p, t, V::dup(-0.18628806f));
  p = V::fma(p, t, V::dup(0.09678418f));
  p = V::fma(p, t, V::dup(0.37409196f));
  p = V::fma(p, t, V::dup(1.00002368f));
  p = V::fma(p, t, V::dup(-1.26551223f));
  const T c = V::fma(ql, V::dup(-0.5f), p);
  const T half_erfc = V::mul(V::mul(t, vexp_split<V>(a, c)), V::dup(0.5f));
  return V::select_lt(xc, V::dup(0.f), half_erfc, V::sub(V::dup(1.f), half_erfc));
}

// |x| < 1/sqrt(2) (|z| < 1/2): Phi = 1/2 + erf(z)/2 with the Taylor series of
// erf to z^17 (truncation < 1e-10 relative); avoids the erfc fit's 1e-7 error
// where Phi is computed as 1 - erfc/2.
template <class V>
inline typename V::T gelu_phi_small(typename V::T xc) {
  using T = typename V::T;
  const T zs = V::mul(xc, V::dup(0.70710678118654752f)), z2 = V::mul(zs, zs);
  T e = V::dup(1.6462114365889246e-06f);
  e = V::fma(e, z2, V::dup(-1.492565035840625e-05f));
  e = V::fma(e, z2, V::dup(0.00012055332981789664f));
  e = V::fma(e, z2, V::dup(-0.0008548327023450852f));
  e = V::fma(e, z2, V::dup(0.005223977625442188f));
  e = V::fma(e, z2, V::dup(-0.026866170645131252f));
  e = V::fma(e, z2, V::dup(0.11283791670955126f));
  e = V::fma(e, z2, V::dup(-0.37612638903183754f));
  e = V::fma(e, z2, V::dup(1.1283791670955126f));
  return V::fma(V::mul(zs, e), V::dup(0.5f), V::dup(0.5f));
}

// lanes of ax below this use gelu_phi_small
constexpr float kGeluSmall = 0.70710678118654752f;

template <class V>
inline typename V::T gelu_out(typename V::T x, typename V::T phi) {
  using T = typename V::T;
  T y = V::mul(x, phi);
  y = V::select_gt(x, V::dup(10.f), x, y);                           // Phi rounds to 1 (also x = +inf)
  y = V::select_gt(x, V::dup(kGeluCutoff), y, V::mul(x, V::dup(0.f)));  // reference is -0 for x <= cutoff
  return y;
}

template <class V>
inline typename V::T vgelu(typename V::T x) {
  using T = typename V::T;
  const T xc = gelu_clamp<V>(x);
  const T ax = V::abs(xc);
  T phi = gelu_phi_large<V>(xc, ax);
  phi = V::select_lt(ax, V::dup(kGeluSmall), gelu_phi_small<V>(xc), phi);
  return gelu_out<V>(x, phi);
}

template <class V>
inline typename V::T vsigmoid(typename V::T x) {
  return V::div(V::dup(1.f), V::add(V::dup(1.f), vexp<V>(V::sub(V::dup(0.f), x))));
}

template <class V>
inline typename V::T vtanh(typename V::T x) {
  using T = typename V::T;
  const T ax = V::abs(x);
  const T z = V::mul(x, x);
  T p = V::dup(-5.70498872745e-3f);
  p = V::fma(p, z, V::dup(2.06390887954e-2f));
  p = V::fma(p, z, V::dup(-5.37397155531e-2f));
  p = V::fma(p, z, V::dup(1.33314422036e-1f));
  p = V::fma(p, z, V::dup(-3.33332819422e-1f));
  const T small = V::fma(V::mul(p, z), x, x);
  const T e = vexp<V>(V::mul(V::min(ax, V::dup(9.f)), V::dup(2.f)));
  const T big = V::sub(V::dup(1.f), V::div(V::dup(2.f), V::add(e, V::dup(1.f))));  // tanh(|x|)
  const T sbig = V::select_lt(x, V::dup(0.f), V::sub(V::dup(0.f), big), big);
  return V::select_lt(ax, V::dup(0.625f), small, sbig);
}

// Reference: v * min(max(v + 3, 0), 6) / 6 in float (bit-exact).
template <class V>
inline typename V::T vhardswish(typename V::T x) {
  return V::div(V::mul(x, V::clamp(V::add(x, V::dup(3.f)), V::dup(0.f), V::dup(6.f))), V::dup(6.f));
}

// Unary transcendental activation codes.
enum : int32_t { kUnGelu = 0, kUnSigmoid = 1, kUnTanh = 2, kUnHardswish = 3 };

template <class V>
inline typename V::T vunary(int32_t op, typename V::T x) {
  switch (op) {
    case kUnGelu: return vgelu<V>(x);
    case kUnSigmoid: return vsigmoid<V>(x);
    case kUnTanh: return vtanh<V>(x);
    default: return vhardswish<V>(x);
  }
}

template <class V, int32_t OP>
void leaf_unary_t(const float* x, float* y, int64_t i0, int64_t i1) {
  constexpr int W = V::W;
  int64_t i = i0;
  for (; i + 2 * W <= i1; i += 2 * W) {
    V::store(y + i, vunary<V>(OP, V::load(x + i)));
    V::store(y + i + W, vunary<V>(OP, V::load(x + i + W)));
  }
  for (; i + W <= i1; i += W) V::store(y + i, vunary<V>(OP, V::load(x + i)));
  if (i < i1) {
    float buf[W];
    for (int j = 0; j < W; ++j) buf[j] = i + j < i1 ? x[i + j] : 0.f;
    V::store(buf, vunary<V>(OP, V::load(buf)));
    for (int j = 0; i + j < i1; ++j) y[i + j] = buf[j];
  }
}

// y = f(x) on [i0, i1) for f = gelu, sigmoid, tanh, hardswish.
template <class V>
void leaf_unary(int32_t op, const float* x, float* y, int64_t i0, int64_t i1) {
  switch (op) {
    case kUnGelu: leaf_unary_t<V, kUnGelu>(x, y, i0, i1); break;
    case kUnSigmoid: leaf_unary_t<V, kUnSigmoid>(x, y, i0, i1); break;
    case kUnTanh: leaf_unary_t<V, kUnTanh>(x, y, i0, i1); break;
    default: leaf_unary_t<V, kUnHardswish>(x, y, i0, i1); break;
  }
}

// Softmax over rows [r0, r1) of c floats.
template <class V>
void leaf_softmax_rows(const float* x, float* y, int64_t c, int64_t r0, int64_t r1) {
  using T = typename V::T;
  constexpr int W = V::W;
  for (int64_t r = r0; r < r1; ++r) {
    const float* xr = x + r * c;
    float* yr = y + r * c;
    // max (std::max_element semantics for finite inputs)
    float m = xr[0];
    int64_t i = 0;
    if (c >= W) {
      T mv = V::load(xr);
      for (i = W; i + W <= c; i += W) mv = V::max(mv, V::load(xr + i));
      m = V::hmax(mv);
    }
    for (; i < c; ++i) m = xr[i] > m ? xr[i] : m;
    const T mv = V::dup(m);
    T s0 = V::dup(0.f), s1 = s0;
    i = 0;
    for (; i + 2 * W <= c; i += 2 * W) {
      const T e0 = vexp<V>(V::sub(V::load(xr + i), mv)), e1 = vexp<V>(V::sub(V::load(xr + i + W), mv));
      V::store(yr + i, e0);
      V::store(yr + i + W, e1);
      s0 = V::add(s0, e0);
      s1 = V::add(s1, e1);
    }
    for (; i + W <= c; i += W) {
      const T e = vexp<V>(V::sub(V::load(xr + i), mv));
      V::store(yr + i, e);
      s0 = V::add(s0, e);
    }
    float s = V::hsum(V::add(s0, s1));
    if (i < c) {
      float buf[W];
      for (int j = 0; j < W; ++j) buf[j] = i + j < c ? xr[i + j] : m;
      V::store(buf, vexp<V>(V::sub(V::load(buf), mv)));
      // fixed trip count: unrolled in place (a variable-length copy became a
      // memcpy call that spilled every vexp constant once per row)
      for (int j = 0; j < W; ++j)
        if (i + j < c) {
          yr[i + j] = buf[j];
          s += buf[j];
        }
    }
    const T sv = V::dup(s);
    i = 0;
    for (; i + W <= c; i += W) V::store(yr + i, V::div(V::load(yr + i), sv));
    for (; i < c; ++i) yr[i] = yr[i] / s;
  }
}

// FakeQuant (bit-exact with FakeQuantRef): per element
//   t = x*pre; t = t/s; t = t+z; q = clamp(nearbyint(t), lo, hi); y = q*s.
// Rows [r0, r1) of `inner` elements; row r uses channel r % len (inner > 1),
// or, when inner == 1, element i of a row of `len` elements uses channel i.
template <class V>
void leaf_fake_quant(const float* x, const float* s, const float* z, float* y, int64_t len, int64_t inner, int64_t r0, int64_t r1,
                     float pre, float lo, float hi) {
  using T = typename V::T;
  constexpr int W = V::W;
  const T pv = V::dup(pre), lov = V::dup(lo), hiv = V::dup(hi);
  auto q1 = [&](T xv, T sv, T zv) {
    T t = V::mul(xv, pv);
    t = V::div(t, sv);
    t = V::add(t, zv);
    return V::mul(V::clamp(V::round(t), lov, hiv), sv);
  };
  if (inner == 1) {  // rows of `len` channels
    for (int64_t r = r0; r < r1; ++r) {
      const float* xr = x + r * len;
      float* yr = y + r * len;
      int64_t i = 0;
      for (; i + W <= len; i += W) V::store(yr + i, q1(V::load(xr + i), V::load(s + i), V::load(z + i)));
      if (i < len) {
        float bx[W], bs[W], bz[W];
        for (int j = 0; j < W; ++j) {
          const bool in = i + j < len;
          bx[j] = in ? xr[i + j] : 0.f;
          bs[j] = in ? s[i + j] : 1.f;
          bz[j] = in ? z[i + j] : 0.f;
        }
        V::store(bx, q1(V::load(bx), V::load(bs), V::load(bz)));
        for (int j = 0; i + j < len; ++j) yr[i + j] = bx[j];
      }
    }
    return;
  }
  for (int64_t r = r0; r < r1; ++r) {
    const int64_t ch = r % len;
    const T sv = V::dup(s[ch]), zv = V::dup(z[ch]);
    const float* xr = x + r * inner;
    float* yr = y + r * inner;
    int64_t i = 0;
    for (; i + W <= inner; i += W) V::store(yr + i, q1(V::load(xr + i), sv, zv));
    if (i < inner) {
      float bx[W];
      for (int j = 0; j < W; ++j) bx[j] = i + j < inner ? xr[i + j] : 0.f;
      V::store(bx, q1(V::load(bx), sv, zv));
      for (int j = 0; i + j < inner; ++j) yr[i + j] = bx[j];
    }
  }
}

// y[i] = a[i] + b[i] over [0, n) (AddBroadcast runs; same as leaf_add without activation).
template <class V>
void leaf_add_run(const float* a, const float* b, float* y, int64_t n) {
  constexpr int W = V::W;
  int64_t i = 0;
  for (; i + W <= n; i += W) V::store(y + i, V::add(V::load(a + i), V::load(b + i)));
  for (; i < n; ++i) y[i] = a[i] + b[i];
}

// y[i] = a[i] + b over [0, n) (broadcast scalar).
template <class V>
void leaf_add_scalar_run(const float* a, float b, float* y, int64_t n) {
  constexpr int W = V::W;
  const typename V::T bv = V::dup(b);
  int64_t i = 0;
  for (; i + W <= n; i += W) V::store(y + i, V::add(V::load(a + i), bv));
  for (; i < n; ++i) y[i] = a[i] + b;
}

// y[i] = x[i] * s over [0, n) (MatMul scale epilogue).
template <class V>
void leaf_scale_run(float* y, float s, int64_t n) {
  constexpr int W = V::W;
  const typename V::T sv = V::dup(s);
  int64_t i = 0;
  for (; i + W <= n; i += W) V::store(y + i, V::mul(V::load(y + i), sv));
  for (; i < n; ++i) y[i] = y[i] * s;
}

}  // namespace
}  // namespace fopt
}  // namespace bitino

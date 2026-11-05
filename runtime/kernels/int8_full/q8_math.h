// Scalar integer arithmetic of the full-integer operators (stage int8_full).
//
// Single source of truth for the C++ side; the numpy oracle is
// python/bitino/int8_ops.py and docs/int8.md ("Full-integer execution").
//
//   mulhi(x, M)  = (x*M + 2^30) >> 31                 (AArch64 SQRDMULH for M > 0)
//   rshr(x, n)   = (x + 2^(n-1)) >> n, n > 0; x if 0  (AArch64 SRSHL by -n)
//   requant(x, M, shift) = rshr(mulhi(sat32(x << max(shift,0)), M), max(-shift,0))
//
// M is in [2^30, 2^31) (or 0), shift a signed exponent. Both roundings are half
// up (towards +inf); ">>" is an arithmetic shift. Intermediate products are
// computed in int64, as the NEON instructions do internally.
//
// Binarization thresholds: deq(q) = fp32(fp32(q - z) * s) is non-decreasing in
// q (s > 0), so for a float threshold t
//   deq(q) >= t  <=>  q >= qthr_ge(t),   qthr_ge(t) = min{q in [-128,127] : deq(q) >= t}  (128 if none)
//   deq(q) >  t  <=>  q >= qthr_gt(t),   qthr_gt(t) = min{q : deq(q) > t}
// computed here by definition (scan over the 256 codes).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace bitino {
namespace q8 {

inline int32_t sat32(int64_t v) {
  return static_cast<int32_t>(std::min<int64_t>(std::max<int64_t>(v, INT32_MIN), INT32_MAX));
}

inline int64_t mulhi(int64_t x, int64_t m) { return (x * m + (int64_t(1) << 30)) >> 31; }

inline int64_t rshr(int64_t x, int n) { return n > 0 ? (x + (int64_t(1) << (n - 1))) >> n : x; }

inline int64_t requant(int64_t x, int32_t m, int32_t shift) {
  const int left = shift > 0 ? shift : 0, right = shift < 0 ? -shift : 0;
  const int64_t xs = sat32(x * (int64_t(1) << left));
  return rshr(mulhi(xs, m), right);
}

inline int8_t clamp8(int64_t v, int qmin = -128, int qmax = 127) {
  return static_cast<int8_t>(std::min<int64_t>(std::max<int64_t>(v, qmin), qmax));
}

// fp32 value -> int8: clamp in float, then round half to even (nearbyint in the
// default rounding mode = FCVTNS). Exact for every finite v.
inline int8_t round_f32(float v, int qmin, int qmax) {
  const float c = std::min(std::max(v, static_cast<float>(qmin)), static_cast<float>(qmax));
  return static_cast<int8_t>(std::nearbyint(c));
}

inline float deq(int q, float s, int32_t z) { return static_cast<float>(static_cast<int64_t>(q) - z) * s; }

inline int qthr_ge(float t, float s, int32_t z) {
  for (int q = -128; q <= 127; ++q)
    if (deq(q, s, z) >= t) return q;
  return 128;
}

inline int qthr_gt(float t, float s, int32_t z) {
  for (int q = -128; q <= 127; ++q)
    if (deq(q, s, z) > t) return q;
  return 128;
}

// ---- 8-bit log2 code ("l8"; int8_ops.l8_decode / l8_encode, docs/int8.md "log2 residual stream")
// val(c) = 0 (c == 0), sign(c) * ldexp(kL8D[k & 7], e + (k >> 3) - 16) with k = min(|c|, 127) + 1;
// encode(w): n = 8 (Ew - 127) + #{i : t >= kL8B[i]} (Ew biased exponent, t top 7 mantissa bits of |w|),
// j = clamp(n + 127, 0, 127), code = sign(w) * j.
inline constexpr uint32_t kL8DBits[8] = {0x3F800000u, 0x3F8B95C2u, 0x3F9837F0u, 0x3FA5FED7u,
                                         0x3FB504F3u, 0x3FC5672Au, 0x3FD744FDu, 0x3FEAC0C7u};
inline constexpr int kL8B[8] = {6, 18, 31, 45, 61, 78, 97, 117};

inline float f32_of_bits(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
inline uint32_t bits_of_f32(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

inline float l8_decode(int c, int e) {
  if (c == 0) return 0.f;
  const int k = std::min(c < 0 ? -c : c, 127) + 1;
  const float v = std::ldexp(f32_of_bits(kL8DBits[k & 7]), e + (k >> 3) - 16);
  return c < 0 ? -v : v;
}

inline int l8_kf(int t) {
  int n = 0;
  for (int b : kL8B) n += t >= b;
  return n;
}

inline int8_t l8_encode(float w) {
  const uint32_t b = bits_of_f32(w);
  const int ew = static_cast<int>((b >> 23) & 0xFF), t = static_cast<int>((b >> 16) & 0x7F);
  const int j = std::min(std::max(8 * (ew - 127) + l8_kf(t) + 127, 0), 127);
  return static_cast<int8_t>((b >> 31) ? -j : j);
}

// Dequantization of an int8 operand: linear (s, z) or l8 (exponent e); non-decreasing in q either way.
struct Deq {
  bool l8 = false;
  float s = 1.f;
  int32_t z = 0;
  int e = 0;
  float operator()(int q) const { return l8 ? l8_decode(q, e) : static_cast<float>(static_cast<int64_t>(q) - z) * s; }
};

inline int qthr_ge(float t, const Deq& d) {
  for (int q = -128; q <= 127; ++q)
    if (d(q) >= t) return q;
  return 128;
}

inline int qthr_gt(float t, const Deq& d) {
  for (int q = -128; q <= 127; ++q)
    if (d(q) > t) return q;
  return 128;
}

// Integer binarization of an int8 code with precomputed thresholds:
// ge: q >= tge; gt: q >= tgt; sign_zero: +1 if q >= tgt, -1 if q < tge, else 0; gt0: {0,1} = (q >= tgt).
enum class QBin { kGe, kGt, kSignZero, kGt0 };

inline int qbin(QBin m, int q, int tge, int tgt) {
  switch (m) {
    case QBin::kGe: return q >= tge ? 1 : -1;
    case QBin::kGt: return q >= tgt ? 1 : -1;
    case QBin::kSignZero: return q >= tgt ? 1 : (q < tge ? -1 : 0);
    case QBin::kGt0: return q >= tgt ? 1 : 0;
  }
  return 0;
}

// Rounding division half away from zero (average pooling).
inline int64_t rdiv(int64_t a, int64_t c) { return a >= 0 ? (a + c / 2) / c : -((-a + c / 2) / c); }

}  // namespace q8
}  // namespace bitino

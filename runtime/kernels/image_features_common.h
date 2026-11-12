// Shared arithmetic of the image-feature operators (docs/operators_image_features.md):
// ImageGrayU8, FftLogMagnitude, LocalBinaryPattern.
//
// Everything is IEEE double +, -, *, /, sqrt in a fixed order (the build uses -ffp-contract=off), with no
// libm transcendental: twiddle factors come from a fixed polynomial sin/cos with exact integer octant
// reduction, the logarithm from a fixed polynomial. python/bitino/image_ops.py performs the same
// operations, so the reference kernel, the optimized kernels and the numpy executor are bit-identical
// on every platform. The hex constants below are the ones image_ops.py derives (correctly rounded
// (-1)^k/(2k+1)!, (-1)^k/(2k)!, 1/(2k+1), pi/4, sqrt(2), fdlibm ln2_hi/ln2_lo).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "bitino/error.h"

namespace bitino {
namespace imgf {

constexpr double kSinC[8] = {-0x1.5555555555555p-3, 0x1.1111111111111p-7,  -0x1.a01a01a01a01ap-13, 0x1.71de3a556c734p-19,
                             -0x1.ae64567f544e4p-26, 0x1.6124613a86d09p-33, -0x1.ae7f3e733b81fp-41, 0x1.952c77030ad4ap-49};
constexpr double kCosC[9] = {-0x1.0000000000000p-1, 0x1.5555555555555p-5,  -0x1.6c16c16c16c17p-10,
                             0x1.a01a01a01a01ap-16,  -0x1.27e4fb7789f5cp-22, 0x1.1eed8eff8d898p-29,
                             -0x1.93974a8c07c9dp-37, 0x1.ae7f3e733b81fp-45,  -0x1.6827863b97d97p-53};
constexpr double kLogC[11] = {0x1.5555555555555p-2, 0x1.999999999999ap-3, 0x1.2492492492492p-3, 0x1.c71c71c71c71cp-4,
                              0x1.745d1745d1746p-4, 0x1.3b13b13b13b14p-4, 0x1.1111111111111p-4, 0x1.e1e1e1e1e1e1ep-5,
                              0x1.af286bca1af28p-5, 0x1.8618618618618p-5, 0x1.642c8590b2164p-5};
constexpr double kPi4 = 0x1.921fb54442d18p-1;
constexpr double kSqrt2 = 0x1.6a09e667f3bcdp+0;
constexpr double kLn2Hi = 0x1.62e42fee00000p-1;
constexpr double kLn2Lo = 0x1.a39ef35793c76p-33;

// (cos(2 pi t / n), sin(2 pi t / n)), t = 0..n-1 (image_ops.sincos_2pi).
inline void sincos_2pi(int64_t n, std::vector<double>& cos_t, std::vector<double>& sin_t) {
  cos_t.resize(n);
  sin_t.resize(n);
  for (int64_t t = 0; t < n; ++t) {
    const int64_t o = (8 * t) / n;
    int64_t r = 8 * t - o * n;
    if (o % 2 == 1) r = n - r;
    const double x = (kPi4 * static_cast<double>(r)) / static_cast<double>(n);
    const double z = x * x;
    double p = kSinC[7];
    for (int k = 6; k >= 0; --k) p = kSinC[k] + z * p;
    const double s = x + x * (z * p);
    double q = kCosC[8];
    for (int k = 7; k >= 0; --k) q = kCosC[k] + z * q;
    const double c = 1.0 + z * q;
    double ct, st;
    switch (o) {
      case 0: ct = c; st = s; break;
      case 1: ct = s; st = c; break;
      case 2: ct = -s; st = c; break;
      case 3: ct = -c; st = s; break;
      case 4: ct = -c; st = -s; break;
      case 5: ct = -s; st = -c; break;
      case 6: ct = s; st = -c; break;
      default: ct = c; st = -s; break;
    }
    cos_t[t] = ct;
    sin_t[t] = st;
  }
}

// Natural logarithm of a positive normal finite double (image_ops.log_f64).
inline double log_f64(double x) {
  uint64_t bits;
  std::memcpy(&bits, &x, 8);
  int64_t e = static_cast<int64_t>((bits >> 52) & 0x7ff) - 1023;
  const uint64_t mb = (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  double m;
  std::memcpy(&m, &mb, 8);
  if (m > kSqrt2) {
    m = m * 0.5;
    e += 1;
  }
  const double f = m - 1.0;
  const double s = f / (2.0 + f);
  const double z = s * s;
  double p = kLogC[10];
  for (int k = 9; k >= 0; --k) p = kLogC[k] + z * p;
  const double r = z * p;
  const double t = s + s;
  const double k = static_cast<double>(e);
  return k * kLn2Hi + (t + (t * r + k * kLn2Lo));
}

// Radix order of a 1-D transform: 4s, one 2, then odd primes ascending (image_ops.fft_factors).
inline std::vector<int> fft_factors(int64_t n) {
  std::vector<int> f;
  while (n % 4 == 0) { f.push_back(4); n /= 4; }
  if (n % 2 == 0) { f.push_back(2); n /= 2; }
  for (int64_t p = 3; n > 1; p += 2)
    while (n % p == 0) { f.push_back(static_cast<int>(p)); n /= p; }
  return f;
}

// Plan of one 1-D length-n forward transform: factors and W_n^t = (cos, -sin)(2 pi t / n).
struct FftPlan {
  int64_t n = 0;
  std::vector<int> factors;
  std::vector<double> wr, wi;
  int max_radix = 1;
  void init(int64_t len) {
    n = len;
    factors = fft_factors(len);
    std::vector<double> c, s;
    sincos_2pi(len, c, s);
    wr = c;
    wi.resize(len);
    for (int64_t t = 0; t < len; ++t) wi[t] = -s[t];
    max_radix = 1;
    for (int p : factors) max_radix = p > max_radix ? p : max_radix;
  }
};

// One Stockham decimation-in-frequency stage over `lanes` independent transforms stored as
// x[pos * ld + lane] (pos = 0..N-1). V is a vector of `V::kLanes` doubles (scalar double: imgf::Scalar);
// lanes advance in steps of V::kLanes. Mirrors image_ops.fft_axis0 exactly:
//   a_q = x[k + s*(j + m*q)], b_r = DFT_p(a)_r, y[k + s*(p*j + r)] = (j == 0 || r == 0) ? b_r : b_r * W_n^{j r}.
template <class V, int p>
inline void fft_stage_p(const FftPlan& pl, int64_t n, int64_t s, const double* xr, const double* xi, double* yr,
                        double* yi, int64_t ld, int64_t lanes) {
  const int64_t N = pl.n, m = n / p, tstep = N / n, step = N / p;
  constexpr int64_t L = V::kLanes;
  for (int64_t j = 0; j < m; ++j) {
    for (int64_t k = 0; k < s; ++k) {
      const int64_t in0 = (k + s * j) * ld, ins = s * m * ld;
      const int64_t out0 = (k + s * p * j) * ld, outs = s * ld;
      for (int64_t l = 0; l < lanes; l += L) {
        V br[p], bi[p];
        V ar[p], ai[p];
        if constexpr (p == 2) {
          const V a0r = V::load(xr + in0 + l), a0i = V::load(xi + in0 + l);
          const V a1r = V::load(xr + in0 + ins + l), a1i = V::load(xi + in0 + ins + l);
          br[0] = a0r + a1r; bi[0] = a0i + a1i;
          br[1] = a0r - a1r; bi[1] = a0i - a1i;
        } else if constexpr (p == 4) {
          for (int q = 0; q < 4; ++q) { ar[q] = V::load(xr + in0 + q * ins + l); ai[q] = V::load(xi + in0 + q * ins + l); }
          const V t0r = ar[0] + ar[2], t0i = ai[0] + ai[2];
          const V t1r = ar[0] - ar[2], t1i = ai[0] - ai[2];
          const V t2r = ar[1] + ar[3], t2i = ai[1] + ai[3];
          const V t3r = ar[1] - ar[3], t3i = ai[1] - ai[3];
          br[0] = t0r + t2r; bi[0] = t0i + t2i;
          br[1] = t1r + t3i; bi[1] = t1i - t3r;
          br[2] = t0r - t2r; bi[2] = t0i - t2i;
          br[3] = t1r - t3i; bi[3] = t1i + t3r;
        } else {
          // odd prime p, conjugate-pair form (image_ops.fft_axis0): s_q = a_q + a_{p-q}, d_q = a_q - a_{p-q};
          // b_0 = a_0 + s_1 + ... + s_h; A = a_0 + sum_q s_q cos, B = sum_q d_q sin (angles 2 pi qr/p, q order);
          // b_r = (A.re + B.im, A.im - B.re), b_{p-r} = (A.re - B.im, A.im + B.re)
          constexpr int h = (p - 1) / 2;
          const V x0r = V::load(xr + in0 + l), x0i = V::load(xi + in0 + l);
          for (int q = 1; q < p; ++q) { ar[q] = V::load(xr + in0 + q * ins + l); ai[q] = V::load(xi + in0 + q * ins + l); }
          V sr[(p - 1) / 2], si[(p - 1) / 2], dr[(p - 1) / 2], di[(p - 1) / 2];
          for (int q = 1; q <= h; ++q) {
            sr[q - 1] = ar[q] + ar[p - q];
            si[q - 1] = ai[q] + ai[p - q];
            dr[q - 1] = ar[q] - ar[p - q];
            di[q - 1] = ai[q] - ai[p - q];
          }
          V b0r = x0r, b0i = x0i;
          for (int q = 0; q < h; ++q) {
            b0r = b0r + sr[q];
            b0i = b0i + si[q];
          }
          br[0] = b0r;
          bi[0] = b0i;
          for (int r = 1; r <= h; ++r) {
            const V c1 = V::dup(pl.wr[(r % p) * step]), sn1 = V::dup(-pl.wi[(r % p) * step]);
            V Ar = x0r + sr[0] * c1, Ai = x0i + si[0] * c1;
            V Br = dr[0] * sn1, Bi = di[0] * sn1;
            for (int q = 2; q <= h; ++q) {
              const int e = (q * r) % p;
              const V c = V::dup(pl.wr[e * step]), sn = V::dup(-pl.wi[e * step]);
              Ar = Ar + sr[q - 1] * c;
              Ai = Ai + si[q - 1] * c;
              Br = Br + dr[q - 1] * sn;
              Bi = Bi + di[q - 1] * sn;
            }
            br[r] = Ar + Bi;
            bi[r] = Ai - Br;
            br[p - r] = Ar - Bi;
            bi[p - r] = Ai + Br;
          }
        }
        V::store(yr + out0 + l, br[0]);
        V::store(yi + out0 + l, bi[0]);
        for (int r = 1; r < p; ++r) {
          if (j == 0) {
            V::store(yr + out0 + r * outs + l, br[r]);
            V::store(yi + out0 + r * outs + l, bi[r]);
          } else {
            const int64_t idx = (j * r * tstep) % N;
            const V wr = V::dup(pl.wr[idx]), wi = V::dup(pl.wi[idx]);
            V::store(yr + out0 + r * outs + l, br[r] * wr - bi[r] * wi);
            V::store(yi + out0 + r * outs + l, br[r] * wi + bi[r] * wr);
          }
        }
      }
    }
  }
}

template <class V>
inline void fft_stage(const FftPlan& pl, int p, int64_t n, int64_t s, const double* xr, const double* xi, double* yr,
                      double* yi, int64_t ld, int64_t lanes) {
  switch (p) {
    case 2: fft_stage_p<V, 2>(pl, n, s, xr, xi, yr, yi, ld, lanes); break;
    case 3: fft_stage_p<V, 3>(pl, n, s, xr, xi, yr, yi, ld, lanes); break;
    case 4: fft_stage_p<V, 4>(pl, n, s, xr, xi, yr, yi, ld, lanes); break;
    case 5: fft_stage_p<V, 5>(pl, n, s, xr, xi, yr, yi, ld, lanes); break;
    case 7: fft_stage_p<V, 7>(pl, n, s, xr, xi, yr, yi, ld, lanes); break;
    default: fail("FFT radix ", p, " not supported");
  }
}

// Full 1-D transform of `lanes` columns of [N, ld] (re, im) in place; uses (tr, ti) [N, ld] as scratch.
// Radices > 7 are not supported by the unrolled stage (checked by the op definition).
template <class V>
inline void fft_columns(const FftPlan& pl, double* re, double* im, double* tr, double* ti, int64_t ld, int64_t lanes) {
  double *xr = re, *xi = im, *yr = tr, *yi = ti;
  int64_t n = pl.n, s = 1;
  for (int p : pl.factors) {
    fft_stage<V>(pl, p, n, s, xr, xi, yr, yi, ld, lanes);
    std::swap(xr, yr);
    std::swap(xi, yi);
    n /= p;
    s *= p;
  }
  if (xr != re) {
    for (int64_t i = 0; i < pl.n * ld; ++i) {
      re[i] = xr[i];
      im[i] = xi[i];
    }
  }
}

// Scalar "vector" of one lane (reference kernel and portable builds).
struct Scalar {
  static constexpr int64_t kLanes = 1;
  double v;
  static Scalar load(const double* p) { return {*p}; }
  static void store(double* p, Scalar a) { *p = a.v; }
  static Scalar dup(double x) { return {x}; }
  friend Scalar operator+(Scalar a, Scalar b) { return {a.v + b.v}; }
  friend Scalar operator-(Scalar a, Scalar b) { return {a.v - b.v}; }
  friend Scalar operator*(Scalar a, Scalar b) { return {a.v * b.v}; }
};

// Gray value of one RGB pixel (ImageGrayU8): u_c = trunc(clamp(f32(x_c * scale), 0, 255)).
inline int gray_u8(const float* px, float scale, const int64_t* coeffs, int shift) {
  int64_t acc = int64_t{1} << (shift - 1);
  for (int c = 0; c < 3; ++c) {
    float v = px[c] * scale;
    v = v >= 0.f ? v : 0.f;
    v = v < 255.f ? v : 255.f;
    acc += static_cast<int64_t>(static_cast<int>(v)) * coeffs[c];
  }
  return static_cast<int>(acc >> shift);
}

// Maximum radix the unrolled FFT stage handles.
constexpr int kMaxRadix = 7;

}  // namespace imgf
}  // namespace bitino

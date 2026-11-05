// int8_avx2 routines that need AVX2 only (compiled with -mavx2): activation
// max-abs and quantization, and the VPMADDWD GEMM micro-kernel (int8_avx2.h).
#include "kernels/int8_avx2/int8_avx2.h"

#if !defined(__AVX2__)
#error "int8_gemm_avx2.cc must be compiled with -mavx2 (see CMakeLists.txt)"
#endif

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "kernels/int8_avx2/int8_avx2_tile.h"

namespace bitino {
namespace int8_avx2 {
namespace {

struct MaddDot {
  // a: (a0, a1) int16 pair broadcast; w: lane j = (w[j][2c], w[j][2c+1]).
  static __m256i dot(__m256i acc, __m256i a, __m256i w) { return _mm256_add_epi32(acc, _mm256_madd_epi16(a, w)); }
};

// q = clamp(rne(x / s), -127, 127) for 8 lanes as int32. Clamping before the
// conversion equals clamping the rounded value for every finite x (rne is
// monotone and +-127 are integers), and keeps CVTPS2DQ in range.
inline __m256i quant8(__m256 x, __m256 s) {
  const __m256 lo = _mm256_set1_ps(-127.f), hi = _mm256_set1_ps(127.f);
  return _mm256_cvtps_epi32(_mm256_min_ps(hi, _mm256_max_ps(lo, _mm256_div_ps(x, s))));
}

// 8 int32 in [-127, 127] -> 8 int16 (low 128 bits).
inline __m128i narrow16(__m256i v) { return _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)); }

// 8 int32 in [-127, 127] -> 8 bytes of q + 128 (low 64 bits).
inline __m128i narrow_u8(__m256i v) {
  const __m128i h = _mm_add_epi16(narrow16(v), _mm_set1_epi16(128));
  return _mm_packus_epi16(h, h);
}

}  // namespace

float maxabs(const float* x, int64_t n) {
  const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 m0 = _mm256_setzero_ps(), m1 = m0, m2 = m0, m3 = m0;
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    m0 = _mm256_max_ps(m0, _mm256_and_ps(mask, _mm256_loadu_ps(x + i)));
    m1 = _mm256_max_ps(m1, _mm256_and_ps(mask, _mm256_loadu_ps(x + i + 8)));
    m2 = _mm256_max_ps(m2, _mm256_and_ps(mask, _mm256_loadu_ps(x + i + 16)));
    m3 = _mm256_max_ps(m3, _mm256_and_ps(mask, _mm256_loadu_ps(x + i + 24)));
  }
  for (; i + 8 <= n; i += 8) m0 = _mm256_max_ps(m0, _mm256_and_ps(mask, _mm256_loadu_ps(x + i)));
  if (i < n) {  // tail: zero padded (|0| never raises the max)
    float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(tmp, x + i, static_cast<size_t>(n - i) * sizeof(float));
    m0 = _mm256_max_ps(m0, _mm256_and_ps(mask, _mm256_loadu_ps(tmp)));
  }
  __m256 m = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));
  __m128 r = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
  r = _mm_max_ps(r, _mm_movehl_ps(r, r));
  r = _mm_max_ss(r, _mm_shuffle_ps(r, r, 1));
  return _mm_cvtss_f32(r);
}

void quantize_s16(const float* x, int64_t n, float scale, int16_t* q) {
  const __m256 s = _mm256_set1_ps(scale);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256i a = quant8(_mm256_loadu_ps(x + i), s), b = quant8(_mm256_loadu_ps(x + i + 8), s);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + i), _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xd8));
  }
  for (; i < n; i += 8) {
    float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t len = n - i < 8 ? n - i : 8;
    std::memcpy(tmp, x + i, static_cast<size_t>(len) * sizeof(float));
    int16_t out[8];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), narrow16(quant8(_mm256_loadu_ps(tmp), s)));
    std::memcpy(q + i, out, static_cast<size_t>(len) * sizeof(int16_t));
  }
}

void quantize_u8(const float* x, int64_t n, float scale, uint8_t* q) {
  const __m256 s = _mm256_set1_ps(scale);
  const __m256i bias = _mm256_set1_epi16(128);
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i a = quant8(_mm256_loadu_ps(x + i), s), b = quant8(_mm256_loadu_ps(x + i + 8), s);
    const __m256i c = quant8(_mm256_loadu_ps(x + i + 16), s), d = quant8(_mm256_loadu_ps(x + i + 24), s);
    // packs within 128-bit lanes: ab = [a0-3 b0-3 | a4-7 b4-7], cd likewise;
    // packus(ab, cd) = [a0-3 b0-3 c0-3 d0-3 | a4-7 b4-7 c4-7 d4-7] (dwords).
    const __m256i ab = _mm256_add_epi16(_mm256_packs_epi32(a, b), bias);
    const __m256i cd = _mm256_add_epi16(_mm256_packs_epi32(c, d), bias);
    const __m256i u = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd), order);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + i), u);
  }
  for (; i < n; i += 8) {
    float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t len = n - i < 8 ? n - i : 8;
    std::memcpy(tmp, x + i, static_cast<size_t>(len) * sizeof(float));
    uint8_t out[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), narrow_u8(quant8(_mm256_loadu_ps(tmp), s)));
    std::memcpy(q + i, out, static_cast<size_t>(len));
  }
}

void gemm_madd(const GemmArgs& g, int64_t r0, int64_t rows, int64_t p0, int64_t p1) { gemm<MaddDot>(g, r0, rows, p0, p1); }

void pack_w_kn(const int16_t* q, int64_t k, int64_t n, int64_t ldq, int64_t kc, bool vnni, uint8_t* w, int32_t* corr) {
  const int64_t e = vnni ? 4 : 2;  // k per 4-byte chunk
  const int64_t panels = (n + kNR - 1) / kNR;
  alignas(32) int16_t tmp[4][kNR];
  for (int64_t p = 0; p < panels; ++p) {
    const int64_t j0 = p * kNR, nv = n - j0 < kNR ? n - j0 : kNR;
    __m256i s_lo = _mm256_setzero_si256(), s_hi = _mm256_setzero_si256();
    uint8_t* wp = w + p * kc * 64;
    for (int64_t c = 0; c < kc; ++c) {
      __m256i r[4];
      for (int64_t t = 0; t < e; ++t) {
        const int64_t kk = c * e + t;
        if (kk >= k) {
          r[t] = _mm256_setzero_si256();
        } else if (nv == kNR) {
          r[t] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + kk * ldq + j0));
        } else {
          std::memset(tmp[t], 0, sizeof tmp[t]);
          std::memcpy(tmp[t], q + kk * ldq + j0, static_cast<size_t>(nv) * sizeof(int16_t));
          r[t] = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp[t]));
        }
        if (vnni) {
          s_lo = _mm256_add_epi32(s_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r[t])));
          s_hi = _mm256_add_epi32(s_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r[t], 1)));
        }
      }
      __m256i lo, hi;
      if (vnni) {  // column j: bytes (q[4c][j], q[4c+1][j], q[4c+2][j], q[4c+3][j])
        const __m256i m8 = _mm256_set1_epi16(0xff);
        const __m256i p01 = _mm256_or_si256(_mm256_and_si256(r[0], m8), _mm256_slli_epi16(r[1], 8));
        const __m256i p23 = _mm256_or_si256(_mm256_and_si256(r[2], m8), _mm256_slli_epi16(r[3], 8));
        lo = _mm256_unpacklo_epi16(p01, p23);  // columns 0-3 | 8-11
        hi = _mm256_unpackhi_epi16(p01, p23);  // columns 4-7 | 12-15
      } else {     // column j: int16 (q[2c][j], q[2c+1][j])
        lo = _mm256_unpacklo_epi16(r[0], r[1]);
        hi = _mm256_unpackhi_epi16(r[0], r[1]);
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(wp + c * 64), _mm256_permute2x128_si256(lo, hi, 0x20));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(wp + c * 64 + 32), _mm256_permute2x128_si256(lo, hi, 0x31));
    }
    if (vnni) {  // 128 * column sum, modulo 2^32 (wrapping shift)
      alignas(32) int32_t sums[kNR];
      _mm256_store_si256(reinterpret_cast<__m256i*>(sums), _mm256_slli_epi32(s_lo, 7));
      _mm256_store_si256(reinterpret_cast<__m256i*>(sums + 8), _mm256_slli_epi32(s_hi, 7));
      for (int64_t j = 0; j < nv; ++j) corr[j0 + j] = sums[j];
    }
  }
}

}  // namespace int8_avx2
}  // namespace bitino

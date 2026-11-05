// Shared micro-kernel and epilogue of the int8_avx2 GEMMs, instantiated by
// int8_gemm_avx2.cc (VPMADDWD) and int8_gemm_avxvnni.cc (VPDPBUSD). Everything
// here has internal linkage, so each ISA translation unit keeps its own copy
// compiled with its own flags (see int8_avx2.h).
//
// Epilogue, bit-exact with int8_epilogue() (int8_quant.h):
//   _mm256_cvtepi32_ps == static_cast<float>(int32)     (round to nearest)
//   _mm256_mul_ps / _mm256_add_ps: separate fp32 roundings (these TUs are
//   built without -mfma, and -ffp-contract=off is global)
//   relu : max(v, 0)            = v > 0 ? v : 0                        (MAXPS returns
//   relu6: min(6, max(0, v))    = std::min(std::max(v, 0.f), 6.f)       the 2nd operand
//                                                                       unless v > 0 / 6 < m)
#pragma once

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "kernels/int8_avx2/int8_avx2.h"

// Full unrolling of the fixed-trip loops over rows / vectors / accumulator
// sets keeps the accumulator arrays in registers (without it GCC keeps them in
// memory and stores every accumulator on each k step).
#define BF_UNROLL _Pragma("GCC unroll 16")

namespace bitino {
namespace int8_avx2 {
namespace {

inline __m256i first_lanes(int64_t n) {
  const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(n < 0 ? 0 : n > 8 ? 8 : n)), idx);
}

inline __m256 act_f32x8(__m256 v, int32_t act) {
  const __m256 zero = _mm256_setzero_ps();
  if (act == 1) return _mm256_max_ps(v, zero);
  if (act == 2) return _mm256_min_ps(_mm256_set1_ps(6.f), _mm256_max_ps(zero, v));
  return v;
}

inline __m256i bcast4(const uint8_t* p) {
  int32_t v;
  std::memcpy(&v, p, 4);
  return _mm256_set1_epi32(v);
}

// Independent accumulator sets over K chunks for short tiles (latency of the
// dot instruction ~5 cycles, 2 per cycle).
template <int R>
constexpr int ksets() {
  return R >= 4 ? 1 : R == 1 ? 4 : 2;
}

// Rows [0, R) of a tile starting at A row `ar`, panel `wp`, output `y`
// (row stride cols), columns [o0, o0+16).
template <class Dot, int R>
void tile(const GemmArgs& g, const uint8_t* ar, const int8_t* wp, float* y, int64_t o0) {
  constexpr int S = ksets<R>();
  __m256i acc[S][R][2];
  BF_UNROLL
  for (int s = 0; s < S; ++s)
    BF_UNROLL
    for (int i = 0; i < R; ++i) acc[s][i][0] = acc[s][i][1] = _mm256_setzero_si256();
  const int64_t kc = g.kc, lda = g.lda;
  int64_t c = 0;
  for (; c + S <= kc; c += S) {
    BF_UNROLL
    for (int s = 0; s < S; ++s) {
      const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + (c + s) * 64));
      const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + (c + s) * 64 + 32));
      BF_UNROLL
      for (int i = 0; i < R; ++i) {
        const __m256i a = bcast4(ar + i * lda + 4 * (c + s));
        acc[s][i][0] = Dot::dot(acc[s][i][0], a, w0);
        acc[s][i][1] = Dot::dot(acc[s][i][1], a, w1);
      }
    }
  }
  for (; c < kc; ++c) {
    const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + c * 64));
    const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + c * 64 + 32));
    BF_UNROLL
    for (int i = 0; i < R; ++i) {
      const __m256i a = bcast4(ar + i * lda + 4 * c);
      acc[0][i][0] = Dot::dot(acc[0][i][0], a, w0);
      acc[0][i][1] = Dot::dot(acc[0][i][1], a, w1);
    }
  }
  BF_UNROLL
  for (int s = 1; s < S; ++s)
    BF_UNROLL
    for (int i = 0; i < R; ++i) {
      acc[0][i][0] = _mm256_add_epi32(acc[0][i][0], acc[s][i][0]);
      acc[0][i][1] = _mm256_add_epi32(acc[0][i][1], acc[s][i][1]);
    }
  const int64_t left = g.cols - o0;
  const bool full = left >= 16;
  const __m256i m0 = first_lanes(left), m1 = first_lanes(left - 8);
  BF_UNROLL
  for (int h = 0; h < 2; ++h) {
    const __m256 sc = _mm256_loadu_ps(g.scale + o0 + 8 * h);
    const __m256 bi = _mm256_loadu_ps(g.bias + o0 + 8 * h);
    const __m256i corr = g.corr ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g.corr + o0 + 8 * h)) : _mm256_setzero_si256();
    BF_UNROLL
    for (int i = 0; i < R; ++i) {
      __m256i v = acc[0][i][h];
      if (g.corr) v = _mm256_sub_epi32(v, corr);
      __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v), sc);
      f = act_f32x8(_mm256_add_ps(f, bi), g.act);
      float* dst = y + i * g.cols + o0 + 8 * h;
      if (full) _mm256_storeu_ps(dst, f);
      else _mm256_maskstore_ps(dst, h ? m1 : m0, f);
    }
  }
}

template <class Dot>
void gemm(const GemmArgs& g, int64_t r0, int64_t rows, int64_t p0, int64_t p1) {
  for (int64_t p = p0; p < p1; ++p) {
    const int8_t* wp = g.w + p * g.kc * 64;
    const int64_t o0 = p * kNR;
    int64_t m = 0;
    for (; m + kMR <= rows; m += kMR) tile<Dot, kMR>(g, g.a + (r0 + m) * g.lda, wp, g.y + (r0 + m) * g.cols, o0);
    const uint8_t* ar = g.a + (r0 + m) * g.lda;
    float* y = g.y + (r0 + m) * g.cols;
    switch (rows - m) {
      case 1: tile<Dot, 1>(g, ar, wp, y, o0); break;
      case 2: tile<Dot, 2>(g, ar, wp, y, o0); break;
      case 3: tile<Dot, 3>(g, ar, wp, y, o0); break;
      case 4: tile<Dot, 4>(g, ar, wp, y, o0); break;
      case 5: tile<Dot, 5>(g, ar, wp, y, o0); break;
      default: break;
    }
  }
}

}  // namespace
}  // namespace int8_avx2
}  // namespace bitino

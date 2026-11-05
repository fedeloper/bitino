// AVX2 micro-kernel, epilogue and bit packing of the "binary_avx2" tier.
// Compiled with -mavx2 -mpopcnt (x86-64 only); called only when
// cpu_info() reports avx2 + popcnt. Includes nothing but binary_avx2.h and
// system headers (see binary_avx2.h). -ffp-contract=off applies here as
// everywhere, so the separate _mm256_mul_ps / _mm256_add_ps of the epilogue
// are never fused into FMA and round exactly like the reference.
#include "kernels/binary_avx2/binary_avx2.h"

#if defined(__x86_64__) && defined(__AVX2__)
#include <immintrin.h>

#include <cstring>

namespace bitino {
namespace avx2 {
namespace {

constexpr int64_t kU8Steps = 31;  // a byte lane gains <= 8 per step: 31 * 8 = 248 <= 255

inline __m256i broadcast64(const uint32_t* p) {
  long long v;
  std::memcpy(&v, p, 8);
  return _mm256_set1_epi64x(v);  // vpbroadcastq ymm, m64 (a load-port uop)
}

// Per-byte popcount via two 16-entry nibble lookups (VPSHUFB).
inline __m256i popcount_bytes(__m256i x, __m256i lut, __m256i low4) {
  const __m256i lo = _mm256_and_si256(x, low4);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low4);
  return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}

// NP rows x 2 lane quads (8 output channels). t0[i] / t1[i] receive the
// popcounts of row i against channels 0-3 / 4-7 of the block as 4 x uint64.
template <int NP>
inline void micro_kernel(const uint32_t* const* a, const __m256i* w0, const __m256i* w1, int64_t steps, __m256i* t0, __m256i* t1) {
  const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i zero = _mm256_setzero_si256();
  for (int i = 0; i < NP; ++i) t0[i] = t1[i] = zero;
  for (int64_t s0 = 0; s0 < steps; s0 += kU8Steps) {
    const int64_t s1 = steps - s0 < kU8Steps ? steps : s0 + kU8Steps;
    __m256i c0[NP], c1[NP];
    for (int i = 0; i < NP; ++i) c0[i] = c1[i] = zero;
    for (int64_t s = s0; s < s1; ++s) {
      const __m256i wa = _mm256_load_si256(w0 + s), wb = _mm256_load_si256(w1 + s);
      for (int i = 0; i < NP; ++i) {
        const __m256i x = broadcast64(a[i] + 2 * s);
        c0[i] = _mm256_add_epi8(c0[i], popcount_bytes(_mm256_xor_si256(x, wa), lut, low4));
        c1[i] = _mm256_add_epi8(c1[i], popcount_bytes(_mm256_xor_si256(x, wb), lut, low4));
      }
    }
    for (int i = 0; i < NP; ++i) {
      t0[i] = _mm256_add_epi64(t0[i], _mm256_sad_epu8(c0[i], zero));
      t1[i] = _mm256_add_epi64(t1[i], _mm256_sad_epu8(c1[i], zero));
    }
  }
}

// Activation with exactly the scalar semantics of apply_activation()
// (including NaN and signed zero).
inline __m256 activate(int act, __m256 v) {
  const __m256 zero = _mm256_setzero_ps();
  if (act == 1) return _mm256_and_ps(v, _mm256_cmp_ps(v, zero, _CMP_GT_OQ));  // v > 0 ? v : +0
  if (act == 2) {
    const __m256 six = _mm256_set1_ps(6.f);
    const __m256 m = _mm256_blendv_ps(v, zero, _mm256_cmp_ps(v, zero, _CMP_LT_OQ));  // std::max(v, 0)
    return _mm256_blendv_ps(m, six, _mm256_cmp_ps(six, m, _CMP_LT_OQ));              // std::min(m, 6)
  }
  return v;
}

// Stores the bits of one 8-channel block (bit output) of output row q at channel o.
inline void store_bits(const TileArgs& a, int64_t q, int64_t o, int nvo, uint32_t bits) {
  uint32_t* row = a.yb + q * a.y_stride;
  if (!a.bit_rmw) {
    // o is a multiple of 8: one byte per block (little-endian words).
    auto* bytes = reinterpret_cast<uint8_t*>(row);
    bytes[o >> 3] = static_cast<uint8_t>(bits);
    // Arena slots may hold stale data: the block owning the last channel
    // clears the remaining padding bytes of the last word.
    if (o + kOcBlock >= a.oc_total)
      for (int64_t b = (o >> 3) + 1; b < a.y_stride * 4; ++b) bytes[b] = 0;
  } else {
    const int64_t wi = o >> 5, sh = o & 31;
    row[wi] |= bits << sh;
    if (sh + nvo > 32) row[wi + 1] |= bits >> (32 - sh);
  }
}

template <int NP>
void tile_rows(const TileArgs& a, const uint32_t* rows, const uint64_t* mask, int64_t q0, int64_t r0, int64_t pb, int64_t ob,
               int64_t j0, int64_t j1, int64_t ocg) {
  const int64_t row_words = 2 * a.steps;
  const uint32_t* ap[NP];
  for (int i = 0; i < NP; ++i) ap[i] = rows + (r0 + i) * row_words;
  const __m256i kb = _mm256_set1_epi32(a.k_bits);
  const __m256i pack_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256 zero = _mm256_setzero_ps();
  for (int64_t j = j0; j < j1; j += kOcBlock) {
    const int64_t p = pb + j;  // padded channel index of the block
    const auto* w0 = reinterpret_cast<const __m256i*>(a.wts + (p / 4) * a.steps * 32);
    const __m256i* w1 = w0 + a.steps;
    __m256i t0[NP], t1[NP];
    micro_kernel<NP>(ap, w0, w1, a.steps, t0, t1);
    const __m256 mul = _mm256_loadu_ps(a.mul + p), add = _mm256_loadu_ps(a.add + p);
    const int64_t o = ob + j;
    const int nvo = static_cast<int>(ocg - j < kOcBlock ? ocg - j : kOcBlock);
    for (int i = 0; i < NP; ++i) {
      const __m256i pc = _mm256_permutevar8x32_epi32(_mm256_or_si256(t0[i], _mm256_slli_epi64(t1[i], 32)), pack_idx);
      __m256i dot = _mm256_sub_epi32(kb, _mm256_slli_epi32(pc, 1));
      const int64_t r = r0 + i;
      uint64_t m = mask ? mask[r] : 0;
      if (m && a.pad_scale != 0) {
        __m256i corr = _mm256_setzero_si256();
        for (; m; m &= m - 1)
          corr = _mm256_add_epi32(corr, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.tap_sum + __builtin_ctzll(m) * a.oc_pad + p)));
        dot = _mm256_add_epi32(dot, _mm256_mullo_epi32(corr, _mm256_set1_epi32(a.pad_scale)));
      }
      const int64_t q = q0 + r;
      if (a.out_lim) {  // bit output through integer thresholds on the dot product (no float epilogue)
        const __m256i lim = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.out_lim + p));
        const __m256i neg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.out_neg + p));
        const __m256i one = _mm256_blendv_epi8(_mm256_cmpgt_epi32(lim, dot), _mm256_cmpgt_epi32(dot, lim), neg);
        store_bits(a, q, o, nvo, static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(one))) & ((1u << nvo) - 1));
        continue;
      }
      __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(dot), mul), add);
      v = activate(a.act, v);
      if (a.yf) {
        float* yp = a.yf + q * a.y_stride + o;
        if (nvo == kOcBlock) _mm256_storeu_ps(yp, v);
        else _mm256_maskstore_ps(yp, _mm256_cmpgt_epi32(_mm256_set1_epi32(nvo), lane), v);
      } else {
        // bit = 1 iff !(v >= t) (!(v > t) for out_gt, t = 0 without out_thr); channels beyond nvo stay 0.
        const __m256 t = a.out_thr ? _mm256_loadu_ps(a.out_thr + p) : zero;
        const __m256 nb = a.out_gt ? _mm256_cmp_ps(v, t, _CMP_NGT_UQ) : _mm256_cmp_ps(v, t, _CMP_NGE_UQ);
        store_bits(a, q, o, nvo, static_cast<uint32_t>(_mm256_movemask_ps(nb)) & ((1u << nvo) - 1));
      }
    }
  }
}

// Bit packing of 32 floats: compare masks -> saturating packs -> byte movemask.
template <bool kGt, bool kThr>
inline uint32_t pack32(const float* x, const float* t) {
  __m256i m[4];
  for (int k = 0; k < 4; ++k) {
    const __m256 v = _mm256_loadu_ps(x + 8 * k);
    const __m256 th = kThr ? _mm256_loadu_ps(t + 8 * k) : _mm256_setzero_ps();
    m[k] = _mm256_castps_si256(kGt ? _mm256_cmp_ps(v, th, _CMP_NGT_UQ) : _mm256_cmp_ps(v, th, _CMP_NGE_UQ));
  }
  // packs keep the 128-bit lanes apart: dwords end up as m0[0:4] m1[0:4] m2[0:4] m3[0:4] | m0[4:8] ...
  const __m256i b = _mm256_packs_epi16(_mm256_packs_epi32(m[0], m[1]), _mm256_packs_epi32(m[2], m[3]));
  const __m256i c = _mm256_permutevar8x32_epi32(b, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  return static_cast<uint32_t>(_mm256_movemask_epi8(c));
}

template <bool kGt, bool kThr>
void pack_words_t(const float* x, int64_t words, uint32_t* out, const float* thr) {
  for (int64_t w = 0; w < words; ++w) out[w] = pack32<kGt, kThr>(x + 32 * w, kThr ? thr + 32 * w : nullptr);
}

// ------------------------------------------------------------ ternary BDense
// Ternary planes of 32 floats: nz = (x != t) ordered (NaN -> 0), neg = (x < t).
template <bool kThr>
inline void pack32_ternary(const float* x, const float* t, uint32_t* nz, uint32_t* neg) {
  __m256i a[4], b[4];
  for (int k = 0; k < 4; ++k) {
    const __m256 v = _mm256_loadu_ps(x + 8 * k);
    const __m256 th = kThr ? _mm256_loadu_ps(t + 8 * k) : _mm256_setzero_ps();
    a[k] = _mm256_castps_si256(_mm256_cmp_ps(v, th, _CMP_NEQ_OQ));
    b[k] = _mm256_castps_si256(_mm256_cmp_ps(v, th, _CMP_LT_OQ));
  }
  const __m256i idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m256i pa = _mm256_packs_epi16(_mm256_packs_epi32(a[0], a[1]), _mm256_packs_epi32(a[2], a[3]));
  const __m256i pb = _mm256_packs_epi16(_mm256_packs_epi32(b[0], b[1]), _mm256_packs_epi32(b[2], b[3]));
  *nz = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_permutevar8x32_epi32(pa, idx)));
  *neg = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_permutevar8x32_epi32(pb, idx)));
}

// NP rows x 2 lane quads. kWz: weights have zero entries (nz plane), so both
// popcount(m) and popcount(m & (neg_x ^ neg_w)) are per (row, channel);
// otherwise m = nz_x and only the second popcount depends on the channel.
template <int NP, bool kWz>
inline void ternary_kernel(const uint32_t* const* xn, const uint32_t* const* xs, const __m256i* s0, const __m256i* s1, const __m256i* n0,
                           const __m256i* n1, int64_t steps, __m256i* d0, __m256i* d1, __m256i* m0, __m256i* m1) {
  const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i zero = _mm256_setzero_si256();
  for (int i = 0; i < NP; ++i) d0[i] = d1[i] = m0[i] = m1[i] = zero;
  for (int64_t c0 = 0; c0 < steps; c0 += kU8Steps) {
    const int64_t c1 = steps - c0 < kU8Steps ? steps : c0 + kU8Steps;
    __m256i ad0[NP], ad1[NP], am0[NP], am1[NP];
    for (int i = 0; i < NP; ++i) ad0[i] = ad1[i] = am0[i] = am1[i] = zero;
    for (int64_t s = c0; s < c1; ++s) {
      const __m256i sa = _mm256_load_si256(s0 + s), sb = _mm256_load_si256(s1 + s);
      __m256i na = zero, nb = zero;
      if (kWz) {
        na = _mm256_load_si256(n0 + s);
        nb = _mm256_load_si256(n1 + s);
      }
      for (int i = 0; i < NP; ++i) {
        const __m256i bn = broadcast64(xn[i] + 2 * s), bs = broadcast64(xs[i] + 2 * s);
        const __m256i ma = kWz ? _mm256_and_si256(bn, na) : bn;
        const __m256i mb = kWz ? _mm256_and_si256(bn, nb) : bn;
        ad0[i] = _mm256_add_epi8(ad0[i], popcount_bytes(_mm256_and_si256(ma, _mm256_xor_si256(bs, sa)), lut, low4));
        ad1[i] = _mm256_add_epi8(ad1[i], popcount_bytes(_mm256_and_si256(mb, _mm256_xor_si256(bs, sb)), lut, low4));
        if (kWz) {
          am0[i] = _mm256_add_epi8(am0[i], popcount_bytes(ma, lut, low4));
          am1[i] = _mm256_add_epi8(am1[i], popcount_bytes(mb, lut, low4));
        }
      }
    }
    for (int i = 0; i < NP; ++i) {
      d0[i] = _mm256_add_epi64(d0[i], _mm256_sad_epu8(ad0[i], zero));
      d1[i] = _mm256_add_epi64(d1[i], _mm256_sad_epu8(ad1[i], zero));
      if (kWz) {
        m0[i] = _mm256_add_epi64(m0[i], _mm256_sad_epu8(am0[i], zero));
        m1[i] = _mm256_add_epi64(m1[i], _mm256_sad_epu8(am1[i], zero));
      }
    }
  }
}

template <int NP, bool kWz>
void ternary_rows(const TernaryArgs& a, const uint32_t* xnz, const uint32_t* xneg, int64_t q0, int64_t r0, int64_t j0, int64_t j1,
                  int64_t oc) {
  const int64_t row_words = 2 * a.steps;
  const uint32_t *xn[NP], *xs[NP];
  __m256i pm_row[NP];  // without zero weights: popcount(nz_x) of the row, broadcast
  for (int i = 0; i < NP; ++i) {
    xn[i] = xnz + (r0 + i) * row_words;
    xs[i] = xneg + (r0 + i) * row_words;
    if (!kWz) {
      long long pc = 0;
      for (int64_t q = 0; q < a.steps; ++q) {
        long long v;
        std::memcpy(&v, xn[i] + 2 * q, 8);
        pc += _mm_popcnt_u64(static_cast<unsigned long long>(v));
      }
      pm_row[i] = _mm256_set1_epi32(static_cast<int>(pc));
    }
  }
  const __m256i pack_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (int64_t j = j0; j < j1; j += kOcBlock) {
    const auto* s0 = reinterpret_cast<const __m256i*>(a.wneg + (j / 4) * a.steps * 32);
    const auto* n0 = kWz ? reinterpret_cast<const __m256i*>(a.wnz + (j / 4) * a.steps * 32) : nullptr;
    __m256i d0[NP], d1[NP], m0[NP], m1[NP];
    ternary_kernel<NP, kWz>(xn, xs, s0, s0 + a.steps, n0, kWz ? n0 + a.steps : nullptr, a.steps, d0, d1, m0, m1);
    const __m256 mul = _mm256_loadu_ps(a.mul + j), add = _mm256_loadu_ps(a.add + j);
    const int nvo = static_cast<int>(oc - j < kOcBlock ? oc - j : kOcBlock);
    for (int i = 0; i < NP; ++i) {
      const __m256i pd = _mm256_permutevar8x32_epi32(_mm256_or_si256(d0[i], _mm256_slli_epi64(d1[i], 32)), pack_idx);
      const __m256i pm = kWz ? _mm256_permutevar8x32_epi32(_mm256_or_si256(m0[i], _mm256_slli_epi64(m1[i], 32)), pack_idx) : pm_row[i];
      const __m256i dot = _mm256_sub_epi32(pm, _mm256_slli_epi32(pd, 1));
      __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(dot), mul), add);
      v = activate(a.act, v);
      float* yp = a.y + (q0 + r0 + i) * a.y_stride + j;
      if (nvo == kOcBlock) _mm256_storeu_ps(yp, v);
      else _mm256_maskstore_ps(yp, _mm256_cmpgt_epi32(_mm256_set1_epi32(nvo), lane), v);
    }
  }
}

template <bool kWz>
void ternary_tile_t(const TernaryArgs& a, const uint32_t* xnz, const uint32_t* xneg, int64_t q0, int64_t np, int64_t j0, int64_t j1,
                    int64_t oc) {
  // Zero weights double the accumulators: 2 rows per micro-tile (8 accumulators).
  constexpr int kRows = kWz ? 2 : 4;
  int64_t r = 0;
  for (; r + kRows <= np; r += kRows) ternary_rows<kRows, kWz>(a, xnz, xneg, q0, r, j0, j1, oc);
  for (; r < np; ++r) ternary_rows<1, kWz>(a, xnz, xneg, q0, r, j0, j1, oc);
}

// ------------------------------------------------ k-bit x binary (QBConv2D)
// NP rows x 2 lane quads, KB planes. Per step a byte lane gains at most
// 8 * (2^KB - 1) (popcount(bit_b & w) << b summed over planes), so the uint8
// chunk is 255 / (8 * (2^KB - 1)) steps (KB = 2: 10).
template <int NP, int KB>
inline void qbinary_kernel(const uint32_t* const* a, int64_t row_words, const __m256i* w0, const __m256i* w1, int64_t steps, __m256i* t0,
                           __m256i* t1) {
  constexpr int64_t kChunk = 255 / (8 * ((1 << KB) - 1));
  const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i zero = _mm256_setzero_si256();
  for (int i = 0; i < NP; ++i) t0[i] = t1[i] = zero;
  for (int64_t s0 = 0; s0 < steps; s0 += kChunk) {
    const int64_t s1 = steps - s0 < kChunk ? steps : s0 + kChunk;
    __m256i c0[NP], c1[NP];
    for (int i = 0; i < NP; ++i) c0[i] = c1[i] = zero;
    for (int64_t s = s0; s < s1; ++s) {
      const __m256i wa = _mm256_load_si256(w0 + s), wb = _mm256_load_si256(w1 + s);
      for (int i = 0; i < NP; ++i)
        for (int b = 0; b < KB; ++b) {
          const __m256i x = broadcast64(a[i] + b * row_words + 2 * s);
          // Shifting a byte count <= 8 left by b <= 3 stays inside its byte.
          __m256i pa = popcount_bytes(_mm256_and_si256(x, wa), lut, low4);
          __m256i pb = popcount_bytes(_mm256_and_si256(x, wb), lut, low4);
          if (b) {
            pa = _mm256_slli_epi16(pa, b);
            pb = _mm256_slli_epi16(pb, b);
          }
          c0[i] = _mm256_add_epi8(c0[i], pa);
          c1[i] = _mm256_add_epi8(c1[i], pb);
        }
    }
    for (int i = 0; i < NP; ++i) {
      t0[i] = _mm256_add_epi64(t0[i], _mm256_sad_epu8(c0[i], zero));
      t1[i] = _mm256_add_epi64(t1[i], _mm256_sad_epu8(c1[i], zero));
    }
  }
}

template <int NP, int KB>
void qbinary_rows(const QTileArgs& a, const uint32_t* rows, int64_t q0, int64_t r0, int64_t j0, int64_t j1, int64_t oc) {
  const int64_t row_words = 2 * a.steps;
  const uint32_t* ap[NP];
  __m256i xsum[NP];  // X = Σ_b 2^b popcount(plane b), per row
  for (int i = 0; i < NP; ++i) {
    ap[i] = rows + (r0 + i) * KB * row_words;
    long long x = 0;
    for (int b = 0; b < KB; ++b) {
      long long pc = 0;
      for (int64_t q = 0; q < a.steps; ++q) {
        long long v;
        std::memcpy(&v, ap[i] + b * row_words + 2 * q, 8);
        pc += _mm_popcnt_u64(static_cast<unsigned long long>(v));
      }
      x += pc << b;
    }
    xsum[i] = _mm256_set1_epi32(static_cast<int>(x));
  }
  const __m256i pack_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (int64_t j = j0; j < j1; j += kOcBlock) {
    const auto* w0 = reinterpret_cast<const __m256i*>(a.wts + (j / 4) * a.steps * 32);
    __m256i t0[NP], t1[NP];
    qbinary_kernel<NP, KB>(ap, row_words, w0, w0 + a.steps, a.steps, t0, t1);
    const __m256 mul = _mm256_loadu_ps(a.mul + j), add = _mm256_loadu_ps(a.add + j);
    const int nvo = static_cast<int>(oc - j < kOcBlock ? oc - j : kOcBlock);
    for (int i = 0; i < NP; ++i) {
      const __m256i d = _mm256_permutevar8x32_epi32(_mm256_or_si256(t0[i], _mm256_slli_epi64(t1[i], 32)), pack_idx);
      const __m256i dot = _mm256_sub_epi32(xsum[i], _mm256_slli_epi32(d, 1));
      __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(dot), mul), add);
      v = activate(a.act, v);
      float* yp = a.y + (q0 + r0 + i) * a.y_stride + j;
      if (nvo == kOcBlock) _mm256_storeu_ps(yp, v);
      else _mm256_maskstore_ps(yp, _mm256_cmpgt_epi32(_mm256_set1_epi32(nvo), lane), v);
    }
  }
}

template <int KB>
void qbinary_tile_t(const QTileArgs& a, const uint32_t* rows, int64_t q0, int64_t np, int64_t j0, int64_t j1, int64_t oc) {
  int64_t r = 0;
  for (; r + kRowBlock <= np; r += kRowBlock) qbinary_rows<kRowBlock, KB>(a, rows, q0, r, j0, j1, oc);
  for (; r < np; ++r) qbinary_rows<1, KB>(a, rows, q0, r, j0, j1, oc);
}

// Integer DoReFa levels of 8 floats (dorefa_level(): std::max / std::min
// semantics incl. NaN, one f32 multiply, round half to even via the default
// MXCSR mode; NaN -> 0x80000000 whose low bits are 0, like the scalar cast).
inline __m256i levels8(const float* x, __m256 n) {
  const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.f);
  const __m256 v = _mm256_loadu_ps(x);
  const __m256 m = _mm256_blendv_ps(v, zero, _mm256_cmp_ps(v, zero, _CMP_LT_OQ));  // std::max(x, 0)
  const __m256 c = _mm256_blendv_ps(m, one, _mm256_cmp_ps(one, m, _CMP_LT_OQ));     // std::min(m, 1)
  return _mm256_cvtps_epi32(_mm256_mul_ps(c, n));
}

}  // namespace

void qbinary_tile(const QTileArgs& a, const uint32_t* rows, int64_t q0, int64_t np, int64_t j0, int64_t j1, int64_t oc) {
  switch (a.planes) {
    case 1: qbinary_tile_t<1>(a, rows, q0, np, j0, j1, oc); break;
    case 2: qbinary_tile_t<2>(a, rows, q0, np, j0, j1, oc); break;
    case 3: qbinary_tile_t<3>(a, rows, q0, np, j0, j1, oc); break;
    case 4: qbinary_tile_t<4>(a, rows, q0, np, j0, j1, oc); break;
    default: break;
  }
}

void pack_levels_row(const float* x, int64_t c, float n, int planes, uint32_t* out, int64_t plane_stride) {
  const __m256 nv = _mm256_set1_ps(n);
  const int64_t full = c / 32;
  for (int64_t w = 0; w < full; ++w) {
    __m256i l[4];
    for (int k = 0; k < 4; ++k) l[k] = levels8(x + 32 * w + 8 * k, nv);
    for (int b = 0; b < planes; ++b) {
      uint32_t bits = 0;
      for (int k = 0; k < 4; ++k)  // bit b -> sign bit -> movemask
        bits |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(l[k], 31 - b)))) << (8 * k);
      out[b * plane_stride + w] = bits;
    }
  }
  if (c == full * 32) return;
  // Tail: same vector path on a zero-padded copy (level(0) = 0).
  alignas(32) float tmp[32] = {};
  std::memcpy(tmp, x + 32 * full, static_cast<size_t>(c - 32 * full) * sizeof(float));
  __m256i l[4];
  for (int k = 0; k < 4; ++k) l[k] = levels8(tmp + 8 * k, nv);
  for (int b = 0; b < planes; ++b) {
    uint32_t bits = 0;
    for (int k = 0; k < 4; ++k)
      bits |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(l[k], 31 - b)))) << (8 * k);
    out[b * plane_stride + full] = bits;
  }
}

void ternary_tile(const TernaryArgs& a, const uint32_t* xnz, const uint32_t* xneg, int64_t q0, int64_t np, int64_t j0, int64_t j1,
                  int64_t oc) {
  if (a.wnz) ternary_tile_t<true>(a, xnz, xneg, q0, np, j0, j1, oc);
  else ternary_tile_t<false>(a, xnz, xneg, q0, np, j0, j1, oc);
}

void pack_ternary_row(const float* x, int64_t c, uint32_t* nz, uint32_t* neg, const float* thr) {
  const int64_t full = c / 32;
  for (int64_t w = 0; w < full; ++w) {
    if (thr) pack32_ternary<true>(x + 32 * w, thr + 32 * w, nz + w, neg + w);
    else pack32_ternary<false>(x + 32 * w, nullptr, nz + w, neg + w);
  }
  if (c == full * 32) return;
  uint32_t bn = 0, bs = 0;
  for (int64_t b = 0; b < c - full * 32; ++b) {
    const float v = x[full * 32 + b], t = thr ? thr[full * 32 + b] : 0.f;
    bn |= uint32_t(v > t || v < t) << b;
    bs |= uint32_t(v < t) << b;
  }
  nz[full] = bn;
  neg[full] = bs;
}

void bconv_tile(const TileArgs& a, const uint32_t* rows, const uint64_t* mask, int64_t q0, int64_t np, int64_t pb, int64_t ob,
                int64_t j0, int64_t j1, int64_t ocg) {
  int64_t r = 0;
  for (; r + kRowBlock <= np; r += kRowBlock) tile_rows<4>(a, rows, mask, q0, r, pb, ob, j0, j1, ocg);
  switch (np - r) {
    case 3: tile_rows<3>(a, rows, mask, q0, r, pb, ob, j0, j1, ocg); break;
    case 2: tile_rows<2>(a, rows, mask, q0, r, pb, ob, j0, j1, ocg); break;
    case 1: tile_rows<1>(a, rows, mask, q0, r, pb, ob, j0, j1, ocg); break;
    default: break;
  }
}

void pack_words(const float* x, int64_t words, uint32_t* out, bool gt, const float* thr) {
  if (gt) {
    if (thr) pack_words_t<true, true>(x, words, out, thr);
    else pack_words_t<true, false>(x, words, out, nullptr);
  } else {
    if (thr) pack_words_t<false, true>(x, words, out, thr);
    else pack_words_t<false, false>(x, words, out, nullptr);
  }
}

void pack_row(const float* x, int64_t c, uint32_t* out, bool gt, const float* thr) {
  const int64_t full = c / 32;
  pack_words(x, full, out, gt, thr);
  if (c == full * 32) return;
  uint32_t bits = 0;
  for (int64_t b = 0; b < c - full * 32; ++b) {
    const float v = x[full * 32 + b], t = thr ? thr[full * 32 + b] : 0.f;
    bits |= uint32_t(gt ? !(v > t) : !(v >= t)) << b;
  }
  out[full] = bits;
}

void bmm_pack_row(int mode, const float* x, int64_t k, int64_t w64, uint32_t* nz, uint32_t* neg) {
  const int64_t w32 = 2 * w64, kw = (k + 31) / 32;
  if (mode == 2) {  // sign_zero
    pack_ternary_row(x, k, nz, neg, nullptr);
  } else {
    pack_row(x, k, neg, mode == 1 || mode == 3, nullptr);  // bit = !(x >= 0) (ge) / !(x > 0) (gt, gt0)
    for (int64_t w = 0; w < kw; ++w) {
      const int64_t valid = k - 32 * w;
      const uint32_t lane = valid >= 32 ? 0xffffffffu : ((1u << valid) - 1u);
      if (mode == 3) {  // gt0: {0, 1}, nz = (x > 0), no negative values
        nz[w] = ~neg[w] & lane;
        neg[w] = 0u;
      } else {
        nz[w] = lane;
      }
    }
  }
  for (int64_t w = kw; w < w32; ++w) nz[w] = neg[w] = 0u;
}

void bmm_dot_row(const uint64_t* a, const uint64_t* b, int64_t n, int64_t w64, float scale, float* y) {
  const uint64_t* an = a;
  const uint64_t* as = a + w64;
  for (int64_t j = 0; j < n; ++j) {
    const uint64_t* bn = b + j * 2 * w64;
    const uint64_t* bs = bn + w64;
    int64_t pm = 0, pd = 0;
    for (int64_t w = 0; w < w64; ++w) {
      const uint64_t m = an[w] & bn[w];
      pm += static_cast<int64_t>(_mm_popcnt_u64(m));
      pd += static_cast<int64_t>(_mm_popcnt_u64(m & (as[w] ^ bs[w])));
    }
    y[j] = static_cast<float>(static_cast<int32_t>(pm - 2 * pd)) * scale;
  }
}

void bmm_dot_rows_cols(const uint64_t* a, int64_t m, const uint64_t* bc, int64_t n, int64_t npad, int64_t w64, float scale, float* y) {
  const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low4 = _mm256_set1_epi8(0x0f);
  auto bytecnt = [&](__m256i v) {
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v, low4)),
                           _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), low4)));
  };
  const __m256i zero = _mm256_setzero_si256();
  const __m256i lo32 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256 sv = _mm256_set1_ps(scale);
  for (int64_t r = 0; r < m; ++r) {
    const uint64_t* an = a + r * 2 * w64;
    const uint64_t* as = an + w64;
    float* yr = y + r * n;
    for (int64_t j = 0; j < n; j += 8) {
      __m256i pm[2] = {zero, zero}, pd[2] = {zero, zero};
      for (int64_t w0 = 0; w0 < w64; w0 += 31) {
        const int64_t w1 = w64 - w0 < 31 ? w64 : w0 + 31;
        __m256i cm[2] = {zero, zero}, cd[2] = {zero, zero};
        for (int64_t w = w0; w < w1; ++w) {
          const __m256i va = _mm256_set1_epi64x(static_cast<long long>(an[w]));
          const __m256i vs = _mm256_set1_epi64x(static_cast<long long>(as[w]));
          const uint64_t* nzp = bc + (w * 2) * npad + j;
          const uint64_t* ngp = nzp + npad;
          for (int h = 0; h < 2; ++h) {
            const __m256i bn = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nzp + 4 * h));
            const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ngp + 4 * h));
            const __m256i mm = _mm256_and_si256(va, bn);
            cm[h] = _mm256_add_epi8(cm[h], bytecnt(mm));
            cd[h] = _mm256_add_epi8(cd[h], bytecnt(_mm256_and_si256(mm, _mm256_xor_si256(vs, bs))));
          }
        }
        for (int h = 0; h < 2; ++h) {
          pm[h] = _mm256_add_epi64(pm[h], _mm256_sad_epu8(cm[h], zero));
          pd[h] = _mm256_add_epi64(pd[h], _mm256_sad_epu8(cd[h], zero));
        }
      }
      // acc = pm - 2 pd per 64-bit lane (|acc| < 2^31); low dwords of the 8 columns -> floats
      const __m256i acc0 = _mm256_sub_epi64(pm[0], _mm256_slli_epi64(pd[0], 1));
      const __m256i acc1 = _mm256_sub_epi64(pm[1], _mm256_slli_epi64(pd[1], 1));
      const __m128i i0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(acc0, lo32));
      const __m128i i1 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(acc1, lo32));
      const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(i1, i0)), sv);
      if (j + 8 <= n) {
        _mm256_storeu_ps(yr + j, v);
      } else {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, v);
        for (int64_t t = 0; j + t < n; ++t) yr[j + t] = tmp[t];
      }
    }
  }
}

}  // namespace avx2
}  // namespace bitino

#endif  // __x86_64__ && __AVX2__

// Bit-plane BMatMul kernels (bit-exact with bmatmul.reference):
//   bmatmul.bitplane  tier binary_scalar, portable (64-bit popcount)
//   bmatmul.neon      tier neon, AArch64 (VCNT, register-blocked 4 x 8 tiles on
//                     32-bit words, b packed in column blocks; see bmmn below)
//   bmatmul.avx2      tier binary_avx2, x86-64 AVX2 + POPCNT (vector packing,
//                     POPCNT dot products; leaves in binary_avx2_simd.cc)
//
// Every binarized operand value v in {-1, 0, +1} (BMode of transformer_common.h)
// is stored as two bit-planes: nz = (v != 0), neg = (v < 0):
//   ge        nz = 1,        neg = !(x >= 0)
//   gt        nz = 1,        neg = !(x > 0)
//   sign_zero nz = (x != 0), neg = (x < 0)        (NaN: 0)
//   gt0       nz = (x > 0),  neg = 0              ({0, 1})
// For two such operands, with m = nz_a & nz_b,
//   sum_k a_k b_k = popcount(m) - 2 popcount(m & (neg_a ^ neg_b))
// exactly (padding bits are 0 in nz). Rows of `a` and columns of `b` (rows of
// b^T; a non-transposed b is transposed in thread scratch first) are packed as
// {nz[w64], neg[w64]} 64-bit words; y = float(acc) * float(scale) as the
// reference. Portable / AVX2: work items are batch entries (pack a and b, then all
// dots); NEON: see BMatMulNeon.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

#include "kernels/bitpack.h"
#include "ops/op_defs.h"
#include "ops/transformer_common.h"

#if defined(__x86_64__)
#include "kernels/binary_avx2/binary_avx2.h"
#endif
#if defined(__aarch64__)
#include <arm_neon.h>

#include "kernels/binary_neon/bitpack_neon.h"
#endif

namespace bitino {
namespace {

// Scalar packing of one operand row (k floats) into {nz, neg} planes of w64 words.
void pack_row_scalar(BMode mode, const float* x, int64_t k, int64_t w64, uint64_t* nz, uint64_t* neg) {
  for (int64_t w = 0; w < w64; ++w) {
    uint64_t bn = 0, bs = 0;
    const int64_t lim = std::min<int64_t>(64, k - 64 * w);
    for (int64_t b = 0; b < lim; ++b) {
      const int v = bmode_apply(mode, x[64 * w + b]);
      bn |= uint64_t(v != 0) << b;
      bs |= uint64_t(v < 0) << b;
    }
    nz[w] = bn;
    neg[w] = bs;
  }
}

struct BmmPortable {
  static void pack(BMode mode, const float* x, int64_t k, int64_t w64, uint64_t* row) {
    pack_row_scalar(mode, x, k, w64, row, row + w64);
  }
  static void dot_row(const uint64_t* a, const uint64_t* b, int64_t n, int64_t w64, float scale, float* y) {
    const uint64_t* an = a;
    const uint64_t* as = a + w64;
    for (int64_t j = 0; j < n; ++j) {
      const uint64_t* bn = b + j * 2 * w64;
      const uint64_t* bs = bn + w64;
      int32_t pm = 0, pd = 0;
      for (int64_t w = 0; w < w64; ++w) {
        const uint64_t m = an[w] & bn[w];
        pm += popcount64(m);
        pd += popcount64(m & (as[w] ^ bs[w]));
      }
      y[j] = static_cast<float>(pm - 2 * pd) * scale;
    }
  }
  static int64_t col_words(int64_t, int64_t) { return 0; }
  static void dot_rows(const uint64_t* a, int64_t m, const uint64_t* b, int64_t n, int64_t w64, float scale, float* y, uint64_t*) {
    for (int64_t r = 0; r < m; ++r) dot_row(a + r * 2 * w64, b, n, w64, scale, y + r * n);
  }
};

#if defined(__aarch64__)
// bmatmul.neon: register-blocked bit-plane GEMM on 32-bit words (w32 = ceil(k/32)).
//
// Rows of a are {nz[w32], neg[w32]}. Columns of b are packed in blocks of 8: for
// block jb and word w, the nz words of the 8 columns, then their neg words (16
// words = four uint32x4), padding columns all zero. A tile of R rows x 8 columns
// keeps the per-byte VCNT counts (<= 8 per word) in uint8 lanes for up to kChunk
// words, then folds them into one uint16 per column. Per word and 4 columns:
//   general       m = nz_a & nz_b, d = m & (neg_a ^ neg_b): pm += cnt(m), pd += cnt(d)
//   row of a full nz_a covers every valid bit, so pm = popcount(nz_b) (per column,
//                 from packing) and d = nz_b & (neg_a ^ neg_b)
//   block of b full  symmetric: pm = popcount(nz_a) of the row, d = nz_a & (neg_a ^ neg_b)
//   both full     pm = k, d = neg_a ^ neg_b (neg is a subset of nz, so its padding bits are 0)
// The full forms take 4 (both: 3) instead of 7 vector ops and give the same integers
// (padding bits of nz and neg are zero on both sides). y = float(pm - 2 pd) * scale as the reference;
// |pm - 2 pd| <= k <= kMaxK, exact in int16 and in float.
namespace bmmn {

constexpr int kNB = 8;              // columns per block
constexpr int64_t kChunk = 15;      // words per uint8 round: 15 * 8 = 120 per byte, pairwise sums <= 240
constexpr int64_t kMaxK = 32767;    // larger k: portable kernel
enum Mode { kGeneral, kAFull, kBFull, kBothFull };

inline uint32_t lane_mask(int64_t k, int64_t w) {
  const int64_t valid = k - 32 * w;
  return valid >= 32 ? 0xffffffffu : ((1u << valid) - 1u);
}

// One operand row (k floats) -> nz[w32], neg[w32] (NEON compare + narrow packing).
inline void pack_row32(BMode mode, const float* x, int64_t k, int64_t w32, uint32_t* nz, uint32_t* neg) {
  if (mode == BMode::kSignZero) {
    pack_ternary_row_neon(x, k, nz, neg, nullptr);
    return;
  }
  // bit = !(x >= 0) (ge) or !(x > 0) (gt, gt0), NaN -> 1, as bmode_apply; padding bits 0
  pack_row_f32_neon(x, k, neg, mode == BMode::kGe ? BinarizeMode::kGe : BinarizeMode::kGt, nullptr);
  for (int64_t w = 0; w < w32; ++w) {
    const uint32_t lane = lane_mask(k, w);
    if (mode == BMode::kGt0) {  // {0, 1}: nz = (x > 0), no negatives
      nz[w] = ~neg[w] & lane;
      neg[w] = 0u;
    } else {
      nz[w] = lane;
    }
  }
}

// popcount(nz) and whether nz has every valid bit set.
inline void nz_stats(const uint32_t* nz, int64_t k, int64_t w32, int32_t* count, bool* full) {
  bool f = true;
  for (int64_t w = 0; w < w32; ++w) f = f && nz[w] == lane_mask(k, w);
  int32_t c = static_cast<int32_t>(k);
  if (!f) {
    c = 0;
    for (int64_t w = 0; w < w32; ++w) c += popcount32(nz[w]);
  }
  *count = c;
  *full = f;
}

// nz / neg lane masks of 4 floats, exactly bmode_apply (NaN: sign_zero 0, ge/gt -1, gt0 0).
template <BMode M>
inline void masks4(float32x4_t v, uint32x4_t* nz, uint32x4_t* neg) {
  if (M == BMode::kSignZero) {
    const uint32x4_t lt = vcltzq_f32(v);
    *nz = vorrq_u32(vcgtzq_f32(v), lt);
    *neg = lt;
  } else if (M == BMode::kGe) {
    *nz = vdupq_n_u32(~0u);
    *neg = vmvnq_u32(vcgezq_f32(v));
  } else if (M == BMode::kGt) {
    *nz = vdupq_n_u32(~0u);
    *neg = vmvnq_u32(vcgtzq_f32(v));
  } else {
    *nz = vcgtzq_f32(v);
    *neg = vdupq_n_u32(0u);
  }
}

// Eight full columns j0..j0+7 of a row-major b[k][n] (b points at b[0][j0]) -> one
// block, without a transpose: rows of a word are visited from the highest bit down
// and shifted in with SLI (acc = acc << 1 | mask & 1).
template <BMode M>
void pack_block_cols(const float* b, int64_t n, int64_t k, int64_t w32, uint32_t* blk) {
  for (int64_t w = 0; w < w32; ++w) {
    uint32x4_t z0 = vdupq_n_u32(0u), z1 = z0, s0 = z0, s1 = z0;
    const int64_t r1 = std::min<int64_t>(k, 32 * w + 32);
    for (int64_t r = r1 - 1; r >= 32 * w; --r) {
      const float* p = b + r * n;
      uint32x4_t mz0, ms0, mz1, ms1;
      masks4<M>(vld1q_f32(p), &mz0, &ms0);
      masks4<M>(vld1q_f32(p + 4), &mz1, &ms1);
      z0 = vsliq_n_u32(mz0, z0, 1);
      z1 = vsliq_n_u32(mz1, z1, 1);
      s0 = vsliq_n_u32(ms0, s0, 1);
      s1 = vsliq_n_u32(ms1, s1, 1);
    }
    uint32_t* o = blk + 16 * w;
    vst1q_u32(o, z0);
    vst1q_u32(o + 4, z1);
    vst1q_u32(o + 8, s0);
    vst1q_u32(o + 12, s1);
  }
}

// Columns j0..j0+nc-1 (nc < 8) of a row-major b[k][n] -> one block (scalar).
inline void pack_block_cols_scalar(BMode mode, const float* b, int64_t n, int64_t k, int64_t w32, int nc, uint32_t* blk) {
  std::memset(blk, 0, sizeof(uint32_t) * 16 * w32);
  for (int c = 0; c < nc; ++c)
    for (int64_t r = 0; r < k; ++r) {
      const int v = bmode_apply(mode, b[r * n + c]);
      const uint32_t bit = 1u << (r & 31);
      if (v != 0) blk[16 * (r >> 5) + c] |= bit;
      if (v < 0) blk[16 * (r >> 5) + 8 + c] |= bit;
    }
}

// Adds the counts of words [w0, w1) (w1 - w0 <= kChunk) of R rows x one block into
// cm16 / cd16 (one uint16 per column; set instead of added when `first`).
template <int R, Mode M>
inline void tile_round(const uint32_t* a, int64_t astride, int64_t w32, const uint32_t* blk, int64_t w0, int64_t w1, bool first,
                       uint16x8_t* cm16, uint16x8_t* cd16) {
  uint8x16_t cm[R][2], cd[R][2];
  for (int r = 0; r < R; ++r) cm[r][0] = cm[r][1] = cd[r][0] = cd[r][1] = vdupq_n_u8(0);
  for (int64_t w = w0; w < w1; ++w) {
    const uint32_t* bw = blk + 16 * w;
    const uint32x4_t bn[2] = {vld1q_u32(bw), vld1q_u32(bw + 4)};
    const uint32x4_t bs[2] = {vld1q_u32(bw + 8), vld1q_u32(bw + 12)};
    for (int r = 0; r < R; ++r) {
      const uint32_t* ar = a + r * astride;
      const uint32x4_t as = vld1q_dup_u32(ar + w32 + w);
      const uint32x4_t an = (M == kAFull || M == kBothFull) ? vdupq_n_u32(0u) : vld1q_dup_u32(ar + w);
      for (int v = 0; v < 2; ++v) {
        const uint32x4_t x = veorq_u32(as, bs[v]);
        uint32x4_t d;
        if (M == kBothFull) {
          d = x;
        } else if (M == kAFull) {
          d = vandq_u32(x, bn[v]);
        } else if (M == kBFull) {
          d = vandq_u32(x, an);
        } else {
          const uint32x4_t m = vandq_u32(an, bn[v]);
          d = vandq_u32(x, m);
          cm[r][v] = vaddq_u8(cm[r][v], vcntq_u8(vreinterpretq_u8_u32(m)));
        }
        cd[r][v] = vaddq_u8(cd[r][v], vcntq_u8(vreinterpretq_u8_u32(d)));
      }
    }
  }
  // bytes 4c..4c+3 of half v count column 4v+c: pairwise widen, then pairwise add
  // across both halves gives the 8 columns in order
  for (int r = 0; r < R; ++r) {
    const uint16x8_t sd = vpaddq_u16(vpaddlq_u8(cd[r][0]), vpaddlq_u8(cd[r][1]));
    cd16[r] = first ? sd : vaddq_u16(cd16[r], sd);
    if (M == kGeneral) {
      const uint16x8_t sm = vpaddq_u16(vpaddlq_u8(cm[r][0]), vpaddlq_u8(cm[r][1]));
      cm16[r] = first ? sm : vaddq_u16(cm16[r], sm);
    }
  }
}

// R rows of a (row r at a + r * astride) x one block of 8 columns -> y (nc valid columns).
template <int R, Mode M, bool kUnit>
inline void tile(const uint32_t* a, int64_t astride, int64_t w32, const uint32_t* blk, const uint16_t* colnz, const int32_t* rownz,
                 float scale, float* y, int64_t ystride, int nc) {
  uint16x8_t cm16[R], cd16[R];
  if (w32 <= kChunk) {
    tile_round<R, M>(a, astride, w32, blk, 0, w32, true, cm16, cd16);
  } else {
    for (int64_t w0 = 0; w0 < w32; w0 += kChunk)
      tile_round<R, M>(a, astride, w32, blk, w0, std::min<int64_t>(w32, w0 + kChunk), w0 == 0, cm16, cd16);
  }
  for (int r = 0; r < R; ++r) {
    uint16x8_t base;
    if (M == kGeneral) base = cm16[r];
    else if (M == kAFull) base = vld1q_u16(colnz);
    else if (M == kBFull) base = vdupq_n_u16(static_cast<uint16_t>(rownz[r]));
    else base = vdupq_n_u16(static_cast<uint16_t>(rownz[r]));  // both full: rownz = k
    const int16x8_t acc = vreinterpretq_s16_u16(vmlsq_n_u16(base, cd16[r], 2));  // pm - 2 pd
    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_high_s16(acc));
    if (!kUnit) {  // x * 1.0f == x: skipped for scale 1
      f0 = vmulq_n_f32(f0, scale);
      f1 = vmulq_n_f32(f1, scale);
    }
    float* yr = y + r * ystride;
    if (nc == kNB) {
      vst1q_f32(yr, f0);
      vst1q_f32(yr + 4, f1);
    } else {
      float t[kNB];
      vst1q_f32(t, f0);
      vst1q_f32(t + 4, f1);
      std::memcpy(yr, t, sizeof(float) * nc);
    }
  }
}

template <int R, bool kUnit>
inline void tile_mode(Mode mode, const uint32_t* a, int64_t astride, int64_t w32, const uint32_t* blk, const uint16_t* colnz,
                      const int32_t* rownz, float scale, float* y, int64_t ystride, int nc) {
  switch (mode) {
    case kBothFull: tile<R, kBothFull, kUnit>(a, astride, w32, blk, colnz, rownz, scale, y, ystride, nc); break;
    case kAFull: tile<R, kAFull, kUnit>(a, astride, w32, blk, colnz, rownz, scale, y, ystride, nc); break;
    case kBFull: tile<R, kBFull, kUnit>(a, astride, w32, blk, colnz, rownz, scale, y, ystride, nc); break;
    default: tile<R, kGeneral, kUnit>(a, astride, w32, blk, colnz, rownz, scale, y, ystride, nc); break;
  }
}

template <int R>
inline void tile_any(Mode mode, const uint32_t* a, int64_t astride, int64_t w32, const uint32_t* blk, const uint16_t* colnz,
                     const int32_t* rownz, float scale, bool unit, float* y, int64_t ystride, int nc) {
  if (unit) tile_mode<R, true>(mode, a, astride, w32, blk, colnz, rownz, scale, y, ystride, nc);
  else tile_mode<R, false>(mode, a, astride, w32, blk, colnz, rownz, scale, y, ystride, nc);
}


// int8 operands (int8_full I8BMatMul, kernels/int8_full/int8_full_neon.cc): the planes of the integer
// binarization of int8 codes q with thresholds T in [-128, 128] (q8::qbin):
//   neg = (q < tneg)     (sign_zero: tneg = T_ge; ge: T_ge; gt: T_gt; gt0: none)
//   pos = (q >= tpos)    (sign_zero / gt0: tpos = T_gt)
//   nz  = pos | neg (sign_zero), all valid bits (ge, gt), pos (gt0)
// i.e. exactly the planes of the {-1, 0, +1} values the float path would pack. T = 128 (an empty
// upper set) does not fit int8 lanes and is handled by flags; T = -128 compares correctly.
struct OpI8 {
  BMode mode = BMode::kSignZero;
  int8_t tneg = 0, tpos = 0;
  bool neg_all = false, pos_none = false;
  void init(BMode m, int tn, int tp) {
    mode = m;
    neg_all = tn >= 128;
    pos_none = tp >= 128;
    tneg = static_cast<int8_t>(std::min(tn, 127));
    tpos = static_cast<int8_t>(std::min(tp, 127));
  }
  int value(int q) const {  // binarized value of one code
    const bool ng = mode != BMode::kGt0 && (neg_all || q < tneg);
    if (mode == BMode::kGe || mode == BMode::kGt) return ng ? -1 : 1;
    const bool ps = !pos_none && q >= tpos;
    if (mode == BMode::kGt0) return ps ? 1 : 0;
    return ps ? 1 : (ng ? -1 : 0);
  }
  // byte masks (0 / 0xFF) of 16 codes
  inline void masks16(int8x16_t q, uint8x16_t* nz, uint8x16_t* neg) const {
    const uint8x16_t ng = mode == BMode::kGt0 ? vdupq_n_u8(0) : (neg_all ? vdupq_n_u8(0xFF) : vcltq_s8(q, vdupq_n_s8(tneg)));
    if (mode == BMode::kGe || mode == BMode::kGt) {
      *nz = vdupq_n_u8(0xFF);
    } else {
      const uint8x16_t ps = pos_none ? vdupq_n_u8(0) : vcgeq_s8(q, vdupq_n_s8(tpos));
      *nz = mode == BMode::kGt0 ? ps : vorrq_u8(ps, ng);
    }
    *neg = ng;
  }
  inline void masks8(int8x8_t q, uint8x8_t* nz, uint8x8_t* neg) const {
    uint8x16_t z, n;
    masks16(vcombine_s8(q, q), &z, &n);
    *nz = vget_low_u8(z);
    *neg = vget_low_u8(n);
  }
};

// One int8 operand row (k codes) -> nz[w32], neg[w32]: 32 codes per word, byte masks folded with the bit
// weights by pairwise adds (both planes at once); a partial last word bit by bit (padding bits 0).
inline void pack_row32_i8(const OpI8& op, const int8_t* x, int64_t k, int64_t w32, uint32_t* nz, uint32_t* neg) {
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t bw = vld1q_u8(kW);
  int64_t w = 0;
  for (; 32 * w + 32 <= k; ++w) {
    uint8x16_t z0, n0, z1, n1;
    op.masks16(vld1q_s8(x + 32 * w), &z0, &n0);
    op.masks16(vld1q_s8(x + 32 * w + 16), &z1, &n1);
    uint8x16_t r = vpaddq_u8(vpaddq_u8(vandq_u8(z0, bw), vandq_u8(z1, bw)), vpaddq_u8(vandq_u8(n0, bw), vandq_u8(n1, bw)));
    r = vpaddq_u8(r, r);  // u32 lanes: nz word, neg word
    nz[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 0);
    neg[w] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 1);
  }
  if (w < w32) {
    uint32_t bn = 0, bs = 0;
    for (int64_t b = 0; 32 * w + b < k; ++b) {
      const int v = op.value(x[32 * w + b]);
      bn |= uint32_t(v != 0) << b;
      bs |= uint32_t(v < 0) << b;
    }
    nz[w] = bn;
    neg[w] = bs;
  }
}

// Full column blocks of a row-major int8 b[k][n] (b points at b[0][j0]): 16 columns (two blocks, blk0 and
// blk1) or 8 columns (blk0). Per group of 8 rows, OR-ing each row's byte masks AND (1 << row) gives one byte
// per column whose bit i is row i; four groups are zipped into the 32-bit column words (little-endian:
// byte g holds rows 8g..8g+7). Rows past k contribute nothing.
template <int kCols>
inline void pack_block_cols_i8(const OpI8& op, const int8_t* b, int64_t n, int64_t k, int64_t w32, uint32_t* blk0, uint32_t* blk1) {
  for (int64_t w = 0; w < w32; ++w) {
    uint8x16_t gz[4], gs[4];
    for (int g = 0; g < 4; ++g) {
      uint8x16_t az = vdupq_n_u8(0), as = vdupq_n_u8(0);
      const int64_t rb = 32 * w + 8 * g;
      const int nr = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(8, k - rb)));
      for (int i = 0; i < nr; ++i) {
        uint8x16_t mz, ms;
        if (kCols == 16) {
          op.masks16(vld1q_s8(b + (rb + i) * n), &mz, &ms);
        } else {
          uint8x8_t z8, s8;
          op.masks8(vld1_s8(b + (rb + i) * n), &z8, &s8);
          mz = vcombine_u8(z8, z8);
          ms = vcombine_u8(s8, s8);
        }
        const uint8x16_t bit = vdupq_n_u8(static_cast<uint8_t>(1u << i));
        az = vorrq_u8(az, vandq_u8(mz, bit));
        as = vorrq_u8(as, vandq_u8(ms, bit));
      }
      gz[g] = az;
      gs[g] = as;
    }
    auto words = [](const uint8x16_t* g, uint32x4_t* out) {  // out[0..3]: columns 0-3, 4-7, 8-11, 12-15
      const uint8x16x2_t lo = vzipq_u8(g[0], g[1]), hi = vzipq_u8(g[2], g[3]);
      for (int h = 0; h < 2; ++h) {
        const uint16x8_t a = vreinterpretq_u16_u8(lo.val[h]), c = vreinterpretq_u16_u8(hi.val[h]);
        out[2 * h] = vreinterpretq_u32_u16(vzip1q_u16(a, c));
        out[2 * h + 1] = vreinterpretq_u32_u16(vzip2q_u16(a, c));
      }
    };
    uint32x4_t z[4], s[4];
    words(gz, z);
    words(gs, s);
    uint32_t* o = blk0 + 16 * w;
    vst1q_u32(o, z[0]);
    vst1q_u32(o + 4, z[1]);
    vst1q_u32(o + 8, s[0]);
    vst1q_u32(o + 12, s[1]);
    if (kCols == 16) {
      uint32_t* o1 = blk1 + 16 * w;
      vst1q_u32(o1, z[2]);
      vst1q_u32(o1 + 4, z[3]);
      vst1q_u32(o1 + 8, s[2]);
      vst1q_u32(o1 + 12, s[3]);
    }
  }
}

// Columns j0..j0+nc-1 (nc < 8) of a row-major int8 b[k][n] -> one block (scalar).
inline void pack_block_cols_scalar_i8(const OpI8& op, const int8_t* b, int64_t n, int64_t k, int64_t w32, int nc, uint32_t* blk) {
  std::memset(blk, 0, sizeof(uint32_t) * 16 * w32);
  for (int c = 0; c < nc; ++c)
    for (int64_t r = 0; r < k; ++r) {
      const int v = op.value(b[r * n + c]);
      const uint32_t bit = 1u << (r & 31);
      if (v != 0) blk[16 * (r >> 5) + c] |= bit;
      if (v < 0) blk[16 * (r >> 5) + 8 + c] |= bit;
    }
}

}  // namespace bmmn
#endif

#if defined(__x86_64__)
struct BmmAvx2 {
  static void pack(BMode mode, const float* x, int64_t k, int64_t w64, uint64_t* row) {
    // 32-bit word planes in place of the 64-bit ones (little-endian: identical bits)
    uint32_t* nz = reinterpret_cast<uint32_t*>(row);
    uint32_t* neg = reinterpret_cast<uint32_t*>(row + w64);
    avx2::bmm_pack_row(static_cast<int>(mode), x, k, w64, nz, neg);
  }
  static void dot_row(const uint64_t* a, const uint64_t* b, int64_t n, int64_t w64, float scale, float* y) {
    avx2::bmm_dot_row(a, b, n, w64, scale, y);
  }
  // Column layout [w][plane][npad] for the 8-column VPSHUFB dot (bmm_dot_rows_cols).
  static int64_t col_words(int64_t n, int64_t w64) { return (n + 7) / 8 * 8 * 2 * w64; }
  static void dot_rows(const uint64_t* a, int64_t m, const uint64_t* b, int64_t n, int64_t w64, float scale, float* y, uint64_t* cols) {
    const int64_t npad = (n + 7) / 8 * 8;
    for (int64_t j = 0; j < n; ++j)
      for (int64_t w = 0; w < w64; ++w) {
        cols[(w * 2) * npad + j] = b[j * 2 * w64 + w];
        cols[(w * 2 + 1) * npad + j] = b[j * 2 * w64 + w64 + w];
      }
    avx2::bmm_dot_rows_cols(a, m, cols, n, npad, w64, scale, y);
  }
};
#endif

template <class B>
class BMatMulBitplane : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    ma_ = parse_bmode(ctx.attrs().get_string("a_binarize", "sign_zero"));
    mb_ = parse_bmode(ctx.attrs().get_string("b_binarize", "sign_zero"));
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    w64_ = (g_.k + 63) / 64;
    const int threads = ctx.pool->num_threads();
    slots_ = threads;
    const size_t per = static_cast<size_t>((g_.m + g_.n) * 2 * w64_);
    planes_.reset(new uint64_t[per * slots_]());
    plane_stride_ = per;
    bt_.assign(g_.transpose_b ? 0 : static_cast<size_t>(g_.n * g_.k) * slots_, 0.f);
    col_words_ = B::col_words(g_.n, w64_);
    cols_.assign(static_cast<size_t>(col_words_) * slots_, 0);  // padding columns stay zero
    busy_.reset(new std::atomic<bool>[slots_]);
    for (int i = 0; i < slots_; ++i) busy_[i].store(false);
    // Small problems run on the calling thread.
    grain_ = g_.batch * g_.m * g_.n * w64_ < 32768 ? std::max<int64_t>(g_.batch, 1) : 1;
  }
  size_t prepacked_bytes() const override { return 0; }

  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t W = w64_, row = 2 * W;
    ctx.pool->parallel_for(g.batch, grain_, [&](int64_t b0, int64_t b1) {
      const int s = acquire();
      uint64_t* ap = planes_.get() + plane_stride_ * s;
      uint64_t* bp = ap + g.m * row;
      float* bt = g.transpose_b ? nullptr : bt_.data() + static_cast<size_t>(g.n * g.k) * s;
      for (int64_t bi = b0; bi < b1; ++bi) {
        for (int64_t r = 0; r < g.m; ++r) B::pack(ma_, a + (bi * g.m + r) * g.k, g.k, W, ap + r * row);
        const float* bb = b + bi * g.k * g.n;
        if (!g.transpose_b) {  // b[k][n] -> bt[n][k]
          for (int64_t k = 0; k < g.k; ++k)
            for (int64_t j = 0; j < g.n; ++j) bt[j * g.k + k] = bb[k * g.n + j];
          bb = bt;
        }
        for (int64_t j = 0; j < g.n; ++j) B::pack(mb_, bb + j * g.k, g.k, W, bp + j * row);
        B::dot_rows(ap, g.m, bp, g.n, W, scale_, y + bi * g.m * g.n, cols_.data() + col_words_ * s);
      }
      busy_[s].store(false, std::memory_order_release);
    });
  }

 private:
  int acquire() {
    for (;;)
      for (int i = 0; i < slots_; ++i) {
        bool expected = false;
        if (busy_[i].compare_exchange_strong(expected, true, std::memory_order_acquire)) return i;
      }
  }

  MatMulGeom g_;
  BMode ma_ = BMode::kSignZero, mb_ = BMode::kSignZero;
  float scale_ = 1.f;
  int64_t w64_ = 1, grain_ = 1;
  int slots_ = 1;
  size_t plane_stride_ = 0;
  std::unique_ptr<uint64_t[]> planes_;
  std::vector<float> bt_;
  int64_t col_words_ = 0;
  std::vector<uint64_t> cols_;
  std::unique_ptr<std::atomic<bool>[]> busy_;
};

#if defined(__aarch64__)
// bmatmul.neon driver (tiles in bmmn above). Per batch entry, b is packed into
// column blocks once (phase 1, parallel over batch entries); then work items of
// (batch entry, row range) pack their rows of a and run the R x 8 tiles (phase 2),
// so single-batch attention matmuls also use several threads. Integer counts:
// the result does not depend on the split.
class BMatMulNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    if (g_.k > bmmn::kMaxK) {  // int16 tile sums would overflow: portable kernel
      fallback_.reset(new BMatMulBitplane<BmmPortable>());
      fallback_->prepare(ctx);
      return;
    }
    ma_ = parse_bmode(ctx.attrs().get_string("a_binarize", "sign_zero"));
    mb_ = parse_bmode(ctx.attrs().get_string("b_binarize", "sign_zero"));
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    unit_ = scale_ == 1.f;
    // int8 operands (only through make_bmatmul_neon_kernel(): bmatmul_supported requires f32): integer
    // thresholds of each operand's binarization from the I8BMatMul wrapper
    i8_ = ctx.in(0).dtype() == DType::kI8;
    if (i8_) {
      BF_CHECK(ctx.in(1).dtype() == DType::kI8, "bmatmul.neon: int8 a needs int8 b");
      const Json& at = ctx.attrs();
      opa_.init(ma_, static_cast<int>(at.get_int("i8_a_tneg", 0)), static_cast<int>(at.get_int("i8_a_tpos", 0)));
      opb_.init(mb_, static_cast<int>(at.get_int("i8_b_tneg", 0)), static_cast<int>(at.get_int("i8_b_tpos", 0)));
    }
    w32_ = (g_.k + 31) / 32;
    nblk_ = (g_.n + bmmn::kNB - 1) / bmmn::kNB;
    const size_t bm = static_cast<size_t>(g_.batch * g_.m), bb = static_cast<size_t>(g_.batch * nblk_);
    apack_.assign(bm * 2 * w32_, 0u);
    rownz_.assign(bm, 0);
    rowfull_.assign(bm, 0);
    bpack_.assign(bb * 16 * w32_, 0u);
    colnz_.assign(bb * bmmn::kNB, 0);
    blkfull_.assign(bb, 0);
    brows_.assign(g_.transpose_b ? static_cast<size_t>(g_.batch * g_.n) * 2 * w32_ : 0, 0u);
    // row ranges (multiples of 4 rows): one per batch entry on one thread, else ~4 items per thread
    const int threads = ctx.pool->num_threads();
    const int64_t groups = (g_.m + 3) / 4;
    int64_t chunks = 1;
    if (threads > 1) chunks = std::max<int64_t>(1, std::min<int64_t>(groups, (4 * threads + g_.batch - 1) / std::max<int64_t>(g_.batch, 1)));
    rows_per_item_ = std::max<int64_t>(4, (groups + chunks - 1) / chunks * 4);
    items_per_batch_ = std::max<int64_t>(1, (g_.m + rows_per_item_ - 1) / rows_per_item_);
    const int64_t work = g_.batch * g_.m * nblk_ * bmmn::kNB * w32_;  // word-column products
    small_ = work < 65536;
  }
  size_t prepacked_bytes() const override { return 0; }

  void run(const KernelContext& ctx) override {
    if (fallback_) return fallback_->run(ctx);
    const void* a = ctx.in(0).raw();
    const float* b = i8_ ? nullptr : ctx.in(1).data<float>();
    const int8_t* b8 = i8_ ? ctx.in(1).data<int8_t>() : nullptr;
    float* y = ctx.out(0).data<float>();
    if (g_.batch == 0 || g_.m == 0 || g_.n == 0) return;
    ctx.pool->parallel_for(g_.batch, small_ ? g_.batch : 1, [&](int64_t b0, int64_t b1) {
      for (int64_t bi = b0; bi < b1; ++bi) {
        if (i8_) pack_b_i8(b8 + bi * g_.k * g_.n, bi);
        else pack_b(b + bi * g_.k * g_.n, bi);
      }
    });
    const int64_t items = g_.batch * items_per_batch_;
    ctx.pool->parallel_for(items, small_ ? items : 1, [&](int64_t i0, int64_t i1) {
      for (int64_t it = i0; it < i1; ++it) {
        const int64_t bi = it / items_per_batch_, r0 = (it % items_per_batch_) * rows_per_item_;
        rows(a, y, bi, r0, std::min(g_.m, r0 + rows_per_item_));
      }
    });
  }

 private:
  void pack_b(const float* bb, int64_t bi) {
    const int64_t k = g_.k, n = g_.n, W = w32_;
    uint32_t* blk0 = bpack_.data() + static_cast<size_t>(bi * nblk_) * 16 * W;
    if (g_.transpose_b) {  // b[n][k]: rows are the columns
      uint32_t* br = brows_.data() + static_cast<size_t>(bi * n) * 2 * W;
      for (int64_t j = 0; j < n; ++j) bmmn::pack_row32(mb_, bb + j * k, k, W, br + j * 2 * W, br + j * 2 * W + W);
      for (int64_t jb = 0; jb < nblk_; ++jb) {
        uint32_t* blk = blk0 + jb * 16 * W;
        const int nc = static_cast<int>(std::min<int64_t>(bmmn::kNB, n - jb * bmmn::kNB));
        if (nc < bmmn::kNB) std::memset(blk, 0, sizeof(uint32_t) * 16 * W);
        for (int c = 0; c < nc; ++c) {
          const uint32_t* r = br + (jb * bmmn::kNB + c) * 2 * W;
          for (int64_t w = 0; w < W; ++w) {
            blk[16 * w + c] = r[w];
            blk[16 * w + 8 + c] = r[W + w];
          }
        }
      }
    } else {  // b[k][n]
      for (int64_t jb = 0; jb < nblk_; ++jb) {
        uint32_t* blk = blk0 + jb * 16 * W;
        const float* bc = bb + jb * bmmn::kNB;
        const int nc = static_cast<int>(std::min<int64_t>(bmmn::kNB, n - jb * bmmn::kNB));
        if (nc < bmmn::kNB) bmmn::pack_block_cols_scalar(mb_, bc, n, k, W, nc, blk);
        else if (mb_ == BMode::kSignZero) bmmn::pack_block_cols<BMode::kSignZero>(bc, n, k, W, blk);
        else if (mb_ == BMode::kGe) bmmn::pack_block_cols<BMode::kGe>(bc, n, k, W, blk);
        else if (mb_ == BMode::kGt) bmmn::pack_block_cols<BMode::kGt>(bc, n, k, W, blk);
        else bmmn::pack_block_cols<BMode::kGt0>(bc, n, k, W, blk);
      }
    }
    block_stats(bi);
  }

  // int8 b (pack_b with the integer binarization of bmmn::OpI8)
  void pack_b_i8(const int8_t* bb, int64_t bi) {
    const int64_t k = g_.k, n = g_.n, W = w32_;
    uint32_t* blk0 = bpack_.data() + static_cast<size_t>(bi * nblk_) * 16 * W;
    if (g_.transpose_b) {  // b[n][k]: rows are the columns
      uint32_t* br = brows_.data() + static_cast<size_t>(bi * n) * 2 * W;
      for (int64_t j = 0; j < n; ++j) bmmn::pack_row32_i8(opb_, bb + j * k, k, W, br + j * 2 * W, br + j * 2 * W + W);
      for (int64_t jb = 0; jb < nblk_; ++jb) {
        uint32_t* blk = blk0 + jb * 16 * W;
        const int nc = static_cast<int>(std::min<int64_t>(bmmn::kNB, n - jb * bmmn::kNB));
        if (nc < bmmn::kNB) std::memset(blk, 0, sizeof(uint32_t) * 16 * W);
        for (int c = 0; c < nc; ++c) {
          const uint32_t* r = br + (jb * bmmn::kNB + c) * 2 * W;
          for (int64_t w = 0; w < W; ++w) {
            blk[16 * w + c] = r[w];
            blk[16 * w + 8 + c] = r[W + w];
          }
        }
      }
    } else {  // b[k][n]
      for (int64_t jb = 0; jb < nblk_;) {
        uint32_t* blk = blk0 + jb * 16 * W;
        const int8_t* bc = bb + jb * bmmn::kNB;
        const int64_t left = n - jb * bmmn::kNB;
        if (left >= 2 * bmmn::kNB) {  // two full blocks
          bmmn::pack_block_cols_i8<16>(opb_, bc, n, k, W, blk, blk + 16 * W);
          jb += 2;
        } else if (left >= bmmn::kNB) {
          bmmn::pack_block_cols_i8<8>(opb_, bc, n, k, W, blk, nullptr);
          jb += 1;
        } else {
          bmmn::pack_block_cols_scalar_i8(opb_, bc, n, k, W, static_cast<int>(left), blk);
          jb += 1;
        }
      }
    }
    block_stats(bi);
  }

  // per-column popcount(nz) and block "full" flags of batch entry bi
  void block_stats(int64_t bi) {
    const int64_t k = g_.k, W = w32_;
    const uint32_t* blk0 = bpack_.data() + static_cast<size_t>(bi * nblk_) * 16 * W;
    for (int64_t jb = 0; jb < nblk_; ++jb) {
      const uint32_t* blk = blk0 + jb * 16 * W;
      uint16_t* cz = colnz_.data() + (bi * nblk_ + jb) * bmmn::kNB;
      bool full = true;
      for (int c = 0; c < bmmn::kNB; ++c) {
        int32_t cnt = 0;
        for (int64_t w = 0; w < W; ++w) {
          cnt += popcount32(blk[16 * w + c]);
          full = full && blk[16 * w + c] == bmmn::lane_mask(k, w);
        }
        cz[c] = static_cast<uint16_t>(cnt);
      }
      blkfull_[bi * nblk_ + jb] = full;
    }
  }

  void rows(const void* av, float* y, int64_t bi, int64_t r0, int64_t r1) {
    const int64_t k = g_.k, n = g_.n, W = w32_, row = 2 * W;
    uint32_t* ap = apack_.data() + static_cast<size_t>(bi * g_.m) * row;
    int32_t* rnz = rownz_.data() + bi * g_.m;
    uint8_t* rfull = rowfull_.data() + bi * g_.m;
    for (int64_t r = r0; r < r1; ++r) {
      if (i8_) bmmn::pack_row32_i8(opa_, static_cast<const int8_t*>(av) + (bi * g_.m + r) * k, k, W, ap + r * row, ap + r * row + W);
      else bmmn::pack_row32(ma_, static_cast<const float*>(av) + (bi * g_.m + r) * k, k, W, ap + r * row, ap + r * row + W);
      bool full;
      bmmn::nz_stats(ap + r * row, k, W, &rnz[r], &full);
      rfull[r] = full;
    }
    const uint32_t* blk0 = bpack_.data() + static_cast<size_t>(bi * nblk_) * 16 * W;
    const uint16_t* cz0 = colnz_.data() + bi * nblk_ * bmmn::kNB;
    const uint8_t* bfull = blkfull_.data() + bi * nblk_;
    float* yb = y + bi * g_.m * n;
    int64_t r = r0;
    for (; r + 4 <= r1; r += 4) {
      const bool afull = rfull[r] && rfull[r + 1] && rfull[r + 2] && rfull[r + 3];
      for (int64_t jb = 0; jb < nblk_; ++jb) {
        const bmmn::Mode mode = afull ? (bfull[jb] ? bmmn::kBothFull : bmmn::kAFull) : (bfull[jb] ? bmmn::kBFull : bmmn::kGeneral);
        const int nc = static_cast<int>(std::min<int64_t>(bmmn::kNB, n - jb * bmmn::kNB));
        bmmn::tile_any<4>(mode, ap + r * row, row, W, blk0 + jb * 16 * W, cz0 + jb * bmmn::kNB, rnz + r, scale_, unit_,
                          yb + r * n + jb * bmmn::kNB, n, nc);
      }
    }
    for (; r < r1; ++r)
      for (int64_t jb = 0; jb < nblk_; ++jb) {
        const bmmn::Mode mode = rfull[r] ? (bfull[jb] ? bmmn::kBothFull : bmmn::kAFull) : (bfull[jb] ? bmmn::kBFull : bmmn::kGeneral);
        const int nc = static_cast<int>(std::min<int64_t>(bmmn::kNB, n - jb * bmmn::kNB));
        bmmn::tile_any<1>(mode, ap + r * row, row, W, blk0 + jb * 16 * W, cz0 + jb * bmmn::kNB, rnz + r, scale_, unit_,
                          yb + r * n + jb * bmmn::kNB, n, nc);
      }
  }

  MatMulGeom g_;
  BMode ma_ = BMode::kSignZero, mb_ = BMode::kSignZero;
  float scale_ = 1.f;
  bool unit_ = true, small_ = true, i8_ = false;
  bmmn::OpI8 opa_, opb_;
  int64_t w32_ = 1, nblk_ = 0, rows_per_item_ = 4, items_per_batch_ = 1;
  std::vector<uint32_t> apack_, bpack_, brows_;
  std::vector<int32_t> rownz_;
  std::vector<uint8_t> rowfull_, blkfull_;
  std::vector<uint16_t> colnz_;
  std::unique_ptr<Kernel> fallback_;
};
#endif

bool bmatmul_supported(const KernelContext& ctx) {
  return ctx.in(0).dtype() == DType::kF32 && ctx.in(1).dtype() == DType::kF32 && ctx.in(0).dim(-1) > 0;
}

}  // namespace

#if defined(__aarch64__)
// Factory for the int8_full NEON I8BMatMul kernel (kernels/int8_full/int8_full_neon.cc).
std::unique_ptr<Kernel> make_bmatmul_neon_kernel() { return make_kernel<BMatMulNeon>(); }
#endif

void register_bmatmul_kernels(OpRegistry& reg) {
  reg.add_kernel({"BMatMul", "bmatmul.bitplane", "binary_scalar", 25, 0, bmatmul_supported, make_kernel<BMatMulBitplane<BmmPortable>>});
#if defined(__aarch64__)
  reg.add_kernel({"BMatMul", "bmatmul.neon", "neon", 35, kFeatNeon, bmatmul_supported, make_kernel<BMatMulNeon>});
#endif
#if defined(__x86_64__)
  reg.add_kernel({"BMatMul", "bmatmul.avx2", "binary_avx2", 35, kFeatAvx2 | kFeatPopcnt, bmatmul_supported,
                  make_kernel<BMatMulBitplane<BmmAvx2>>});
#endif
}

}  // namespace bitino

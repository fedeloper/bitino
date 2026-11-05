// Bit packing helpers shared by binary kernels.
// Convention (see tensor.h): bit 1 = -1, bit 0 = +1, LSB-first within uint32
// words, padding bits zero.
#pragma once

#include <cstdint>
#include <cstring>

#include "ops/ops_common.h"

namespace bitino {

// Packs one row of `c` floats. kSignZero is not representable and must be
// rejected by the caller.
inline void pack_row_f32(const float* x, int64_t c, uint32_t* out, BinarizeMode mode, const float* thr) {
  int64_t words = bit_words(c);
  for (int64_t w = 0; w < words; ++w) {
    uint32_t bits = 0;
    int64_t base = w * 32;
    int64_t lim = c - base < 32 ? c - base : 32;
    if (mode == BinarizeMode::kGt) {
      for (int64_t b = 0; b < lim; ++b) bits |= uint32_t(!(x[base + b] > (thr ? thr[base + b] : 0.f))) << b;
    } else {
      for (int64_t b = 0; b < lim; ++b) bits |= uint32_t(!(x[base + b] >= (thr ? thr[base + b] : 0.f))) << b;
    }
    out[w] = bits;
  }
}

inline int bit_value(const uint32_t* row, int64_t c) { return (row[c >> 5] >> (c & 31)) & 1u ? -1 : 1; }

inline int popcount32(uint32_t v) { return __builtin_popcount(v); }
inline int popcount64(uint64_t v) { return __builtin_popcountll(v); }

}  // namespace bitino

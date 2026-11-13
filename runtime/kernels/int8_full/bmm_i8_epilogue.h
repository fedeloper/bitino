// int8 output epilogue of the bmatmul.neon tiles for the int8_full I8BMatMul (kernels/binary_scalar/bmatmul_bitplane.cc,
// make_bmatmul_neon_i8_kernel; filled by kernels/int8_full/int8_full_neon.cc): per row r and column j of a batch entry
//   acc = c_ab * (A B)[r][j] + rowterm[r] + colterm[j]   (int32; plain products: acc = (A B)[r][j])
//   v = fp32(fp32(float(acc) * mul[j]) + add[j]),  y = clamp(rne(v), lo, hi)
// (int8_full::BMatMulParams::store). Column arrays are padded to whole 8-column blocks (bmatmul_i8_col_pad).
#pragma once

#include <cstdint>

namespace bitino {

struct BmmI8Epilogue {
  bool plain = true;
  int32_t c_ab = 1;
  const int32_t* rowterm = nullptr;  // [batch * m]: c_rs * rowsum (not plain)
  const int32_t* colterm = nullptr;  // [batch * col_pad]: c_cs * colsum + c0 (not plain)
  const float* mul = nullptr;        // [col_pad] (mul_vec) or [1]
  const float* add = nullptr;        // [col_pad] (add_vec), [1] or null
  bool mul_vec = false, add_vec = false;
  int lo = -128, hi = 127;
};

}  // namespace bitino

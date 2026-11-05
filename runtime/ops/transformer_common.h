// Shared helpers for the transformer operators (Gather, LayerNorm, MatMul,
// BMatMul, AddBroadcast, Transpose). Contracts: docs/operators.md
// § Transformer operators. Kept separate from ops_common.h so the CNN
// binarization enum (used by optimized BConv2D/BDense kernels) is unchanged.
#pragma once

#include <string>

#include "bitino/error.h"
#include "bitino/model.h"

namespace bitino {

// Operand binarization for BMatMul. Values are exact small integers.
enum class BMode {
  kGe,        // +1 if x >= 0 else -1           (larq sign, sign(0) = +1)
  kGt,        // +1 if x >  0 else -1           (sign(0) = -1)
  kSignZero,  // torch.sign: +1 / 0 / -1
  kGt0,       // Heaviside {0,1}: 1 if x > 0 else 0 (BiBERT ZMeanBinaryQuantizer: sign() with -1 -> 0)
};

inline BMode parse_bmode(const std::string& m) {
  if (m == "ge") return BMode::kGe;
  if (m == "gt") return BMode::kGt;
  if (m == "sign_zero") return BMode::kSignZero;
  if (m == "gt0") return BMode::kGt0;
  fail("unknown binarize mode '", m, "' (expected ge|gt|sign_zero|gt0)");
}

inline int bmode_apply(BMode m, float x) {
  switch (m) {
    case BMode::kGe: return x >= 0.f ? 1 : -1;
    case BMode::kGt: return x > 0.f ? 1 : -1;
    case BMode::kSignZero: return x > 0.f ? 1 : (x < 0.f ? -1 : 0);
    case BMode::kGt0: return x > 0.f ? 1 : 0;
  }
  return 0;
}

// Geometry of a batched matmul a[..., M, K] x b[..., K, N] (or b[..., N, K]
// with transpose_b). Leading (batch) dims must be equal; no broadcasting.
struct MatMulGeom {
  int64_t batch = 1, m = 0, k = 0, n = 0;
  bool transpose_b = false;
  Shape out;
};

inline MatMulGeom matmul_geometry(const Json& attrs, const Shape& a, const Shape& b) {
  BF_CHECK(a.size() >= 2 && a.size() == b.size(), "matmul operands need equal rank >= 2, got ", shape_str(a), " and ",
           shape_str(b));
  MatMulGeom g;
  g.transpose_b = attrs.get_int("transpose_b", 0) != 0;
  size_t r = a.size();
  for (size_t d = 0; d + 2 < r; ++d) {
    BF_CHECK(a[d] == b[d], "matmul batch dims differ: ", shape_str(a), " vs ", shape_str(b), " (no broadcasting)");
    g.batch *= a[d];
  }
  g.m = a[r - 2];
  g.k = a[r - 1];
  int64_t bk = g.transpose_b ? b[r - 1] : b[r - 2];
  g.n = g.transpose_b ? b[r - 2] : b[r - 1];
  BF_CHECK(bk == g.k, "matmul K mismatch: ", shape_str(a), " x ", shape_str(b), g.transpose_b ? " (transpose_b)" : "");
  g.out = a;
  g.out[r - 1] = g.n;
  return g;
}

}  // namespace bitino

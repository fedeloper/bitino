// Operator semantics for transformer models: attribute validation and shape
// inference. Contracts: docs/operators.md § Transformer operators.
#include <algorithm>

#include "ops/op_defs.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

void set_out(Outs& outs, DType dt, Shape s) {
  BF_CHECK(!outs.empty(), "missing output");
  outs[0]->dtype = dt;
  outs[0]->shape = std::move(s);
}

const TensorInfo& req(const Ins& ins, size_t i, const char* what) {
  BF_CHECK(i < ins.size() && ins[i], "missing required input ", i, " (", what, ")");
  return *ins[i];
}

bool has(const Ins& ins, size_t i) { return i < ins.size() && ins[i]; }

// Gather(table, indices, scale?, zero_mask?) -> f32[indices.shape..., D]
void infer_gather(const Node&, const Ins& ins, Outs& outs) {
  const auto& t = req(ins, 0, "table");
  const auto& idx = req(ins, 1, "indices");
  BF_CHECK(t.shape.size() == 2, "Gather table must be [V, D], got ", shape_str(t.shape));
  BF_CHECK(t.dtype == DType::kF32 || t.dtype == DType::kBit, "Gather table must be f32 or bit");
  BF_CHECK(idx.dtype == DType::kI32, "Gather indices must be i32, got ", dtype_name(idx.dtype));
  if (has(ins, 2)) {
    BF_CHECK(t.dtype == DType::kBit, "Gather scale requires a bit table");
    BF_CHECK(ins[2]->dtype == DType::kF32 && ins[2]->shape == Shape{t.shape[0]}, "Gather scale must be f32[V]");
  }
  if (has(ins, 3)) {
    BF_CHECK(t.dtype == DType::kBit, "Gather zero_mask requires a bit table");
    BF_CHECK(ins[3]->dtype == DType::kBit && ins[3]->shape == t.shape, "Gather zero_mask must be bit with the table's shape");
  }
  Shape out = idx.shape;
  out.push_back(t.shape[1]);
  set_out(outs, DType::kF32, out);
}

void check_last_dim_vec(const Ins& ins, size_t i, int64_t d, const char* what) {
  if (!has(ins, i)) return;
  BF_CHECK(ins[i]->dtype == DType::kF32 && ins[i]->shape == Shape{d}, what, " must be f32[", d, "]");
}

// LayerNorm(x, gamma?, beta?) over the last axis.
void infer_layernorm(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32 && !x.shape.empty(), "LayerNorm expects f32 input");
  int64_t d = x.shape.back();
  check_last_dim_vec(ins, 1, d, "gamma");
  check_last_dim_vec(ins, 2, d, "beta");
  BF_CHECK(n.attrs.get_number("eps", 1e-5) >= 0.0, "LayerNorm eps must be >= 0");
  set_out(outs, DType::kF32, x.shape);
}

// MatMul(a, b): both f32 (runtime or constant).
void infer_matmul(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req(ins, 0, "a");
  const auto& b = req(ins, 1, "b");
  BF_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32, "MatMul expects f32 operands");
  auto g = matmul_geometry(n.attrs, a.shape, b.shape);
  n.attrs.get_number("scale", 1.0);
  set_out(outs, DType::kF32, g.out);
}

// BMatMul(a, b): binarize both operands, exact integer accumulation, * scale.
void infer_bmatmul(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req(ins, 0, "a");
  const auto& b = req(ins, 1, "b");
  BF_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32, "BMatMul expects f32 operands (binarized internally)");
  auto g = matmul_geometry(n.attrs, a.shape, b.shape);
  parse_bmode(n.attrs.get_string("a_binarize", "sign_zero"));
  parse_bmode(n.attrs.get_string("b_binarize", "sign_zero"));
  BF_CHECK(g.k < (int64_t(1) << 24), "BMatMul K too large for exact f32 conversion of the accumulator");
  set_out(outs, DType::kF32, g.out);
}

// AddBroadcast(a, b): b broadcast (right-aligned, dims equal or 1) to a's shape.
void infer_add_broadcast(const Node&, const Ins& ins, Outs& outs) {
  const auto& a = req(ins, 0, "a");
  const auto& b = req(ins, 1, "b");
  BF_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32, "AddBroadcast expects f32");
  BF_CHECK(b.shape.size() <= a.shape.size(), "AddBroadcast: rank(b) must be <= rank(a)");
  size_t off = a.shape.size() - b.shape.size();
  for (size_t d = 0; d < b.shape.size(); ++d)
    BF_CHECK(b.shape[d] == 1 || b.shape[d] == a.shape[off + d], "AddBroadcast: ", shape_str(b.shape),
             " is not broadcastable to ", shape_str(a.shape));
  set_out(outs, DType::kF32, a.shape);
}

void infer_transpose(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32, "Transpose supports f32");
  auto perm = n.attrs.get_ints("perm", {});
  BF_CHECK(perm.size() == x.shape.size() && perm.size() <= 8, "Transpose perm must list every axis (rank <= 8)");
  std::vector<bool> seen(perm.size(), false);
  Shape out(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    BF_CHECK(perm[i] >= 0 && perm[i] < static_cast<int64_t>(perm.size()) && !seen[perm[i]], "invalid Transpose perm");
    seen[perm[i]] = true;
    out[i] = x.shape[perm[i]];
  }
  set_out(outs, DType::kF32, out);
}

}  // namespace

void register_transformer_op_defs(OpRegistry& reg) {
  reg.add_op({"Gather", infer_gather});
  reg.add_op({"LayerNorm", infer_layernorm});
  reg.add_op({"MatMul", infer_matmul});
  reg.add_op({"BMatMul", infer_bmatmul});
  reg.add_op({"AddBroadcast", infer_add_broadcast});
  reg.add_op({"Transpose", infer_transpose});
}

}  // namespace bitino

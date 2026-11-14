// Operator semantics of the int8_full ops for accuracy-carrying glue (2026-11-13; docs/int8.md "int8_full ops for
// accuracy-carrying glue"; numpy oracle python/bitino/int8_ops.py): I8ChannelShift (ShiftViT token mixer),
// I8BinarizeDyn (INSTA-BNN instance-aware threshold) and I8SimAM (parameter-free attention). Attribute validation and
// shape inference; the int64 bounds the kernels rely on are checked here.
#include <cmath>

#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

bool has(const Ins& ins, size_t i) { return i < ins.size() && ins[i]; }

const TensorInfo& req_i8_4d(const Ins& ins, size_t i, const char* op, const char* what) {
  BF_CHECK(has(ins, i), op, ": missing input ", i, " (", what, ")");
  const auto& t = *ins[i];
  BF_CHECK(t.dtype == DType::kI8 && t.shape.size() == 4, op, ": ", what, " must be i8 [N, H, W, C], got ", dtype_name(t.dtype),
           shape_str(t.shape));
  return t;
}

void check_vec(const Ins& ins, size_t i, DType dt, int64_t n, const char* op, const char* what) {
  BF_CHECK(has(ins, i), op, ": missing ", what);
  BF_CHECK(ins[i]->dtype == dt && num_elements(ins[i]->shape) == n, op, ": ", what, " must be ", dtype_name(dt), "[", n, "], got ",
           dtype_name(ins[i]->dtype), shape_str(ins[i]->shape));
}

void set_out(Outs& outs, DType dt, const Shape& s) {
  BF_CHECK(!outs.empty() && outs[0], "missing output");
  outs[0]->dtype = dt;
  outs[0]->shape = s;
}

// I8ChannelShift: x i8 [N, H, W, C]; groups = flat [c0, c1, dy, dx, ...] (0 <= c0 < c1 <= C, ranges disjoint),
// fill_q in [-128, 127]; y i8 of x's shape.
void infer_channel_shift(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8_4d(ins, 0, "I8ChannelShift", "x");
  const int64_t c = x.shape[3];
  const std::vector<int64_t> g = n.attrs.get_ints("groups", {});
  BF_CHECK(g.size() % 4 == 0, "I8ChannelShift: groups must be [c0, c1, dy, dx] quadruples");
  std::vector<int> used(static_cast<size_t>(c), 0);
  for (size_t k = 0; k < g.size(); k += 4) {
    BF_CHECK(0 <= g[k] && g[k] < g[k + 1] && g[k + 1] <= c, "I8ChannelShift: channel range out of [0, C]");
    BF_CHECK(std::llabs(g[k + 2]) <= (int64_t(1) << 20) && std::llabs(g[k + 3]) <= (int64_t(1) << 20), "I8ChannelShift: shift too large");
    for (int64_t ch = g[k]; ch < g[k + 1]; ++ch) {
      BF_CHECK(!used[ch], "I8ChannelShift: overlapping channel ranges");
      used[ch] = 1;
    }
  }
  const int64_t f = n.attrs.get_int("fill_q", 0);
  BF_CHECK(f >= -128 && f <= 127, "I8ChannelShift: fill_q out of range");
  set_out(outs, DType::kI8, x.shape);
}

// I8BinarizeDyn: x i8 [N, H, W, C], T0 / A / B i32[C], o i8 [N, 1, 1, C]? with Ko i32[C]; attributes x_zero, o_zero
// (|.| <= 2^16), shift F (0..40), sqrt_shift G (0..20); output a bit tensor of x's shape. Bounds (int8_ops.binarize_dyn_bound_ok):
// HW^2 255^2 4^G < 2^62, HW 2^F 255 < 2^62, max|T0| + max|A| 255 HW + max|B| 255 HW 2^G + max|Ko| 255 < 2^62.
void infer_binarize_dyn(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8_4d(ins, 0, "I8BinarizeDyn", "x");
  const int64_t c = x.shape[3], hw = x.shape[1] * x.shape[2];
  check_vec(ins, 1, DType::kI32, c, "I8BinarizeDyn", "T0");
  check_vec(ins, 2, DType::kI32, c, "I8BinarizeDyn", "A");
  check_vec(ins, 3, DType::kI32, c, "I8BinarizeDyn", "B");
  if (has(ins, 4)) {
    BF_CHECK(ins[4]->dtype == DType::kI8 && ins[4]->shape == Shape({x.shape[0], 1, 1, c}), "I8BinarizeDyn: o must be i8 [N, 1, 1, C]");
    check_vec(ins, 5, DType::kI32, c, "I8BinarizeDyn", "Ko");
  }
  const int64_t F = n.attrs.get_int("shift", 0), G = n.attrs.get_int("sqrt_shift", 0);
  BF_CHECK(F >= 0 && F <= 40 && G >= 0 && G <= 20, "I8BinarizeDyn: shift / sqrt_shift out of range");
  for (const char* k : {"x_zero", "o_zero"})
    BF_CHECK(std::llabs(n.attrs.get_int(k, 0)) <= (int64_t(1) << 16), "I8BinarizeDyn: ", k, " out of range");
  BF_CHECK(hw >= 1 && hw < (int64_t(1) << 20), "I8BinarizeDyn: spatial size out of range");
  const double l62 = 4.611686018427388e18, h = static_cast<double>(hw);
  BF_CHECK(h * h * 65025.0 * std::ldexp(1.0, static_cast<int>(2 * G)) < l62 && h * std::ldexp(1.0, static_cast<int>(F)) * 255.0 < l62,
           "I8BinarizeDyn: statistics exceed the int64 bound");
  BF_CHECK(!outs.empty() && outs[0], "I8BinarizeDyn: missing output");
  outs[0]->dtype = DType::kBit;
  outs[0]->shape = x.shape;
}

// I8SimAM: x i8 [N, H, W, C] (|x_zero| <= 128), tab i8 [1, 256]; attributes lam_q (1 .. 2^40), g_zero, mult, shift,
// y_zero, qmin, qmax; y i8 of x's shape.
void infer_simam(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req_i8_4d(ins, 0, "I8SimAM", "x");
  BF_CHECK(has(ins, 1) && ins[1]->dtype == DType::kI8 && ins[1]->shape == Shape({1, 256}), "I8SimAM: tab must be i8 [1, 256]");
  BF_CHECK(std::llabs(n.attrs.get_int("x_zero", 0)) <= 128, "I8SimAM: |x_zero| must be <= 128");
  const int64_t L = n.attrs.get_int("lam_q", 1);
  BF_CHECK(L >= 1 && L <= (int64_t(1) << 40), "I8SimAM: lam_q out of range");
  BF_CHECK(x.shape[1] * x.shape[2] < (int64_t(1) << 20), "I8SimAM: spatial size too large");
  BF_CHECK(n.attrs.get_int("mult", 0) >= 0 && n.attrs.get_int("mult", 0) < (int64_t(1) << 31), "I8SimAM: mult out of range");
  const int64_t sh = n.attrs.get_int("shift", 0);
  BF_CHECK(sh >= -62 && sh <= 30, "I8SimAM: shift out of range");
  const int64_t qmin = n.attrs.get_int("qmin", -128), qmax = n.attrs.get_int("qmax", 127);
  BF_CHECK(qmin >= -128 && qmin <= qmax && qmax <= 127, "I8SimAM: qmin / qmax out of range");
  set_out(outs, DType::kI8, x.shape);
}

}  // namespace

void register_int8_glue_op_defs(OpRegistry& reg) {
  reg.add_op({"I8ChannelShift", infer_channel_shift});
  reg.add_op({"I8BinarizeDyn", infer_binarize_dyn});
  reg.add_op({"I8SimAM", infer_simam});
}

}  // namespace bitino

// Operator semantics: attribute validation and shape inference.
#include "ops/op_defs.h"

#include "ops/ops_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

void set_out(Outs& outs, size_t i, DType dt, Shape s) {
  BF_CHECK(i < outs.size(), "missing output ", i);
  outs[i]->dtype = dt;
  outs[i]->shape = std::move(s);
}

const TensorInfo& req(const Ins& ins, size_t i, const char* what) {
  BF_CHECK(i < ins.size() && ins[i], "missing required input ", i, " (", what, ")");
  return *ins[i];
}

void check_vec(const Ins& ins, size_t i, int64_t len, const char* what) {
  if (i >= ins.size() || !ins[i]) return;
  BF_CHECK(ins[i]->dtype == DType::kF32 && ins[i]->shape.size() == 1 && ins[i]->shape[0] == len,
           what, " must be f32[", len, "], got ", dtype_name(ins[i]->dtype), shape_str(ins[i]->shape));
}

// "output": "f32" | "bit" of BConv2D / BDense (kernels/bit_output.h). Bit output stores
// bit = 1 (-1) iff !(v >= t[c]) ("output_binarize" ge, default) or !(v > t[c]) (gt), v the
// final float output (after the fused epilogue, if any), t the f32[oc] input at index
// "output_threshold_input" (after the `fixed` inputs) or 0: the next layer's binarization.
DType output_dtype(const Node& n, const Ins& ins, int64_t oc, size_t fixed) {
  const std::string od = n.attrs.get_string("output", "f32");
  if (od == "f32") {
    BF_CHECK(!n.attrs.contains("output_binarize") && !n.attrs.contains("output_threshold_input"),
             "output_binarize / output_threshold_input need output 'bit'");
    return DType::kF32;
  }
  BF_CHECK(od == "bit", n.op, " output must be 'f32' or 'bit'");
  const std::string ob = n.attrs.get_string("output_binarize", "ge");
  BF_CHECK(ob == "ge" || ob == "gt", "output_binarize must be 'ge' or 'gt', got '", ob, "'");
  if (n.attrs.contains("output_threshold_input")) {
    const int64_t k = n.attrs.get_int("output_threshold_input", -1);
    BF_CHECK(k >= static_cast<int64_t>(fixed) && k < static_cast<int64_t>(ins.size()) && ins[static_cast<size_t>(k)],
             "output_threshold_input ", k, " must name an input after the ", fixed, " fixed ones");
    check_vec(ins, static_cast<size_t>(k), oc, "output threshold");
  }
  return DType::kBit;
}

void infer_conv2d(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  const auto& w = req(ins, 1, "filter");
  BF_CHECK(x.dtype == DType::kF32 && w.dtype == DType::kF32, "Conv2D expects f32 input and filter");
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  check_vec(ins, 2, g.oc, "bias");
  parse_activation(n.attrs);
  set_out(outs, 0, DType::kF32, {g.n, g.oh, g.ow, g.oc});
}

void infer_bconv2d(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  const auto& w = req(ins, 1, "filter");
  BF_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBit, "BConv2D input must be f32 or bit");
  BF_CHECK(w.dtype == DType::kBit, "BConv2D filter must be bit-packed (dtype bit), got ", dtype_name(w.dtype));
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  check_vec(ins, 2, g.oc, "post_mul");
  check_vec(ins, 3, g.oc, "post_add");
  check_vec(ins, 4, g.ic, "threshold");
  double pv = n.attrs.get_number("pad_value", 0.0);
  BF_CHECK(pv == 0.0 || pv == 1.0 || pv == -1.0, "pad_value must be -1, 0 or +1, got ", pv);
  BinarizeMode m = parse_binarize(n.attrs);
  BF_CHECK(x.dtype == DType::kF32 || (m != BinarizeMode::kSignZero && !(ins.size() > 4 && ins[4])),
           "bit input cannot use thresholds or sign_zero binarization");
  parse_activation(n.attrs);
  set_out(outs, 0, output_dtype(n, ins, g.oc, 5), {g.n, g.oh, g.ow, g.oc});
}

void infer_dense(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  const auto& w = req(ins, 1, "weights");
  BF_CHECK(w.shape.size() == 2, "Dense weights must be [O, K]");
  BF_CHECK(!x.shape.empty(), "Dense input must have rank >= 1");
  int64_t k = x.shape.back();
  BF_CHECK(w.shape[1] == k, "Dense K mismatch: input ", shape_str(x.shape), " weights ", shape_str(w.shape));
  bool binary = n.op == "BDense";
  if (binary) {
    BF_CHECK(w.dtype == DType::kBit, "BDense weights must be bit-packed");
    check_vec(ins, 2, w.shape[0], "post_mul");
    check_vec(ins, 3, w.shape[0], "post_add");
    check_vec(ins, 4, k, "threshold");
    BF_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBit, "BDense input must be f32 or bit");
    BF_CHECK(x.dtype == DType::kF32 || (parse_binarize(n.attrs) != BinarizeMode::kSignZero && !(ins.size() > 4 && ins[4])),
             "bit input cannot use thresholds or sign_zero binarization");
    parse_binarize(n.attrs);
    // Optional w_zero (input 5): bit[O, K]; a set bit (the "-1" encoding) marks an exact
    // zero weight (ternary weights, e.g. torch.sign of a centered weight). Restricted to
    // sign_zero nodes: every packed +-1 kernel already rejects sign_zero, so only
    // ternary-aware kernels (reference, ternary_bitplane) can be selected.
    if (ins.size() > 5 && ins[5]) {
      BF_CHECK(ins[5]->dtype == DType::kBit && ins[5]->shape == w.shape, "BDense w_zero must be bit with the weights' shape");
      BF_CHECK(n.attrs.get_string("input_binarize", "ge") == "sign_zero", "BDense w_zero requires input_binarize=sign_zero");
    }
  } else {
    BF_CHECK(x.dtype == DType::kF32 && w.dtype == DType::kF32, "Dense expects f32");
    check_vec(ins, 2, w.shape[0], "bias");
  }
  parse_activation(n.attrs);
  Shape out(x.shape.begin(), x.shape.end() - 1);
  out.push_back(w.shape[0]);
  set_out(outs, 0, binary ? output_dtype(n, ins, w.shape[0], 6) : DType::kF32, out);
}

void infer_same_f32(const Node&, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32, "expected f32 input");
  set_out(outs, 0, DType::kF32, x.shape);
}

void infer_channel_affine(const Node& n, const Ins& ins, Outs& outs) {
  infer_same_f32(n, ins, outs);
  int64_t c = ins[0]->shape.back();
  check_vec(ins, 1, c, "scale");
  check_vec(ins, 2, c, "bias");
}

void infer_activation(const Node& n, const Ins& ins, Outs& outs) {
  infer_same_f32(n, ins, outs);
  std::string t = n.attrs.get_string("type", "");
  static const char* kTypes[] = {"relu", "relu6", "hardtanh", "prelu", "leaky_relu", "sigmoid", "hardswish", "tanh", "gelu", "sign", "sign_zero"};
  bool ok = false;
  for (const char* k : kTypes) ok |= t == k;
  BF_CHECK(ok, "unknown activation type '", t, "'");
  if (t == "prelu") check_vec(ins, 1, ins[0]->shape.back(), "slope");
}

void infer_add(const Node& n, const Ins& ins, Outs& outs) {
  const auto& a = req(ins, 0, "a");
  const auto& b = req(ins, 1, "b");
  BF_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32, "Add expects f32");
  BF_CHECK(a.shape == b.shape, "Add shape mismatch ", shape_str(a.shape), " vs ", shape_str(b.shape));
  parse_activation(n.attrs);
  set_out(outs, 0, DType::kF32, a.shape);
}

void infer_concat(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(!ins.empty(), "Concat needs inputs");
  Shape s = req(ins, 0, "input0").shape;
  int64_t axis = n.attrs.get_int("axis", -1);
  if (axis < 0) axis += static_cast<int64_t>(s.size());
  BF_CHECK(axis >= 0 && axis < static_cast<int64_t>(s.size()), "bad concat axis");
  s[axis] = 0;
  for (size_t i = 0; i < ins.size(); ++i) {
    const auto& t = req(ins, i, "input");
    BF_CHECK(t.dtype == DType::kF32 && t.shape.size() == s.size(), "Concat inputs must be f32 with equal rank");
    for (size_t d = 0; d < s.size(); ++d)
      if (static_cast<int64_t>(d) != axis) BF_CHECK(t.shape[d] == ins[0]->shape[d], "Concat dim mismatch");
    s[axis] += t.shape[axis];
  }
  set_out(outs, 0, DType::kF32, s);
}

void infer_pool(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32, "pool expects f32");
  auto g = pool_geometry(n.attrs, x.shape);
  set_out(outs, 0, DType::kF32, {g.n, g.oh, g.ow, g.c});
}

void infer_global_avg_pool(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32 && x.shape.size() == 4, "GlobalAvgPool expects f32 NHWC");
  if (n.attrs.get_int("keepdims", 0)) set_out(outs, 0, DType::kF32, {x.shape[0], 1, 1, x.shape[3]});
  else set_out(outs, 0, DType::kF32, {x.shape[0], x.shape[3]});
}

void infer_reshape(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  auto s = n.attrs.get_ints("shape", {});
  int64_t known = 1, infer_at = -1;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == -1) {
      BF_CHECK(infer_at < 0, "at most one -1 in reshape");
      infer_at = static_cast<int64_t>(i);
    } else {
      known *= s[i];
    }
  }
  if (infer_at >= 0) s[infer_at] = num_elements(x.shape) / known;
  BF_CHECK(num_elements(s) == num_elements(x.shape), "reshape element count mismatch");
  BF_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kI8, "Reshape supports f32 and i8");
  set_out(outs, 0, x.dtype, s);
}

void infer_pad(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  auto p = n.attrs.get_ints("pads", {});  // [before_0, after_0, before_1, after_1, ...]
  BF_CHECK(p.size() == 2 * x.shape.size(), "Pad needs 2 values per dim");
  Shape s = x.shape;
  for (size_t d = 0; d < s.size(); ++d) s[d] += p[2 * d] + p[2 * d + 1];
  set_out(outs, 0, DType::kF32, s);
}

}  // namespace

void register_op_defs(OpRegistry& reg) {
  reg.add_op({"Conv2D", infer_conv2d});
  reg.add_op({"BConv2D", infer_bconv2d});
  reg.add_op({"Dense", infer_dense});
  reg.add_op({"BDense", infer_dense});
  reg.add_op({"ChannelAffine", infer_channel_affine});
  reg.add_op({"Activation", infer_activation});
  reg.add_op({"Add", infer_add});
  reg.add_op({"Concat", infer_concat});
  reg.add_op({"AvgPool2D", infer_pool});
  reg.add_op({"MaxPool2D", infer_pool});
  reg.add_op({"GlobalAvgPool", infer_global_avg_pool});
  reg.add_op({"Reshape", infer_reshape});
  reg.add_op({"Softmax", infer_same_f32});
  reg.add_op({"Pad", infer_pad});
}

}  // namespace bitino

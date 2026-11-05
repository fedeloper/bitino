// Generic operators added for BHViT (binarized hybrid ViT): attribute validation
// and shape inference. Contracts: docs/operators_bhvit.md.
//   FakeQuant  - per-channel uniform quantize-dequantize (LSQ-style, e.g. 2-bit attention probabilities)
//   ReduceMean - mean over one axis (channel AvgPool1d, mean over windows)
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

const TensorInfo& req(const Ins& ins, size_t i, const char* what) {
  BF_CHECK(i < ins.size() && ins[i], "missing required input ", i, " (", what, ")");
  return *ins[i];
}

int64_t norm_axis(int64_t axis, size_t rank) {
  if (axis < 0) axis += static_cast<int64_t>(rank);
  BF_CHECK(axis >= 0 && axis < static_cast<int64_t>(rank), "axis out of range for rank ", rank);
  return axis;
}

// FakeQuant(x, scale[C], zero_point[C]) with C = x.shape[axis]:
//   y = clamp(nearbyint(((x * pre_scale) / scale[c]) + zero_point[c]), qmin, qmax) * scale[c]
void infer_fake_quant(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32 && !x.shape.empty(), "FakeQuant expects f32 input");
  int64_t axis = norm_axis(n.attrs.get_int("axis", -1), x.shape.size());
  int64_t c = x.shape[axis];
  for (size_t i = 1; i <= 2; ++i) {
    const auto& t = req(ins, i, i == 1 ? "scale" : "zero_point");
    BF_CHECK(t.dtype == DType::kF32 && t.shape == Shape{c}, "FakeQuant ", i == 1 ? "scale" : "zero_point", " must be f32[", c, "]");
  }
  int64_t qmin = n.attrs.get_int("qmin", 0), qmax = n.attrs.get_int("qmax", 1);
  BF_CHECK(qmin <= qmax, "FakeQuant needs qmin <= qmax");
  n.attrs.get_number("pre_scale", 1.0);
  BF_CHECK(!outs.empty(), "missing output");
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = x.shape;
}

void infer_reduce_mean(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  BF_CHECK(x.dtype == DType::kF32 && !x.shape.empty(), "ReduceMean expects f32 input");
  int64_t axis = norm_axis(n.attrs.get_int("axis", -1), x.shape.size());
  BF_CHECK(x.shape[axis] > 0, "ReduceMean over an empty axis");
  Shape s = x.shape;
  if (n.attrs.get_int("keepdims", 0)) s[axis] = 1;
  else s.erase(s.begin() + axis);
  BF_CHECK(!outs.empty(), "missing output");
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = s;
}

}  // namespace

void register_bhvit_op_defs(OpRegistry& reg) {
  reg.add_op({"FakeQuant", infer_fake_quant});
  reg.add_op({"ReduceMean", infer_reduce_mean});
}

}  // namespace bitino

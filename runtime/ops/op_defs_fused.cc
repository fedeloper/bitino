// OpDef of the internal FusedElementwise operator (ops/fused_elementwise.h),
// created by the plan-time rewrite in src/graph_rewrite.cc.
#include "ops/fused_elementwise.h"
#include "ops/op_defs.h"

namespace bitino {
namespace {

void infer_fused_elementwise(const Node& n, const std::vector<const TensorInfo*>& ins, std::vector<TensorInfo*>& outs) {
  const std::vector<FeInstr> prog = parse_fe_program(n.attrs);
  fe_stack_depth(prog);
  const TensorInfo* shape_src = nullptr;
  auto in = [&](int32_t i, const char* what) -> const TensorInfo& {
    BF_CHECK(i >= 0 && static_cast<size_t>(i) < ins.size() && ins[i], "FusedElementwise: bad ", what, " input ", i);
    return *ins[i];
  };
  for (const FeInstr& i : prog)
    if (i.op == FeOp::kLoad) {
      const TensorInfo& t = in(i.a, "load");
      BF_CHECK(t.dtype == DType::kF32, "FusedElementwise: f32 operands only");
      if (!shape_src) shape_src = &t;
      BF_CHECK(t.shape == shape_src->shape, "FusedElementwise: operand shapes differ: ", shape_str(t.shape), " vs ",
               shape_str(shape_src->shape));
    }
  Shape out_shape;
  if (n.attrs.contains("out_shape")) out_shape = n.attrs.get_ints("out_shape", {});
  BF_CHECK((shape_src && !shape_src->shape.empty()) || !out_shape.empty(), "FusedElementwise: no tensor operand");
  if (out_shape.empty()) out_shape = shape_src->shape;
  BF_CHECK(!shape_src || shape_src->shape == out_shape, "FusedElementwise: operands ", shape_str(shape_src->shape), " vs output ",
           shape_str(out_shape));
  const Shape& s = out_shape;
  const int64_t c = s.back();
  for (const FeInstr& i : prog) {
    if (i.op == FeOp::kLoadGather) {
      const TensorInfo& t = in(i.a, "gather table");
      const TensorInfo& ids = in(i.b, "gather ids");
      BF_CHECK(t.dtype == DType::kF32 && t.shape.size() == 2 && t.shape[1] == i.act && i.act > 0 && c % i.act == 0,
               "FusedElementwise: gather table must be f32 [V, D] with C % D == 0");
      BF_CHECK(ids.dtype == DType::kI32 && num_elements(ids.shape) * i.act == num_elements(s), "FusedElementwise: gather ids");
    } else if (i.op == FeOp::kLoadTile) {
      const TensorInfo& t = in(i.a, "tile source");
      BF_CHECK(t.dtype == DType::kF32 && i.act > 0 && t.shape.back() == i.act && c % i.act == 0 &&
                   num_elements(t.shape) * (c / i.act) == num_elements(s),
               "FusedElementwise: tile source must be f32 [..., c0] with C % c0 == 0");
    }
  }
  auto vec = [&](int32_t i, const char* what) {
    if (i < 0) return;
    const TensorInfo& t = in(i, what);
    BF_CHECK(t.dtype == DType::kF32 && t.shape.size() == 1 && t.shape[0] == c, "FusedElementwise: ", what, " must be f32[", c, "]");
  };
  for (const FeInstr& i : prog) {
    if (i.op == FeOp::kAffine) {
      vec(i.a, "scale");
      vec(i.b, "bias");
    } else if (i.op == FeOp::kPrelu) {
      BF_CHECK(i.a >= 0, "FusedElementwise: prelu needs a slope");
      vec(i.a, "slope");
    } else if (i.op == FeOp::kGate) {
      const TensorInfo& g = in(i.a, "gate");
      BF_CHECK(s.size() == 4 && g.dtype == DType::kF32 && num_elements(g.shape) == s[0] * c && g.shape.back() == c,
               "FusedElementwise: gate must be [N, C] / [N, 1, 1, C] for ", shape_str(s));
    }
  }
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = s;
}

}  // namespace

void register_fused_ops(OpRegistry& reg) { reg.add_op({"FusedElementwise", infer_fused_elementwise}); }

}  // namespace bitino

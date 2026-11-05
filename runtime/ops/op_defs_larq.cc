// Operators added for the remaining Larq Zoo models (see docs/operators_larq.md):
//   QBConv2D / QBDense : k-bit (DoReFa) activations x binary weights, integer accumulation.
//   ChannelGate        : per-(sample, channel) gating x * sigmoid(g) (squeeze-excite style
//                        data-dependent scaling of Real-to-Binary Net).
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
  BF_CHECK(ins[i]->dtype == DType::kF32 && ins[i]->shape.size() == 1 && ins[i]->shape[0] == len, what, " must be f32[",
           len, "], got ", dtype_name(ins[i]->dtype), shape_str(ins[i]->shape));
}

void check_act_quant(const Node& n) {
  int64_t k = n.attrs.get_int("act_bits", 0);
  BF_CHECK(k >= 1 && k <= 8, n.op, ": act_bits must be in [1, 8], got ", k);
  std::string q = n.attrs.get_string("act_quant", "dorefa");
  BF_CHECK(q == "dorefa", n.op, ": unsupported act_quant '", q, "' (only 'dorefa')");
}

void infer_qbconv2d(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  const auto& w = req(ins, 1, "filter");
  BF_CHECK(x.dtype == DType::kF32, "QBConv2D input must be f32");
  BF_CHECK(w.dtype == DType::kBit, "QBConv2D filter must be bit-packed (dtype bit), got ", dtype_name(w.dtype));
  BF_CHECK(ins.size() <= 4, "QBConv2D takes at most 4 inputs (x, w, post_mul, post_add)");
  auto g = conv_geometry(n.attrs, x.shape, w.shape);
  check_vec(ins, 2, g.oc, "post_mul");
  check_vec(ins, 3, g.oc, "post_add");
  double pv = n.attrs.get_number("pad_value", 0.0);
  BF_CHECK(pv == 0.0, "QBConv2D supports only pad_value 0 (zero padding of the quantized input), got ", pv);
  check_act_quant(n);
  parse_activation(n.attrs);
  set_out(outs, 0, DType::kF32, {g.n, g.oh, g.ow, g.oc});
}

void infer_qbdense(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  const auto& w = req(ins, 1, "weights");
  BF_CHECK(x.dtype == DType::kF32, "QBDense input must be f32");
  BF_CHECK(w.dtype == DType::kBit && w.shape.size() == 2, "QBDense weights must be bit[O, K]");
  BF_CHECK(ins.size() <= 4, "QBDense takes at most 4 inputs (x, w, post_mul, post_add)");
  int64_t k = x.shape.back();
  BF_CHECK(w.shape[1] == k, "QBDense K mismatch: input ", shape_str(x.shape), " weights ", shape_str(w.shape));
  check_vec(ins, 2, w.shape[0], "post_mul");
  check_vec(ins, 3, w.shape[0], "post_add");
  check_act_quant(n);
  parse_activation(n.attrs);
  BF_CHECK(!x.shape.empty(), "QBDense input must have rank >= 1");
  Shape out(x.shape.begin(), x.shape.end() - 1);
  out.push_back(w.shape[0]);
  set_out(outs, 0, DType::kF32, out);
}

void infer_channel_gate(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = req(ins, 0, "input");
  const auto& g = req(ins, 1, "gate");
  BF_CHECK(x.dtype == DType::kF32 && g.dtype == DType::kF32, "ChannelGate expects f32 inputs");
  BF_CHECK(x.shape.size() == 4, "ChannelGate input must be NHWC, got ", shape_str(x.shape));
  bool flat = g.shape.size() == 2 && g.shape[0] == x.shape[0] && g.shape[1] == x.shape[3];
  bool keep = g.shape.size() == 4 && g.shape[0] == x.shape[0] && g.shape[1] == 1 && g.shape[2] == 1 &&
              g.shape[3] == x.shape[3];
  BF_CHECK(flat || keep, "ChannelGate gate must be [N, C] or [N, 1, 1, C] for input ", shape_str(x.shape), ", got ",
           shape_str(g.shape));
  std::string t = n.attrs.get_string("gate", "sigmoid");
  BF_CHECK(t == "sigmoid" || t == "none", "ChannelGate gate must be 'sigmoid' or 'none', got '", t, "'");
  set_out(outs, 0, DType::kF32, x.shape);
}

}  // namespace

void register_larq_ops(OpRegistry& reg) {
  reg.add_op({"QBConv2D", infer_qbconv2d});
  reg.add_op({"QBDense", infer_qbdense});
  reg.add_op({"ChannelGate", infer_channel_gate});
}

}  // namespace bitino

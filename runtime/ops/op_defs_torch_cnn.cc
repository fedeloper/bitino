// Operator needed by PyTorch binary CNNs (BNext ELM attention). Contract: docs/operators_torch_cnn.md.
//
//   Mul(a, b):  y = a * b, same shapes (elementwise product of two runtime tensors).
//
// Everything else on the BNext / RBONN path reuses existing operators: the info-recoupling
// mix scale*a + (1-scale)*b is ChannelAffine(a, scale) + ChannelAffine(b, 1-scale) -> Add (same
// rounding sequence as PyTorch eager), and the squeeze-and-excitation gate is ChannelGate
// (gate=sigmoid, docs/operators_larq.md).
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

void infer_mul(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(ins.size() == 2 && ins[0] && ins[1], "Mul needs two inputs");
  const auto& a = *ins[0];
  const auto& b = *ins[1];
  BF_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32, "Mul expects f32 inputs");
  BF_CHECK(a.shape == b.shape, "Mul: shapes must match, got ", shape_str(a.shape), " and ", shape_str(b.shape));
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = a.shape;
}

}  // namespace

void register_torch_cnn_ops(OpRegistry& reg) { reg.add_op({"Mul", infer_mul}); }

}  // namespace bitino

// FusedElementwise: an internal operator created by the plan-time rewrite
// (src/graph_rewrite.cc), never stored in a .bitino. It evaluates a tree of
// elementwise operators in one pass, as a stack program over equally shaped
// f32 tensors [..., C] (C = last dim). Instructions, in post order:
//
//   load   a            push input a (same shape as the output)
//   affine a=s b=bias   top = top * (s ? s[c] : 1) + (bias ? bias[c] : 0)   (ChannelAffine)
//   add    act          pop b, pop a, push act(a + b)                       (Add, fused activation)
//   mul                 pop b, pop a, push a * b                            (Mul)
//   relu                top = top > 0 ? top : 0                             (Activation relu)
//   clamp  lo hi        top = min(max(top, lo), hi)                         (relu6, hardtanh)
//   leaky  alpha        top = top >= 0 ? top : top * alpha                  (leaky_relu)
//   prelu  a=slope      top = top >= 0 ? top : top * slope[c]               (prelu)
//   sign                top = top >= 0 ? 1 : -1                             (sign)
//   sign0               top = top > 0 ? 1 : (top < 0 ? -1 : 0)              (sign_zero)
//   gate   a=g b=sig    top = top * s[n][c], s = sig ? float(1/(1+exp(-(double)g))) : g
//                       (ChannelGate; g is [N, C] or [N, 1, 1, C], n = batch of the row)
//   loadg  a=table b=ids act=D   push rows gathered from table [V, D] (f32): output row r
//                       (C floats) = table[ids[r*C/D + q]] for q < C/D, zero row for ids
//                       outside [0, V) (Gather, optionally between Reshapes; BHViT roll)
//   loadt  a=x act=c0    push x's rows of c0 floats repeated C/c0 times (Concat of one
//                       tensor with itself along the last axis; BHViT repeat shortcut)
//
// Each instruction is exactly the float operation of the reference kernel of
// the operator it replaces, so the fused node is bit-identical to the
// unfused nodes. Program encoding in the node attributes:
//   "program": [[op, a, b, act, lo, hi, alpha], ...]
#pragma once

#include <cstdint>
#include <vector>

#include "bitino/error.h"
#include "bitino/json.h"

namespace bitino {

enum class FeOp : int32_t { kLoad = 0, kAffine, kAdd, kMul, kRelu, kClamp, kLeaky, kPrelu, kSign, kSignZero, kGate, kLoadGather, kLoadTile };

inline bool fe_is_load(FeOp op) { return op == FeOp::kLoad || op == FeOp::kLoadGather || op == FeOp::kLoadTile; }

struct FeInstr {
  FeOp op = FeOp::kLoad;
  int32_t a = -1, b = -1;  // input indices (load / affine scale+bias / prelu slope / gate); gate: b = sigmoid flag
  int32_t act = 0;         // add: fused activation (0 none, 1 relu, 2 relu6)
  float lo = 0.f, hi = 0.f, alpha = 0.f;
};

inline Json fe_program_json(const std::vector<FeInstr>& prog) {
  Json::Array arr;
  for (const FeInstr& i : prog)
    arr.push_back(Json(Json::Array{Json(static_cast<int>(i.op)), Json(i.a), Json(i.b), Json(i.act), Json(static_cast<double>(i.lo)),
                                   Json(static_cast<double>(i.hi)), Json(static_cast<double>(i.alpha))}));
  return Json(arr);
}

inline std::vector<FeInstr> parse_fe_program(const Json& attrs) {
  std::vector<FeInstr> prog;
  const Json& arr = attrs["program"];
  for (size_t k = 0; k < arr.size(); ++k) {
    const Json& e = arr[k];
    BF_CHECK(e.is_array() && e.size() == 7, "FusedElementwise: malformed instruction ", k);
    FeInstr i;
    const int64_t op = e[0].as_int();
    BF_CHECK(op >= 0 && op <= static_cast<int64_t>(FeOp::kLoadTile), "FusedElementwise: unknown opcode ", op);
    i.op = static_cast<FeOp>(op);
    i.a = static_cast<int32_t>(e[1].as_int());
    i.b = static_cast<int32_t>(e[2].as_int());
    i.act = static_cast<int32_t>(e[3].as_int());
    i.lo = static_cast<float>(e[4].as_number());
    i.hi = static_cast<float>(e[5].as_number());
    i.alpha = static_cast<float>(e[6].as_number());
    prog.push_back(i);
  }
  return prog;
}

// Maximum stack depth of a program; throws if it under/overflows or leaves
// anything but one value.
inline int fe_stack_depth(const std::vector<FeInstr>& prog) {
  int d = 0, m = 0;
  for (const FeInstr& i : prog) {
    if (fe_is_load(i.op)) d += 1;
    else if (i.op == FeOp::kAdd || i.op == FeOp::kMul) d -= 1;
    BF_CHECK(d >= 1, "FusedElementwise: stack underflow");
    m = d > m ? d : m;
  }
  BF_CHECK(d == 1, "FusedElementwise: program must leave one value");
  return m;
}

}  // namespace bitino

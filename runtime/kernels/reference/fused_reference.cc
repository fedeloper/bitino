// Reference kernel of the internal FusedElementwise operator
// (ops/fused_elementwise.h): evaluates the program element by element with
// the scalar expressions of the reference kernels it replaces
// (ChannelAffineRef, AddRef, MulRef, ActivationRef, ChannelGateRef), so it is
// bit-identical to running those nodes one after the other.
#include <algorithm>
#include <cmath>
#include <vector>

#include "ops/fused_elementwise.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

class FusedElementwiseRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    prog_ = parse_fe_program(ctx.attrs());
    fe_stack_depth(prog_);
    gate_.assign(prog_.size(), {});
    for (size_t k = 0; k < prog_.size(); ++k)
      if (prog_[k].op == FeOp::kGate) gate_[k].assign(static_cast<size_t>(ctx.in(prog_[k].a).size()), 0.f);
  }
  void run(const KernelContext& ctx) override {
    Tensor& yt = ctx.out(0);
    const int64_t c = yt.dim(-1), n = yt.size();
    const int64_t per_batch = yt.dim(0) > 0 ? n / yt.dim(0) : n;
    for (size_t k = 0; k < prog_.size(); ++k)
      if (prog_[k].op == FeOp::kGate) {
        const float* g = ctx.in(prog_[k].a).data<float>();
        for (size_t i = 0; i < gate_[k].size(); ++i)
          gate_[k][i] = prog_[k].b ? static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(g[i])))) : g[i];
      }
    auto in = [&](int32_t i) { return ctx.in(i).data<float>(); };
    float* y = yt.data<float>();
    std::vector<float> st(prog_.size() + 1);
    for (int64_t e = 0; e < n; ++e) {
      const int64_t ch = e % c;
      int sp = 0;
      for (size_t k = 0; k < prog_.size(); ++k) {
        const FeInstr& i = prog_[k];
        float& v = st[sp > 0 ? sp - 1 : 0];
        switch (i.op) {
          case FeOp::kLoad: st[sp++] = in(i.a)[e]; break;
          case FeOp::kLoadGather: {  // row e / c, channel ch -> table[ids[...]][ch % D] (0 out of range)
            const int64_t per = c / i.act, id = ctx.in(i.b).data<int32_t>()[(e / c) * per + ch / i.act];
            const int64_t v = ctx.in(i.a).dim(0);
            st[sp++] = (id < 0 || id >= v) ? 0.f : in(i.a)[id * i.act + ch % i.act];
            break;
          }
          case FeOp::kLoadTile: st[sp++] = in(i.a)[(e / c) * i.act + ch % i.act]; break;
          case FeOp::kAffine: v = v * (i.a >= 0 ? in(i.a)[ch] : 1.f) + (i.b >= 0 ? in(i.b)[ch] : 0.f); break;
          case FeOp::kAdd: {
            const float b = st[--sp];
            st[sp - 1] = apply_activation(static_cast<Activation>(i.act), st[sp - 1] + b);
            break;
          }
          case FeOp::kMul: {
            const float b = st[--sp];
            st[sp - 1] = st[sp - 1] * b;
            break;
          }
          case FeOp::kRelu: v = v > 0.f ? v : 0.f; break;
          case FeOp::kClamp: v = std::min(std::max(v, i.lo), i.hi); break;
          case FeOp::kLeaky: v = v >= 0.f ? v : v * i.alpha; break;
          case FeOp::kPrelu: v = v >= 0.f ? v : v * in(i.a)[ch]; break;
          case FeOp::kSign: v = v >= 0.f ? 1.f : -1.f; break;
          case FeOp::kSignZero: v = v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); break;
          case FeOp::kGate: v = v * gate_[k][(e / per_batch) * c + ch]; break;
        }
      }
      y[e] = st[0];
    }
  }

 private:
  std::vector<FeInstr> prog_;
  std::vector<std::vector<float>> gate_;
};

}  // namespace

void register_fused_reference_kernels(OpRegistry& reg) {
  reg.add_kernel({"FusedElementwise", "fused_elementwise.reference", "reference", 0, 0, nullptr, make_kernel<FusedElementwiseRef>});
}

}  // namespace bitino

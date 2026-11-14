// Float (stage reference / neon / optimized) ops for accuracy-carrying glue (2026-11-14, the research notes (not published)):
// InstanceThreshold (INSTA-BNN) and ChannelShift (ShiftViT). Bit-identical to the numpy executor
// (python/bitino/executor.py instance_threshold / channel_shift):
//   InstanceThreshold(x, t0, a, b[, off]): per (n, c), in double, sums over the H x W positions in row-major order:
//     s1 = sum x, s2 = sum x * x, mu = s1 / HW, var = s2 / HW - mu * mu, sd = sqrt(max(var, 0)),
//     th = t0[c] + a[c] * mu + b[c] * sd (+ off[n, c]),   y = float(double(x) - th)
//   ChannelShift(x; groups [c0, c1, dy, dx] ...): y[n, h, w, c] = x[n, h - dy, w - dx, c] for c in a group (0 outside the
//     map), other channels copied.
// The int8_full forms are I8BinarizeDyn and I8ChannelShift (ops/op_defs_int8_glue.cc).
#include <cmath>
#include <vector>

#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

void infer_instance_threshold(const Node&, const Ins& ins, Outs& outs) {
  BF_CHECK(ins.size() >= 4 && ins[0] && ins[1] && ins[2] && ins[3], "InstanceThreshold needs x, t0, a, b");
  const auto& x = *ins[0];
  BF_CHECK(x.dtype == DType::kF32 && x.shape.size() == 4, "InstanceThreshold: x must be f32 [N, H, W, C]");
  const int64_t c = x.shape[3];
  for (size_t i = 1; i < 4; ++i)
    BF_CHECK(ins[i]->dtype == DType::kF32 && num_elements(ins[i]->shape) == c, "InstanceThreshold: t0 / a / b must be f32[C]");
  if (ins.size() > 4 && ins[4])
    BF_CHECK(ins[4]->dtype == DType::kF32 && num_elements(ins[4]->shape) == x.shape[0] * c, "InstanceThreshold: off must be f32 [N, 1, 1, C]");
  BF_CHECK(!outs.empty() && outs[0], "missing output");
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = x.shape;
}

void infer_channel_shift_f(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(!ins.empty() && ins[0], "ChannelShift needs x");
  const auto& x = *ins[0];
  BF_CHECK(x.dtype == DType::kF32 && x.shape.size() == 4, "ChannelShift: x must be f32 [N, H, W, C]");
  auto g = n.attrs.get_ints("groups", {});
  BF_CHECK(g.size() % 4 == 0, "ChannelShift: groups must be [c0, c1, dy, dx] quadruples (flat)");
  for (size_t i = 0; i < g.size(); i += 4) BF_CHECK(0 <= g[i] && g[i] < g[i + 1] && g[i + 1] <= x.shape[3], "ChannelShift: bad channel range");
  BF_CHECK(!outs.empty() && outs[0], "missing output");
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = x.shape;
}

class InstanceThresholdRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t n = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3);
    const float *t0 = ctx.in(1).data<float>(), *a = ctx.in(2).data<float>(), *b = ctx.in(3).data<float>();
    const float* off = ctx.has_in(4) ? ctx.in(4).data<float>() : nullptr;
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    std::vector<double> s1(static_cast<size_t>(c)), s2(static_cast<size_t>(c)), th(static_cast<size_t>(c));
    for (int64_t s = 0; s < n; ++s) {
      std::fill(s1.begin(), s1.end(), 0.0);
      std::fill(s2.begin(), s2.end(), 0.0);
      const float* xs = xp + s * hw * c;
      for (int64_t p = 0; p < hw; ++p)
        for (int64_t ch = 0; ch < c; ++ch) {
          const double v = static_cast<double>(xs[p * c + ch]);
          s1[ch] += v;
          s2[ch] += v * v;
        }
      for (int64_t ch = 0; ch < c; ++ch) {
        const double mu = s1[ch] / static_cast<double>(hw);
        const double var = s2[ch] / static_cast<double>(hw) - mu * mu;
        const double sd = std::sqrt(var > 0.0 ? var : 0.0);
        double t = static_cast<double>(t0[ch]) + static_cast<double>(a[ch]) * mu + static_cast<double>(b[ch]) * sd;
        if (off) t = t + static_cast<double>(off[s * c + ch]);
        th[ch] = t;
      }
      float* ys = y + s * hw * c;
      for (int64_t p = 0; p < hw; ++p)
        for (int64_t ch = 0; ch < c; ++ch) ys[p * c + ch] = static_cast<float>(static_cast<double>(xs[p * c + ch]) - th[ch]);
    }
  }
};

class ChannelShiftRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t n = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3);
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    std::vector<int> grp(static_cast<size_t>(c), -1);
    auto g = ctx.attrs().get_ints("groups", {});
    for (size_t i = 0; i < g.size(); i += 4)
      for (int64_t ch = g[i]; ch < g[i + 1]; ++ch) grp[ch] = static_cast<int>(i / 4);
    for (int64_t s = 0; s < n; ++s)
      for (int64_t r = 0; r < h; ++r)
        for (int64_t q = 0; q < w; ++q)
          for (int64_t ch = 0; ch < c; ++ch) {
            float v;
            if (grp[ch] < 0) {
              v = xp[((s * h + r) * w + q) * c + ch];
            } else {
              const int64_t dy = g[grp[ch] * 4 + 2], dx = g[grp[ch] * 4 + 3], rr = r - dy, qq = q - dx;
              v = (rr >= 0 && rr < h && qq >= 0 && qq < w) ? xp[((s * h + rr) * w + qq) * c + ch] : 0.f;
            }
            y[((s * h + r) * w + q) * c + ch] = v;
          }
  }
};

}  // namespace

void register_glue_float(OpRegistry& reg) {
  reg.add_op({"InstanceThreshold", infer_instance_threshold});
  reg.add_op({"ChannelShift", infer_channel_shift_f});
  reg.add_kernel({"InstanceThreshold", "instance_threshold.reference", "reference", 0, 0, nullptr, make_kernel<InstanceThresholdRef>});
  reg.add_kernel({"ChannelShift", "channel_shift.reference", "reference", 0, 0, nullptr, make_kernel<ChannelShiftRef>});
}

}  // namespace bitino

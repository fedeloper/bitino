// Reference kernels for the BHViT operators (docs/operators_bhvit.md). The numpy
// executor (python/bitino/executor.py) performs the same IEEE operations in the
// same order, so both are bit-identical:
//   * FakeQuant: float32 steps t = x*pre; t = t/scale; t = t+zp; nearbyint (ties to even);
//     clamp; y = r*scale (no FMA: the build uses -ffp-contract=off);
//   * ReduceMean: double accumulation in index order, one division by the count, rounded once.
#include <algorithm>
#include <cfenv>
#include <cmath>

#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

struct AxisSplit {
  int64_t outer = 1, len = 1, inner = 1;
};

AxisSplit split_axis(const Shape& s, int64_t axis) {
  if (axis < 0) axis += static_cast<int64_t>(s.size());
  AxisSplit a;
  for (int64_t d = 0; d < axis; ++d) a.outer *= s[d];
  a.len = s[axis];
  for (size_t d = axis + 1; d < s.size(); ++d) a.inner *= s[d];
  return a;
}

class FakeQuantRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    sp_ = split_axis(ctx.in(0).shape(), ctx.attrs().get_int("axis", -1));
    pre_ = static_cast<float>(ctx.attrs().get_number("pre_scale", 1.0));
    qmin_ = static_cast<float>(ctx.attrs().get_int("qmin", 0));
    qmax_ = static_cast<float>(ctx.attrs().get_int("qmax", 1));
    // nearbyint honours the current rounding mode; the contract is round-half-to-even.
    BF_CHECK(std::fegetround() == FE_TONEAREST, "FakeQuant requires the default rounding mode");
  }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const float* s = ctx.in(1).data<float>();
    const float* z = ctx.in(2).data<float>();
    float* y = ctx.out(0).data<float>();
    const AxisSplit sp = sp_;
    const float pre = pre_, lo = qmin_, hi = qmax_;
    ctx.pool->parallel_for(sp.outer * sp.len, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t c = r % sp.len;
        const float sc = s[c], zc = z[c];
        const float* xr = x + r * sp.inner;
        float* yr = y + r * sp.inner;
        for (int64_t i = 0; i < sp.inner; ++i) {
          float t = xr[i] * pre;
          t = t / sc;
          t = t + zc;
          float q = std::nearbyint(t);
          q = std::min(std::max(q, lo), hi);
          yr[i] = q * sc;
        }
      }
    });
  }

 private:
  AxisSplit sp_;
  float pre_ = 1.f, qmin_ = 0.f, qmax_ = 1.f;
};

class ReduceMeanRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { sp_ = split_axis(ctx.in(0).shape(), ctx.attrs().get_int("axis", -1)); }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const AxisSplit sp = sp_;
    const double cnt = static_cast<double>(sp.len);
    ctx.pool->parallel_for(sp.outer, 1, [&](int64_t o0, int64_t o1) {
      for (int64_t o = o0; o < o1; ++o)
        for (int64_t i = 0; i < sp.inner; ++i) {
          double acc = 0.0;
          for (int64_t k = 0; k < sp.len; ++k) acc += static_cast<double>(x[(o * sp.len + k) * sp.inner + i]);
          y[o * sp.inner + i] = static_cast<float>(acc / cnt);
        }
    });
  }

 private:
  AxisSplit sp_;
};

}  // namespace

void register_bhvit_reference_kernels(OpRegistry& reg) {
  reg.add_kernel({"FakeQuant", "fake_quant.reference", "reference", 0, 0, nullptr, make_kernel<FakeQuantRef>});
  reg.add_kernel({"ReduceMean", "reduce_mean.reference", "reference", 0, 0, nullptr, make_kernel<ReduceMeanRef>});
}

}  // namespace bitino

// INT8 reference kernels ("int8_reference" tier): dynamic-range quantized
// Conv2D (any groups) and Dense. Straightforward loops that follow the
// contract in kernels/int8_quant.h literally; they are the oracle for the
// "int8_neon" tier, which must match them bit for bit.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/int8_quant.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

// Per-tensor quantization of the whole input (scalar).
float quantize_tensor(const float* x, int64_t n, int8_t* q) {
  float m = 0.f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
  float sx = int8_scale(m);
  for (int64_t i = 0; i < n; ++i) q[i] = int8_quantize(x[i], sx);
  return sx;
}

// ------------------------------------------------------------------ Conv2D
class Int8Conv2DRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
    int64_t k = g_.kh * g_.kw * g_.kic;
    wq_.resize(static_cast<size_t>(g_.oc * k));
    sw_.resize(static_cast<size_t>(g_.oc));
    int8_quantize_rows(ctx.in(1).data<float>(), g_.oc, k, wq_.data(), sw_.data());
    xq_.resize(static_cast<size_t>(g_.n * g_.ih * g_.iw * g_.ic));
  }
  size_t prepacked_bytes() const override { return wq_.size() + sw_.size() * 4; }

  void run(const KernelContext& ctx) override {
    const Conv2DGeom g = g_;
    const float sx = quantize_tensor(ctx.in(0).data<float>(), static_cast<int64_t>(xq_.size()), xq_.data());
    const int qpad = int8_quantize(pad_value_, sx);
    const float* bias = opt_f32(ctx, 2);
    float* y = ctx.out(0).data<float>();
    const int64_t ocg = g.oc / g.groups;
    ctx.pool->parallel_for(g.n * g.oh, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        int64_t n = r / g.oh, oy = r % g.oh;
        for (int64_t ox = 0; ox < g.ow; ++ox) {
          float* yp = y + ((n * g.oh + oy) * g.ow + ox) * g.oc;
          for (int64_t o = 0; o < g.oc; ++o) {
            int64_t grp = o / ocg;
            int32_t acc = 0;
            for (int64_t ky = 0; ky < g.kh; ++ky) {
              int64_t iy = oy * g.sh - g.pt + ky * g.dh;
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                int64_t ix = ox * g.sw - g.pl + kx * g.dw;
                bool inside = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
                const int8_t* wp = &wq_[((o * g.kh + ky) * g.kw + kx) * g.kic];
                const int8_t* xp = inside ? &xq_[((n * g.ih + iy) * g.iw + ix) * g.ic + grp * g.kic] : nullptr;
                for (int64_t c = 0; c < g.kic; ++c) acc += (xp ? xp[c] : qpad) * wp[c];
              }
            }
            yp[o] = int8_epilogue(acc, sx, sw_[o], bias ? bias[o] : 0.f, act_);
          }
        }
      }
    });
  }

 private:
  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  float pad_value_ = 0;
  std::vector<int8_t> wq_, xq_;
  std::vector<float> sw_;
};

// ------------------------------------------------------------------- Dense
class Int8DenseRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    const Tensor& w = ctx.in(1);
    k_ = w.dim(1);
    o_ = w.dim(0);
    wq_.resize(static_cast<size_t>(o_ * k_));
    sw_.resize(static_cast<size_t>(o_));
    int8_quantize_rows(w.data<float>(), o_, k_, wq_.data(), sw_.data());
    xq_.resize(static_cast<size_t>(ctx.in(0).size()));
  }
  size_t prepacked_bytes() const override { return wq_.size() + sw_.size() * 4; }

  void run(const KernelContext& ctx) override {
    const int64_t k = k_, o_n = o_, rows = static_cast<int64_t>(xq_.size()) / k;
    const float sx = quantize_tensor(ctx.in(0).data<float>(), static_cast<int64_t>(xq_.size()), xq_.data());
    const float* bias = opt_f32(ctx, 2);
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(rows * o_n, 16, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        int64_t r = i / o_n, o = i % o_n;
        int32_t acc = 0;
        for (int64_t j = 0; j < k; ++j) acc += xq_[r * k + j] * wq_[o * k + j];
        y[i] = int8_epilogue(acc, sx, sw_[o], bias ? bias[o] : 0.f, act_);
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
  int64_t k_ = 0, o_ = 0;
  std::vector<int8_t> wq_, xq_;
  std::vector<float> sw_;
};

bool conv_supported(const KernelContext& ctx) {
  if (!int8_allowed(ctx)) return false;
  const Shape& w = ctx.in(1).shape();
  return w[1] * w[2] * w[3] <= kInt8MaxK;
}

bool dense_supported(const KernelContext& ctx) { return int8_allowed(ctx) && ctx.in(1).dim(1) <= kInt8MaxK; }

}  // namespace

// INT8 kernels are opt-in (lossy) and, when allowed, preferred over float
// kernels: priorities are above every float tier.
void register_int8_kernels(OpRegistry& reg) {
  // Priority 30: above the float reference (0) but below float_opt (40/45). The
  // scalar int8 oracle must never displace optimized float kernels in mixed tiers.
  reg.add_kernel({"Conv2D", "conv2d.int8_reference", "int8_reference", 30, 0, conv_supported, make_kernel<Int8Conv2DRef>});
  reg.add_kernel({"Dense", "dense.int8_reference", "int8_reference", 30, 0, dense_supported, make_kernel<Int8DenseRef>});
}

}  // namespace bitino

// Reference kernels for QBConv2D, QBDense and ChannelGate (contracts in
// docs/operators_larq.md). Literal loops; integer accumulation for the k-bit x
// binary ops so results are exact and order independent.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/bitpack.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

// DoReFa activation quantizer, integer level: round_half_even(clip(x, 0, 1) * n)
// evaluated in float32 (the multiply is a single rounded f32 op; the build uses
// -ffp-contract=off). nearbyint uses the default round-to-nearest-even mode, like
// tf.round / np.round.
inline int dorefa_level(float x, float n) {
  float c = std::min(std::max(x, 0.f), 1.f);
  return static_cast<int>(std::nearbyint(c * n));
}

float levels_of(const KernelContext& ctx) {
  int64_t k = ctx.attrs().get_int("act_bits", 0);
  return static_cast<float>((1 << k) - 1);
}

// ---------------------------------------------------------------- QBConv2D
class QBConv2DRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    n_ = levels_of(ctx);
    const Tensor& w = ctx.in(1);
    int64_t rows = g_.oc * g_.kh * g_.kw, words = bit_words(g_.kic);
    wq_.resize(static_cast<size_t>(rows * g_.kic));
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t c = 0; c < g_.kic; ++c)
        wq_[r * g_.kic + c] = static_cast<int8_t>(bit_value(w.data<uint32_t>() + r * words, c));
    xq_.resize(static_cast<size_t>(g_.n * g_.ih * g_.iw * g_.ic));
  }
  size_t prepacked_bytes() const override { return wq_.size(); }

  void run(const KernelContext& ctx) override {
    const Conv2DGeom g = g_;
    const float* x = ctx.in(0).data<float>();
    int64_t total = g.n * g.ih * g.iw * g.ic;
    for (int64_t i = 0; i < total; ++i) xq_[i] = static_cast<uint8_t>(dorefa_level(x[i], n_));
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
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
              if (iy < 0 || iy >= g.ih) continue;  // zero padding: level 0 contributes nothing
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                int64_t ix = ox * g.sw - g.pl + kx * g.dw;
                if (ix < 0 || ix >= g.iw) continue;
                const int8_t* wp = &wq_[((o * g.kh + ky) * g.kw + kx) * g.kic];
                const uint8_t* xp = &xq_[((n * g.ih + iy) * g.iw + ix) * g.ic + grp * g.kic];
                for (int64_t c = 0; c < g.kic; ++c) acc += static_cast<int32_t>(xp[c]) * wp[c];
              }
            }
            float v = static_cast<float>(acc);
            if (mul) v *= mul[o];
            if (add) v += add[o];
            yp[o] = apply_activation(act_, v);
          }
        }
      }
    });
  }

 private:
  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  float n_ = 1;
  std::vector<int8_t> wq_;
  std::vector<uint8_t> xq_;
};

// ----------------------------------------------------------------- QBDense
class QBDenseRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    n_ = levels_of(ctx);
    const Tensor& w = ctx.in(1);
    k_ = w.dim(1);
    o_ = w.dim(0);
    int64_t words = bit_words(k_);
    wq_.resize(static_cast<size_t>(o_ * k_));
    for (int64_t o = 0; o < o_; ++o)
      for (int64_t j = 0; j < k_; ++j) wq_[o * k_ + j] = static_cast<int8_t>(bit_value(w.data<uint32_t>() + o * words, j));
    xq_.resize(static_cast<size_t>(ctx.in(0).size()));
  }
  size_t prepacked_bytes() const override { return wq_.size(); }

  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    int64_t rows = x.size() / k_;
    for (int64_t i = 0; i < x.size(); ++i) xq_[i] = static_cast<uint8_t>(dorefa_level(x.data<float>()[i], n_));
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    float* y = ctx.out(0).data<float>();
    const int64_t k = k_, on = o_;
    ctx.pool->parallel_for(rows * on, 16, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        int64_t r = i / on, o = i % on;
        const uint8_t* xp = &xq_[r * k];
        const int8_t* wp = &wq_[o * k];
        int32_t acc = 0;
        for (int64_t j = 0; j < k; ++j) acc += static_cast<int32_t>(xp[j]) * wp[j];
        float v = static_cast<float>(acc);
        if (mul) v *= mul[o];
        if (add) v += add[o];
        y[i] = apply_activation(act_, v);
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
  float n_ = 1;
  int64_t k_ = 0, o_ = 0;
  std::vector<int8_t> wq_;
  std::vector<uint8_t> xq_;
};

// ------------------------------------------------------------- ChannelGate
class ChannelGateRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    sigmoid_ = ctx.attrs().get_string("gate", "sigmoid") == "sigmoid";
    scale_.resize(static_cast<size_t>(ctx.in(1).size()));
  }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const float* g = ctx.in(1).data<float>();
    int64_t n = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3);
    // Gate values: sigmoid evaluated in double, rounded once to f32.
    for (int64_t i = 0; i < n * c; ++i)
      scale_[i] = sigmoid_ ? static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(g[i])))) : g[i];
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    for (int64_t b = 0; b < n; ++b)
      for (int64_t p = 0; p < hw; ++p)
        for (int64_t ch = 0; ch < c; ++ch) {
          int64_t i = (b * hw + p) * c + ch;
          y[i] = xp[i] * scale_[b * c + ch];
        }
  }

 private:
  bool sigmoid_ = true;
  std::vector<float> scale_;
};

}  // namespace

void register_larq_kernels(OpRegistry& reg) {
  auto add = [&](const char* op, const char* name, std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, name, "reference", 0, 0, nullptr, create});
  };
  add("QBConv2D", "qbconv2d.reference", make_kernel<QBConv2DRef>);
  add("QBDense", "qbdense.reference", make_kernel<QBDenseRef>);
  add("ChannelGate", "channel_gate.reference", make_kernel<ChannelGateRef>);
}

}  // namespace bitino

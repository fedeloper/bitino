// Reference ("oracle") kernels. Straightforward loops that follow the
// operator contracts literally; every optimized kernel is tested against them.
// Binary ops accumulate in exact integer arithmetic on unpacked ±1 values.
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

#include "kernels/bit_output.h"
#include "kernels/bitpack.h"
#include "kernels/float_opt/eltwise_program.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

// ---------------------------------------------------------------- Conv2D f32
class Conv2DRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
  }
  void run(const KernelContext& ctx) override {
    const Conv2DGeom g = g_;
    const float* x = ctx.in(0).data<float>();
    const float* w = ctx.in(1).data<float>();
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
            double acc = bias ? bias[o] : 0.0;
            for (int64_t ky = 0; ky < g.kh; ++ky) {
              int64_t iy = oy * g.sh - g.pt + ky * g.dh;
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                int64_t ix = ox * g.sw - g.pl + kx * g.dw;
                bool inside = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
                const float* wp = w + ((o * g.kh + ky) * g.kw + kx) * g.kic;
                for (int64_t c = 0; c < g.kic; ++c) {
                  float v = inside ? x[((n * g.ih + iy) * g.iw + ix) * g.ic + grp * g.kic + c] : pad_value_;
                  acc += static_cast<double>(v) * wp[c];
                }
              }
            }
            yp[o] = apply_activation(act_, static_cast<float>(acc));
          }
        }
      }
    });
  }

 private:
  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  float pad_value_ = 0;
};

// ----------------------------------------------------------- BConv2D (oracle)
// Unpacks weights to int8 ±1 and binarizes the input to int8 {-1,0,+1}, then
// performs a direct convolution with int32 accumulation.
class BConv2DRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    mode_ = parse_binarize(ctx.attrs());
    pad_value_ = static_cast<int>(ctx.attrs().get_number("pad_value", 0.0));
    bout_.parse(ctx);
    const Tensor& w = ctx.in(1);
    int64_t rows = g_.oc * g_.kh * g_.kw, words = bit_words(g_.kic);
    wq_.resize(static_cast<size_t>(rows * g_.kic));
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t c = 0; c < g_.kic; ++c)
        wq_[r * g_.kic + c] = static_cast<int8_t>(bit_value(w.data<uint32_t>() + r * words, c));
    xq_.resize(static_cast<size_t>(g_.n * g_.ih * g_.iw * g_.ic));
    if (bout_.active) yf_.assign(static_cast<size_t>(g_.n * g_.oh * g_.ow * g_.oc), 0.f);
    epi_.prepare_epilogue(ctx, g_.oh * g_.ow);
  }
  size_t prepacked_bytes() const override { return wq_.size(); }

  // Bit output: the float output (and its epilogue) is computed in yf_ first,
  // then binarized row by row (kernels/bit_output.h).
  void run(const KernelContext& ctx) override {
    float* y = bout_.active ? yf_.data() : ctx.out(0).data<float>();
    conv(ctx, y);
    const int64_t pixels = g_.n * g_.oh * g_.ow;
    if (epi_.active()) {  // fused elementwise epilogue (graph_rewrite), after the whole convolution
      epi_.begin(ctx);
      ctx.pool->parallel_for(pixels, 256, [&](int64_t r0, int64_t r1) { epi_.apply(y, r0, r1); });
    }
    if (bout_.active) pack_output_rows(y, 0, pixels, g_.oc, ctx.out(0).data<uint32_t>(), bout_.mode, bout_.threshold(ctx));
  }

 private:
  void conv(const KernelContext& ctx, float* y) {
    const Conv2DGeom g = g_;
    const Tensor& xt = ctx.in(0);
    const float* thr = opt_f32(ctx, 4);
    // Binarize input.
    int64_t pix = g.n * g.ih * g.iw;
    if (xt.dtype() == DType::kF32) {
      const float* x = xt.data<float>();
      for (int64_t p = 0; p < pix; ++p)
        for (int64_t c = 0; c < g.ic; ++c)
          xq_[p * g.ic + c] = static_cast<int8_t>(binarize(mode_, x[p * g.ic + c], thr ? thr[c] : 0.f));
    } else {
      const uint32_t* xb = xt.data<uint32_t>();
      int64_t words = bit_words(g.ic);
      for (int64_t p = 0; p < pix; ++p)
        for (int64_t c = 0; c < g.ic; ++c) xq_[p * g.ic + c] = static_cast<int8_t>(bit_value(xb + p * words, c));
    }
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    const int64_t ocg = g.oc / g.groups;
    ctx.pool->parallel_for(g.n * g.oh, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        int64_t n = r / g.oh, oy = r % g.oh;
        for (int64_t ox = 0; ox < g.ow; ++ox) {
          int64_t opix = (n * g.oh + oy) * g.ow + ox;
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
                for (int64_t c = 0; c < g.kic; ++c) acc += (xp ? xp[c] : pad_value_) * wp[c];
              }
            }
            float v = static_cast<float>(acc);
            if (mul) v *= mul[o];
            if (add) v += add[o];
            y[opix * g.oc + o] = apply_activation(act_, v);
          }
        }
      }
    });
  }

  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  int pad_value_ = 0;
  BitOutput bout_;
  std::vector<int8_t> wq_, xq_;
  std::vector<float> yf_;  // float output before binarization (bit output only)
  EltwiseProgram epi_;
};

// --------------------------------------------------------------- Dense / BDense
class DenseRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { act_ = parse_activation(ctx.attrs()); }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& w = ctx.in(1);
    int64_t k = w.dim(1), o_n = w.dim(0), rows = x.size() / k;
    const float* bias = opt_f32(ctx, 2);
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(rows * o_n, 16, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        int64_t r = i / o_n, o = i % o_n;
        double acc = bias ? bias[o] : 0.0;
        for (int64_t j = 0; j < k; ++j) acc += static_cast<double>(x.data<float>()[r * k + j]) * w.data<float>()[o * k + j];
        y[i] = apply_activation(act_, static_cast<float>(acc));
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
};

// Weights are unpacked once to int8 {-1, 0, +1} (0 where the optional w_zero
// mask, input 5, marks a zero weight); each input row is binarized once per run.
class BDenseRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    mode_ = parse_binarize(ctx.attrs());
    const Tensor& w = ctx.in(1);
    k_ = w.dim(1);
    o_ = w.dim(0);
    rows_ = ctx.in(0).size() / k_;
    int64_t words = bit_words(k_);
    wq_.resize(static_cast<size_t>(o_ * k_));
    const uint32_t* wz = ctx.has_in(5) ? ctx.in(5).data<uint32_t>() : nullptr;
    for (int64_t o = 0; o < o_; ++o)
      for (int64_t j = 0; j < k_; ++j)
        wq_[o * k_ + j] = (wz && bit_value(wz + o * words, j) < 0) ? 0 : static_cast<int8_t>(bit_value(w.data<uint32_t>() + o * words, j));
    xq_.resize(static_cast<size_t>(rows_ * k_));
    bout_.parse(ctx);
    if (bout_.active) yf_.assign(static_cast<size_t>(rows_ * o_), 0.f);
    const Tensor& y = ctx.out(0);
    epi_.prepare_epilogue(ctx, y.dim(0) > 0 ? rows_ / y.dim(0) : rows_);
  }
  size_t prepacked_bytes() const override { return wq_.size(); }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t k = k_, o_n = o_, words = bit_words(k);
    const float* thr = opt_f32(ctx, 4);
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    for (int64_t r = 0; r < rows_; ++r)
      for (int64_t j = 0; j < k; ++j)
        xq_[r * k + j] = static_cast<int8_t>(x.dtype() == DType::kF32 ? binarize(mode_, x.data<float>()[r * k + j], thr ? thr[j] : 0.f)
                                                                     : bit_value(x.data<uint32_t>() + r * words, j));
    float* y = bout_.active ? yf_.data() : ctx.out(0).data<float>();  // bit output: float rows first
    ctx.pool->parallel_for(rows_ * o_n, 16, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        int64_t r = i / o_n, o = i % o_n;
        const int8_t* xr = &xq_[r * k];
        const int8_t* wr = &wq_[o * k];
        int32_t acc = 0;
        for (int64_t j = 0; j < k; ++j) acc += xr[j] * wr[j];
        float v = static_cast<float>(acc);
        if (mul) v *= mul[o];
        if (add) v += add[o];
        y[i] = apply_activation(act_, v);
      }
    });
    if (epi_.active()) {  // fused elementwise epilogue (graph_rewrite), after the whole product
      epi_.begin(ctx);
      ctx.pool->parallel_for(rows_, 64, [&](int64_t r0, int64_t r1) { epi_.apply(y, r0, r1); });
    }
    if (bout_.active) pack_output_rows(y, 0, rows_, o_n, ctx.out(0).data<uint32_t>(), bout_.mode, bout_.threshold(ctx));
  }

 private:
  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  int64_t k_ = 0, o_ = 0, rows_ = 0;
  BitOutput bout_;
  std::vector<int8_t> wq_, xq_;
  std::vector<float> yf_;  // float output before binarization (bit output only)
  EltwiseProgram epi_;
};

// ------------------------------------------------------------ elementwise ops
class ChannelAffineRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    int64_t c = x.dim(-1), n = x.size();
    const float* s = opt_f32(ctx, 1);
    const float* b = opt_f32(ctx, 2);
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    for (int64_t i = 0; i < n; ++i) y[i] = xp[i] * (s ? s[i % c] : 1.f) + (b ? b[i % c] : 0.f);
  }
};

class ActivationRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    type_ = ctx.attrs().get_string("type", "");
    lo_ = static_cast<float>(ctx.attrs().get_number("min", -1.0));
    hi_ = static_cast<float>(ctx.attrs().get_number("max", 1.0));
    alpha_ = static_cast<float>(ctx.attrs().get_number("alpha", 0.01));
  }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    int64_t c = x.dim(-1), n = x.size();
    const float* xp = x.data<float>();
    const float* slope = opt_f32(ctx, 1);
    float* y = ctx.out(0).data<float>();
    for (int64_t i = 0; i < n; ++i) {
      float v = xp[i];
      if (type_ == "relu") v = v > 0.f ? v : 0.f;
      else if (type_ == "relu6") v = std::min(std::max(v, 0.f), 6.f);
      else if (type_ == "hardtanh") v = std::min(std::max(v, lo_), hi_);
      else if (type_ == "prelu") v = v >= 0.f ? v : v * slope[i % c];
      else if (type_ == "leaky_relu") v = v >= 0.f ? v : v * alpha_;
      else if (type_ == "sigmoid") v = 1.f / (1.f + std::exp(-v));
      else if (type_ == "tanh") v = static_cast<float>(std::tanh(static_cast<double>(v)));
      else if (type_ == "hardswish") v = v * std::min(std::max(v + 3.f, 0.f), 6.f) / 6.f;
      else if (type_ == "gelu") {  // double, torch/BERT formula x*0.5*(1+erf(x/sqrt(2)))
        double d = v;
        v = static_cast<float>(d * 0.5 * (1.0 + std::erf(d / std::sqrt(2.0))));
      }
      else if (type_ == "sign") v = v >= 0.f ? 1.f : -1.f;
      else if (type_ == "sign_zero") v = v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f);  // torch.sign
      y[i] = v;
    }
  }

 private:
  std::string type_;
  float lo_ = -1, hi_ = 1, alpha_ = 0.01f;
};

class AddRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { act_ = parse_activation(ctx.attrs()); }
  void run(const KernelContext& ctx) override {
    int64_t n = ctx.in(0).size();
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    for (int64_t i = 0; i < n; ++i) y[i] = apply_activation(act_, a[i] + b[i]);
  }

 private:
  Activation act_ = Activation::kNone;
};

class ConcatRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Shape& os = ctx.out(0).shape();
    int64_t axis = ctx.attrs().get_int("axis", -1);
    if (axis < 0) axis += static_cast<int64_t>(os.size());
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < axis; ++d) outer *= os[d];
    for (size_t d = axis + 1; d < os.size(); ++d) inner *= os[d];
    float* y = ctx.out(0).data<float>();
    int64_t out_stride = os[axis] * inner, off = 0;
    for (size_t i = 0; i < ctx.inputs.size(); ++i) {
      const Tensor& t = ctx.in(i);
      int64_t len = t.dim(static_cast<int>(axis)) * inner;
      for (int64_t o = 0; o < outer; ++o) std::copy_n(t.data<float>() + o * len, len, y + o * out_stride + off);
      off += len;
    }
  }
};

class MaxPoolRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    auto g = pool_geometry(ctx.attrs(), ctx.in(0).shape());
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    for (int64_t n = 0; n < g.n; ++n)
      for (int64_t oy = 0; oy < g.oh; ++oy)
        for (int64_t ox = 0; ox < g.ow; ++ox)
          for (int64_t c = 0; c < g.c; ++c) {
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t ky = 0; ky < g.kh; ++ky)
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                int64_t iy = oy * g.sh - g.pt + ky, ix = ox * g.sw - g.pl + kx;
                if (iy < 0 || iy >= g.ih || ix < 0 || ix >= g.iw) continue;
                m = std::max(m, x[((n * g.ih + iy) * g.iw + ix) * g.c + c]);
              }
            y[((n * g.oh + oy) * g.ow + ox) * g.c + c] = m;
          }
  }
};

class AvgPoolRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    auto g = pool_geometry(ctx.attrs(), ctx.in(0).shape());
    bool include_pad = ctx.attrs().get_int("count_include_pad", 0) != 0;
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    for (int64_t n = 0; n < g.n; ++n)
      for (int64_t oy = 0; oy < g.oh; ++oy)
        for (int64_t ox = 0; ox < g.ow; ++ox)
          for (int64_t c = 0; c < g.c; ++c) {
            double s = 0;
            int64_t cnt = 0;
            for (int64_t ky = 0; ky < g.kh; ++ky)
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                int64_t iy = oy * g.sh - g.pt + ky, ix = ox * g.sw - g.pl + kx;
                if (iy < 0 || iy >= g.ih || ix < 0 || ix >= g.iw) {
                  cnt += include_pad;
                  continue;
                }
                s += x[((n * g.ih + iy) * g.iw + ix) * g.c + c];
                ++cnt;
              }
            y[((n * g.oh + oy) * g.ow + ox) * g.c + c] = static_cast<float>(s / std::max<int64_t>(cnt, 1));
          }
  }
};

class GlobalAvgPoolRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    int64_t n = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3);
    float* y = ctx.out(0).data<float>();
    for (int64_t b = 0; b < n; ++b)
      for (int64_t ch = 0; ch < c; ++ch) {
        double s = 0;
        for (int64_t p = 0; p < hw; ++p) s += x.data<float>()[(b * hw + p) * c + ch];
        y[b * c + ch] = static_cast<float>(s / hw);
      }
  }
};

class CopyRef : public Kernel {  // Reshape
 public:
  void run(const KernelContext& ctx) override {
    // The session aliases a Reshape's output to its input buffer (zero-copy)
    // unless the output is a graph output; only then is there anything to copy.
    if (ctx.in(0).raw() == ctx.out(0).raw()) return;
    std::memcpy(ctx.out(0).raw(), ctx.in(0).raw(), ctx.in(0).bytes());  // f32, i8 or bit (same last dimension: same storage)
  }
};

class SoftmaxRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    int64_t c = x.dim(-1), rows = x.size() / c;
    for (int64_t r = 0; r < rows; ++r) {
      const float* xp = x.data<float>() + r * c;
      float* y = ctx.out(0).data<float>() + r * c;
      float m = *std::max_element(xp, xp + c);
      double s = 0;
      for (int64_t i = 0; i < c; ++i) s += std::exp(static_cast<double>(xp[i] - m));
      for (int64_t i = 0; i < c; ++i) y[i] = static_cast<float>(std::exp(static_cast<double>(xp[i] - m)) / s);
    }
  }
};

class PadRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    Tensor& y = ctx.out(0);
    auto p = ctx.attrs().get_ints("pads", {});
    float value = static_cast<float>(ctx.attrs().get_number("value", 0.0));
    std::fill_n(y.data<float>(), y.size(), value);
    int r = x.rank();
    std::vector<int64_t> idx(r, 0);
    for (int64_t i = 0; i < x.size(); ++i) {
      int64_t rem = i, off = 0;
      for (int d = r - 1; d >= 0; --d) {
        idx[d] = rem % x.dim(d);
        rem /= x.dim(d);
      }
      for (int d = 0; d < r; ++d) off = off * y.dim(d) + idx[d] + p[2 * d];
      y.data<float>()[off] = x.data<float>()[i];
    }
  }
};

}  // namespace

void register_reference_kernels(OpRegistry& reg) {
  auto add = [&](const char* op, const char* name, std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, name, "reference", 0, 0, nullptr, create});
  };
  add("Conv2D", "conv2d.reference", make_kernel<Conv2DRef>);
  add("BConv2D", "bconv2d.reference", make_kernel<BConv2DRef>);
  add("Dense", "dense.reference", make_kernel<DenseRef>);
  add("BDense", "bdense.reference", make_kernel<BDenseRef>);
  add("ChannelAffine", "channel_affine.reference", make_kernel<ChannelAffineRef>);
  add("Activation", "activation.reference", make_kernel<ActivationRef>);
  add("Add", "add.reference", make_kernel<AddRef>);
  add("Concat", "concat.reference", make_kernel<ConcatRef>);
  add("MaxPool2D", "maxpool2d.reference", make_kernel<MaxPoolRef>);
  add("AvgPool2D", "avgpool2d.reference", make_kernel<AvgPoolRef>);
  add("GlobalAvgPool", "global_avg_pool.reference", make_kernel<GlobalAvgPoolRef>);
  add("Reshape", "reshape.reference", make_kernel<CopyRef>);
  add("Softmax", "softmax.reference", make_kernel<SoftmaxRef>);
  add("Pad", "pad.reference", make_kernel<PadRef>);
}

}  // namespace bitino

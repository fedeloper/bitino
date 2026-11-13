// Reference kernel of the internal FusedElementwise operator
// (ops/fused_elementwise.h): evaluates the program element by element with
// the scalar expressions of the reference kernels it replaces
// (ChannelAffineRef, AddRef, MulRef, ActivationRef, ChannelGateRef), so it is
// bit-identical to running those nodes one after the other.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/bitpack.h"
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
    // reduce (pool_fused_outputs): the elementwise result of shape out_shape is evaluated into a buffer, then the
    // channel means are taken as the float_opt GlobalAvgPool kernel takes them (float sums in pixel order)
    const bool reduce = ctx.attrs().contains("reduce");
    const Shape es = reduce ? Shape(ctx.attrs().get_ints("out_shape", {})) : yt.shape();
    const int64_t c = es.back(), n = num_elements(es);
    const int64_t per_batch = es[0] > 0 ? n / es[0] : n;
    for (size_t k = 0; k < prog_.size(); ++k)
      if (prog_[k].op == FeOp::kGate) {
        const float* g = ctx.in(prog_[k].a).data<float>();
        for (size_t i = 0; i < gate_[k].size(); ++i)
          gate_[k][i] = prog_[k].b ? static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(g[i])))) : g[i];
      }
    auto in = [&](int32_t i) { return ctx.in(i).data<float>(); };
    std::vector<float> full(reduce ? static_cast<size_t>(n) : 0);
    float* y = reduce ? full.data() : yt.data<float>();
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
    if (reduce) {
      float* m = yt.data<float>();
      const float count = static_cast<float>(per_batch / c);
      for (int64_t b = 0; b < es[0]; ++b)
        for (int64_t ch = 0; ch < c; ++ch) {
          float s = 0.f;
          for (int64_t p = 0; p < per_batch / c; ++p) s += y[b * per_batch + p * c + ch];
          m[b * c + ch] = s / count;
        }
    }
  }

 private:
  std::vector<FeInstr> prog_;
  std::vector<std::vector<float>> gate_;
};

// PackBits: pack_row_f32 per row (the binary reference kernels' input packing).
class PackBitsRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t c = x.dim(-1), rows = c > 0 ? x.size() / c : 0, words = bit_words(c);
    const float* thr = ctx.has_in(1) ? ctx.in(1).data<float>() : nullptr;
    const BinarizeMode mode = parse_binarize(ctx.attrs());
    uint32_t* y = ctx.out(0).data<uint32_t>();
    for (int64_t r = 0; r < rows; ++r) pack_row_f32(x.data<float>() + r * c, c, y + r * words, mode, thr);
  }
};

// AvgPool2DPrelu: ActivationRef's prelu, then AvgPoolRef's average (double sums), per output; output 1 = the average
// of x itself.
class AvgPoolPreluRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Pool2DGeom g = pool_geometry(ctx.attrs(), ctx.in(0).shape());
    const bool include_pad = ctx.attrs().get_int("count_include_pad", 0) != 0;
    const float* x = ctx.in(0).data<float>();
    const float* sl = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    float* y2 = ctx.outputs.size() > 1 && ctx.outputs[1] ? ctx.out(1).data<float>() : nullptr;
    for (int64_t n = 0; n < g.n; ++n)
      for (int64_t oy = 0; oy < g.oh; ++oy)
        for (int64_t ox = 0; ox < g.ow; ++ox)
          for (int64_t c = 0; c < g.c; ++c) {
            double s = 0, t = 0;
            int64_t cnt = 0;
            for (int64_t ky = 0; ky < g.kh; ++ky)
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                const int64_t iy = oy * g.sh - g.pt + ky, ix = ox * g.sw - g.pl + kx;
                if (iy < 0 || iy >= g.ih || ix < 0 || ix >= g.iw) {
                  cnt += include_pad;
                  continue;
                }
                const float v = x[((n * g.ih + iy) * g.iw + ix) * g.c + c];
                s += v >= 0.f ? v : v * sl[c];
                t += v;
                ++cnt;
              }
            const int64_t o = ((n * g.oh + oy) * g.ow + ox) * g.c + c;
            y[o] = static_cast<float>(s / std::max<int64_t>(cnt, 1));
            if (y2) y2[o] = static_cast<float>(t / std::max<int64_t>(cnt, 1));
          }
  }
};

}  // namespace

void register_fused_reference_kernels(OpRegistry& reg) {
  reg.add_kernel({"AvgPool2DPrelu", "avgpool2d_prelu.reference", "reference", 0, 0, nullptr, make_kernel<AvgPoolPreluRef>});
  reg.add_kernel({"FusedElementwise", "fused_elementwise.reference", "reference", 0, 0, nullptr, make_kernel<FusedElementwiseRef>});
  reg.add_kernel({"PackBits", "pack_bits.reference", "reference", 0, 0, nullptr, make_kernel<PackBitsRef>});
}

}  // namespace bitino

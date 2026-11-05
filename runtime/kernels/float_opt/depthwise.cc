// float_opt depthwise Conv2D: groups == ic == oc (channel multiplier 1,
// filter [C][kh][kw][1]). Direct convolution over channel vectors; the
// arithmetic is the shared leaf in float_opt_leaf.h (4-lane NEON/portable
// here, 8-lane AVX2 in float_opt_avx2.cc).
//
// The filter is repacked in prepare() to [kh*kw][Cp] (Cp = C rounded up to the
// backend's vector width, zero padded) so each tap's weights for a channel
// block are one contiguous vector load; bias is padded the same way. Work is
// split over output rows (n, oy).
#include "kernels/float_opt/float_opt.h"
#include "kernels/float_opt/float_opt_leaf.h"
#include "kernels/float_opt/float_simd.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__x86_64__)
#include "kernels/float_opt/float_opt_avx2.h"
#endif

namespace bitino {
namespace {

using namespace fopt;

// Layers below kInlineMacs (depthwise runs at a few GMAC/s, so ~30-40 µs)
// run on the calling thread (see parallel_grain()).
constexpr int64_t kInlineMacs = int64_t(1) << 17;

struct Simd4Dw {
  static constexpr int kLanes = Simd4::W;
  static void rows(const DwArgs& d, int64_t r0, int64_t r1) { leaf_depthwise_rows<Simd4>(d, r0, r1); }
};

#if defined(__x86_64__)
struct Avx2Dw {
  static constexpr int kLanes = fopt_avx2::kLanes;
  static void rows(const DwArgs& d, int64_t r0, int64_t r1) { fopt_avx2::depthwise_rows(d, r0, r1); }
};
#endif

template <class B>
class DepthwiseConv : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    act_ = parse_activation(ctx.attrs());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
    c_ = g_.ic;
    cp_ = round_up(c_, B::kLanes);
    const int64_t taps = g_.kh * g_.kw;
    wbuf_ = aligned_alloc_shared(static_cast<size_t>((taps + 1) * cp_) * sizeof(float));
    float* w = static_cast<float*>(wbuf_.get());
    const float* src = ctx.in(1).data<float>();  // [C][taps]
    for (int64_t ch = 0; ch < c_; ++ch)
      for (int64_t t = 0; t < taps; ++t) w[t * cp_ + ch] = src[ch * taps + t];
    const float* bias = opt_f32(ctx, 2);
    for (int64_t ch = 0; ch < c_; ++ch) w[taps * cp_ + ch] = bias ? bias[ch] : 0.f;
    // Horizontal interior [ox_lo_, ox_hi_): every kx tap in bounds. The set
    // is contiguous; when it is empty every pixel takes the border path.
    ox_lo_ = ox_hi_ = 0;
    bool found = false;
    for (int64_t ox = 0; ox < g_.ow; ++ox) {
      const int64_t ix0 = ox * g_.sw - g_.pl;
      if (ix0 >= 0 && ix0 + g_.dw * (g_.kw - 1) < g_.iw) {
        if (!found) ox_lo_ = ox;
        found = true;
        ox_hi_ = ox + 1;
      }
    }
  }
  size_t prepacked_bytes() const override { return static_cast<size_t>((g_.kh * g_.kw + 1) * cp_) * sizeof(float); }

  void run(const KernelContext& ctx) override {
    const Conv2DGeom& g = g_;
    DwArgs d;
    d.x = ctx.in(0).data<float>();
    d.y = ctx.out(0).data<float>();
    d.w = static_cast<const float*>(wbuf_.get());
    d.ih = g.ih;
    d.iw = g.iw;
    d.c = c_;
    d.cp = cp_;
    d.oh = g.oh;
    d.ow = g.ow;
    d.kh = g.kh;
    d.kw = g.kw;
    d.sh = g.sh;
    d.sw = g.sw;
    d.dh = g.dh;
    d.dw = g.dw;
    d.pt = g.pt;
    d.pl = g.pl;
    d.ox_lo = ox_lo_;
    d.ox_hi = ox_hi_;
    d.pad_value = pad_value_;
    d.act = static_cast<int32_t>(act_);
    const int64_t rows = g.n * g.oh;
    ctx.pool->parallel_for(rows, parallel_grain(rows, rows * g.ow * c_ * g.kh * g.kw, kInlineMacs),
                           [&](int64_t r0, int64_t r1) { B::rows(d, r0, r1); });
  }

 private:
  Conv2DGeom g_;
  Activation act_ = Activation::kNone;
  float pad_value_ = 0;
  int64_t c_ = 0, cp_ = 0;
  int64_t ox_lo_ = 0, ox_hi_ = 0;
  std::shared_ptr<void> wbuf_;  // [kh*kw][Cp] weights, then [Cp] bias
};

bool depthwise_supported(const KernelContext& ctx) {
  if (ctx.in(0).dtype() != DType::kF32 || ctx.in(1).dtype() != DType::kF32) return false;
  const Shape& x = ctx.in(0).shape();
  const Shape& w = ctx.in(1).shape();
  const int64_t groups = ctx.attrs().get_int("groups", 1);
  // groups == 1 (ic == 1) is handled by the GEMM kernel.
  return groups > 1 && x[3] == groups && w[0] == groups && w[3] == 1;
}

}  // namespace

void register_float_opt_depthwise_kernels(OpRegistry& reg) {
  reg.add_kernel({"Conv2D", "conv2d_depthwise.float_opt", "float_opt", 45, kFloatOptFeatures, depthwise_supported,
                  make_kernel<DepthwiseConv<Simd4Dw>>});
#if defined(__x86_64__)
  reg.add_kernel({"Conv2D", "conv2d_depthwise.float_opt_avx2", "float_opt", kFloatOptAvx2Priority + 5, kFloatOptAvx2Features,
                  depthwise_supported, make_kernel<DepthwiseConv<Avx2Dw>>});
#endif
}

}  // namespace bitino

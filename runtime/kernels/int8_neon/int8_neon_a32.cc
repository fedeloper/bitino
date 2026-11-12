// "int8_neon" tier on 32-bit ARM (armeabi-v7a / AArch32, BF_NEON_A32): dynamic-range INT8 Conv2D (groups == 1)
// and Dense, bit-exact with the "int8_reference" tier (kernels/int8_quant.h), as conv2d.int8_neon_a32 /
// dense.int8_neon_a32 (priority 140, like the AArch64 SDOT kernels). AArch32 has no dot products: the GEMM is the
// VMULL.S8 / VMLAL.S8 / VPADAL.S16 tile of int8_gemm_a32.h (quantized weights are in [-127, 127], so its int16 pair
// sums are exact).
//
// Per run():
//   1. scale_x: max |x| on the fp32 bit patterns (unsigned max of x & 0x7fffffff: the IEEE max of |x| for finite
//      inputs; a vector float max would flush denormals to zero on AArch32 NEON).
//   2. qx = clamp(rne(x / scale_x), -127, 127) (quantize_fast below: exact, see there).
//   3. per 8 output rows (pixels / Dense rows): im2col rows of raw qx codes (padding taps qpad, K zero-padded to a
//      multiple of 16), packed as quads, int32 sums by the 4x2 tiles for every pair of output channels.
//   4. y = act(float(acc) * (scale_x * scale_w[o]) + bias[o]): NEON when every scale_x * scale_w[o] is 0 or at
//      least 2^-100 in magnitude and every bias is 0 or normal (then no operand or result of the multiply and the
//      add is a denormal, so the flush-to-zero of AArch32 NEON changes nothing; checked per run), else scalar.
// Only built on 32-bit ARM with NEON; registered from int8_neon_kernels.cc.
#include "kernels/neon_a32.h"

#if BF_NEON_A32
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "kernels/int8_neon/int8_gemm_a32.h"
#include "kernels/int8_neon/int8_gemm_epilogue.h"
#include "kernels/int8_quant.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace int8_neon_a32 {
namespace {

// max |x| for finite x (bit patterns of non-negative floats order like unsigned integers)
float maxabs(const float* x, int64_t n) {
  const uint32x4_t mask = vdupq_n_u32(0x7fffffffu);
  uint32x4_t m0 = vdupq_n_u32(0), m1 = m0, m2 = m0, m3 = m0;
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    m0 = vmaxq_u32(m0, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i)), mask));
    m1 = vmaxq_u32(m1, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i + 4)), mask));
    m2 = vmaxq_u32(m2, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i + 8)), mask));
    m3 = vmaxq_u32(m3, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i + 12)), mask));
  }
  uint32_t m = vmaxvq_u32(vmaxq_u32(vmaxq_u32(m0, m1), vmaxq_u32(m2, m3)));
  for (; i < n; ++i) {
    uint32_t u;
    std::memcpy(&u, x + i, 4);
    m = std::max(m, u & 0x7fffffffu);
  }
  float f;
  std::memcpy(&f, &m, 4);
  return f;
}

// q = clamp(rne(x / s), -127, 127) with the IEEE quotient of the reference (int8_quantize), 16 values per step.
// The quotient t = x * (1/s) (one NEON multiply by the fp32 reciprocal) is within 3 ulp of fl(x / s) (relative
// error <= 2^-24 for the reciprocal and for the product, plus the rounding of fl(x / s)). Where it matters the two
// round to the same integer: |t| >= 129 clamps both (|x / s| > 128.9); otherwise ulp(t) <= 2^-16, the two differ by
// less than 2^-14, and when t is at least 2^-12 away from a half-integer no half-integer lies between them (or on
// fl(x / s)), so rne gives the same integer (a product t that underflows is flushed to 0 by NEON: then |x / s| is
// below 2^-125 and rounds to 0 as well). Groups of 16 holding a value |t| < 129 within 2^-12 of a half-integer take
// the exact quotient: vdivq_f32 of neon_a32.h (four IEEE VFP divisions, or the multiply by the exact reciprocal of a
// power-of-two divisor, whose flushed denormal quotients round to 0 like the IEEE ones) and the NEON rounding; inputs
// with exact half-integer quotients are common (larq_zoo images x = p / 127.5 - 1 with s = 2 / 255), so this path
// costs what the plain division loop did. Groups holding a denormal x (flushed to 0 by NEON although x / s need not
// round to 0 when s is tiny too) use the scalar reference expression, and so does a scale that is not normal or has
// no normal reciprocal (tensors of tiny values). One across-lane reduction per 16 values (a NEON-to-core transfer
// stalls the A53 pipeline).
void quantize_fast(const float* x, int64_t n, float s, int8_t* q) {
  const float r = 1.f / s;
  if (std::fpclassify(s) != FP_NORMAL || std::fpclassify(r) != FP_NORMAL) {
    for (int64_t i = 0; i < n; ++i) q[i] = int8_quantize(x[i], s);
    return;
  }
  const float32x4_t vr = vdupq_n_f32(r);
  const float32x4_t lim = vdupq_n_f32(129.f), half = vdupq_n_f32(0.5f), win = vdupq_n_f32(1.f / 4096.f);
  const uint32x4_t expm = vdupq_n_u32(0x7f800000u), zero = vdupq_n_u32(0);
  const int8x16_t lo = vdupq_n_s8(-127);
  const uint32x4_t mant = vdupq_n_u32(0x007fffffu);
  const float32x4_t vs = vdupq_n_f32(s);
  // rounded quotient of 4 values; lanes near a half-integer (|t| < 129 and | |t - rne(t)| - 0.5 | < 2^-12) or-ed into
  // *near, denormal x into *den
  auto conv4 = [&](float32x4_t xv, uint32x4_t* near, uint32x4_t* den) {
    const float32x4_t t = vmulq_f32(xv, vr);
    const float32x4_t rt = bf_a32_rndn(t);
    const float32x4_t d = vabsq_f32(vsubq_f32(vabsq_f32(vsubq_f32(t, rt)), half));
    const uint32x4_t ux = vreinterpretq_u32_f32(xv);
    *den = vorrq_u32(*den, vandq_u32(vceqq_u32(vandq_u32(ux, expm), zero), vtstq_u32(ux, mant)));
    *near = vorrq_u32(*near, vandq_u32(vcltq_f32(vabsq_f32(t), lim), vcltq_f32(d, win)));
    return vcvtq_s32_f32(rt);
  };
  auto exact4 = [&](const float* p) { return vcvtq_s32_f32(bf_a32_rndn(vdivq_f32(vld1q_f32(p), vs))); };
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint32x4_t near = zero, den = zero;
    int32x4_t a = conv4(vld1q_f32(x + i), &near, &den), b = conv4(vld1q_f32(x + i + 4), &near, &den);
    int32x4_t c = conv4(vld1q_f32(x + i + 8), &near, &den), d = conv4(vld1q_f32(x + i + 12), &near, &den);
    if (vmaxvq_u32(vorrq_u32(near, den)) != 0) {
      if (vmaxvq_u32(den) != 0) {
        for (int64_t j = i; j < i + 16; ++j) q[j] = int8_quantize(x[j], s);
        continue;
      }
      a = exact4(x + i);
      b = exact4(x + i + 4);
      c = exact4(x + i + 8);
      d = exact4(x + i + 12);
    }
    const int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)), cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
    vst1q_s8(q + i, vmaxq_s8(vcombine_s8(vqmovn_s16(ab), vqmovn_s16(cd)), lo));
  }
  for (; i < n; ++i) q[i] = int8_quantize(x[i], s);
}

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

// Weights and per-run state shared by Conv2D and Dense: W as column pairs [O/2][kp/16][2][16], per-channel scales.
class GemmA32 {
 public:
  void init(const float* w, int64_t o, int64_t k, const float* bias, Activation act) {
    o_ = o;
    k_ = k;
    kp_ = (k + 15) / 16 * 16;
    npair_ = (o + 1) / 2;
    ldy_ = (2 * npair_ + 3) / 4 * 4;
    std::vector<int8_t> wq(static_cast<size_t>(o * k));
    sw_.assign(static_cast<size_t>(o), 0.f);
    int8_quantize_rows(w, o, k, wq.data(), sw_.data());
    wpack_.assign(static_cast<size_t>(npair_ * 2 * kp_), 0);
    for (int64_t p = 0; p < npair_; ++p)
      a32gemm::pack_pair(wq.data() + 2 * p * k, 2 * p + 1 < o ? wq.data() + (2 * p + 1) * k : nullptr, k, kp_, wpack_.data() + p * 2 * kp_);
    act_ = act;
    bias_.assign(static_cast<size_t>(ldy_), 0.f);
    bias_ok_ = true;
    for (int64_t j = 0; j < o; ++j) {
      bias_[j] = bias ? bias[j] : 0.f;
      bias_ok_ = bias_ok_ && (bias_[j] == 0.f || std::fpclassify(bias_[j]) == FP_NORMAL);
    }
    scale_.assign(static_cast<size_t>(ldy_), 0.f);
  }
  int64_t kp() const { return kp_; }
  size_t prepacked_bytes() const { return wpack_.size() + sw_.size() * 4; }

  // per-run epilogue constants; true when the NEON epilogue is exact (see the file comment)
  bool set_scale(float sx) {
    bool vec = bias_ok_;
    for (int64_t j = 0; j < o_; ++j) {
      scale_[j] = sx * sw_[j];
      const float a = std::fabs(scale_[j]);
      vec = vec && (a == 0.f || a >= 0x1p-100f);
    }
    return vec;
  }

  // rows rp[0..nr) (kp bytes each) -> y rows (o_ floats each, row stride o_)
  void rows(const int8_t* const* rp, int nr, int8_t* apack, int32_t* acc, float* y, bool vec) const {
    const int nq = (nr + 3) / 4;
    for (int qd = 0; qd < nq; ++qd) a32gemm::pack_quad(rp + 4 * qd, std::min(4, nr - 4 * qd), kp_, apack + qd * 4 * kp_);
    const int32_t n16 = static_cast<int32_t>(kp_ / 16), ldy = static_cast<int32_t>(ldy_);
    const int8_t* wp = wpack_.data();
    for (int64_t p = 0; p < npair_; ++p, wp += 2 * kp_)
      for (int qd = 0; qd < nq; ++qd) a32gemm::tile_x2(std::min(4, nr - 4 * qd), apack + qd * 4 * kp_, wp, n16, acc + 4 * qd * ldy_ + 2 * p, ldy);
    const int act = act_ == Activation::kRelu ? 1 : act_ == Activation::kRelu6 ? 2 : 0;
    for (int i = 0; i < nr; ++i) {
      const int32_t* a = acc + i * ldy_;
      float* yr = y + i * o_;
      int64_t j = 0;
      if (vec)
        for (; j + 4 <= o_; j += 4) {
          const float32x4_t v = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(a + j)), vld1q_f32(scale_.data() + j)), vld1q_f32(bias_.data() + j));
          vst1q_f32(yr + j, int8_neon::int8_act_f32x4(v, act));
        }
      for (; j < o_; ++j) yr[j] = apply_activation(act_, static_cast<float>(a[j]) * scale_[j] + bias_[j]);
    }
  }
  int64_t ldy() const { return ldy_; }

 private:
  int64_t o_ = 0, k_ = 0, kp_ = 0, npair_ = 0, ldy_ = 0;
  Activation act_ = Activation::kNone;
  bool bias_ok_ = true;
  std::vector<int8_t> wpack_;
  std::vector<float> sw_, bias_, scale_;
};

// per-thread scratch of one 8-row tile
struct Scratch {
  std::vector<int8_t> rowbuf, apack;
  std::vector<int32_t> acc;
  void fit(int64_t kp, int64_t ldy) {
    if (static_cast<int64_t>(rowbuf.size()) < 8 * kp) rowbuf.assign(static_cast<size_t>(8 * kp), 0);
    if (static_cast<int64_t>(apack.size()) < 8 * kp) apack.resize(static_cast<size_t>(8 * kp));
    if (static_cast<int64_t>(acc.size()) < 8 * ldy) acc.resize(static_cast<size_t>(8 * ldy));
  }
};

class Int8Conv2DA32 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    pad_value_ = static_cast<float>(ctx.attrs().get_number("pad_value", 0.0));
    k_ = g_.kh * g_.kw * g_.ic;
    gemm_.init(ctx.in(1).data<float>(), g_.oc, k_, opt_f32(ctx, 2), parse_activation(ctx.attrs()));
    xq_.assign(static_cast<size_t>(g_.n * g_.ih * g_.iw * g_.ic), 0);
    rows_ = g_.n * g_.oh * g_.ow;
    direct_ = g_.kh == 1 && g_.kw == 1 && g_.sh == 1 && g_.sw == 1 && g_.pt == 0 && g_.pl == 0 && g_.pb == 0 && g_.pr == 0 && k_ % 16 == 0;
  }
  size_t prepacked_bytes() const override { return gemm_.prepacked_bytes(); }
  std::vector<int> released_inputs() const override { return {1}; }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const int64_t n = static_cast<int64_t>(xq_.size());
    const float sx = int8_scale(maxabs(x, n));
    quantize_fast(x, n, sx, xq_.data());
    const int8_t qpad = int8_quantize(pad_value_, sx);
    const bool vec = gemm_.set_scale(sx);
    float* y = ctx.out(0).data<float>();
    const int64_t kp = gemm_.kp(), tiles = (rows_ + 7) / 8;
    ctx.pool->parallel_for(tiles, 1, [&](int64_t t0, int64_t t1) {
      thread_local Scratch sc;
      sc.fit(kp, gemm_.ldy());
      const int8_t* rp[8];
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t r0 = t * 8;
        const int nr = static_cast<int>(std::min<int64_t>(8, rows_ - r0));
        if (direct_) {
          for (int i = 0; i < nr; ++i) rp[i] = xq_.data() + (r0 + i) * k_;
        } else {
          const uint32_t hw = static_cast<uint32_t>(g_.oh * g_.ow), ow = static_cast<uint32_t>(g_.ow);
          const uint32_t nn = bf_a32_udiv(static_cast<uint32_t>(r0), hw), rem = static_cast<uint32_t>(r0) - nn * hw;
          const uint32_t oy0 = bf_a32_udiv(rem, ow);
          int64_t b = nn, oy = oy0, ox = rem - oy0 * ow;
          for (int i = 0; i < nr; ++i) {
            rp[i] = im2col(b, oy, ox, qpad, kp, sc.rowbuf.data() + i * kp);
            if (++ox == g_.ow) {
              ox = 0;
              if (++oy == g_.oh) oy = 0, ++b;
            }
          }
        }
        gemm_.rows(rp, nr, sc.apack.data(), sc.acc.data(), y + r0 * g_.oc, vec);
      }
    });
  }

 private:
  // patch row of output pixel (b, oy, ox): K codes (padding taps qpad), then zeros up to kp (the thread's scratch is
  // shared by every layer, so the padding is rewritten)
  const int8_t* im2col(int64_t b, int64_t oy, int64_t ox, int8_t qpad, int64_t kp, int8_t* buf) const {
    const Conv2DGeom& g = g_;
    std::memset(buf + k_, 0, static_cast<size_t>(kp - k_));
    const int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl, ic = g.ic;
    const int8_t* img = xq_.data() + b * g.ih * g.iw * ic;
    if (g.dw == 1 && iy0 >= 0 && ix0 >= 0 && iy0 + g.dh * (g.kh - 1) < g.ih && ix0 + g.kw - 1 < g.iw) {
      const int64_t run = g.kw * ic;
      for (int64_t ky = 0; ky < g.kh; ++ky) a32gemm::copy_bytes(buf + ky * run, img + ((iy0 + ky * g.dh) * g.iw + ix0) * ic, run);
      return buf;
    }
    int64_t kk = 0;
    for (int64_t ky = 0; ky < g.kh; ++ky)
      for (int64_t kx = 0; kx < g.kw; ++kx, kk += ic) {
        const int64_t iy = iy0 + ky * g.dh, ix = ix0 + kx * g.dw;
        if (iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw) a32gemm::copy_bytes(buf + kk, img + (iy * g.iw + ix) * ic, ic);
        else std::memset(buf + kk, qpad, static_cast<size_t>(ic));
      }
    return buf;
  }

  Conv2DGeom g_{};
  float pad_value_ = 0.f;
  int64_t k_ = 0, rows_ = 0;
  bool direct_ = false;
  GemmA32 gemm_;
  std::vector<int8_t> xq_;
};

class Int8DenseA32 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Tensor& w = ctx.in(1);
    k_ = w.dim(1);
    o_ = w.dim(0);
    gemm_.init(w.data<float>(), o_, k_, opt_f32(ctx, 2), parse_activation(ctx.attrs()));
    xq_.assign(static_cast<size_t>(ctx.in(0).size()), 0);
    rows_ = ctx.in(0).size() / k_;
  }
  size_t prepacked_bytes() const override { return gemm_.prepacked_bytes(); }
  std::vector<int> released_inputs() const override { return {1}; }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const int64_t n = static_cast<int64_t>(xq_.size());
    const float sx = int8_scale(maxabs(x, n));
    quantize_fast(x, n, sx, xq_.data());
    const bool vec = gemm_.set_scale(sx);
    float* y = ctx.out(0).data<float>();
    const int64_t kp = gemm_.kp(), tiles = (rows_ + 7) / 8;
    ctx.pool->parallel_for(tiles, 1, [&](int64_t t0, int64_t t1) {
      thread_local Scratch sc;
      sc.fit(kp, gemm_.ldy());
      const int8_t* rp[8];
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t r0 = t * 8;
        const int nr = static_cast<int>(std::min<int64_t>(8, rows_ - r0));
        for (int i = 0; i < nr; ++i) {
          if (k_ == kp) {
            rp[i] = xq_.data() + (r0 + i) * k_;
          } else {
            std::memcpy(sc.rowbuf.data() + i * kp, xq_.data() + (r0 + i) * k_, static_cast<size_t>(k_));
            std::memset(sc.rowbuf.data() + i * kp + k_, 0, static_cast<size_t>(kp - k_));
            rp[i] = sc.rowbuf.data() + i * kp;
          }
        }
        gemm_.rows(rp, nr, sc.apack.data(), sc.acc.data(), y + r0 * o_, vec);
      }
    });
  }

 private:
  int64_t k_ = 0, o_ = 0, rows_ = 0;
  GemmA32 gemm_;
  std::vector<int8_t> xq_;
};

bool conv_supported(const KernelContext& ctx) {
  if (!int8_allowed(ctx)) return false;
  if (ctx.attrs().get_int("groups", 1) != 1) return false;
  const Shape& w = ctx.in(1).shape();
  return w[1] * w[2] * w[3] <= kInt8MaxK;
}

bool dense_supported(const KernelContext& ctx) { return int8_allowed(ctx) && ctx.in(1).dim(1) <= kInt8MaxK; }

}  // namespace
}  // namespace int8_neon_a32

void register_int8_neon_a32_kernels(OpRegistry& reg) {
  using namespace int8_neon_a32;
  reg.add_kernel({"Conv2D", "conv2d.int8_neon_a32", "int8_neon", 140, kFeatNeon, conv_supported, make_kernel<Int8Conv2DA32>});
  reg.add_kernel({"Dense", "dense.int8_neon_a32", "int8_neon", 140, kFeatNeon, dense_supported, make_kernel<Int8DenseA32>});
}

}  // namespace bitino
#endif

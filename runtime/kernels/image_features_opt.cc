// Optimized kernels of FftLogMagnitude and LocalBinaryPattern (docs/operators_image_features.md).
//
// Bit-identical to the reference kernels by construction: every output element goes through the same IEEE
// double operations in the same order; the optimization is only in data layout and vectorization *across*
// independent transforms / pixels (2-lane float64 NEON on AArch64, a portable 2-lane struct elsewhere):
//   * FftLogMagnitude: blocks of kBlock columns are transformed together along H (lanes = columns), the
//     rows k1 = 0..H/2 are stored transposed and transformed along W in blocks of kBlock rows (lanes = rows);
//     magnitude, logarithm and the final division run on 2-lane vectors; the value of (k1, k2) is also
//     stored at the mirrored position (H-k1, W-k2) (the reference copies the magnitude there; equal).
//   * LocalBinaryPattern: the image is copied into a zero-padded double buffer (the reference returns 0
//     outside), so the bilinear neighbours of consecutive columns are consecutive in memory; this needs
//     floor(c + cp) = c + floor(cp) and ceil(c + cp) = c + ceil(cp) for every column (checked in supports()).
// Registered in tier "neon" (AArch64) or "binary_scalar" (other targets: the same code with the portable
// 2-lane struct), so the bit-exact stage `neon` uses them.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/image_features_common.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace bitino {
namespace {

#if defined(__aarch64__)
struct F64x2 {
  static constexpr int64_t kLanes = 2;
  float64x2_t v;
  static F64x2 load(const double* p) { return {vld1q_f64(p)}; }
  static void store(double* p, F64x2 a) { vst1q_f64(p, a.v); }
  static F64x2 dup(double x) { return {vdupq_n_f64(x)}; }
  friend F64x2 operator+(F64x2 a, F64x2 b) { return {vaddq_f64(a.v, b.v)}; }
  friend F64x2 operator-(F64x2 a, F64x2 b) { return {vsubq_f64(a.v, b.v)}; }
  friend F64x2 operator*(F64x2 a, F64x2 b) { return {vmulq_f64(a.v, b.v)}; }
};

// out[i] = f32((mul * log_f64(sqrt(re^2 + im^2) + eps)) / div) for two lanes.
inline void log_mag2(const double* re, const double* im, double eps, double mul, double div, float* out) {
  const float64x2_t r = vld1q_f64(re), i = vld1q_f64(im);
  const float64x2_t x = vaddq_f64(vsqrtq_f64(vaddq_f64(vmulq_f64(r, r), vmulq_f64(i, i))), vdupq_n_f64(eps));
  const uint64x2_t bits = vreinterpretq_u64_f64(x);
  int64x2_t e = vsubq_s64(vreinterpretq_s64_u64(vandq_u64(vshrq_n_u64(bits, 52), vdupq_n_u64(0x7ff))), vdupq_n_s64(1023));
  float64x2_t m = vreinterpretq_f64_u64(vorrq_u64(vandq_u64(bits, vdupq_n_u64(0x000fffffffffffffULL)), vdupq_n_u64(0x3ff0000000000000ULL)));
  const uint64x2_t big = vcgtq_f64(m, vdupq_n_f64(imgf::kSqrt2));
  m = vbslq_f64(big, vmulq_f64(m, vdupq_n_f64(0.5)), m);
  e = vsubq_s64(e, vreinterpretq_s64_u64(big));  // all-ones mask = -1
  const float64x2_t f = vsubq_f64(m, vdupq_n_f64(1.0));
  const float64x2_t s = vdivq_f64(f, vaddq_f64(vdupq_n_f64(2.0), f));
  const float64x2_t z = vmulq_f64(s, s);
  float64x2_t p = vdupq_n_f64(imgf::kLogC[10]);
  for (int k = 9; k >= 0; --k) p = vaddq_f64(vdupq_n_f64(imgf::kLogC[k]), vmulq_f64(z, p));
  const float64x2_t rr = vmulq_f64(z, p);
  const float64x2_t t = vaddq_f64(s, s);
  const float64x2_t kf = vcvtq_f64_s64(e);
  const float64x2_t lg = vaddq_f64(vmulq_f64(kf, vdupq_n_f64(imgf::kLn2Hi)),
                                   vaddq_f64(t, vaddq_f64(vmulq_f64(t, rr), vmulq_f64(kf, vdupq_n_f64(imgf::kLn2Lo)))));
  const float64x2_t v = vdivq_f64(vmulq_f64(vdupq_n_f64(mul), lg), vdupq_n_f64(div));
  vst1_f32(out, vcvt_f32_f64(v));
}
// Per-lane bit patterns of (d >= 0) for the LBP kernel (NaN -> 0, -0 -> 1, as the scalar compare).
struct Bits2 {
  uint64x2_t v = vdupq_n_u64(0);
  void or_ge0(F64x2 d, int i) { v = vorrq_u64(v, vshlq_u64(vshrq_n_u64(vcgezq_f64(d.v), 63), vdupq_n_s64(i))); }
  uint32_t get0() const { return static_cast<uint32_t>(vgetq_lane_u64(v, 0)); }
  uint32_t get1() const { return static_cast<uint32_t>(vgetq_lane_u64(v, 1)); }
};
constexpr const char* kTier = "neon";
constexpr uint32_t kFeatures = kFeatNeon;
#else
struct F64x2 {
  static constexpr int64_t kLanes = 2;
  double v[2];
  static F64x2 load(const double* p) { return {{p[0], p[1]}}; }
  static void store(double* p, F64x2 a) { p[0] = a.v[0]; p[1] = a.v[1]; }
  static F64x2 dup(double x) { return {{x, x}}; }
  friend F64x2 operator+(F64x2 a, F64x2 b) { return {{a.v[0] + b.v[0], a.v[1] + b.v[1]}}; }
  friend F64x2 operator-(F64x2 a, F64x2 b) { return {{a.v[0] - b.v[0], a.v[1] - b.v[1]}}; }
  friend F64x2 operator*(F64x2 a, F64x2 b) { return {{a.v[0] * b.v[0], a.v[1] * b.v[1]}}; }
};

inline void log_mag2(const double* re, const double* im, double eps, double mul, double div, float* out) {
  for (int l = 0; l < 2; ++l)
    out[l] = static_cast<float>((mul * imgf::log_f64(std::sqrt(re[l] * re[l] + im[l] * im[l]) + eps)) / div);
}
struct Bits2 {
  uint32_t p[2] = {0, 0};
  void or_ge0(F64x2 d, int i) {
    p[0] |= static_cast<uint32_t>(d.v[0] >= 0) << i;
    p[1] |= static_cast<uint32_t>(d.v[1] >= 0) << i;
  }
  uint32_t get0() const { return p[0]; }
  uint32_t get1() const { return p[1]; }
};
constexpr const char* kTier = "binary_scalar";
constexpr uint32_t kFeatures = 0;
#endif

constexpr double kColIdx[2] = {0.0, 1.0};
constexpr int64_t kBlock = 8;  // lanes per block (multiple of F64x2::kLanes); 4 buffers of N x 8 doubles

class FftLogMagnitudeOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Shape& s = ctx.in(0).shape();
    h_ = s[1];
    w_ = s[2];
    hh_ = h_ / 2 + 1;
    hhp_ = (hh_ + kBlock - 1) / kBlock * kBlock;
    ph_.init(h_);
    pw_.init(w_);
    eps_ = ctx.attrs().get_number("eps", 1e-9);
    mul_ = ctx.attrs().get_number("mul", 20.0);
    div_ = ctx.attrs().get_number("div", 255.0);
    shift_ = ctx.attrs().get_int("shift", 1) != 0;
    zr_.assign(w_ * hhp_, 0.0);
    zi_.assign(w_ * hhp_, 0.0);
    nb1_ = (w_ + kBlock - 1) / kBlock;
    nb2_ = hhp_ / kBlock;
    slot_ = 4 * std::max(h_, w_) * kBlock;
    scratch_.assign(std::max(nb1_, nb2_) * slot_, 0.0);
  }

  void run(const KernelContext& ctx) override {
    const int64_t h = h_, w = w_, hh = hh_, hhp = hhp_;
    for (int64_t b = 0; b < ctx.in(0).shape()[0]; ++b) {
      const float* x = ctx.in(0).data<float>() + b * h * w;
      float* y = ctx.out(0).data<float>() + b * h * w;
      // pass 1: columns along H, blocks of kBlock columns; keep rows k1 < hh, transposed into z[c][k1]
      ctx.pool->parallel_for(nb1_, 1, [&](int64_t i0, int64_t i1) {
        double* re = scratch_.data() + i0 * slot_;
        double* im = re + h * kBlock;
        double* tr = im + h * kBlock;
        double* ti = tr + h * kBlock;
        for (int64_t blk = i0; blk < i1; ++blk) {
          const int64_t c0 = blk * kBlock, nc = std::min(kBlock, w - c0);
          for (int64_t r = 0; r < h; ++r) {
            for (int64_t l = 0; l < kBlock; ++l) re[r * kBlock + l] = l < nc ? static_cast<double>(x[r * w + c0 + l]) : 0.0;
            for (int64_t l = 0; l < kBlock; ++l) im[r * kBlock + l] = 0.0;
          }
          imgf::fft_columns<F64x2>(ph_, re, im, tr, ti, kBlock, kBlock);
          for (int64_t l = 0; l < nc; ++l)
            for (int64_t k1 = 0; k1 < hh; ++k1) {
              zr_[(c0 + l) * hhp + k1] = re[k1 * kBlock + l];
              zi_[(c0 + l) * hhp + k1] = im[k1 * kBlock + l];
            }
        }
      });
      // pass 2: rows k1 < hh along W, blocks of kBlock rows; log-magnitude; fftshift + mirror on store
      ctx.pool->parallel_for(nb2_, 1, [&](int64_t i0, int64_t i1) {
        double* re = scratch_.data() + i0 * slot_;
        double* im = re + w * kBlock;
        double* tr = im + w * kBlock;
        double* ti = tr + w * kBlock;
        float vals[kBlock];
        for (int64_t blk = i0; blk < i1; ++blk) {
          const int64_t k0 = blk * kBlock, nk = std::min(kBlock, hh - k0);
          for (int64_t c = 0; c < w; ++c)
            for (int64_t l = 0; l < kBlock; ++l) {
              re[c * kBlock + l] = zr_[c * hhp + k0 + l];
              im[c * kBlock + l] = zi_[c * hhp + k0 + l];
            }
          imgf::fft_columns<F64x2>(pw_, re, im, tr, ti, kBlock, kBlock);
          for (int64_t k2 = 0; k2 < w; ++k2) {
            for (int64_t l = 0; l < kBlock; l += 2)
              log_mag2(re + k2 * kBlock + l, im + k2 * kBlock + l, eps_, mul_, div_, vals + l);
            const int64_t c = shift_ ? (k2 + w / 2) % w : k2;
            const int64_t k2m = (w - k2) % w;
            const int64_t cm = shift_ ? (k2m + w / 2) % w : k2m;
            for (int64_t l = 0; l < nk; ++l) {
              const int64_t k1 = k0 + l;
              const int64_t r = shift_ ? (k1 + h / 2) % h : k1;
              y[r * w + c] = vals[l];
              const int64_t k1m = h - k1;  // mirrored row (only rows k1m >= hh are not computed directly)
              if (k1 >= 1 && k1m >= hh) {
                const int64_t rm = shift_ ? (k1m + h / 2) % h : k1m;
                y[rm * w + cm] = vals[l];
              }
            }
          }
        }
      });
    }
  }

 private:
  int64_t h_ = 0, w_ = 0, hh_ = 0, hhp_ = 0, nb1_ = 0, nb2_ = 0, slot_ = 0;
  imgf::FftPlan ph_, pw_;
  double eps_ = 1e-9, mul_ = 20.0, div_ = 255.0;
  bool shift_ = true;
  std::vector<double> zr_, zi_, scratch_;
};

struct LbpGeom {
  std::vector<double> rp, cp;
  int64_t pad_r = 0, pad_c = 0;
};

LbpGeom lbp_geom(const Json& attrs) {
  LbpGeom g;
  const auto& rp = attrs["rp"];
  const auto& cp = attrs["cp"];
  for (size_t i = 0; i < rp.size(); ++i) {
    g.rp.push_back(rp[i].as_number());
    g.cp.push_back(cp[i].as_number());
    g.pad_r = std::max<int64_t>(g.pad_r, static_cast<int64_t>(std::ceil(std::fabs(g.rp.back()))) + 1);
    g.pad_c = std::max<int64_t>(g.pad_c, static_cast<int64_t>(std::ceil(std::fabs(g.cp.back()))) + 1);
  }
  return g;
}

// Columns' neighbours are contiguous: floor/ceil(c + cp) = c + floor/ceil(cp) for every column c.
bool lbp_opt_supported(const KernelContext& ctx) {
  if (ctx.in(0).shape().size() != 4) return false;
  const int64_t cols = ctx.in(0).shape()[2];
  LbpGeom g = lbp_geom(ctx.attrs());
  if (g.rp.size() > 24) return false;
  for (double cp : g.cp) {
    const double f = std::floor(cp), cl = std::ceil(cp);
    for (int64_t c = 0; c < cols; ++c) {
      const double cc = static_cast<double>(c) + cp;
      if (std::floor(cc) != static_cast<double>(c) + f || std::ceil(cc) != static_cast<double>(c) + cl) return false;
    }
  }
  return true;
}

class LocalBinaryPatternOpt : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = lbp_geom(ctx.attrs());
    uniform_ = ctx.attrs().get_string("method", "uniform") == "uniform";
    div_ = ctx.attrs().get_number("div", 1.0);
    rows_ = ctx.in(0).shape()[1];
    cols_ = ctx.in(0).shape()[2];
    cols_v_ = (cols_ + 1) / 2 * 2;
    ld_ = cols_v_ + 2 * g_.pad_c + 2;
    pad_.assign((rows_ + 2 * g_.pad_r) * ld_, 0.0);
    const int P = static_cast<int>(g_.rp.size());
    lut_.clear();
    if (P <= 16)
      for (uint32_t pat = 0; pat < (1u << P); ++pat) lut_.push_back(code_of(pat, P, uniform_, div_));
    floor_c_.clear();
    ceil_c_.clear();
    for (double cp : g_.cp) {
      floor_c_.push_back(static_cast<int64_t>(std::floor(cp)));
      ceil_c_.push_back(static_cast<int64_t>(std::ceil(cp)));
    }
  }

  void run(const KernelContext& ctx) override {
    const int64_t rows = rows_, cols = cols_, ld = ld_, pr = g_.pad_r, pc = g_.pad_c;
    const int P = static_cast<int>(g_.rp.size());
    for (int64_t b = 0; b < ctx.in(0).shape()[0]; ++b) {
      const float* img = ctx.in(0).data<float>() + b * rows * cols;
      float* y = ctx.out(0).data<float>() + b * rows * cols;
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) pad_[(r + pr) * ld + c + pc] = static_cast<double>(img[r * cols + c]);
      const double* base = pad_.data() + pr * ld + pc;  // base[r * ld + c] = image, 0 outside
      ctx.pool->parallel_for(rows, 1, [&](int64_t r0, int64_t r1) {
        const double* rmin[24];
        const double* rmax[24];
        double omdr[24], vdr[24];
        for (int64_t r = r0; r < r1; ++r) {
          for (int i = 0; i < P; ++i) {  // per row: the reference's rr, floor, ceil, dr
            const double rr = static_cast<double>(r) + g_.rp[i];
            const double fr = std::floor(rr), cr = std::ceil(rr);
            const double dr = rr - fr;
            omdr[i] = 1.0 - dr;
            vdr[i] = dr;
            rmin[i] = base + static_cast<int64_t>(fr) * ld + floor_c_[i];
            rmax[i] = base + static_cast<int64_t>(cr) * ld + floor_c_[i];
          }
          const F64x2 one = F64x2::dup(1.0);
          for (int64_t c = 0; c < cols_v_; c += 2) {
            const F64x2 cvec = F64x2::load(kColIdx) + F64x2::dup(static_cast<double>(c));
            const F64x2 center = F64x2::load(base + r * ld + c);
            Bits2 pat;
            for (int i = 0; i < P; ++i) {
              // dc = cc - floor(cc) with floor(cc) = c + floor(cp) (supports()), as the reference
              const F64x2 dc = (cvec + F64x2::dup(g_.cp[i])) - (cvec + F64x2::dup(static_cast<double>(floor_c_[i])));
              const int64_t dt = ceil_c_[i] - floor_c_[i];
              const F64x2 tl = F64x2::load(rmin[i] + c), tr = F64x2::load(rmin[i] + c + dt);
              const F64x2 bl = F64x2::load(rmax[i] + c), br = F64x2::load(rmax[i] + c + dt);
              const F64x2 top = (one - dc) * tl + dc * tr;
              const F64x2 bottom = (one - dc) * bl + dc * br;
              const F64x2 tex = F64x2::dup(omdr[i]) * top + F64x2::dup(vdr[i]) * bottom;
              pat.or_ge0(tex - center, i);
            }
            y[r * cols + c] = code(pat.get0());
            if (c + 1 < cols) y[r * cols + c + 1] = code(pat.get1());
          }
        }
      });
    }
  }

 private:
  LbpGeom g_;
  bool uniform_ = true;
  double div_ = 1.0;
  int64_t rows_ = 0, cols_ = 0, cols_v_ = 0, ld_ = 0;
  std::vector<int64_t> floor_c_, ceil_c_;
  std::vector<double> pad_;
  std::vector<float> lut_;  // output for every bit pattern (P <= 16), same double division as the reference

  static float code_of(uint32_t pat, int P, bool uniform, double div) {
    int64_t lbp = 0;
    if (uniform) {
      int changes = 0, ones = 0;
      for (int i = 0; i < P - 1; ++i) changes += ((pat >> i) & 1) != ((pat >> (i + 1)) & 1);
      for (int i = 0; i < P; ++i) ones += (pat >> i) & 1;
      lbp = changes <= 2 ? ones : P + 1;
    } else {
      lbp = static_cast<int64_t>(pat);
    }
    return static_cast<float>(static_cast<double>(lbp) / div);
  }
  float code(uint32_t pat) const {
    return lut_.empty() ? code_of(pat, static_cast<int>(g_.rp.size()), uniform_, div_) : lut_[pat];
  }
};

// Concat along the last axis of inputs with a few channels each (e.g. RGB + two feature planes): one pass over the
// pixels instead of one memcpy per pixel and input. Exact (copies).
bool concat_small_supported(const KernelContext& ctx) {
  const Shape& os = ctx.out(0).shape();
  int64_t axis = ctx.attrs().get_int("axis", -1);
  if (axis < 0) axis += static_cast<int64_t>(os.size());
  if (os.empty() || axis != static_cast<int64_t>(os.size()) - 1 || ctx.inputs.size() > 8) return false;
  for (size_t i = 0; i < ctx.inputs.size(); ++i)
    if (!ctx.inputs[i] || ctx.in(i).dtype() != DType::kF32 || ctx.in(i).shape().back() > 8) return false;
  return true;
}

class ConcatSmallChannels : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const int64_t ni = static_cast<int64_t>(ctx.inputs.size());
    const float* src[8];
    int64_t len[8];
    int64_t total = 0;
    for (int64_t i = 0; i < ni; ++i) {
      src[i] = ctx.in(i).data<float>();
      len[i] = ctx.in(i).shape().back();
      total += len[i];
    }
    float* y = ctx.out(0).data<float>();
    const int64_t outer = ctx.out(0).size() / total;
    if (ni == 3 && len[0] == 3 && len[1] == 1 && len[2] == 1) {  // RGB + two planes (the FTL features)
      const float *a = src[0], *b = src[1], *c = src[2];
      ctx.pool->parallel_for(outer, 4096, [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; ++o) {
          float* d = y + o * 5;
          d[0] = a[3 * o];
          d[1] = a[3 * o + 1];
          d[2] = a[3 * o + 2];
          d[3] = b[o];
          d[4] = c[o];
        }
      });
      return;
    }
    ctx.pool->parallel_for(outer, 4096, [&](int64_t o0, int64_t o1) {
      for (int64_t o = o0; o < o1; ++o) {
        float* d = y + o * total;
        for (int64_t i = 0; i < ni; ++i) {
          const int64_t l = len[i];
          const float* s = src[i] + o * l;
          for (int64_t k = 0; k < l; ++k) d[k] = s[k];
          d += l;
        }
      }
    });
  }
};

#if defined(__aarch64__)
// ImageGrayU8 on 4 pixels per step: the reference's float32 product, IEEE selects for the clamp (NaN -> 0),
// truncating conversion and the integer fixed-point sum (int32; supports() checks it cannot overflow).
bool gray_neon_supported(const KernelContext& ctx) {
  auto c = ctx.attrs().get_ints("coeffs", {});
  const int64_t shift = ctx.attrs().get_int("shift", 15);
  return c.size() == 3 && 255 * (c[0] + c[1] + c[2]) + (int64_t{1} << (shift - 1)) < (int64_t{1} << 31);
}

class ImageGrayU8Neon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    auto c = ctx.attrs().get_ints("coeffs", {});
    for (int i = 0; i < 3; ++i) coeffs_[i] = c[i];
    shift_ = static_cast<int>(ctx.attrs().get_int("shift", 15));
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 255.0));
  }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.out(0).size();
    const float32x4_t sc = vdupq_n_f32(scale_), zero = vdupq_n_f32(0.f), hi = vdupq_n_f32(255.f);
    const int32x4_t rnd = vdupq_n_s32(1 << (shift_ - 1)), sh = vdupq_n_s32(-shift_);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
      const float32x4x3_t px = vld3q_f32(x + 3 * i);
      int32x4_t acc = rnd;
      for (int c = 0; c < 3; ++c) {
        float32x4_t v = vmulq_f32(px.val[c], sc);
        v = vbslq_f32(vcgeq_f32(v, zero), v, zero);
        v = vbslq_f32(vcltq_f32(v, hi), v, hi);
        acc = vmlaq_n_s32(acc, vcvtq_s32_f32(v), static_cast<int32_t>(coeffs_[c]));
      }
      vst1q_f32(y + i, vcvtq_f32_s32(vshlq_s32(acc, sh)));
    }
    for (; i < n; ++i) y[i] = static_cast<float>(imgf::gray_u8(x + 3 * i, scale_, coeffs_, shift_));
  }

 private:
  int64_t coeffs_[3] = {0, 0, 0};
  int shift_ = 15;
  float scale_ = 255.f;
};
#endif

}  // namespace

void register_image_feature_opt_kernels(OpRegistry& reg) {
  reg.add_kernel({"FftLogMagnitude", "fft_log_magnitude.opt", kTier, 30, kFeatures, nullptr, make_kernel<FftLogMagnitudeOpt>});
  reg.add_kernel({"LocalBinaryPattern", "local_binary_pattern.opt", kTier, 30, kFeatures, lbp_opt_supported,
                  make_kernel<LocalBinaryPatternOpt>});
  // priority 45: also above concat.float_opt (40), which copies each pixel's channels with one memcpy per input
  reg.add_kernel({"Concat", "concat.small_channels", kTier, 45, kFeatures, concat_small_supported, make_kernel<ConcatSmallChannels>});
#if defined(__aarch64__)
  reg.add_kernel({"ImageGrayU8", "image_gray_u8.neon", "neon", 30, kFeatNeon, gray_neon_supported, make_kernel<ImageGrayU8Neon>});
#endif
}

}  // namespace bitino

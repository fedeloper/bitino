// Reference kernels of the image-feature operators (docs/operators_image_features.md): literal per-pixel /
// per-transform loops over the shared arithmetic of kernels/image_features_common.h. The numpy executor
// (python/bitino/image_ops.py) performs the same IEEE operations in the same order (bit-identical).
#include <cmath>
#include <vector>

#include "kernels/image_features_common.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

class ImageGrayU8Ref : public Kernel {
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
    for (int64_t i = 0; i < n; ++i) y[i] = static_cast<float>(imgf::gray_u8(x + 3 * i, scale_, coeffs_, shift_));
  }

 private:
  int64_t coeffs_[3] = {0, 0, 0};
  int shift_ = 15;
  float scale_ = 255.f;
};

// |FFT2| by 1-D transforms: every column along H, then the rows k1 = 0..H/2 along W; rows k1 > H/2 mirrored.
class FftLogMagnitudeRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Shape& s = ctx.in(0).shape();
    h_ = s[1];
    w_ = s[2];
    ph_.init(h_);
    pw_.init(w_);
    eps_ = ctx.attrs().get_number("eps", 1e-9);
    mul_ = ctx.attrs().get_number("mul", 20.0);
    div_ = ctx.attrs().get_number("div", 255.0);
    shift_ = ctx.attrs().get_int("shift", 1) != 0;
    const int64_t len = std::max(h_, w_);
    re_.resize(len);
    im_.resize(len);
    tr_.resize(len);
    ti_.resize(len);
    yr_.resize(h_ * w_);
    yi_.resize(h_ * w_);
    mag_.resize(h_ * w_);
  }
  void run(const KernelContext& ctx) override {
    const int64_t h = h_, w = w_, hh = h / 2 + 1;
    for (int64_t b = 0; b < ctx.in(0).shape()[0]; ++b) {
      const float* x = ctx.in(0).data<float>() + b * h * w;
      float* y = ctx.out(0).data<float>() + b * h * w;
      for (int64_t c = 0; c < w; ++c) {
        for (int64_t r = 0; r < h; ++r) {
          re_[r] = static_cast<double>(x[r * w + c]);
          im_[r] = 0.0;
        }
        imgf::fft_columns<imgf::Scalar>(ph_, re_.data(), im_.data(), tr_.data(), ti_.data(), 1, 1);
        for (int64_t r = 0; r < h; ++r) {
          yr_[r * w + c] = re_[r];
          yi_[r * w + c] = im_[r];
        }
      }
      for (int64_t k1 = 0; k1 < hh; ++k1) {
        for (int64_t c = 0; c < w; ++c) {
          re_[c] = yr_[k1 * w + c];
          im_[c] = yi_[k1 * w + c];
        }
        imgf::fft_columns<imgf::Scalar>(pw_, re_.data(), im_.data(), tr_.data(), ti_.data(), 1, 1);
        for (int64_t k2 = 0; k2 < w; ++k2) mag_[k1 * w + k2] = std::sqrt(re_[k2] * re_[k2] + im_[k2] * im_[k2]);
      }
      for (int64_t k1 = hh; k1 < h; ++k1)
        for (int64_t k2 = 0; k2 < w; ++k2) mag_[k1 * w + k2] = mag_[(h - k1) * w + (w - k2) % w];
      for (int64_t k1 = 0; k1 < h; ++k1) {
        const int64_t r = shift_ ? (k1 + h / 2) % h : k1;
        for (int64_t k2 = 0; k2 < w; ++k2) {
          const int64_t c = shift_ ? (k2 + w / 2) % w : k2;
          const double v = (mul_ * imgf::log_f64(mag_[k1 * w + k2] + eps_)) / div_;
          y[r * w + c] = static_cast<float>(v);
        }
      }
    }
  }

 private:
  int64_t h_ = 0, w_ = 0;
  imgf::FftPlan ph_, pw_;
  double eps_ = 1e-9, mul_ = 20.0, div_ = 255.0;
  bool shift_ = true;
  std::vector<double> re_, im_, tr_, ti_, yr_, yi_, mag_;
};

// scikit-image _local_binary_pattern with bilinear_interpolation(mode 'C', cval 0), literally.
class LocalBinaryPatternRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const auto& rp = ctx.attrs()["rp"];
    const auto& cp = ctx.attrs()["cp"];
    rp_.clear();
    cp_.clear();
    for (size_t i = 0; i < rp.size(); ++i) {
      rp_.push_back(rp[i].as_number());
      cp_.push_back(cp[i].as_number());
    }
    uniform_ = ctx.attrs().get_string("method", "uniform") == "uniform";
    div_ = ctx.attrs().get_number("div", 1.0);
  }
  void run(const KernelContext& ctx) override {
    const Shape& s = ctx.in(0).shape();
    const int64_t rows = s[1], cols = s[2];
    const int P = static_cast<int>(rp_.size());
    for (int64_t b = 0; b < s[0]; ++b) {
      const float* img = ctx.in(0).data<float>() + b * rows * cols;
      float* y = ctx.out(0).data<float>() + b * rows * cols;
      auto px = [&](long r, long c) -> double {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return 0.0;
        return static_cast<double>(img[r * cols + c]);
      };
      int bits[32];
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          const double center = static_cast<double>(img[r * cols + c]);
          for (int i = 0; i < P; ++i) {
            const double rr = static_cast<double>(r) + rp_[i];
            const double cc = static_cast<double>(c) + cp_[i];
            const long minr = static_cast<long>(std::floor(rr)), minc = static_cast<long>(std::floor(cc));
            const long maxr = static_cast<long>(std::ceil(rr)), maxc = static_cast<long>(std::ceil(cc));
            const double dr = rr - static_cast<double>(minr);
            const double dc = cc - static_cast<double>(minc);
            const double top = (1 - dc) * px(minr, minc) + dc * px(minr, maxc);
            const double bottom = (1 - dc) * px(maxr, minc) + dc * px(maxr, maxc);
            const double tex = (1 - dr) * top + dr * bottom;
            bits[i] = (tex - center >= 0) ? 1 : 0;
          }
          int64_t lbp = 0;
          if (uniform_) {
            int changes = 0, ones = 0;
            for (int i = 0; i < P - 1; ++i) changes += bits[i] != bits[i + 1];
            for (int i = 0; i < P; ++i) ones += bits[i];
            lbp = changes <= 2 ? ones : P + 1;
          } else {
            for (int i = 0; i < P; ++i) lbp += static_cast<int64_t>(bits[i]) << i;
          }
          y[r * cols + c] = static_cast<float>(static_cast<double>(lbp) / div_);
        }
      }
    }
  }

 private:
  std::vector<double> rp_, cp_;
  bool uniform_ = true;
  double div_ = 1.0;
};

}  // namespace

void register_image_feature_reference_kernels(OpRegistry& reg) {
  reg.add_kernel({"ImageGrayU8", "image_gray_u8.reference", "reference", 0, 0, nullptr, make_kernel<ImageGrayU8Ref>});
  reg.add_kernel({"FftLogMagnitude", "fft_log_magnitude.reference", "reference", 0, 0, nullptr, make_kernel<FftLogMagnitudeRef>});
  reg.add_kernel({"LocalBinaryPattern", "local_binary_pattern.reference", "reference", 0, 0, nullptr,
                  make_kernel<LocalBinaryPatternRef>});
}

}  // namespace bitino

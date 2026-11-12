// Bit-plane kernels for ternary ({-1, 0, +1}) operands, i.e. torch.sign
// semantics (input_binarize = sign_zero) that the classic XNOR/popcount
// kernels cannot represent.
//
// Encoding (64-bit words, element c -> word c/64, bit c%64, padding bits 0):
//   nz  plane: 1 iff the value is non-zero
//   neg plane: 1 iff the value is negative   (same polarity as the bit dtype)
// For x ternary and w = ±1 with an optional zero mask:
//   m   = nz_x & nz_w
//   x·w = popcount(m) - 2 * popcount(m & (neg_x ^ neg_w))
// which is exact. Registered in tier "binary_scalar" so the existing benchmark
// stages pick it up; results are bit-identical to bdense.reference (tests).
#include <cstring>
#include <vector>

#include "kernels/bitpack.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

const float* opt_in(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

// Repacks a uint32 bit row (bit dtype) into uint64 words.
inline void to_words64(const uint32_t* src, int64_t words32, uint64_t* dst, int64_t words64) {
  for (int64_t q = 0; q < words64; ++q) {
    uint64_t lo = 2 * q < words32 ? src[2 * q] : 0u;
    uint64_t hi = 2 * q + 1 < words32 ? src[2 * q + 1] : 0u;
    dst[q] = lo | (hi << 32);
  }
}

class BDenseTernaryBitplane : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    const Tensor& w = ctx.in(1);
    o_ = w.dim(0);
    k_ = w.dim(1);
    rows_ = ctx.in(0).size() / k_;
    w32_ = bit_words(k_);
    w64_ = (k_ + 63) / 64;
    wneg_.assign(static_cast<size_t>(o_ * w64_), 0);
    wnz_.assign(static_cast<size_t>(o_ * w64_), 0);
    has_wz_ = ctx.has_in(5);
    std::vector<uint64_t> tmp(static_cast<size_t>(w64_));
    for (int64_t o = 0; o < o_; ++o) {
      to_words64(w.data<uint32_t>() + o * w32_, w32_, &wneg_[o * w64_], w64_);
      if (has_wz_) {
        to_words64(ctx.in(5).data<uint32_t>() + o * w32_, w32_, tmp.data(), w64_);
        for (int64_t q = 0; q < w64_; ++q) {
          int64_t valid = k_ - q * 64;
          uint64_t lane = valid >= 64 ? ~0ull : ((1ull << valid) - 1);
          wnz_[o * w64_ + q] = ~tmp[q] & lane;
        }
      }
    }
    xneg_.assign(static_cast<size_t>(rows_ * w64_), 0);
    xnz_.assign(static_cast<size_t>(rows_ * w64_), 0);
  }
  size_t prepacked_bytes() const override { return (wneg_.size() + (has_wz_ ? wnz_.size() : 0)) * 8; }

  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    const float* thr = opt_in(ctx, 4);
    for (int64_t r = 0; r < rows_; ++r) {
      const float* xr = x + r * k_;
      for (int64_t q = 0; q < w64_; ++q) {
        uint64_t nz = 0, neg = 0;
        int64_t lim = k_ - q * 64 < 64 ? k_ - q * 64 : 64;
        BF_A32_IEEE_LOOP
        for (int64_t b = 0; b < lim; ++b) {
          int64_t c = q * 64 + b;
          float t = thr ? thr[c] : 0.f;
          nz |= uint64_t(xr[c] > t || xr[c] < t) << b;  // NaN -> 0, like binarize()
          neg |= uint64_t(xr[c] < t) << b;
        }
        xnz_[r * w64_ + q] = nz;
        xneg_[r * w64_ + q] = neg;
      }
    }
    const float* mul = opt_in(ctx, 2);
    const float* add = opt_in(ctx, 3);
    float* y = ctx.out(0).data<float>();
    const int64_t W = w64_;
    ctx.pool->parallel_for(o_, 16, [&](int64_t o0, int64_t o1) {
      for (int64_t r = 0; r < rows_; ++r) {
        const uint64_t* xn = &xnz_[r * W];
        const uint64_t* xs = &xneg_[r * W];
        for (int64_t o = o0; o < o1; ++o) {
          const uint64_t* ws = &wneg_[o * W];
          int32_t pm = 0, pd = 0;
          if (has_wz_) {
            const uint64_t* wn = &wnz_[o * W];
            for (int64_t q = 0; q < W; ++q) {
              uint64_t m = xn[q] & wn[q];
              pm += popcount64(m);
              pd += popcount64(m & (xs[q] ^ ws[q]));
            }
          } else {
            for (int64_t q = 0; q < W; ++q) {
              pm += popcount64(xn[q]);
              pd += popcount64(xn[q] & (xs[q] ^ ws[q]));
            }
          }
          float v = static_cast<float>(pm - 2 * pd);
          if (mul) v *= mul[o];
          if (add) v += add[o];
          y[r * o_ + o] = apply_activation(act_, v);
        }
      }
    });
  }

 private:
  Activation act_ = Activation::kNone;
  int64_t o_ = 0, k_ = 0, rows_ = 0, w32_ = 0, w64_ = 0;
  bool has_wz_ = false;
  std::vector<uint64_t> wneg_, wnz_, xneg_, xnz_;
};

bool bdense_ternary_supported(const KernelContext& ctx) {
  return ctx.in(0).dtype() == DType::kF32 && ctx.attrs().get_string("input_binarize", "ge") == "sign_zero" &&
         !ctx.attrs().contains("epilogue") &&  // no fused-epilogue support (never hosted: graph_rewrite skips sign_zero)
         ctx.attrs().get_string("output", "f32") == "f32";  // no bit output (never set on sign_zero BDense either)
}

}  // namespace

void register_ternary_kernels(OpRegistry& reg) {
  reg.add_kernel({"BDense", "bdense.ternary_bitplane", "binary_scalar", 25, 0, bdense_ternary_supported,
                  make_kernel<BDenseTernaryBitplane>});
}

}  // namespace bitino

// Portable bit-packed binary kernels.
//
//  * "packed_naive"  (benchmark stage 3): bit packing + XOR/popcount over
//    32-bit words, one (pixel, channel) at a time. No blocking.
//  * "binary_scalar" (stage 4): 64-bit popcounts, 2-pixel x 4-channel register
//    blocking, interior/border split so the padding correction is only
//    evaluated on border pixels.
// Both support grouped convolution (BConvGroupLayout in bconv_plan.h): the
// patch matrix holds one row per (group, pixel).
// Both are plain C++; the compiler may auto-vectorize parts of them.
#include <algorithm>

#include "kernels/bconv_plan.h"
#include "kernels/float_opt/eltwise_program.h"
#include "ops/op_defs.h"

namespace bitino {
namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

class BConv2DPackedNaive : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { plan_.build(ctx); }
  size_t prepacked_bytes() const override { return plan_.prepacked_bytes(); }
  void run(const KernelContext& ctx) override {
    const uint32_t* pt = plan_.prepare_patches(ctx, opt_f32(ctx, 4));
    const uint32_t* w = plan_.weights();
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    const BConvPlan& p = plan_;
    int64_t pixels = p.g.n * p.g.oh * p.g.ow;
    ctx.pool->parallel_for(pixels, 16, [&](int64_t q0, int64_t q1) {
      for (int64_t q = q0; q < q1; ++q) {
        for (int64_t o = 0; o < p.g.oc; ++o) {
          const uint32_t* a = p.patch_row(pt, q, o);
          const uint32_t* b = w + o * p.k_words_pad;
          int32_t pc = 0;
          for (int64_t k = 0; k < p.k_words; ++k) pc += popcount32(a[k] ^ b[k]);
          int32_t dot = static_cast<int32_t>(p.k_bits) - 2 * pc + p.pad_correction(q, o);
          p.store(ctx, q, o, dot, mul, add);
        }
      }
    });
  }

 private:
  BConvPlan plan_;
};

class BConv2DScalar : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    plan_.build(ctx);
    epi_.prepare_epilogue(ctx, plan_.g.oh * plan_.g.ow);
    if (plan_.bit_out && epi_.active()) {  // bit output after an epilogue: float rows first
      yf_.assign(static_cast<size_t>(plan_.pixels * plan_.g.oc), 0.f);
      plan_.float_rows = yf_.data();
    }
  }
  size_t prepacked_bytes() const override { return plan_.prepacked_bytes(); }
  void run(const KernelContext& ctx) override {
    conv(ctx);
    if (epi_.active()) {  // fused elementwise epilogue (graph_rewrite)
      epi_.begin(ctx);
      float* y = plan_.float_rows ? plan_.float_rows : ctx.out(0).data<float>();
      ctx.pool->parallel_for(plan_.pixels, 64, [&](int64_t r0, int64_t r1) { epi_.apply(y, r0, r1); });
      if (plan_.float_rows) run_bit_output(ctx);
    }
  }

 private:
  // Bit output after an epilogue: binarize the final float rows (kernels/bit_output.h).
  void run_bit_output(const KernelContext& ctx) {
    const float* thr = plan_.bout.threshold(ctx);
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    ctx.pool->parallel_for(plan_.pixels, 64, [&](int64_t r0, int64_t r1) {
      pack_output_rows(plan_.float_rows, r0, r1, plan_.g.oc, yb, plan_.bout.mode, thr);
    });
  }

  void conv(const KernelContext& ctx) {
    const uint32_t* pt = plan_.prepare_patches(ctx, opt_f32(ctx, 4));
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    const BConvPlan& p = plan_;
    const int64_t pixels = p.g.n * p.g.oh * p.g.ow;
    const int64_t kq = p.k_words_pad / 2;  // uint64 words per row (k_words_pad % 4 == 0)
    const auto* w64 = reinterpret_cast<const uint64_t*>(p.weights());
    const int64_t ocg = p.L.ocg, gstride = pixels * kq;  // uint64 words between the patch matrices of two groups
    const int32_t kbits = static_cast<int32_t>(p.k_bits);
    // Parallelize over pairs of pixels.
    ctx.pool->parallel_for((pixels + 1) / 2, 8, [&](int64_t b0, int64_t b1) {
      for (int64_t blk = b0; blk < b1; ++blk) {
        int64_t q0 = blk * 2;
        bool two = q0 + 1 < pixels;
        bool interior = p.pixel_is_interior(q0) && (!two || p.pixel_is_interior(q0 + 1));
        for (int64_t gi = 0; gi < p.g.groups; ++gi) {
          const uint64_t* x0 = reinterpret_cast<const uint64_t*>(pt) + gi * gstride + q0 * kq;
          const uint64_t* x1 = two ? x0 + kq : x0;
          const int64_t oe = (gi + 1) * ocg;
          int64_t o = gi * ocg;
          for (; o + 4 <= oe; o += 4) {
            const uint64_t* w0 = w64 + o * kq;
            const uint64_t* w1 = w0 + kq;
            const uint64_t* w2 = w1 + kq;
            const uint64_t* w3 = w2 + kq;
            int32_t c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            for (int64_t k = 0; k < kq; ++k) {
              uint64_t a0 = x0[k], a1 = x1[k];
              c00 += popcount64(a0 ^ w0[k]); c01 += popcount64(a0 ^ w1[k]);
              c02 += popcount64(a0 ^ w2[k]); c03 += popcount64(a0 ^ w3[k]);
              c10 += popcount64(a1 ^ w0[k]); c11 += popcount64(a1 ^ w1[k]);
              c12 += popcount64(a1 ^ w2[k]); c13 += popcount64(a1 ^ w3[k]);
            }
            int32_t r0[4] = {c00, c01, c02, c03}, r1[4] = {c10, c11, c12, c13};
            for (int j = 0; j < 4; ++j) {
              int32_t d0 = kbits - 2 * r0[j], d1 = kbits - 2 * r1[j];
              if (!interior) {
                d0 += p.pad_correction(q0, o + j);
                if (two) d1 += p.pad_correction(q0 + 1, o + j);
              }
              p.store(ctx, q0, o + j, d0, mul, add);
              if (two) p.store(ctx, q0 + 1, o + j, d1, mul, add);
            }
          }
          for (; o < oe; ++o) {
            const uint64_t* w0 = w64 + o * kq;
            int32_t c0 = 0, c1 = 0;
            for (int64_t k = 0; k < kq; ++k) {
              c0 += popcount64(x0[k] ^ w0[k]);
              c1 += popcount64(x1[k] ^ w0[k]);
            }
            p.store(ctx, q0, o, kbits - 2 * c0 + (interior ? 0 : p.pad_correction(q0, o)), mul, add);
            if (two) p.store(ctx, q0 + 1, o, kbits - 2 * c1 + (interior ? 0 : p.pad_correction(q0 + 1, o)), mul, add);
          }
        }
      }
    });
  }

  BConvPlan plan_;
  EltwiseProgram epi_;
  std::vector<float> yf_;
};

// BDense: treat as a 1x1 convolution over rows.
class BDenseScalar : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    act_ = parse_activation(ctx.attrs());
    mode_ = parse_binarize(ctx.attrs());
    k_ = ctx.in(1).dim(1);
    words_ = bit_words(k_);
    rows_ = ctx.in(0).size() / k_;
    xbits_.assign(static_cast<size_t>(rows_ * words_), 0);
    bout_.parse(ctx);
    if (bout_.active) yf_.assign(static_cast<size_t>(rows_ * ctx.in(1).dim(0)), 0.f);
    const Tensor& y = ctx.out(0);
    epi_.prepare_epilogue(ctx, y.dim(0) > 0 ? rows_ / y.dim(0) : rows_);
  }
  void run(const KernelContext& ctx) override {
    const float* thr = opt_f32(ctx, 4);
    const uint32_t* xb;
    if (ctx.in(0).dtype() == DType::kF32) {
      for (int64_t r = 0; r < rows_; ++r) pack_row_f32(ctx.in(0).data<float>() + r * k_, k_, &xbits_[r * words_], mode_, thr);
      xb = xbits_.data();
    } else {
      xb = ctx.in(0).data<uint32_t>();
    }
    const uint32_t* w = ctx.in(1).data<uint32_t>();
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    int64_t on = ctx.in(1).dim(0);
    float* y = bout_.active ? yf_.data() : ctx.out(0).data<float>();  // bit output: float rows first
    ctx.pool->parallel_for(on, 16, [&](int64_t o0, int64_t o1) {
      for (int64_t r = 0; r < rows_; ++r)
        for (int64_t o = o0; o < o1; ++o) {
          int32_t pc = 0;
          for (int64_t q = 0; q < words_; ++q) pc += popcount32(xb[r * words_ + q] ^ w[o * words_ + q]);
          float v = static_cast<float>(k_ - 2 * pc);
          if (mul) v *= mul[o];
          if (add) v += add[o];
          y[r * on + o] = apply_activation(act_, v);
        }
    });
    if (epi_.active()) {  // fused elementwise epilogue (graph_rewrite), after the whole product
      epi_.begin(ctx);
      ctx.pool->parallel_for(rows_, 64, [&](int64_t r0, int64_t r1) { epi_.apply(y, r0, r1); });
    }
    if (bout_.active) run_bit_output(ctx, y, on);
  }

 private:
  // Bit output: binarize the final float rows (kernels/bit_output.h).
  void run_bit_output(const KernelContext& ctx, const float* y, int64_t on) {
    const float* thr = bout_.threshold(ctx);
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    ctx.pool->parallel_for(rows_, 64, [&](int64_t r0, int64_t r1) { pack_output_rows(y, r0, r1, on, yb, bout_.mode, thr); });
  }

  Activation act_ = Activation::kNone;
  BinarizeMode mode_ = BinarizeMode::kGe;
  int64_t k_ = 0, words_ = 0, rows_ = 0;
  BitOutput bout_;
  std::vector<uint32_t> xbits_;
  std::vector<float> yf_;  // float output before binarization (bit output only)
  EltwiseProgram epi_;
};

bool bdense_supported(const KernelContext& ctx) { return ctx.attrs().get_string("input_binarize", "ge") != "sign_zero"; }
bool bconv_naive_supported(const KernelContext& ctx) { return bconv_packed_grouped_supported(ctx) && !has_epilogue(ctx); }

}  // namespace

void register_binary_scalar_kernels(OpRegistry& reg) {
  reg.add_kernel({"BConv2D", "bconv2d.packed_naive", "packed_naive", 10, 0, bconv_naive_supported, make_kernel<BConv2DPackedNaive>});
  reg.add_kernel({"BConv2D", "bconv2d.binary_scalar", "binary_scalar", 20, 0, bconv_packed_grouped_supported, make_kernel<BConv2DScalar>});
  reg.add_kernel({"BDense", "bdense.binary_scalar", "binary_scalar", 20, 0, bdense_supported, make_kernel<BDenseScalar>});
}

}  // namespace bitino

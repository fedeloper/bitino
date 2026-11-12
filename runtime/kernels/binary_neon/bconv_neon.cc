// AArch64 NEON bit-packed BConv2D (benchmark stage 5).
//
// GEMM view: patches [P][Kw] x weights [O][Kw] (Kw multiple of 4 words).
// Micro-kernel: 4 pixels x 4 output channels, 128 bits per step:
//   acc[i][j] += pairwise_widen(vcnt(a_i XOR w_j))     (uint16x8)
// Each step adds at most 16 per uint16 lane, so rows of up to 4095 steps
// (524,160 bits) cannot overflow; supports() enforces this bound.
#include "kernels/bconv_plan.h"
#include "kernels/float_opt/eltwise_program.h"
#include "kernels/neon_a32.h"
#include "ops/op_defs.h"

#if defined(__aarch64__) || BF_NEON_A32
#include <arm_neon.h>
#endif

namespace bitino {

#if defined(__aarch64__) || BF_NEON_A32
namespace {

const float* opt_f32(const KernelContext& ctx, size_t i) { return ctx.has_in(i) ? ctx.in(i).data<float>() : nullptr; }

constexpr int64_t kMaxSteps = 4095;

// Computes popcount(x_i XOR w_j) for 4 rows x 4 rows over `steps` 128-bit steps.
inline void popcount_4x4(const uint32_t* const x[4], const uint32_t* const w[4], int64_t steps, int32_t out[4][4]) {
  uint16x8_t acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = vdupq_n_u16(0);
  for (int64_t s = 0; s < steps; ++s) {
    uint8x16_t a0 = vld1q_u8(reinterpret_cast<const uint8_t*>(x[0] + 4 * s));
    uint8x16_t a1 = vld1q_u8(reinterpret_cast<const uint8_t*>(x[1] + 4 * s));
    uint8x16_t a2 = vld1q_u8(reinterpret_cast<const uint8_t*>(x[2] + 4 * s));
    uint8x16_t a3 = vld1q_u8(reinterpret_cast<const uint8_t*>(x[3] + 4 * s));
    for (int j = 0; j < 4; ++j) {
      uint8x16_t b = vld1q_u8(reinterpret_cast<const uint8_t*>(w[j] + 4 * s));
      acc[0][j] = vpadalq_u8(acc[0][j], vcntq_u8(veorq_u8(a0, b)));
      acc[1][j] = vpadalq_u8(acc[1][j], vcntq_u8(veorq_u8(a1, b)));
      acc[2][j] = vpadalq_u8(acc[2][j], vcntq_u8(veorq_u8(a2, b)));
      acc[3][j] = vpadalq_u8(acc[3][j], vcntq_u8(veorq_u8(a3, b)));
    }
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) out[i][j] = static_cast<int32_t>(vaddlvq_u16(acc[i][j]));
}

inline int32_t popcount_1x1(const uint32_t* x, const uint32_t* w, int64_t steps) {
  uint16x8_t acc = vdupq_n_u16(0);
  for (int64_t s = 0; s < steps; ++s)
    acc = vpadalq_u8(acc, vcntq_u8(veorq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(x + 4 * s)),
                                            vld1q_u8(reinterpret_cast<const uint8_t*>(w + 4 * s)))));
  return static_cast<int32_t>(vaddlvq_u16(acc));
}

class BConv2DNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { plan_.build(ctx); }
  size_t prepacked_bytes() const override { return plan_.prepacked_bytes(); }
  void run(const KernelContext& ctx) override {
    const uint32_t* pt = plan_.prepare_patches(ctx, opt_f32(ctx, 4));
    const float* mul = opt_f32(ctx, 2);
    const float* add = opt_f32(ctx, 3);
    const BConvPlan& p = plan_;
    const int64_t pixels = p.g.n * p.g.oh * p.g.ow;
    const int64_t kwp = p.k_words_pad, steps = kwp / 4, oc = p.g.oc;
    const int32_t kbits = static_cast<int32_t>(p.k_bits);
    const uint32_t* wts = p.weights();
    ctx.pool->parallel_for((pixels + 3) / 4, 4, [&](int64_t b0, int64_t b1) {
      for (int64_t blk = b0; blk < b1; ++blk) {
        int64_t q0 = blk * 4;
        int nvalid = static_cast<int>(std::min<int64_t>(4, pixels - q0));
        const uint32_t* xr[4];
        bool interior = true;
        for (int i = 0; i < 4; ++i) {
          int64_t q = q0 + std::min(i, nvalid - 1);
          xr[i] = pt + q * kwp;
          if (i < nvalid) interior = interior && p.pixel_is_interior(q);
        }
        int64_t o = 0;
        for (; o + 4 <= oc; o += 4) {
          const uint32_t* wr[4] = {wts + o * kwp, wts + (o + 1) * kwp, wts + (o + 2) * kwp, wts + (o + 3) * kwp};
          int32_t pc[4][4];
          popcount_4x4(xr, wr, steps, pc);
          for (int i = 0; i < nvalid; ++i)
            for (int j = 0; j < 4; ++j) {
              int32_t d = kbits - 2 * pc[i][j];
              if (!interior) d += p.pad_correction(q0 + i, o + j);
              p.store(ctx, q0 + i, o + j, d, mul, add);
            }
        }
        for (; o < oc; ++o)
          for (int i = 0; i < nvalid; ++i) {
            int32_t d = kbits - 2 * popcount_1x1(xr[i], wts + o * kwp, steps);
            if (!interior) d += p.pad_correction(q0 + i, o);
            p.store(ctx, q0 + i, o, d, mul, add);
          }
      }
    });
  }

 private:
  BConvPlan plan_;
};

bool bconv_neon_supported(const KernelContext& ctx) {
  if (has_epilogue(ctx)) return false;  // no fused-epilogue support (bconv2d.neon_opt has it)
  if (!bconv_packed_supported(ctx)) return false;
  const Shape& w = ctx.in(1).shape();
  int64_t kwp = (w[1] * w[2] * bit_words(w[3]) + 3) / 4 * 4;
  return kwp / 4 <= kMaxSteps;
}

}  // namespace

void register_binary_neon_kernels(OpRegistry& reg) {
  reg.add_kernel({"BConv2D", "bconv2d.neon", "neon", 30, kFeatNeon, bconv_neon_supported, make_kernel<BConv2DNeon>});
}

#else

void register_binary_neon_kernels(OpRegistry&) {}

#endif

}  // namespace bitino

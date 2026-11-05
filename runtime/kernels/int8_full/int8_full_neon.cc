// NEON kernels of the full-integer I8* operators (tier int8_full_neon).
//
// Every kernel is bit-identical to the int8_full_reference kernel of the same
// operator (tests/cpp/test_int8_full.cc); contracts in docs/int8.md
// ("Full-integer execution: int8_full").
//
//  * Fixed-point requantization uses SQSHL (saturating left shift), SQRDMULH
//    and SRSHL by a negative amount: exactly q8::requant (q8_math.h).
//  * fp32 epilogues use separate FMUL / FADD (no FMLA) and FCVTNS after the
//    same operations as the reference; clamping the converted integer equals
//    clamping in float first (q8::round_f32) for finite values.
//  * Binary layers (I8BDense, I8BConv2D, I8BMatMul) run the NEON binary kernels
//    (bdense.neon, bdense.ternary_neon, bconv2d.neon_opt, bmatmul.neon) on the
//    exact integer binarization of the int8 input: ge/gt inputs are packed into
//    bits here, sign_zero BDense and BMatMul operands are read as int8 codes by
//    the binary kernels (given the integer thresholds), and the binary kernels
//    store the int8 output clamp(rne(v)) of their unchanged fp32 value v.
//  * I8Conv2D / I8Dense run an int32 SDOT GEMM (own micro-kernel, int8_neon tile
//    layout) on raw codes, then subtract x_zero * sum_k w[o] and requantize;
//    taken whenever the int32 accumulator is provably exact (see
//    gemm_sdot_supported), incl. grouped convolutions.
#include "ops/op_defs.h"

#if defined(__aarch64__)
#include <arm_neon.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "kernels/int8_full/int8_full.h"
#include "kernels/int8_full/l8_neon.h"
#include "kernels/int8_full/q8_math.h"
#include "kernels/int8_neon/int8_gemm.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"
#endif

namespace bitino {

#if defined(__aarch64__)

// Binary NEON kernels (binary_neon/bconv_neon_opt.cc, binary_scalar/bmatmul_bitplane.cc).
std::unique_ptr<Kernel> make_bconv2d_neon_opt_kernel();
std::unique_ptr<Kernel> make_bdense_neon_kernel();
std::unique_ptr<Kernel> make_bdense_ternary_neon_kernel();
std::unique_ptr<Kernel> make_bmatmul_neon_kernel();

// SMMLA tiles (int8_full_gemm_i8mm.cc, +i8mm TU; called only with kFeatI8mm)
namespace int8_full_i8mm {
void tiles_i32(const int8_t* a, const int8_t* w, int64_t kc8, int64_t nob, int32_t* y, int64_t ldy, int nr);
void pack_tile_rows(const int8_t* const rp[8], int64_t kp, int8_t* dst);
}  // namespace int8_full_i8mm

namespace {

using q8::clamp8;

// ------------------------------------------------------------ vector helpers
// q8::requant on 4 lanes: rshr(mulhi(sat32(x << l), M), r).
inline int32x4_t requant4(int32x4_t x, int32x4_t m, int32x4_t lsh, int32x4_t neg_r) {
  return vrshlq_s32(vqrdmulhq_s32(vqshlq_s32(x, lsh), m), neg_r);
}

struct Rq {  // scalar multiplier broadcast
  int32x4_t m, l, nr;
  Rq(int32_t mult, int32_t shift)
      : m(vdupq_n_s32(mult)), l(vdupq_n_s32(shift > 0 ? shift : 0)), nr(vdupq_n_s32(shift < 0 ? shift : 0)) {}
  int32x4_t operator()(int32x4_t x) const { return requant4(x, m, l, nr); }
};

// 16 int32 lanes (4 vectors) + zero point -> clamp -> 16 int8.
inline int8x16_t narrow16(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d, int16x8_t lo, int16x8_t hi) {
  int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
  int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
  ab = vminq_s16(vmaxq_s16(ab, lo), hi);
  cd = vminq_s16(vmaxq_s16(cd, lo), hi);
  return vcombine_s8(vmovn_s16(ab), vmovn_s16(cd));
}

// fp32 -> int32 round half to even (FCVTNS); with a saturating narrow and a clamp this is q8::round_f32 for finite values.
inline int32x4_t rne4(float32x4_t v) { return vcvtnq_s32_f32(v); }

// Widening loads of 16 int8 into 4 int32x4.
inline void widen16(int8x16_t q, int32x4_t* w) {
  const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_s8(vget_high_s8(q));
  w[0] = vmovl_s16(vget_low_s16(l));
  w[1] = vmovl_s16(vget_high_s16(l));
  w[2] = vmovl_s16(vget_low_s16(h));
  w[3] = vmovl_s16(vget_high_s16(h));
}

constexpr int64_t kSerial = 64 * 1024;  // elements below which elementwise kernels stay on the calling thread

// ------------------------------------------------------------- I8Quantize
class QuantizeNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const float s = static_cast<float>(ctx.attrs().get_number("scale", 1));
    const float z = static_cast<float>(ctx.attrs().get_int("zero", 0));
    const float* x = ctx.in(0).data<float>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t n = ctx.in(0).size();
    ctx.pool->parallel_for((n + 15) / 16, kSerial / 16, [x, y, n, s, z](int64_t b0, int64_t b1) {  // by value: no reloads after stores
      const float32x4_t vs = vdupq_n_f32(s), vz = vdupq_n_f32(z), lo = vdupq_n_f32(-128.f), hi = vdupq_n_f32(127.f);
      auto one = [&](float32x4_t v) {
        v = vaddq_f32(vrndnq_f32(vdivq_f32(v, vs)), vz);  // nearbyint(x / s) + z
        return vcvtq_s32_f32(vminq_f32(vmaxq_f32(v, lo), hi));
      };
      const int16x8_t l16 = vdupq_n_s16(-128), h16 = vdupq_n_s16(127);
      for (int64_t b = b0; b < b1; ++b) {
        const int64_t i = b * 16;
        if (i + 16 <= n) {
          vst1q_s8(y + i, narrow16(one(vld1q_f32(x + i)), one(vld1q_f32(x + i + 4)), one(vld1q_f32(x + i + 8)),
                                   one(vld1q_f32(x + i + 12)), l16, h16));
        } else {
          for (int64_t j = i; j < n; ++j) y[j] = int8_full::quantize1(x[j], s, z);
        }
      }
    });
  }
};

// ------------------------------------------------------------------ I8Lut
// One 256-entry table: TBL/TBX over four 64-byte quarters, index = q + 128.
class LutNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const int8_t* q = ctx.in(0).data<int8_t>();
    const auto* tab = reinterpret_cast<const uint8_t*>(ctx.in(1).data<int8_t>());
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t n = ctx.in(0).size();
    ctx.pool->parallel_for((n + 15) / 16, kSerial / 16, [q, tab, y, n](int64_t b0, int64_t b1) {
      // tables loaded here (not captured by reference): they stay in registers over the loop
      const uint8x16x4_t t0 = vld1q_u8_x4(tab), t1 = vld1q_u8_x4(tab + 64), t2 = vld1q_u8_x4(tab + 128), t3 = vld1q_u8_x4(tab + 192);
      const uint8x16_t k64 = vdupq_n_u8(64), k80 = vdupq_n_u8(0x80);
      for (int64_t b = b0; b < b1; ++b) {
        const int64_t i = b * 16;
        if (i + 16 <= n) {
          const uint8x16_t idx = veorq_u8(vreinterpretq_u8_s8(vld1q_s8(q + i)), k80);
          uint8x16_t r = vqtbl4q_u8(t0, idx);
          r = vqtbx4q_u8(r, t1, vsubq_u8(idx, k64));
          r = vqtbx4q_u8(r, t2, vsubq_u8(idx, vdupq_n_u8(128)));
          r = vqtbx4q_u8(r, t3, vsubq_u8(idx, vdupq_n_u8(192)));
          vst1q_s8(y + i, vreinterpretq_s8_u8(r));
        } else {
          for (int64_t j = i; j < n; ++j) y[j] = static_cast<int8_t>(tab[q[j] + 128]);
        }
      }
    });
  }
};
bool lut_neon_supported(const KernelContext& ctx) { return ctx.in(1).dtype() == DType::kI8 && ctx.in(1).dim(0) == 1; }

// ------------------------------------------------------------------ I8Add
// TFLite Add: xa = requant((qa - za) << ls, am, as), xb likewise, y = clamp(zy + requant(xa + xb, om, os)).
// The input multipliers have no left shift (add_neon_supported), so their SQSHL is skipped; a scalar b
// (one element broadcast to every element) has its xb computed once.
class AddNeon : public Kernel {
 public:
  // b of a's shape, one row of the last dimension broadcast over every row (brow_ = row length, a
  // multiple of 16 so that a 16-element block never straddles two rows), or one element (bscalar_)
  void prepare(const KernelContext& ctx) override {
    bscalar_ = ctx.in(1).size() == 1 && ctx.in(0).size() > 1;
    brow_ = ctx.in(0).shape() == ctx.in(1).shape() || bscalar_ ? 0 : ctx.in(1).size();
  }
  void run(const KernelContext& ctx) override {
    const Json& a = ctx.attrs();
    const bool olsh = a.get_int("out_shift", 0) > 0, as = a.get_int("a_shift", 0) != 0, bs = a.get_int("b_shift", 0) != 0;
    const bool full = a.get_int("qmin", -128) == -128 && a.get_int("qmax", 127) == 127;
    // Same shapes, full output range, at most one input rescaled by a shift (TFLite: the larger scale gets shift 0):
    // the specialized loop; everything else the general one.
    if (!bscalar_ && !brow_ && full && !(as && bs)) {
      if (olsh) {
        if (as) run_same<true, true, false>(ctx);
        else if (bs) run_same<true, false, true>(ctx);
        else run_same<true, false, false>(ctx);
      } else {
        if (as) run_same<false, true, false>(ctx);
        else if (bs) run_same<false, false, true>(ctx);
        else run_same<false, false, false>(ctx);
      }
      return;
    }
    if (olsh) run_t<true>(ctx);
    else run_t<false>(ctx);
  }

 private:
  // Same-shape inputs, qmin -128 / qmax 127. The same integer operations per element as run_t: (q - z) in int16
  // lanes (|q - z| <= 255) widened to int32, << left_shift, SQRDMULH, SRSHL by the input shift (skipped when it is
  // 0: a rounding shift by 0 is the identity), sum, output requantization, saturating narrows (the clamp to
  // [-128, 127] is the saturation of the last narrow).
  template <bool kOutLsh, bool kAs, bool kBs>
  void run_same(const KernelContext& ctx) const {
    const Json& a = ctx.attrs();
    const int32_t za = static_cast<int32_t>(a.get_int("a_zero", 0)), zb = static_cast<int32_t>(a.get_int("b_zero", 0));
    const int32_t zy = static_cast<int32_t>(a.get_int("y_zero", 0));
    const int ls = static_cast<int>(a.get_int("left_shift", 20));
    const int32_t am = static_cast<int32_t>(a.get_int("a_mult", 0)), as = static_cast<int32_t>(a.get_int("a_shift", 0));
    const int32_t bm = static_cast<int32_t>(a.get_int("b_mult", 0)), bs = static_cast<int32_t>(a.get_int("b_shift", 0));
    const int32_t om = static_cast<int32_t>(a.get_int("out_mult", 0)), os = static_cast<int32_t>(a.get_int("out_shift", 0));
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t n = ctx.in(0).size();
    ctx.pool->parallel_for((n + 15) / 16, kSerial / 16, [=](int64_t b0, int64_t b1) {
      const int32x4_t mam = vdupq_n_s32(am), mbm = vdupq_n_s32(bm), nra = vdupq_n_s32(std::min(as, 0)), nrb = vdupq_n_s32(std::min(bs, 0));
      const Rq ro(om, os);
      const int8x8_t za8 = vdup_n_s8(static_cast<int8_t>(za)), zb8 = vdup_n_s8(static_cast<int8_t>(zb));
      const int8x16_t za16 = vdupq_n_s8(static_cast<int8_t>(za)), zb16 = vdupq_n_s8(static_cast<int8_t>(zb));
      const int32x4_t vzy = vdupq_n_s32(zy), vls = vdupq_n_s32(ls);
      auto term = [&](int16x8_t d, int k, int32x4_t m, int32x4_t nr, bool sh) {
        const int32x4_t w = (k & 1) ? vmovl_high_s16(d) : vmovl_s16(vget_low_s16(d));
        const int32x4_t t = vqrdmulhq_s32(vshlq_s32(w, vls), m);
        return sh ? vrshlq_s32(t, nr) : t;
      };
      for (int64_t b = b0; b < b1; ++b) {
        const int64_t i = b * 16;
        if (i + 16 > n) {
          for (int64_t j = i; j < n; ++j) {
            const int64_t xa = q8::requant((qa[j] - za) * (int64_t(1) << ls), am, as);
            const int64_t xb = q8::requant((qb[j] - zb) * (int64_t(1) << ls), bm, bs);
            y[j] = clamp8(zy + q8::requant(xa + xb, om, os), -128, 127);
          }
          continue;
        }
        // (q - z) as int16: vsubl of the int8 codes and the int8 zero point (in range: add_neon_supported)
        const int8x16_t va = vld1q_s8(qa + i), vb = vld1q_s8(qb + i);
        const int16x8_t da[2] = {vsubl_s8(vget_low_s8(va), za8), vsubl_high_s8(va, za16)};
        const int16x8_t db[2] = {vsubl_s8(vget_low_s8(vb), zb8), vsubl_high_s8(vb, zb16)};
        int32x4_t r[4];
        for (int k = 0; k < 4; ++k) {
          const int32x4_t s = vaddq_s32(term(da[k >> 1], k, mam, nra, kAs), term(db[k >> 1], k, mbm, nrb, kBs));
          r[k] = vqaddq_s32(kOutLsh ? ro(s) : vrshlq_s32(vqrdmulhq_s32(s, ro.m), ro.nr), vzy);
        }
        const int16x8_t ab = vqmovn_high_s32(vqmovn_s32(r[0]), r[1]), cd = vqmovn_high_s32(vqmovn_s32(r[2]), r[3]);
        vst1q_s8(y + i, vqmovn_high_s16(vqmovn_s16(ab), cd));
      }
    });
  }

  template <bool kOutLsh>
  void run_t(const KernelContext& ctx) const {
    const Json& a = ctx.attrs();
    const int32_t za = static_cast<int32_t>(a.get_int("a_zero", 0)), zb = static_cast<int32_t>(a.get_int("b_zero", 0));
    const int32_t zy = static_cast<int32_t>(a.get_int("y_zero", 0));
    const int ls = static_cast<int>(a.get_int("left_shift", 20));
    const int32_t am = static_cast<int32_t>(a.get_int("a_mult", 0)), as = static_cast<int32_t>(a.get_int("a_shift", 0));
    const int32_t bm = static_cast<int32_t>(a.get_int("b_mult", 0)), bs = static_cast<int32_t>(a.get_int("b_shift", 0));
    const int32_t om = static_cast<int32_t>(a.get_int("out_mult", 0)), os = static_cast<int32_t>(a.get_int("out_shift", 0));
    const int lo = static_cast<int>(a.get_int("qmin", -128)), hi = static_cast<int>(a.get_int("qmax", 127));
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t n = ctx.in(0).size();
    const bool bsc = bscalar_;
    const int64_t brow = brow_;
    // scalar b: xb is the same for every element
    const int32_t xb_s = bsc ? static_cast<int32_t>(q8::requant((qb[0] - zb) * (int64_t(1) << ls), bm, bs)) : 0;
    ctx.pool->parallel_for((n + 15) / 16, kSerial / 16, [&](int64_t b0, int64_t b1) {
      const int32x4_t mam = vdupq_n_s32(am), mbm = vdupq_n_s32(bm), nra = vdupq_n_s32(std::min(as, 0)), nrb = vdupq_n_s32(std::min(bs, 0));
      const Rq ro(om, os);
      const int32x4_t vza = vdupq_n_s32(za), vzb = vdupq_n_s32(zb), vzy = vdupq_n_s32(zy), vls = vdupq_n_s32(ls), vxb = vdupq_n_s32(xb_s);
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      for (int64_t b = b0; b < b1; ++b) {
        const int64_t i = b * 16;
        if (i + 16 > n) {
          for (int64_t j = i; j < n; ++j) {
            const int64_t xa = q8::requant((qa[j] - za) * (int64_t(1) << ls), am, as);
            const int64_t xb = bsc ? xb_s : q8::requant((qb[brow ? j % brow : j] - zb) * (int64_t(1) << ls), bm, bs);
            y[j] = clamp8(zy + q8::requant(xa + xb, om, os), lo, hi);
          }
          continue;
        }
        int32x4_t wa[4], wb[4], r[4];
        widen16(vld1q_s8(qa + i), wa);
        if (!bsc) widen16(vld1q_s8(qb + (brow ? i % brow : i)), wb);
        for (int k = 0; k < 4; ++k) {
          // |q - z| << 20 fits int32 (|q - z| <= 255 for in-range zero points), as the reference's
          // int64 product before sat32 (requant applies no further left shift for these multipliers).
          const int32x4_t xa = vrshlq_s32(vqrdmulhq_s32(vshlq_s32(vsubq_s32(wa[k], vza), vls), mam), nra);
          const int32x4_t xb = bsc ? vxb : vrshlq_s32(vqrdmulhq_s32(vshlq_s32(vsubq_s32(wb[k], vzb), vls), mbm), nrb);
          const int32x4_t s = vaddq_s32(xa, xb);
          r[k] = vqaddq_s32(kOutLsh ? ro(s) : vrshlq_s32(vqrdmulhq_s32(s, ro.m), ro.nr), vzy);
        }
        vst1q_s8(y + i, narrow16(r[0], r[1], r[2], r[3], vlo, vhi));
      }
    });
  }

  int64_t brow_ = 0;
  bool bscalar_ = false;
};
// Same shapes, a broadcast row, or a scalar b; zero points within int8 (so (q - z) << left_shift cannot
// overflow int32) and no left shift in the input multipliers (TFLite Add: real multipliers <= 1/2).
bool add_neon_supported(const KernelContext& ctx) {
  const Json& a = ctx.attrs();
  auto in8 = [](int64_t z) { return z >= -128 && z <= 127; };
  const Tensor& ta = ctx.in(0);
  const Tensor& tb = ctx.in(1);
  const bool row = tb.size() == ta.dim(-1) && tb.dim(-1) == ta.dim(-1) && ta.dim(-1) % 16 == 0;
  return (ta.shape() == tb.shape() || row || tb.size() == 1) && in8(a.get_int("a_zero", 0)) && in8(a.get_int("b_zero", 0)) &&
         a.get_int("left_shift", 20) <= 22 && a.get_int("a_shift", 0) <= 0 && a.get_int("b_shift", 0) <= 0;
}

// ------------------------------------------------------------ I8AffineSum
// Rows of C channels; 16 channels per step: acc = B[c] + sum_i (q_i - z_i) * K[i][c] in int32 lanes (MLA),
// then requantized. The quantizer bounds |acc| < 2^31 (N*255*max|K| <= 2^28, |B| <= 2^30).
// Fast path, when prepare() proves |B[c]| + sum_i |K[i][c]| (128 + |z_i|) < 2^31 for every channel (so the
// exact acc and every reordering of its integer sum fit int32): the zero points are folded into the bias,
// B'[c] = B[c] - sum_i z_i K[i][c], and acc = B'[c] + sum_i q_i K[i][c]; inputs whose coefficients all fit
// int16 use widening SMLAL on int16 codes (no int32 widening of q), the others int32 MLA. Requantization
// skips SQSHL for a non-positive shift and the clamp for the full [-128, 127] range (saturating narrows).
class AffineSumNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    nin_ = ctx.inputs.size() - 2;
    const Tensor& k = ctx.in(nin_);
    c_ = ctx.in(0).dim(-1);
    // expand K and B to full channel vectors (one copy per input), padded to a multiple of 16
    cp_ = (c_ + 15) / 16 * 16;
    kfull_.assign(nin_ * cp_, 0);
    bfull_.assign(cp_, 0);
    const int32_t* K = k.data<int32_t>();
    const int32_t* B = ctx.in(nin_ + 1).data<int32_t>();
    const int64_t kc = k.dim(1);
    for (size_t i = 0; i < nin_; ++i)
      for (int64_t ch = 0; ch < c_; ++ch) kfull_[i * cp_ + ch] = K[i * kc + (kc == 1 ? 0 : ch)];
    for (int64_t ch = 0; ch < c_; ++ch) bfull_[ch] = B[kc == 1 ? 0 : ch];
    zs_ = ctx.attrs().get_ints("zeros", {});
    // fast path: exactness bound, folded bias, int16 coefficient rows
    fast_ = true;
    bfold_.assign(cp_, 0);
    for (int64_t ch = 0; ch < c_ && fast_; ++ch) {
      int64_t bound = std::abs(static_cast<int64_t>(bfull_[ch])), b = bfull_[ch];
      for (size_t i = 0; i < nin_; ++i) {
        bound += std::abs(static_cast<int64_t>(kfull_[i * cp_ + ch])) * (128 + std::abs(zs_[i]));
        b -= zs_[i] * static_cast<int64_t>(kfull_[i * cp_ + ch]);
      }
      fast_ = bound < (int64_t(1) << 31);
      bfold_[ch] = static_cast<int32_t>(b);
    }
    // inputs whose coefficients all fit int16 first (order_), their rows in kh_, the others' rows in k32_
    order_.clear();
    n16_ = 0;
    for (int pass = 0; pass < 2; ++pass)
      for (size_t i = 0; i < nin_; ++i) {
        bool fits = true;
        for (int64_t ch = 0; ch < c_; ++ch) fits = fits && kfull_[i * cp_ + ch] >= -32768 && kfull_[i * cp_ + ch] <= 32767;
        if (fits == (pass == 0)) order_.push_back(i);
        if (pass == 0 && fits) ++n16_;
      }
    kh_.assign(n16_ * cp_, 0);
    k32_.assign((nin_ - n16_) * cp_, 0);
    for (size_t j = 0; j < nin_; ++j)
      for (int64_t ch = 0; ch < c_; ++ch) {
        const int32_t v = kfull_[order_[j] * cp_ + ch];
        if (j < n16_) kh_[j * cp_ + ch] = static_cast<int16_t>(v);
        else k32_[(j - n16_) * cp_ + ch] = v;
      }
  }
  void run(const KernelContext& ctx) override {
    const int32_t s = static_cast<int32_t>(ctx.attrs().get_int("shift", 0));
    const bool full = ctx.attrs().get_int("qmin", -128) == -128 && ctx.attrs().get_int("qmax", 127) == 127;
    if (!fast_) return run_general(ctx);
    // 2 or 4 inputs (nearly every I8AffineSum of the zoo): coefficients and bias in registers, rows inner
#define BF_AFS(N16, N32)                                  \
  do {                                                    \
    if (s > 0) {                                          \
      if (full) run_tiled<N16, N32, true, true>(ctx);     \
      else run_tiled<N16, N32, true, false>(ctx);         \
    } else {                                              \
      if (full) run_tiled<N16, N32, false, true>(ctx);    \
      else run_tiled<N16, N32, false, false>(ctx);        \
    }                                                     \
    return;                                               \
  } while (0)
    if (nin_ == 2 && n16_ == 0) BF_AFS(0, 2);
    if (nin_ == 2 && n16_ == 1) BF_AFS(1, 1);
    if (nin_ == 2 && n16_ == 2) BF_AFS(2, 0);
    if (nin_ == 4 && n16_ == 0) BF_AFS(0, 4);
#undef BF_AFS
    if (s > 0) {
      if (full) run_fast<true, true>(ctx);
      else run_fast<true, false>(ctx);
    } else {
      if (full) run_fast<false, true>(ctx);
      else run_fast<false, false>(ctx);
    }
  }

 private:
  // run_fast with N16 int16-coefficient inputs and N32 int32-coefficient inputs (order_) fixed at compile time:
  // per tile of 32 rows and 16-channel block the folded bias and the coefficient rows are loaded once, then every
  // row of the tile runs from registers. Same integer operations per element as run_fast (bias, then the inputs
  // in order_ order; exact int32 sums by the fast-path bound), same requantization and narrowing.
  template <int N16, int N32, bool kLsh, bool kFull>
  void run_tiled(const KernelContext& ctx) const {
    constexpr int N = N16 + N32;
    const int64_t rows = ctx.in(0).size() / c_;
    const int32_t m = static_cast<int32_t>(ctx.attrs().get_int("mult", 0)), s = static_cast<int32_t>(ctx.attrs().get_int("shift", 0));
    const int32_t zy = static_cast<int32_t>(ctx.attrs().get_int("y_zero", 0));
    const int lo = static_cast<int>(ctx.attrs().get_int("qmin", -128)), hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    const int8_t* xp[N];
    for (int j = 0; j < N; ++j) xp[j] = ctx.in(order_[j]).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t c = c_, cp = cp_;
    const int32_t* bf = bfold_.data();
    const int32_t* k32 = k32_.data();
    const int16_t* kh = kh_.data();
    constexpr int64_t kRt = 32;  // rows per tile
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, kSerial / std::max<int64_t>(c * N, 1)), [=, &ctx](int64_t r0, int64_t r1) {
      const Rq rq(m, s);
      const int32x4_t vzy = vdupq_n_s32(zy);
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      for (int64_t rt = r0; rt < r1; rt += kRt) {
        const int64_t re = std::min(r1, rt + kRt);
        int64_t ch = 0;
        for (; ch + 16 <= c; ch += 16) {
          int32x4_t b[4];
          for (int k = 0; k < 4; ++k) b[k] = vld1q_s32(bf + ch + 4 * k);
          int16x8_t ka[N16 > 0 ? N16 : 1], kb[N16 > 0 ? N16 : 1];
          for (int i = 0; i < N16; ++i) {
            ka[i] = vld1q_s16(kh + i * cp + ch);
            kb[i] = vld1q_s16(kh + i * cp + ch + 8);
          }
          int32x4_t kw[N32 > 0 ? N32 : 1][4];
          for (int i = 0; i < N32; ++i)
            for (int k = 0; k < 4; ++k) kw[i][k] = vld1q_s32(k32 + i * cp + ch + 4 * k);
          for (int64_t r = rt; r < re; ++r) {
            const int64_t off = r * c + ch;
            int32x4_t acc[4] = {b[0], b[1], b[2], b[3]};
            for (int i = 0; i < N16; ++i) {
              const int8x16_t q = vld1q_s8(xp[i] + off);
              const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
              acc[0] = vmlal_s16(acc[0], vget_low_s16(l), vget_low_s16(ka[i]));
              acc[1] = vmlal_high_s16(acc[1], l, ka[i]);
              acc[2] = vmlal_s16(acc[2], vget_low_s16(h), vget_low_s16(kb[i]));
              acc[3] = vmlal_high_s16(acc[3], h, kb[i]);
            }
            for (int i = 0; i < N32; ++i) {
              const int8x16_t q = vld1q_s8(xp[N16 + i] + off);
              const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
              acc[0] = vmlaq_s32(acc[0], vmovl_s16(vget_low_s16(l)), kw[i][0]);
              acc[1] = vmlaq_s32(acc[1], vmovl_high_s16(l), kw[i][1]);
              acc[2] = vmlaq_s32(acc[2], vmovl_s16(vget_low_s16(h)), kw[i][2]);
              acc[3] = vmlaq_s32(acc[3], vmovl_high_s16(h), kw[i][3]);
            }
            for (int k = 0; k < 4; ++k) acc[k] = vqaddq_s32(kLsh ? rq(acc[k]) : vrshlq_s32(vqrdmulhq_s32(acc[k], rq.m), rq.nr), vzy);
            if (kFull) {
              const int16x8_t ab = vqmovn_high_s32(vqmovn_s32(acc[0]), acc[1]), cd = vqmovn_high_s32(vqmovn_s32(acc[2]), acc[3]);
              vst1q_s8(y + off, vqmovn_high_s16(vqmovn_s16(ab), cd));
            } else {
              vst1q_s8(y + off, narrow16(acc[0], acc[1], acc[2], acc[3], vlo, vhi));
            }
          }
        }
        for (; ch < c; ++ch)
          for (int64_t r = rt; r < re; ++r) {
            const int64_t base = r * c;
            int64_t acc = bfull_[ch];
            for (size_t i = 0; i < nin_; ++i) acc += (ctx.in(i).data<int8_t>()[base + ch] - zs_[i]) * static_cast<int64_t>(kfull_[i * cp_ + ch]);
            y[base + ch] = clamp8(zy + q8::requant(acc, m, s), lo, hi);
          }
      }
    });
  }

  template <bool kLsh, bool kFull>
  void run_fast(const KernelContext& ctx) const {
    const int64_t rows = ctx.in(0).size() / c_;
    const int32_t m = static_cast<int32_t>(ctx.attrs().get_int("mult", 0)), s = static_cast<int32_t>(ctx.attrs().get_int("shift", 0));
    const int32_t zy = static_cast<int32_t>(ctx.attrs().get_int("y_zero", 0));
    const int lo = static_cast<int>(ctx.attrs().get_int("qmin", -128)), hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    std::vector<const int8_t*> x(nin_), xo(nin_);
    for (size_t i = 0; i < nin_; ++i) x[i] = ctx.in(i).data<int8_t>();
    for (size_t j = 0; j < nin_; ++j) xo[j] = x[order_[j]];
    int8_t* y = ctx.out(0).data<int8_t>();
    const size_t nin = nin_, n16 = n16_;
    const int64_t c = c_, cp = cp_;
    const int32_t* bf = bfold_.data();
    const int32_t* k32 = k32_.data();
    const int16_t* kh = kh_.data();
    const int8_t* const* xp = xo.data();
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, kSerial / std::max<int64_t>(c * static_cast<int64_t>(nin), 1)), [&](int64_t r0, int64_t r1) {
      const Rq rq(m, s);
      const int32x4_t vzy = vdupq_n_s32(zy);
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t base = r * c;
        int64_t ch = 0;
        for (; ch + 16 <= c; ch += 16) {
          const int64_t off = base + ch;
          int32x4_t acc[4];
          for (int k = 0; k < 4; ++k) acc[k] = vld1q_s32(bf + ch + 4 * k);
          const int16_t* kr16 = kh + ch;
          for (size_t i = 0; i < n16; ++i, kr16 += cp) {
            const int8x16_t q = vld1q_s8(xp[i] + off);
            const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
            const int16x8_t ka = vld1q_s16(kr16), kb = vld1q_s16(kr16 + 8);
            acc[0] = vmlal_s16(acc[0], vget_low_s16(l), vget_low_s16(ka));
            acc[1] = vmlal_high_s16(acc[1], l, ka);
            acc[2] = vmlal_s16(acc[2], vget_low_s16(h), vget_low_s16(kb));
            acc[3] = vmlal_high_s16(acc[3], h, kb);
          }
          const int32_t* kr = k32 + ch;
          for (size_t i = n16; i < nin; ++i, kr += cp) {
            const int8x16_t q = vld1q_s8(xp[i] + off);
            const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
            acc[0] = vmlaq_s32(acc[0], vmovl_s16(vget_low_s16(l)), vld1q_s32(kr));
            acc[1] = vmlaq_s32(acc[1], vmovl_high_s16(l), vld1q_s32(kr + 4));
            acc[2] = vmlaq_s32(acc[2], vmovl_s16(vget_low_s16(h)), vld1q_s32(kr + 8));
            acc[3] = vmlaq_s32(acc[3], vmovl_high_s16(h), vld1q_s32(kr + 12));
          }
          for (int k = 0; k < 4; ++k) acc[k] = vqaddq_s32(kLsh ? rq(acc[k]) : vrshlq_s32(vqrdmulhq_s32(acc[k], rq.m), rq.nr), vzy);
          if (kFull) {
            const int16x8_t ab = vqmovn_high_s32(vqmovn_s32(acc[0]), acc[1]), cd = vqmovn_high_s32(vqmovn_s32(acc[2]), acc[3]);
            vst1q_s8(y + base + ch, vqmovn_high_s16(vqmovn_s16(ab), cd));
          } else {
            vst1q_s8(y + base + ch, narrow16(acc[0], acc[1], acc[2], acc[3], vlo, vhi));
          }
        }
        for (; ch < c; ++ch) {
          int64_t acc = bfull_[ch];
          for (size_t i = 0; i < nin; ++i) acc += (x[i][base + ch] - zs_[i]) * static_cast<int64_t>(kfull_[i * cp_ + ch]);
          y[base + ch] = clamp8(zy + q8::requant(acc, m, s), lo, hi);
        }
      }
    });
  }

  void run_general(const KernelContext& ctx) const {
    const int64_t rows = ctx.in(0).size() / c_;
    const int32_t m = static_cast<int32_t>(ctx.attrs().get_int("mult", 0)), s = static_cast<int32_t>(ctx.attrs().get_int("shift", 0));
    const int32_t zy = static_cast<int32_t>(ctx.attrs().get_int("y_zero", 0));
    const int lo = static_cast<int>(ctx.attrs().get_int("qmin", -128)), hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    std::vector<const int8_t*> x(nin_);
    for (size_t i = 0; i < nin_; ++i) x[i] = ctx.in(i).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, kSerial / std::max<int64_t>(c_ * static_cast<int64_t>(nin_), 1)), [&](int64_t r0, int64_t r1) {
      const Rq rq(m, s);
      const int32x4_t vzy = vdupq_n_s32(zy);
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t base = r * c_;
        int64_t ch = 0;
        for (; ch + 16 <= c_; ch += 16) {
          int32x4_t acc[4];
          for (int k = 0; k < 4; ++k) acc[k] = vld1q_s32(bfull_.data() + ch + 4 * k);
          for (size_t i = 0; i < nin_; ++i) {
            const int16x8_t z = vdupq_n_s16(static_cast<int16_t>(zs_[i]));
            const int8x16_t q = vld1q_s8(x[i] + base + ch);
            const int16x8_t l = vsubq_s16(vmovl_s8(vget_low_s8(q)), z), h = vsubq_s16(vmovl_s8(vget_high_s8(q)), z);
            const int32_t* kr = kfull_.data() + i * cp_ + ch;
            acc[0] = vmlaq_s32(acc[0], vmovl_s16(vget_low_s16(l)), vld1q_s32(kr));
            acc[1] = vmlaq_s32(acc[1], vmovl_s16(vget_high_s16(l)), vld1q_s32(kr + 4));
            acc[2] = vmlaq_s32(acc[2], vmovl_s16(vget_low_s16(h)), vld1q_s32(kr + 8));
            acc[3] = vmlaq_s32(acc[3], vmovl_s16(vget_high_s16(h)), vld1q_s32(kr + 12));
          }
          for (int k = 0; k < 4; ++k) acc[k] = vqaddq_s32(rq(acc[k]), vzy);
          vst1q_s8(y + base + ch, narrow16(acc[0], acc[1], acc[2], acc[3], vlo, vhi));
        }
        for (; ch < c_; ++ch) {
          int64_t acc = bfull_[ch];
          for (size_t i = 0; i < nin_; ++i) acc += (x[i][base + ch] - zs_[i]) * static_cast<int64_t>(kfull_[i * cp_ + ch]);
          y[base + ch] = clamp8(zy + q8::requant(acc, m, s), lo, hi);
        }
      }
    });
  }

  size_t nin_ = 0;
  int64_t c_ = 0, cp_ = 0;
  bool fast_ = false;
  std::vector<int32_t> kfull_, bfull_, bfold_;
  std::vector<int32_t> k32_;
  std::vector<int16_t> kh_;
  std::vector<size_t> order_;
  size_t n16_ = 0;
  std::vector<int64_t> zs_;
};
// zero points within int16 (q - z computed in int16 lanes)
bool affine_sum_neon_supported(const KernelContext& ctx) {
  for (auto z : ctx.attrs().get_ints("zeros", {}))
    if (z < -32000 || z > 32000) return false;
  return true;
}

// ------------------------------------------------------------ I8LayerNorm
class LayerNormNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t d = x.dim(-1), rows = x.size() / d;
    const int32_t xz = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    const double xs = ctx.attrs().get_number("x_scale", 1.0), eps = ctx.attrs().get_number("eps", 1e-5);
    const float* G = ctx.in(1).data<float>();
    const float* B = ctx.in(2).data<float>();
    const int lo = static_cast<int>(ctx.attrs().get_int("qmin", -128)), hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    const int8_t* q = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, 16384 / std::max<int64_t>(d, 1)), [&](int64_t r0, int64_t r1) {
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      for (int64_t r = r0; r < r1; ++r) {
        const int8_t* qr = q + r * d;
        // raw sums of q and q^2 (int32: d <= 16384, |q| <= 128)
        int32x4_t s1v = vdupq_n_s32(0), s2v = vdupq_n_s32(0);
        int64_t i = 0;
        for (; i + 16 <= d; i += 16) {
          const int8x16_t v = vld1q_s8(qr + i);
          const int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_s8(vget_high_s8(v));
          s1v = vpadalq_s16(s1v, l);
          s1v = vpadalq_s16(s1v, h);
          s2v = vmlal_s16(s2v, vget_low_s16(l), vget_low_s16(l));
          s2v = vmlal_s16(s2v, vget_high_s16(l), vget_high_s16(l));
          s2v = vmlal_s16(s2v, vget_low_s16(h), vget_low_s16(h));
          s2v = vmlal_s16(s2v, vget_high_s16(h), vget_high_s16(h));
        }
        int64_t sq = vaddvq_s32(s1v), sq2 = vaddvq_s32(s2v);
        for (; i < d; ++i) {
          sq += qr[i];
          sq2 += qr[i] * qr[i];
        }
        // S1 = sum (q - z), S2 = sum (q - z)^2
        const int64_t s1 = sq - d * xz, s2 = sq2 - 2 * int64_t(xz) * sq + d * int64_t(xz) * xz;
        const float rf = int8_full::layernorm_row_factor(d, s1, s2, xs, eps);
        // u = d*(q - z) - S1 = d*q - (S1 + d*z): exact in int32 and in fp32
        const int32x4_t vd = vdupq_n_s32(static_cast<int32_t>(d)), vc = vdupq_n_s32(static_cast<int32_t>(s1 + d * xz));
        const float32x4_t vr = vdupq_n_f32(rf);
        int8_t* yr = y + r * d;
        i = 0;
        for (; i + 16 <= d; i += 16) {
          int32x4_t w[4], o[4];
          widen16(vld1q_s8(qr + i), w);
          for (int k = 0; k < 4; ++k) {
            const float32x4_t u = vcvtq_f32_s32(vsubq_s32(vmulq_s32(w[k], vd), vc));
            const float32x4_t v = vmulq_f32(u, vr);
            const float32x4_t t = vaddq_f32(vmulq_f32(v, vld1q_f32(G + i + 4 * k)), vld1q_f32(B + i + 4 * k));
            o[k] = rne4(t);
          }
          vst1q_s8(yr + i, narrow16(o[0], o[1], o[2], o[3], vlo, vhi));
        }
        for (; i < d; ++i) {
          const float u = static_cast<float>(d * (qr[i] - xz) - s1);
          const float v = u * rf;
          const float w = v * G[i];
          yr[i] = q8::round_f32(w + B[i], lo, hi);
        }
      }
    });
  }
};

// -------------------------------------------------------------- I8Softmax
class SoftmaxNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const int32_t* tab = ctx.in(1).data<int32_t>();
    int64_t total = 0;
    planes_ = true;
    for (int d = 0; d < 256; ++d) {
      planes_ = planes_ && tab[d] >= 0 && tab[d] < (1 << 24);
      total = std::max<int64_t>(total, tab[d]);
    }
    // the vector row sum is uint32: N * max entry must stay below 2^31
    planes_ = planes_ && total * ctx.in(0).dim(-1) < (int64_t(1) << 31);
    if (!planes_) return;
    uint8_t b[3][256];
    for (int d = 0; d < 256; ++d)
      for (int k = 0; k < 3; ++k) b[k][d] = static_cast<uint8_t>((tab[d] >> (8 * k)) & 0xff);
    for (int qd = 0; qd < 4; ++qd) {
      p0_[qd] = vld1q_u8_x4(b[0] + 64 * qd);
      p1_[qd] = vld1q_u8_x4(b[1] + 64 * qd);
      p2_[qd] = vld1q_u8_x4(b[2] + 64 * qd);
    }
  }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t n = x.dim(-1), rows = x.size() / n;
    const int32_t* tab = ctx.in(1).data<int32_t>();
    const bool f32 = ctx.attrs().get_string("output", "i8") == "f32";
    const double num = f32 ? 1.0 : ctx.attrs().get_number("inv_scale", 1.0);
    const int32_t zy = static_cast<int32_t>(ctx.attrs().get_int("y_zero", 0));
    const int lo = static_cast<int>(ctx.attrs().get_int("qmin", -128)), hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    const int8_t* q = x.data<int8_t>();
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, 8192 / std::max<int64_t>(n, 1)), [&](int64_t r0, int64_t r1) {
      // per-thread row buffer of exp values (grows once per thread, never in steady state)
      thread_local std::vector<int32_t> ebuf;
      if (static_cast<int64_t>(ebuf.size()) < n) ebuf.resize(static_cast<size_t>(n));
      int32_t* e = ebuf.data();
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      const int32x4_t vzy = vdupq_n_s32(zy);
      for (int64_t r = r0; r < r1; ++r) {
        const int8_t* qr = q + r * n;
        int8x16_t mv = vdupq_n_s8(-128);
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) mv = vmaxq_s8(mv, vld1q_s8(qr + i));
        int m = vmaxvq_s8(mv);
        for (; i < n; ++i) m = std::max<int>(m, qr[i]);
        int64_t s = 0;
        i = 0;
        if (planes_) {
          // e = E[m - q] via three 256-byte planes of the table (entries < 2^24), TBL/TBX on 64-byte quarters
          const uint8x16_t vm = vreinterpretq_u8_s8(vdupq_n_s8(static_cast<int8_t>(m)));
          const uint8x16_t k64 = vdupq_n_u8(64), k128 = vdupq_n_u8(128), k192 = vdupq_n_u8(192);
          uint32x4_t vs = vdupq_n_u32(0);
          auto look = [&](const uint8x16x4_t* t, uint8x16_t d, uint8x16_t d1, uint8x16_t d2, uint8x16_t d3) {
            uint8x16_t r = vqtbl4q_u8(t[0], d);
            r = vqtbx4q_u8(r, t[1], d1);
            r = vqtbx4q_u8(r, t[2], d2);
            return vqtbx4q_u8(r, t[3], d3);
          };
          for (; i + 16 <= n; i += 16) {
            const uint8x16_t d = vsubq_u8(vm, vreinterpretq_u8_s8(vld1q_s8(qr + i)));  // m - q in [0, 255]
            const uint8x16_t d1 = vsubq_u8(d, k64), d2 = vsubq_u8(d, k128), d3 = vsubq_u8(d, k192);
            const uint8x16_t b0 = look(p0_, d, d1, d2, d3), b1 = look(p1_, d, d1, d2, d3), b2 = look(p2_, d, d1, d2, d3);
            const uint16x8_t z1 = vreinterpretq_u16_u8(vzip1q_u8(b0, b1)), z2 = vreinterpretq_u16_u8(vzip2q_u8(b0, b1));
            const uint16x8_t c1 = vmovl_u8(vget_low_u8(b2)), c2 = vmovl_u8(vget_high_u8(b2));
            const uint32x4_t e0 = vreinterpretq_u32_u16(vzip1q_u16(z1, c1)), e1 = vreinterpretq_u32_u16(vzip2q_u16(z1, c1));
            const uint32x4_t e2 = vreinterpretq_u32_u16(vzip1q_u16(z2, c2)), e3 = vreinterpretq_u32_u16(vzip2q_u16(z2, c2));
            vst1q_u32(reinterpret_cast<uint32_t*>(e + i), e0);
            vst1q_u32(reinterpret_cast<uint32_t*>(e + i + 4), e1);
            vst1q_u32(reinterpret_cast<uint32_t*>(e + i + 8), e2);
            vst1q_u32(reinterpret_cast<uint32_t*>(e + i + 12), e3);
            vs = vaddq_u32(vs, vaddq_u32(vaddq_u32(e0, e1), vaddq_u32(e2, e3)));  // row sum < 2^31 (quantizer)
          }
          s = vaddvq_u32(vs);
        }
        for (; i < n; ++i) {
          e[i] = tab[m - qr[i]];
          s += e[i];
        }
        const float rr = static_cast<float>(num / static_cast<double>(s));
        const float32x4_t vr = vdupq_n_f32(rr);
        i = 0;
        if (f32) {
          float* yr = ctx.out(0).data<float>() + r * n;
          for (; i + 4 <= n; i += 4) vst1q_f32(yr + i, vmulq_f32(vcvtq_f32_s32(vld1q_s32(e + i)), vr));
          for (; i < n; ++i) yr[i] = static_cast<float>(e[i]) * rr;
          continue;
        }
        int8_t* yr = ctx.out(0).data<int8_t>() + r * n;
        for (; i + 16 <= n; i += 16) {
          int32x4_t o[4];
          for (int k = 0; k < 4; ++k)
            o[k] = vqaddq_s32(rne4(vmulq_f32(vcvtq_f32_s32(vld1q_s32(e + i + 4 * k)), vr)), vzy);
          vst1q_s8(yr + i, narrow16(o[0], o[1], o[2], o[3], vlo, vhi));
        }
        for (; i < n; ++i) {
          const float v = static_cast<float>(e[i]) * rr;
          const float t = std::nearbyint(v) + static_cast<float>(zy);
          yr[i] = static_cast<int8_t>(std::min(std::max(t, static_cast<float>(lo)), static_cast<float>(hi)));
        }
      }
    });
  }
 private:
  bool planes_ = false;
  uint8x16x4_t p0_[4], p1_[4], p2_[4];
};
// The vector path adds y_zero in int32 after FCVTNS; the reference adds it in fp32 after
// nearbyint. They agree when |nearbyint(v)| < 2^24 (exact float add) and both clamp otherwise,
// provided |y_zero| < 2^23 (true for every zero point the quantizer emits, |z| < 2^20).
bool softmax_neon_supported(const KernelContext& ctx) {
  const int64_t z = ctx.attrs().get_int("y_zero", 0);
  return z > -(int64_t(1) << 23) && z < (int64_t(1) << 23);
}

// ----------------------------------------------------- int8 -> binarization
// Thresholds of a binarized int8 operand as int16 (range [-128, 128]).
struct BinThr {
  q8::QBin mode = q8::QBin::kGe;
  std::vector<int16_t> tge, tgt;  // size 1 (scalar) or C
};

// bits of one row: bit 1 (= -1) iff q < T (T = tge for ge, tgt for gt; scalar or per channel).
void pack_row_bits_i8(const int8_t* q, int64_t c, const int16_t* t, bool per_channel, uint32_t* out) {
  const int64_t words = bit_words(c);
  std::memset(out, 0, sizeof(uint32_t) * words);
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t bitw = vld1q_u8(kW);
  const int16x8_t ts = vdupq_n_s16(t[0]);
  int64_t i = 0;
  for (; i + 16 <= c; i += 16) {
    const int8x16_t v = vld1q_s8(q + i);
    const int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_s8(vget_high_s8(v));
    const uint16x8_t ml = vcltq_s16(l, per_channel ? vld1q_s16(t + i) : ts);
    const uint16x8_t mh = vcltq_s16(h, per_channel ? vld1q_s16(t + i + 8) : ts);
    const uint8x16_t m = vandq_u8(vcombine_u8(vmovn_u16(ml), vmovn_u16(mh)), bitw);
    const uint32_t bits = static_cast<uint32_t>(vaddv_u8(vget_low_u8(m))) | (static_cast<uint32_t>(vaddv_u8(vget_high_u8(m))) << 8);
    out[i >> 5] |= bits << (i & 31);
  }
  for (; i < c; ++i)
    if (q[i] < t[per_channel ? i : 0]) out[i >> 5] |= 1u << (i & 31);
}

// Same bits when every threshold T is at most 127 (T >= -128 always), compared in int8 lanes (t8 = T as int8; per
// channel, padded, or one value): 64 channels -> 2 words per step (CMGT, AND with the bit weights, three
// pairwise-add levels), then whole words of 32, then the remaining channels as above.
void pack_row_bits_i8x(const int8_t* q, int64_t c, const int8_t* t8, const int16_t* t, bool per_channel, uint32_t* out) {
  static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t bitw = vld1q_u8(kW);
  const int8x16_t ts = vdupq_n_s8(t8[0]);
  auto mask = [&](int64_t i) { return vandq_u8(vcltq_s8(vld1q_s8(q + i), per_channel ? vld1q_s8(t8 + i) : ts), bitw); };
  int64_t i = 0;
  for (; i + 64 <= c; i += 64) {
    uint8x16_t r = vpaddq_u8(vpaddq_u8(mask(i), mask(i + 16)), vpaddq_u8(mask(i + 32), mask(i + 48)));
    r = vpaddq_u8(r, r);
    vst1_u8(reinterpret_cast<uint8_t*>(out + (i >> 5)), vget_low_u8(r));
  }
  for (; i + 32 <= c; i += 32) {
    uint8x16_t r = vpaddq_u8(mask(i), mask(i + 16));
    r = vpaddq_u8(r, r);
    r = vpaddq_u8(r, r);
    out[i >> 5] = vgetq_lane_u32(vreinterpretq_u32_u8(r), 0);
  }
  if (i < c) pack_row_bits_i8(q + i, c - i, per_channel ? t + i : t, per_channel, out + (i >> 5));
}

BinThr bin_thresholds(const std::string& mode, const q8::Deq& d, const float* thr, int64_t c) {
  BinThr b;
  b.mode = mode == "gt" ? q8::QBin::kGt : (mode == "sign_zero" ? q8::QBin::kSignZero : (mode == "gt0" ? q8::QBin::kGt0 : q8::QBin::kGe));
  const int64_t n = thr ? c : 1;
  b.tge.resize(static_cast<size_t>(n) + 8);  // + 8: vector loads past the end stay in the buffer
  b.tgt.resize(static_cast<size_t>(n) + 8);
  for (int64_t i = 0; i < n; ++i) {
    const float t = thr ? thr[i] : 0.f;
    b.tge[i] = static_cast<int16_t>(q8::qthr_ge(t, d));
    b.tgt[i] = static_cast<int16_t>(q8::qthr_gt(t, d));
  }
  b.tge.resize(static_cast<size_t>(n));
  b.tgt.resize(static_cast<size_t>(n));
  return b;
}

// A sub-kernel run on scratch tensors (the binary NEON kernels on the binarized int8 input).
struct Sub {
  Node node;
  std::vector<Tensor> owned;  // scratch tensors referenced by ctx
  KernelContext ctx;
  std::unique_ptr<Kernel> k;
};

// ------------------------------------------------------------ I8BDense / I8BConv2D
class BinaryLayerNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    conv_ = ctx.node->op != "I8BDense";
    res_ = ctx.node->op == "I8BConv2DRes";
    const Tensor& x = ctx.in(0);
    bit_in_ = x.dtype() == DType::kBit;
    c_ = x.dim(-1);
    pix_ = x.size() / c_;
    const std::string mode = ctx.attrs().get_string("input_binarize", "ge");
    ternary_ = mode == "sign_zero" && !bit_in_;
    if (!bit_in_)
      thr_ = bin_thresholds(mode, int8_full::deq_of(ctx.attrs(), "x"), ctx.has_in(4) ? ctx.in(4).data<float>() : nullptr, c_);
    if (!ternary_ && !bit_in_) {  // ge/gt: x' = ±1 with T = tge / tgt; packed bits with 'ge' at 0
      const std::vector<int16_t>& t = thr_.mode == q8::QBin::kGt ? thr_.tgt : thr_.tge;
      tsel_.assign(t.begin(), t.end());
      tsel_.resize(t.size() + 16);
      t8ok_ = true;
      for (int16_t v : t) t8ok_ = t8ok_ && v <= 127;  // T >= -128 always: int8 lanes hold every threshold
      t8_.assign(t.size() + 64, 0);
      if (t8ok_)
        for (size_t i = 0; i < t.size(); ++i) t8_[i] = static_cast<int8_t>(t[i]);
    }
    sub_.node.op = conv_ ? "BConv2D" : "BDense";
    sub_.node.name = ctx.node->name + "/binary";
    Json attrs = Json::Object{};
    for (const auto& kv : ctx.attrs().as_object())
      if (kv.first != "x_scale" && kv.first != "x_zero" && kv.first != "qmin" && kv.first != "qmax" && kv.first != "output" &&
          kv.first != "input_binarize" && kv.first != "x_code" && kv.first != "x_exp" && kv.first != "y_exp")
        attrs[kv.first] = kv.second;  // I8BConv2DRes: r_code / r_exp / r_scale / r_zero / bit_binarize go to the kernel
    attrs["input_binarize"] = ternary_ ? "sign_zero" : "ge";
    attrs["i8_qmin"] = ctx.attrs().get_int("qmin", -128);
    attrs["i8_qmax"] = ctx.attrs().get_int("qmax", 127);
    if (res_) attrs["i8_res"] = 1;
    sub_.node.attrs = attrs;
    Shape xs = x.shape();
    sub_.owned.reserve(2);
    if (ternary_) {
      // bdense.ternary_neon reads the codes (int32 thresholds [tge..., tgt...] as input 4) and writes the
      // node's output (int8 or f32) itself
      const int64_t nt = static_cast<int64_t>(thr_.tge.size());
      sub_.owned.push_back(Tensor::allocate(DType::kI32, Shape{2 * nt}));
      int32_t* t = sub_.owned[0].data<int32_t>();
      for (int64_t i = 0; i < nt; ++i) {
        t[i] = thr_.tge[i];
        t[nt + i] = thr_.tgt[i];
      }
    } else if (!bit_in_) {
      sub_.owned.push_back(Tensor::allocate(DType::kBit, xs));
    }
    sub_.ctx.node = &sub_.node;
    sub_.ctx.pool = ctx.pool;
    sub_.ctx.cpu = ctx.cpu;
    // the binary kernel writes the node's output (int8: clamp(rne(v)) of its float value v; or f32) itself
    sub_.ctx.outputs = {ctx.outputs[0]};
    if (ternary_) sub_.ctx.inputs = {ctx.inputs[0], ctx.inputs[1], ctx.inputs[2], ctx.inputs[3], &sub_.owned[0]};
    else sub_.ctx.inputs = {bit_in_ ? ctx.inputs[0] : &sub_.owned[0], ctx.inputs[1], ctx.inputs[2], ctx.inputs[3], nullptr};
    if (res_) {  // residual, slope, out_bias, bit_thr at the node's positions; bit output as out(1)
      for (size_t i = 5; i < ctx.inputs.size(); ++i) sub_.ctx.inputs.push_back(ctx.inputs[i]);
      if (ctx.outputs.size() > 1) sub_.ctx.outputs.push_back(ctx.outputs[1]);
    } else if (ctx.has_in(5)) {
      sub_.ctx.inputs.push_back(ctx.inputs[5]);
    }
    if (conv_) sub_.k = make_bconv2d_neon_opt_kernel();
    else sub_.k = ternary_ ? make_bdense_ternary_neon_kernel() : make_bdense_neon_kernel();
    sub_.k->prepare(sub_.ctx);
  }
  size_t prepacked_bytes() const override { return sub_.k ? sub_.k->prepacked_bytes() : 0; }
  // the binary kernel reads the weights (input 1 here and there) only in prepare()
  std::vector<int> released_inputs() const override { return sub_.k ? sub_.k->released_inputs() : std::vector<int>{}; }

  void run(const KernelContext& ctx) override {
    const int8_t* q = ctx.in(0).data<int8_t>();
    const bool pc = thr_.tge.size() > 1;
    if (ternary_ || bit_in_) {
      sub_.ctx.inputs[0] = ctx.inputs[0];
      rebind(ctx);
      sub_.k->run(sub_.ctx);
      return;
    }
    {
      uint32_t* xb = sub_.owned[0].data<uint32_t>();
      const int64_t words = bit_words(c_);
      ctx.pool->parallel_for(pix_, std::max<int64_t>(1, kSerial / std::max<int64_t>(c_, 1)), [&](int64_t p0, int64_t p1) {
        if (t8ok_)
          for (int64_t p = p0; p < p1; ++p) pack_row_bits_i8x(q + p * c_, c_, t8_.data(), tsel_.data(), pc, xb + p * words);
        else
          for (int64_t p = p0; p < p1; ++p) pack_row_bits_i8(q + p * c_, c_, tsel_.data(), pc, xb + p * words);
      });
    }
    rebind(ctx);
    sub_.k->run(sub_.ctx);
  }

 private:
  void rebind(const KernelContext& ctx) {
    for (size_t i = 0; i < ctx.outputs.size(); ++i) sub_.ctx.outputs[i] = ctx.outputs[i];
    if (res_)
      for (size_t i = 5; i < ctx.inputs.size(); ++i) sub_.ctx.inputs[i] = ctx.inputs[i];
  }
  bool conv_ = false, ternary_ = false, res_ = false, bit_in_ = false;
  int64_t c_ = 0, pix_ = 0;
  BinThr thr_;
  std::vector<int16_t> tsel_;
  std::vector<int8_t> t8_;
  bool t8ok_ = false;
  Sub sub_;
};

bool binary_layer_neon_supported(const KernelContext& ctx) {
  const std::string mode = ctx.attrs().get_string("input_binarize", "ge");
  if (ctx.node->op == "I8BConv2DRes") {
    // the pre stage (inputs 10-13) runs in the two-phase epilogue only: int16 dot products, K <= 32767
    const bool pre = ctx.has_in(10) || ctx.has_in(11) || ctx.has_in(12) || ctx.has_in(13);
    const Tensor& w = ctx.in(1);
    if (pre && w.dim(1) * w.dim(2) * w.dim(3) > 32767) return false;
    return (mode != "sign_zero" || ctx.in(0).dtype() == DType::kBit) && ctx.attrs().get_int("groups", 1) == 1;
  }
  if (ctx.in(0).dtype() == DType::kBit) return ctx.node->op == "I8BConv2D";
  if (ctx.node->op == "I8BConv2D") return mode != "sign_zero";  // bconv2d.neon_opt: ±1 inputs only
  return !(ctx.has_in(5) && mode != "sign_zero");
}

// -------------------------------------------------------------- I8BMatMul
// bmatmul.neon reads the int8 operands itself (its int8 mode packs the planes of the integer
// binarization q8::qbin straight from the codes, given the thresholds below), so no {-1, 0, +1} float
// copies are made. Row sums of A and column sums of B (c_rs, c_cs) come from the codes as well.
//   neg = q < tneg, pos = q >= tpos:  sign_zero (T_ge, T_gt), ge (T_ge, -), gt (T_gt, -), gt0 (-, T_gt)
inline void bmm_tneg_tpos(const BinThr& t, int* tneg, int* tpos) {
  *tneg = t.mode == q8::QBin::kGt ? t.tgt[0] : t.tge[0];
  *tpos = t.tgt[0];
}

// pos / neg lane masks (0 / -1 in int16 lanes) of 8 codes: the binarized value is neg - pos.
inline void qbin_masks8(int16x8_t q, q8::QBin mode, int16x8_t tge, int16x8_t tgt, int16x8_t* neg, int16x8_t* pos) {
  switch (mode) {
    case q8::QBin::kGe:
      *neg = vreinterpretq_s16_u16(vcltq_s16(q, tge));
      *pos = vmvnq_s16(*neg);
      break;
    case q8::QBin::kGt:
      *neg = vreinterpretq_s16_u16(vcltq_s16(q, tgt));
      *pos = vmvnq_s16(*neg);
      break;
    case q8::QBin::kSignZero:
      *neg = vreinterpretq_s16_u16(vcltq_s16(q, tge));
      *pos = vreinterpretq_s16_u16(vcgeq_s16(q, tgt));
      break;
    case q8::QBin::kGt0:
      *neg = vdupq_n_s16(0);
      *pos = vreinterpretq_s16_u16(vcgeq_s16(q, tgt));
      break;
  }
}

// sum of the binarized values of a row of n codes (|sum| <= n <= 32767: int16 lanes)
int64_t qbin_row_sum(const int8_t* q, int64_t n, const BinThr& t) {
  const int16x8_t tge = vdupq_n_s16(t.tge[0]), tgt = vdupq_n_s16(t.tgt[0]);
  int16x8_t acc = vdupq_n_s16(0);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const int8x16_t v = vld1q_s8(q + i);
    int16x8_t ng, ps;
    qbin_masks8(vmovl_s8(vget_low_s8(v)), t.mode, tge, tgt, &ng, &ps);
    acc = vaddq_s16(acc, vsubq_s16(ng, ps));
    qbin_masks8(vmovl_high_s8(v), t.mode, tge, tgt, &ng, &ps);
    acc = vaddq_s16(acc, vsubq_s16(ng, ps));
  }
  int64_t s = vaddlvq_s16(acc);
  for (; i < n; ++i) s += q8::qbin(t.mode, q[i], t.tge[0], t.tgt[0]);
  return s;
}

// column sums of the binarized values of a row-major [k][n] block of codes (|sum| <= k <= 32767)
void qbin_col_sums(const int8_t* q, int64_t k, int64_t n, const BinThr& t, int64_t* cs) {
  const int16x8_t tge = vdupq_n_s16(t.tge[0]), tgt = vdupq_n_s16(t.tgt[0]);
  int64_t j = 0;
  for (; j + 16 <= n; j += 16) {
    int16x8_t a0 = vdupq_n_s16(0), a1 = vdupq_n_s16(0);
    for (int64_t kk = 0; kk < k; ++kk) {
      const int8x16_t v = vld1q_s8(q + kk * n + j);
      int16x8_t ng, ps;
      qbin_masks8(vmovl_s8(vget_low_s8(v)), t.mode, tge, tgt, &ng, &ps);
      a0 = vaddq_s16(a0, vsubq_s16(ng, ps));
      qbin_masks8(vmovl_high_s8(v), t.mode, tge, tgt, &ng, &ps);
      a1 = vaddq_s16(a1, vsubq_s16(ng, ps));
    }
    int16_t tmp[16];
    vst1q_s16(tmp, a0);
    vst1q_s16(tmp + 8, a1);
    for (int c = 0; c < 16; ++c) cs[j + c] = tmp[c];
  }
  for (; j < n; ++j) {
    int64_t s = 0;
    for (int64_t kk = 0; kk < k; ++kk) s += q8::qbin(t.mode, q[kk * n + j], t.tge[0], t.tgt[0]);
    cs[j] = s;
  }
}

class BMatMulI8Neon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    const Json& a = ctx.attrs();
    const float tha = static_cast<float>(a.get_number("a_threshold", 0.0)), thb = static_cast<float>(a.get_number("b_threshold", 0.0));
    ta_ = bin_thresholds(a.get_string("a_binarize", "sign_zero"), int8_full::deq_of(a, "a"), &tha, 1);
    tb_ = bin_thresholds(a.get_string("b_binarize", "sign_zero"), int8_full::deq_of(a, "b"), &thb, 1);
    p_.reset(new int8_full::BMatMulParams(ctx));
    acc_ = Tensor::allocate(DType::kF32, ctx.out(0).shape());
    rs_.assign(static_cast<size_t>(g_.batch * g_.m), 0);
    cs_.assign(static_cast<size_t>(g_.batch * g_.n), 0);
    colterm_.assign(static_cast<size_t>(g_.batch * g_.n), 0);
    // plain products (acc = ab, v = ab * mul + add): float(int(ab)) == ab for these integers (|ab| <= k < 2^24)
    plain_ = p_->c_ab == 1 && p_->c_rs == 0 && p_->c_cs == 0 && p_->c0 == 0;
    if (p_->c_ab != 0) {
      int atn, atp, btn, btp;
      bmm_tneg_tpos(ta_, &atn, &atp);
      bmm_tneg_tpos(tb_, &btn, &btp);
      sub_.node.op = "BMatMul";
      sub_.node.name = ctx.node->name + "/binary";
      sub_.node.attrs = Json::Object{{"a_binarize", a.get_string("a_binarize", "sign_zero")},
                                     {"b_binarize", a.get_string("b_binarize", "sign_zero")},
                                     {"transpose_b", static_cast<int64_t>(g_.transpose_b ? 1 : 0)},
                                     {"scale", 1.0},
                                     {"i8_a_tneg", static_cast<int64_t>(atn)},
                                     {"i8_a_tpos", static_cast<int64_t>(atp)},
                                     {"i8_b_tneg", static_cast<int64_t>(btn)},
                                     {"i8_b_tpos", static_cast<int64_t>(btp)}};
      sub_.ctx.node = &sub_.node;
      sub_.ctx.pool = ctx.pool;
      sub_.ctx.cpu = ctx.cpu;
      sub_.ctx.inputs = {ctx.inputs[0], ctx.inputs[1]};
      sub_.ctx.outputs = {&acc_};
      sub_.k = make_bmatmul_neon_kernel();
      sub_.k->prepare(sub_.ctx);
    }
  }

  void run(const KernelContext& ctx) override {
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    const int64_t k = g_.k, m = g_.m, n = g_.n;
    const bool need_rs = p_->c_rs != 0, need_cs = p_->c_cs != 0, need_ab = p_->c_ab != 0;
    if (need_rs)
      ctx.pool->parallel_for(g_.batch * m, std::max<int64_t>(1, kSerial / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) rs_[r] = qbin_row_sum(qa + r * k, k, ta_);
      });
    if (need_cs) {
      if (g_.transpose_b) {
        ctx.pool->parallel_for(g_.batch * n, std::max<int64_t>(1, kSerial / std::max<int64_t>(k, 1)), [&](int64_t r0, int64_t r1) {
          for (int64_t r = r0; r < r1; ++r) cs_[r] = qbin_row_sum(qb + r * k, k, tb_);
        });
      } else {
        for (int64_t bt = 0; bt < g_.batch; ++bt) qbin_col_sums(qb + bt * k * n, k, n, tb_, cs_.data() + bt * n);
      }
    }
    if (need_ab) {
      sub_.ctx.inputs = {ctx.inputs[0], ctx.inputs[1]};
      sub_.k->run(sub_.ctx);
    }
    const float* ab = acc_.data<float>();
    const int8_full::BMatMulParams& p = *p_;
    // column terms c_cs*cs[j] + c0 (int32: |acc| <= K*(|c_ab| + |c_rs| + |c_cs|) + |c0| < 2^31 checked in prepare)
    for (int64_t i = 0, nb = g_.batch * n; i < nb; ++i) colterm_[i] = static_cast<int32_t>(p.c_cs * cs_[i] + p.c0);
    const bool plain = plain_;
    ctx.pool->parallel_for(g_.batch * m, std::max<int64_t>(1, kSerial / std::max<int64_t>(n, 1)), [&](int64_t r0, int64_t r1) {
      const int32x4_t cab = vdupq_n_s32(static_cast<int32_t>(p.c_ab));
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(p.lo)), vhi = vdupq_n_s16(static_cast<int16_t>(p.hi));
      const float32x4_t mul_s = vdupq_n_f32(p.mul[0]), add_s = vdupq_n_f32(p.add ? p.add[0] : 0.f);
      for (int64_t r = r0; r < r1; ++r) {
        const int32_t* ct = colterm_.data() + (r / m) * n;
        const int32x4_t rt = vdupq_n_s32(static_cast<int32_t>(p.c_rs * rs_[r]));
        const float* abr = ab + r * n;
        // acc_j = c_ab*ab[j] + c_rs*rs[r] + (c_cs*cs[j] + c0); v = fp32(fp32(float(acc) * mul[j]) + add[j])
        auto value = [&](int64_t j) {
          float32x4_t f;
          if (plain) {
            f = vld1q_f32(abr + j);
          } else {
            int32x4_t acc = vaddq_s32(rt, vld1q_s32(ct + j));
            if (need_ab) acc = vmlaq_s32(acc, vcvtq_s32_f32(vld1q_f32(abr + j)), cab);
            f = vcvtq_f32_s32(acc);
          }
          float32x4_t v = vmulq_f32(f, p.mul_vec ? vld1q_f32(p.mul + j) : mul_s);
          if (p.add) v = vaddq_f32(v, p.add_vec ? vld1q_f32(p.add + j) : add_s);
          return v;
        };
        int64_t j = 0;
        if (p.f32) {
          float* y = ctx.out(0).data<float>() + r * n;
          for (; j + 4 <= n; j += 4) vst1q_f32(y + j, value(j));
        } else {
          int8_t* y = ctx.out(0).data<int8_t>() + r * n;
          for (; j + 16 <= n; j += 16)
            vst1q_s8(y + j, narrow16(rne4(value(j)), rne4(value(j + 4)), rne4(value(j + 8)), rne4(value(j + 12)), vlo, vhi));
        }
        for (; j < n; ++j) {
          const int64_t abv = need_ab ? static_cast<int64_t>(abr[j]) : 0;
          p.store(ctx, r * n + j, j, p.combine(abv, rs_[r], cs_[(r / m) * n + j]));
        }
      }
    });
  }

 private:
  MatMulGeom g_;
  BinThr ta_, tb_;
  std::unique_ptr<int8_full::BMatMulParams> p_;
  Tensor acc_;
  std::vector<int64_t> rs_, cs_;
  std::vector<int32_t> colterm_;
  bool plain_ = false;
  Sub sub_;
};
bool bmatmul_i8_neon_supported(const KernelContext& ctx) {
  auto g = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
  const Json& a = ctx.attrs();
  const int64_t bound = g.k * (std::abs(a.get_int("c_ab", 1)) + std::abs(a.get_int("c_rs", 0)) + std::abs(a.get_int("c_cs", 0))) +
                        std::abs(a.get_int("c0", 0));
  return g.k <= 32767 && bound < (int64_t(1) << 31);  // bmatmul.neon range (int16 tile sums); int32 combination
}

// ---------------------------------------------------- I8Conv2D / I8Dense (SDOT)
// GEMM per group: rows = output pixels (Dense rows) x columns = the group's output channels, K = taps*kic
// raw int8 codes (padding taps = pad_q), zero-padded to a multiple of 16 bytes. Operands in the int8_neon
// tile layout (int8_gemm.h): A [kc][8 rows][4], W [ocg/8][kc][8 cols][4]. The SDOT micro-kernel below
// keeps int32 sums (the int8_neon one ends in an fp32 epilogue); then
//   acc = raw - x_zero * sum_k w[o]   (int32: exact while |acc| < 2^31, see gemm_sdot_supported)
//   y   = clamp(y_zero + requant(sat32(acc + bias)))   (i8 output; f32 output: the reference expression)
// Bit-identical to Conv2DRef / DenseRef (same integers, same requantization).
constexpr int kC = 4;  // SDOT chunk bytes

// 8 x 8 tiles (R valid rows) over nob column blocks: y[r * ldy + 8 * ob + j] = sum_k a[r][k] * w[8 ob + j][k].
// kc is a multiple of 4. Accumulators c[r][h] = 4 consecutive columns of row r (vdotq_laneq_s32).
template <int R>
void sdot_tiles_i32(const int8_t* __restrict a, const int8_t* __restrict w, int64_t kc, int64_t nob, int32_t* __restrict y, int64_t ldy) {
  for (int64_t ob = 0; ob < nob; ++ob, w += kc * 32) {
    int32x4_t c[8][2];
#pragma clang loop unroll(full)
    for (int r = 0; r < 8; ++r) c[r][0] = c[r][1] = vdupq_n_s32(0);
    const int8_t* ap = a;
    const int8_t* wp = w;
    for (int64_t s = 0; s < kc; ++s, ap += 32, wp += 32) {
      const int8x16_t x0 = vld1q_s8(ap), w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16);
#define BF_SDOT_ROW(r, x, l)                       \
  if (R > r) {                                     \
    c[r][0] = vdotq_laneq_s32(c[r][0], w0, x, l);  \
    c[r][1] = vdotq_laneq_s32(c[r][1], w1, x, l);  \
  }
      BF_SDOT_ROW(0, x0, 0)
      BF_SDOT_ROW(1, x0, 1)
      BF_SDOT_ROW(2, x0, 2)
      BF_SDOT_ROW(3, x0, 3)
      if (R > 4) {
        const int8x16_t x1 = vld1q_s8(ap + 16);
        BF_SDOT_ROW(4, x1, 0)
        BF_SDOT_ROW(5, x1, 1)
        BF_SDOT_ROW(6, x1, 2)
        BF_SDOT_ROW(7, x1, 3)
      }
#undef BF_SDOT_ROW
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      vst1q_s32(y + r * ldy + ob * 8, c[r][0]);
      vst1q_s32(y + r * ldy + ob * 8 + 4, c[r][1]);
    }
  }
}

void sdot_tiles_i32_rows(int rows, const int8_t* a, const int8_t* w, int64_t kc, int64_t nob, int32_t* y, int64_t ldy) {
  switch (rows) {
    case 1: return sdot_tiles_i32<1>(a, w, kc, nob, y, ldy);
    case 2: return sdot_tiles_i32<2>(a, w, kc, nob, y, ldy);
    case 3: return sdot_tiles_i32<3>(a, w, kc, nob, y, ldy);
    case 4: return sdot_tiles_i32<4>(a, w, kc, nob, y, ldy);
    case 5: return sdot_tiles_i32<5>(a, w, kc, nob, y, ldy);
    case 6: return sdot_tiles_i32<6>(a, w, kc, nob, y, ldy);
    case 7: return sdot_tiles_i32<7>(a, w, kc, nob, y, ldy);
    default: return sdot_tiles_i32<8>(a, w, kc, nob, y, ldy);
  }
}

// 8 rows of kp bytes (kp a multiple of 16) -> tile layout [kp/4 chunks][8 rows][4 bytes]: a 4-byte
// transpose with ZIP1/ZIP2 at 32 and 64 bits, 16 bytes of every row per step.
void pack_tile_rows(const int8_t* const rp[8], int64_t kp, int8_t* __restrict dst) {
  for (int64_t k = 0; k < kp; k += 16, dst += 128) {
    uint32x4_t r[8];
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) r[i] = vreinterpretq_u32_s8(vld1q_s8(rp[i] + k));
#pragma clang loop unroll(full)
    for (int h = 0; h < 2; ++h) {  // rows 4h .. 4h+3
      const uint64x2_t a0 = vreinterpretq_u64_u32(vzip1q_u32(r[4 * h], r[4 * h + 1]));
      const uint64x2_t a1 = vreinterpretq_u64_u32(vzip2q_u32(r[4 * h], r[4 * h + 1]));
      const uint64x2_t b0 = vreinterpretq_u64_u32(vzip1q_u32(r[4 * h + 2], r[4 * h + 3]));
      const uint64x2_t b1 = vreinterpretq_u64_u32(vzip2q_u32(r[4 * h + 2], r[4 * h + 3]));
      vst1q_u64(reinterpret_cast<uint64_t*>(dst + 0 * 32 + 16 * h), vzip1q_u64(a0, b0));  // chunk 0
      vst1q_u64(reinterpret_cast<uint64_t*>(dst + 1 * 32 + 16 * h), vzip2q_u64(a0, b0));  // chunk 1
      vst1q_u64(reinterpret_cast<uint64_t*>(dst + 2 * 32 + 16 * h), vzip1q_u64(a1, b1));  // chunk 2
      vst1q_u64(reinterpret_cast<uint64_t*>(dst + 3 * 32 + 16 * h), vzip2q_u64(a1, b1));  // chunk 3
    }
  }
}

// Exact-size byte copy / fill for the short runs of im2col rows (a few to a few hundred bytes): fixed-size
// moves with the last one overlapping, never touching bytes outside [d, d + n) / [s, s + n).
inline __attribute__((always_inline)) void copy_bytes(int8_t* d, const int8_t* s, int64_t n) {
  if (n >= 16) {
    for (int64_t i = 0; i + 16 < n; i += 16) std::memcpy(d + i, s + i, 16);
    std::memcpy(d + n - 16, s + n - 16, 16);
  } else if (n >= 8) {
    std::memcpy(d, s, 8);
    std::memcpy(d + n - 8, s + n - 8, 8);
  } else if (n >= 4) {
    std::memcpy(d, s, 4);
    std::memcpy(d + n - 4, s + n - 4, 4);
  } else if (n >= 2) {
    std::memcpy(d, s, 2);
    std::memcpy(d + n - 2, s + n - 2, 2);
  } else if (n == 1) {
    d[0] = s[0];
  }
}
inline __attribute__((always_inline)) void fill_bytes(int8_t* d, int8_t v, int64_t n) {
  const int8x16_t vv = vdupq_n_s8(v);
  if (n >= 16) {
    for (int64_t i = 0; i + 16 < n; i += 16) vst1q_s8(d + i, vv);
    vst1q_s8(d + n - 16, vv);
  } else {
    for (int64_t i = 0; i < n; ++i) d[i] = v;
  }
}

class GemmSdot : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    dense_ = ctx.node->op == "I8Dense";
    const Tensor& w = ctx.in(1);
    o_ = w.dim(0);
    k_ = w.size() / o_;
    groups_ = 1;
    if (dense_) {
      rows_ = ctx.in(0).size() / k_;
    } else {
      g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), w.shape());
      rows_ = g_.n * g_.oh * g_.ow;
      groups_ = g_.groups;
    }
    ocg_ = o_ / groups_;
    kp_ = (k_ + 15) / 16 * 16;
    kc_ = kp_ / kC;
    obg_ = (ocg_ + int8_neon::kTile - 1) / int8_neon::kTile;
    ldy_ = obg_ * int8_neon::kTile;
    // W per group: [obg][kc][8][4], zero padded; column sums of the int8 weights
    wpack_.assign(static_cast<size_t>(groups_ * obg_ * kc_ * int8_neon::kTile * kC), 0);
    wsum_.assign(static_cast<size_t>(o_), 0);
    // SMMLA (i8mm): W [obg][kp/8 chunks][4 column pairs][2][8] (same bytes per group); BITINO_GEMM_I8MM=0 keeps SDOT
    const char* ge = std::getenv("BITINO_GEMM_I8MM");
    mm_ = ctx.cpu->has(kFeatI8mm) && !(ge && std::string(ge) == "0");
    // SMMLA convolutions with groups 1, dilation_w 1 and a kernel of several taps: K in kernel-row runs, each run of
    // kw*ic bytes zero-padded to rl8_ (a multiple of 8), so that the packed A is loaded in 8-byte chunks straight from
    // the input for interior pixels (pack_runs; no im2col row copy). The padding bytes meet zero weights.
    runs_ = mm_ && !dense_ && groups_ == 1 && g_.dw == 1 && g_.kh * g_.kw > 1;
    if (runs_) {
      rl8_ = (g_.kw * g_.ic + 7) / 8 * 8;
      kc8_ = g_.kh * rl8_ / 8;
      kp_ = (g_.kh * rl8_ + 15) / 16 * 16;
      kc_ = kp_ / kC;
      wpack_.assign(static_cast<size_t>(groups_ * obg_ * kc_ * int8_neon::kTile * kC), 0);
      // a padded copy of small inputs (borders pad_q, filled once): every pixel then reads its runs in place
      const int64_t ph = g_.ih + g_.pt + g_.pb, pw = g_.iw + g_.pl + g_.pr;
      const int64_t bytes = g_.n * ph * pw * g_.ic;
      if ((g_.pt | g_.pb | g_.pl | g_.pr) != 0 && bytes <= (int64_t(4) << 20)) {
        padded_.assign(static_cast<size_t>(bytes + rl8_ + 16),
                       static_cast<int8_t>(ctx.attrs().get_int("pad_q", ctx.attrs().get_int("x_zero", 0))));
        ph_ = ph;
        pw_ = pw;
      }
    } else {
      kc8_ = kp_ / 8;
    }
    const int8_t* wd = w.data<int8_t>();
    for (int64_t o = 0; o < o_; ++o) {
      const int64_t gi = o / ocg_, j = o % ocg_;
      int8_t* wg = wpack_.data() + gi * obg_ * kc_ * 32;
      for (int64_t kk = 0; kk < k_; ++kk) {
        const int64_t kr = runs_ ? kk / (g_.kw * g_.ic) * rl8_ + kk % (g_.kw * g_.ic) : kk;  // K index in the run layout
        if (mm_) wg[((j / 8 * kc8_ + kr / 8) * 8 + j % 8) * 8 + kr % 8] = wd[o * k_ + kk];
        else wg[((j / 8 * kc_ + kk / kC) * 8 + j % 8) * kC + kk % kC] = wd[o * k_ + kk];
        wsum_[o] += wd[o * k_ + kk];
      }
    }
    xz_ = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    bits_ = !dense_ && ctx.has_in(5);
    if (bits_) {  // bit output thresholds by channel (padded)
      bthr_.assign(static_cast<size_t>(o_ + 16), 0);
      std::copy_n(ctx.in(5).data<int32_t>(), o_, bthr_.data());
      bw_ = bit_words(o_);
    }
    padq_ = static_cast<int8_t>(ctx.attrs().get_int("pad_q", xz_));
    // per-channel epilogue constants (i8 output): x_zero*wsum wrapped to int32 as the vector code computes it
    int8_full::GemmEpilogue ep(ctx);
    f32_ = ep.f32;
    l8_ = ep.l8;
    if (l8_) {  // fp32 scale / bias by channel (padded) for the l8 epilogue
      scale_f_.assign(static_cast<size_t>(o_ + 16), 0.f);
      bias_f_.assign(static_cast<size_t>(o_ + 16), 0.f);
      std::copy_n(ep.scale, o_, scale_f_.data());
      if (ep.bias_f) std::copy_n(ep.bias_f, o_, bias_f_.data());
      has_bias_f_ = ep.bias_f != nullptr;
    }
    xzw_.assign(static_cast<size_t>(o_ + 16), 0);
    bias_.assign(static_cast<size_t>(o_ + 16), 0);
    mult_.assign(static_cast<size_t>(o_ + 16), 0);
    lsh_.assign(static_cast<size_t>(o_ + 16), 0);
    nrs_.assign(static_cast<size_t>(o_ + 16), 0);
    any_lsh_ = false;
    for (int64_t o = 0; o < o_; ++o) {
      xzw_[o] = static_cast<int32_t>(static_cast<uint32_t>(xz_) * static_cast<uint32_t>(wsum_[o]));
      if (f32_ || l8_) continue;
      bias_[o] = ep.bias_q ? ep.bias_q[o] : 0;
      mult_[o] = ep.mult[o];
      lsh_[o] = std::max(ep.shift[o], 0);
      nrs_[o] = std::min(ep.shift[o], 0);
      any_lsh_ = any_lsh_ || lsh_[o] > 0;
    }
    // im2col rows: interior pixels of groups == 1, dilation_w == 1 convolutions copy one run of kw*ic bytes
    // per kernel row; 1x1 / stride 1 / no padding convolutions and Dense rows with K % 16 == 0 are read in place
    direct_ = dense_ ? k_ % 16 == 0
                     : (groups_ == 1 && g_.kh == 1 && g_.kw == 1 && g_.sh == 1 && g_.sw == 1 && g_.pt == 0 && g_.pl == 0 &&
                        g_.pb == 0 && g_.pr == 0 && k_ % 16 == 0);
    zero_row_.assign(static_cast<size_t>(kp_), 0);
  }
  size_t prepacked_bytes() const override { return wpack_.size(); }
  std::vector<int> released_inputs() const override { return {1}; }  // weights: prepacked, not read by run()

  void run(const KernelContext& ctx) override {
    const int8_t* x = ctx.in(0).data<int8_t>();
    const int64_t pb = (rows_ + 7) / 8;
    if (!padded_.empty()) {  // interior of the padded copy (the borders keep pad_q)
      const int64_t rowb = g_.iw * g_.ic;
      for (int64_t n = 0; n < g_.n; ++n)
        for (int64_t iy = 0; iy < g_.ih; ++iy)
          std::memcpy(padded_.data() + ((n * ph_ + iy + g_.pt) * pw_ + g_.pl) * g_.ic, x + (n * g_.ih + iy) * rowb, static_cast<size_t>(rowb));
    }
    int8_full::GemmEpilogue ep(ctx);
    ctx.pool->parallel_for(pb, 1, [&](int64_t t0, int64_t t1) {
      thread_local std::vector<int8_t> rowbuf, apack;
      thread_local std::vector<int32_t> acc;
      if (static_cast<int64_t>(rowbuf.size()) < 8 * kp_) rowbuf.resize(static_cast<size_t>(8 * kp_));
      if (static_cast<int64_t>(apack.size()) < kc_ * 32) apack.resize(static_cast<size_t>(kc_ * 32));
      if (static_cast<int64_t>(acc.size()) < 8 * ldy_) acc.resize(static_cast<size_t>(8 * ldy_));
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t r0 = t * 8;
        const int nr = static_cast<int>(std::min<int64_t>(8, rows_ - r0));
        for (int64_t gi = 0; gi < groups_; ++gi) {
          const int8_t* rp[8];
          if (runs_) {
            pack_runs(x, r0, nr, rowbuf.data(), apack.data());
            int8_full_i8mm::tiles_i32(apack.data(), wpack_.data(), kc8_, obg_, acc.data(), ldy_, nr);
          } else if (mm_) {
            for (int i = 0; i < 8; ++i) rp[i] = i < nr ? row_ptr(x, r0 + i, gi, rowbuf.data() + i * kp_) : zero_row_.data();
            int8_full_i8mm::pack_tile_rows(rp, kp_, apack.data());
            int8_full_i8mm::tiles_i32(apack.data(), wpack_.data() + gi * obg_ * kc_ * 32, kc8_, obg_, acc.data(), ldy_, nr);
          } else {
            for (int i = 0; i < 8; ++i) rp[i] = i < nr ? row_ptr(x, r0 + i, gi, rowbuf.data() + i * kp_) : zero_row_.data();
            pack_tile_rows(rp, kp_, apack.data());
            sdot_tiles_i32_rows(nr, apack.data(), wpack_.data() + gi * obg_ * kc_ * 32, kc_, obg_, acc.data(), ldy_);
          }
          if (f32_) {
            for (int i = 0; i < nr; ++i) {
              float* y = ctx.out(0).data<float>() + (r0 + i) * o_ + gi * ocg_;
              const int32_t* a = acc.data() + i * ldy_;
              for (int64_t j = 0; j < ocg_; ++j) {
                const int64_t o = gi * ocg_ + j;
                y[j] = ep.f(static_cast<int64_t>(a[j]) - int64_t(xz_) * wsum_[o], o);
              }
            }
          } else if (l8_) {
            epilogue_l8(acc.data(), nr, gi, ep, ctx.out(0).data<int8_t>() + r0 * o_);
          } else if (any_lsh_) {
            epilogue_i8<true>(acc.data(), nr, gi, ep, ctx.out(0).data<int8_t>() + r0 * o_);
          } else {
            epilogue_i8<false>(acc.data(), nr, gi, ep, ctx.out(0).data<int8_t>() + r0 * o_);
          }
          if (bits_) epilogue_bits(acc.data(), nr, gi, ctx.out(1).data<uint32_t>() + r0 * bw_);
        }
      }
    });
  }

 private:
  // Row r of the GEMM for group gi: kp_ bytes, K real (raw codes, padding taps = pad_q) then zeros.
  const int8_t* row_ptr(const int8_t* x, int64_t r, int64_t gi, int8_t* buf) const {
    if (direct_) return x + r * k_;
    fill_bytes(buf + k_, 0, kp_ - k_);
    if (dense_) {
      copy_bytes(buf, x + r * k_, k_);
      return buf;
    }
    const int64_t n = r / (g_.oh * g_.ow), oy = (r / g_.ow) % g_.oh, ox = r % g_.ow;
    const int64_t iy0 = oy * g_.sh - g_.pt, ix0 = ox * g_.sw - g_.pl;
    const int8_t* img = x + n * g_.ih * g_.iw * g_.ic + gi * g_.kic;
    if (groups_ == 1 && g_.dw == 1 && iy0 >= 0 && ix0 >= 0 && iy0 + g_.dh * (g_.kh - 1) < g_.ih && ix0 + g_.kw - 1 < g_.iw) {
      const int64_t run = g_.kw * g_.ic;
      for (int64_t ky = 0; ky < g_.kh; ++ky) copy_bytes(buf + ky * run, img + ((iy0 + ky * g_.dh) * g_.iw + ix0) * g_.ic, run);
      return buf;
    }
    int64_t kk = 0;
    for (int64_t ky = 0; ky < g_.kh; ++ky)
      for (int64_t kx = 0; kx < g_.kw; ++kx, kk += g_.kic) {
        const int64_t iy = iy0 + ky * g_.dh, ix = ix0 + kx * g_.dw;
        if (iy >= 0 && iy < g_.ih && ix >= 0 && ix < g_.iw)
          copy_bytes(buf + kk, img + (iy * g_.iw + ix) * g_.ic, g_.kic);
        else
          fill_bytes(buf + kk, padq_, g_.kic);
      }
    return buf;
  }

  // SMMLA A tile [kc8_][4 row pairs][2 rows][8 bytes] in the run layout (runs_): chunk (ky, j) of row r = 8 bytes at
  // src[r] + ky * rs[r] + 8 j. Interior rows read the input (the run's padding bytes are the next input bytes, which
  // meet zero weights; the whole rl8_ bytes lie inside the input row); border rows are built in rowbuf (padding taps
  // pad_q, run padding 0); rows >= nr read zero_row_.
  void pack_runs(const int8_t* x, int64_t r0, int nr, int8_t* rowbuf, int8_t* dst) const {
    const int8_t* src[8];
    int64_t rs[8];
    const int64_t ic = g_.ic, run = g_.kw * ic;
    int64_t n = r0 / (g_.oh * g_.ow), oy = (r0 / g_.ow) % g_.oh, ox = r0 % g_.ow - 1;  // advanced per row below
    for (int i = 0; i < 8; ++i) {
      if (i >= nr) {
        src[i] = zero_row_.data();
        rs[i] = 0;
        continue;
      }
      if (++ox == g_.ow) {
        ox = 0;
        if (++oy == g_.oh) {
          oy = 0;
          ++n;
        }
      }
      const int64_t iy0 = oy * g_.sh - g_.pt, ix0 = ox * g_.sw - g_.pl;
      const int8_t* img = x + n * g_.ih * g_.iw * ic;
      if (!padded_.empty() && (ox * g_.sw) * ic + rl8_ <= pw_ * ic) {  // padded copy (run())
        src[i] = padded_.data() + ((n * ph_ + oy * g_.sh) * pw_ + ox * g_.sw) * ic;
        rs[i] = g_.dh * pw_ * ic;
        continue;
      }
      if (iy0 >= 0 && iy0 + g_.dh * (g_.kh - 1) < g_.ih && ix0 >= 0 && ix0 * ic + rl8_ <= g_.iw * ic) {
        src[i] = img + (iy0 * g_.iw + ix0) * ic;
        rs[i] = g_.dh * g_.iw * ic;
        continue;
      }
      int8_t* b = rowbuf + i * kp_;
      for (int64_t ky = 0; ky < g_.kh; ++ky) {
        const int64_t iy = iy0 + ky * g_.dh;
        int8_t* d = b + ky * rl8_;
        for (int64_t kx = 0; kx < g_.kw; ++kx) {
          const int64_t ix = ix0 + kx;
          if (iy >= 0 && iy < g_.ih && ix >= 0 && ix < g_.iw) std::memcpy(d + kx * ic, img + (iy * g_.iw + ix) * ic, static_cast<size_t>(ic));
          else std::memset(d + kx * ic, padq_, static_cast<size_t>(ic));
        }
        std::memset(d + run, 0, static_cast<size_t>(rl8_ - run));
      }
      src[i] = b;
      rs[i] = rl8_;
    }
    const int64_t cpr = rl8_ / 8;
    for (int64_t ky = 0; ky < g_.kh; ++ky) {
      const int8_t* s[8];
      for (int i = 0; i < 8; ++i) s[i] = src[i] + ky * rs[i];
      for (int64_t j = 0; j < cpr; ++j, dst += 64) {
#pragma clang loop unroll(full)
        for (int p = 0; p < 4; ++p) vst1q_s8(dst + 16 * p, vcombine_s8(vld1_s8(s[2 * p] + 8 * j), vld1_s8(s[2 * p + 1] + 8 * j)));
      }
    }
  }

  // output l8: y = l8_encode(fp32(fp32(float(acc - x_zero*wsum)) * scale[o]) + bias[o]), 16 channels per step.
  void epilogue_l8(const int32_t* acc, int nr, int64_t gi, const int8_full::GemmEpilogue& ep, int8_t* y) const {
    const int64_t c0 = gi * ocg_;
    int64_t j = 0;
    for (; j + 16 <= ocg_; j += 16) {
      int32x4_t xzw[4];
      float32x4_t sc[4], bi[4];
      for (int k = 0; k < 4; ++k) {
        const int64_t o = c0 + j + 4 * k;
        xzw[k] = vld1q_s32(xzw_.data() + o);
        sc[k] = vld1q_f32(scale_f_.data() + o);
        bi[k] = vld1q_f32(bias_f_.data() + o);
      }
      // strides, flag and tables in locals: the int8 stores may alias members, which would reload them per row
      const int64_t ldy = ldy_, ys = o_;
      const l8neon::EncTables et = enc_;
      const int32_t* a = acc + j;
      int8_t* yp = y + c0 + j;
      if (has_bias_f_) {
        for (int i = 0; i < nr; ++i, a += ldy, yp += ys) {
          float32x4_t v[4];
          for (int k = 0; k < 4; ++k) v[k] = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vld1q_s32(a + 4 * k), xzw[k])), sc[k]), bi[k]);
          vst1q_s8(yp, l8neon::encode16(v[0], v[1], v[2], v[3], et));
        }
      } else {
        for (int i = 0; i < nr; ++i, a += ldy, yp += ys) {
          float32x4_t v[4];
          for (int k = 0; k < 4; ++k) v[k] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vld1q_s32(a + 4 * k), xzw[k])), sc[k]);
          vst1q_s8(yp, l8neon::encode16(v[0], v[1], v[2], v[3], et));
        }
      }
    }
    for (int i = 0; i < nr; ++i) {
      const int32_t* a = acc + i * ldy_;
      for (int64_t jj = j; jj < ocg_; ++jj) {
        const int64_t o = c0 + jj;
        y[i * o_ + o] = q8::l8_encode(ep.f(static_cast<int64_t>(a[jj]) - int64_t(xz_) * wsum_[o], o));
      }
    }
  }

  // acc[i][j] (rows i < nr of the tile, channels j of group gi) -> y[i * o_ + gi * ocg_ + j]:
  // acc - x_zero*wsum (int32), saturating + bias, requant (SQSHL only when some shift is positive),
  // saturating + y_zero, clamp. Blocks of 16 channels keep their constants in registers over the rows.
  template <bool kLsh>
  void epilogue_i8(const int32_t* acc, int nr, int64_t gi, const int8_full::GemmEpilogue& ep, int8_t* y) const {
    const int64_t c0 = gi * ocg_;
    const int32x4_t vzy = vdupq_n_s32(static_cast<int32_t>(ep.zy));
    const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(ep.lo)), vhi = vdupq_n_s16(static_cast<int16_t>(ep.hi));
    int64_t j = 0;
    for (; j + 16 <= ocg_; j += 16) {
      int32x4_t xzw[4], bias[4], mult[4], lsh[4], nrs[4];
      for (int k = 0; k < 4; ++k) {
        const int64_t o = c0 + j + 4 * k;
        xzw[k] = vld1q_s32(xzw_.data() + o);
        bias[k] = vld1q_s32(bias_.data() + o);
        mult[k] = vld1q_s32(mult_.data() + o);
        if (kLsh) lsh[k] = vld1q_s32(lsh_.data() + o);
        nrs[k] = vld1q_s32(nrs_.data() + o);
      }
      for (int i = 0; i < nr; ++i) {
        const int32_t* a = acc + i * ldy_ + j;
        int32x4_t v[4];
        for (int k = 0; k < 4; ++k) {
          int32x4_t s = vqaddq_s32(vsubq_s32(vld1q_s32(a + 4 * k), xzw[k]), bias[k]);
          if (kLsh) s = vqshlq_s32(s, lsh[k]);
          v[k] = vqaddq_s32(vrshlq_s32(vqrdmulhq_s32(s, mult[k]), nrs[k]), vzy);
        }
        vst1q_s8(y + i * o_ + c0 + j, narrow16(v[0], v[1], v[2], v[3], vlo, vhi));
      }
    }
    for (; j + 4 <= ocg_; j += 4) {  // blocks of 4 channels (grouped convolutions with few channels per group)
      const int64_t o = c0 + j;
      const int32x4_t xzw = vld1q_s32(xzw_.data() + o), bias = vld1q_s32(bias_.data() + o), mult = vld1q_s32(mult_.data() + o);
      const int32x4_t lsh = vld1q_s32(lsh_.data() + o), nrs = vld1q_s32(nrs_.data() + o);
      for (int i = 0; i < nr; ++i) {
        int32x4_t s = vqaddq_s32(vsubq_s32(vld1q_s32(acc + i * ldy_ + j), xzw), bias);
        if (kLsh) s = vqshlq_s32(s, lsh);
        const int32x4_t v = vqaddq_s32(vrshlq_s32(vqrdmulhq_s32(s, mult), nrs), vzy);
        const int16x4_t h = vmin_s16(vmax_s16(vqmovn_s32(v), vget_low_s16(vlo)), vget_low_s16(vhi));
        const uint32_t b4 = vget_lane_u32(vreinterpret_u32_s8(vmovn_s16(vcombine_s16(h, h))), 0);
        std::memcpy(y + i * o_ + o, &b4, 4);
      }
    }
    for (int i = 0; i < nr; ++i) {
      const int32_t* a = acc + i * ldy_;
      for (int64_t jj = j; jj < ocg_; ++jj) {
        const int64_t o = c0 + jj;
        y[i * o_ + o] = ep.q(static_cast<int64_t>(a[jj]) - int64_t(xz_) * wsum_[o], o);
      }
    }
  }

  // bit output (I8Conv2D bit_thr): bit -1 iff !(acc - x_zero*wsum >= bit_thr[o]) (int32, exact: see gemm_sdot_supported).
  // 16 channels per step (compare, select one bit per lane, add across) when the group starts on a 16-channel
  // boundary, else per channel; the tile's bit rows are cleared at group 0.
  __attribute__((noinline)) void epilogue_bits(const int32_t* acc, int nr, int64_t gi, uint32_t* yb) const {
    const int64_t c0 = gi * ocg_;
    if (gi == 0) std::memset(yb, 0, static_cast<size_t>(nr * bw_) * sizeof(uint32_t));
    int64_t j = 0;
    if (c0 % 16 == 0) {
      static const uint32_t kSel[16] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7,
                                        1u << 8, 1u << 9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15};
      uint32x4_t sel[4];
      for (int k = 0; k < 4; ++k) sel[k] = vld1q_u32(kSel + 4 * k);
      for (; j + 16 <= ocg_; j += 16) {
        int32x4_t xzw[4], thr[4];
        for (int k = 0; k < 4; ++k) {
          xzw[k] = vld1q_s32(xzw_.data() + c0 + j + 4 * k);
          thr[k] = vld1q_s32(bthr_.data() + c0 + j + 4 * k);
        }
        for (int i = 0; i < nr; ++i) {
          const int32_t* a = acc + i * ldy_ + j;
          uint32x4_t m = vandq_u32(vcltq_s32(vsubq_s32(vld1q_s32(a), xzw[0]), thr[0]), sel[0]);
          for (int k = 1; k < 4; ++k) m = vorrq_u32(m, vandq_u32(vcltq_s32(vsubq_s32(vld1q_s32(a + 4 * k), xzw[k]), thr[k]), sel[k]));
          const uint16_t h = static_cast<uint16_t>(vaddvq_u32(m));
          std::memcpy(reinterpret_cast<uint8_t*>(yb + i * bw_) + (c0 + j) / 8, &h, sizeof(h));
        }
      }
    }
    for (int i = 0; i < nr; ++i)
      for (int64_t jj = j; jj < ocg_; ++jj) {
        const int64_t o = c0 + jj;
        if (!(static_cast<int64_t>(acc[i * ldy_ + jj]) - int64_t(xz_) * wsum_[o] >= bthr_[o])) yb[i * bw_ + (o >> 5)] |= 1u << (o & 31);
      }
  }

  bool dense_ = false, direct_ = false, f32_ = false, any_lsh_ = false, l8_ = false, has_bias_f_ = false, bits_ = false;
  bool mm_ = false;  // SMMLA tiles (i8mm)
  bool runs_ = false;  // SMMLA with the kernel-row run layout of K (pack_runs)
  int64_t rl8_ = 0, kc8_ = 0;
  std::vector<int8_t> padded_;  // runs_: padded input copy (pack_runs)
  int64_t ph_ = 0, pw_ = 0;
  int64_t bw_ = 0;
  std::vector<int32_t> bthr_;
  std::vector<float> scale_f_, bias_f_;
  l8neon::EncTables enc_;
  Conv2DGeom g_{};
  int64_t o_ = 0, k_ = 0, kp_ = 0, rows_ = 0, kc_ = 0, groups_ = 1, ocg_ = 0, obg_ = 0, ldy_ = 0;
  int32_t xz_ = 0;
  int8_t padq_ = 0;
  std::vector<int8_t> wpack_, zero_row_;
  std::vector<int32_t> wsum_, xzw_, bias_, mult_, lsh_, nrs_;
};
// groups == 1 and K <= 1032 (the range of the previous fp32-accumulator kernel; same integers), and any
// other K or grouped convolution (except the depthwise ones of i8conv2d.neon_depthwise) whose accumulator is
// provably exact in int32: |sum (x - x_zero) w| <= K (128 + |x_zero|) 127 < 2^31 (and so is the raw sum).
bool depthwise_neon_supported(const KernelContext& ctx);
bool gemm_sdot_supported(const KernelContext& ctx) {
  const Tensor& w = ctx.in(1);
  const int64_t k = w.size() / w.dim(0);
  const int64_t xz = ctx.attrs().get_int("x_zero", 0);
  const bool grouped = ctx.node->op == "I8Conv2D" && ctx.attrs().get_int("groups", 1) != 1;
  if (grouped && depthwise_neon_supported(ctx)) return false;
  if ((k > 1032 || grouped) && k * (128 + std::abs(xz)) * 127 >= (int64_t(1) << 31)) return false;
  return true;
}

// ------------------------------------------------------ I8Lut (per channel)

// 16x16 byte transpose (TRN1/TRN2 at 8, 16, 32 and 64 bits); its own inverse.
inline void transpose16x16(uint8x16_t v[16]) {
  uint8x16_t b[16];
  for (int i = 0; i < 16; i += 2) {
    b[i] = vtrn1q_u8(v[i], v[i + 1]);
    b[i + 1] = vtrn2q_u8(v[i], v[i + 1]);
  }
  for (int k = 0; k < 16; k += 4)
    for (int i = 0; i < 2; ++i) {
      const uint16x8_t x = vreinterpretq_u16_u8(b[k + i]), y = vreinterpretq_u16_u8(b[k + i + 2]);
      v[k + i] = vreinterpretq_u8_u16(vtrn1q_u16(x, y));
      v[k + i + 2] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
    }
  for (int k = 0; k < 16; k += 8)
    for (int i = 0; i < 4; ++i) {
      const uint32x4_t x = vreinterpretq_u32_u8(v[k + i]), y = vreinterpretq_u32_u8(v[k + i + 4]);
      b[k + i] = vreinterpretq_u8_u32(vtrn1q_u32(x, y));
      b[k + i + 4] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
    }
  for (int i = 0; i < 8; ++i) {
    const uint64x2_t x = vreinterpretq_u64_u8(b[i]), y = vreinterpretq_u64_u8(b[i + 8]);
    v[i] = vreinterpretq_u8_u64(vtrn1q_u64(x, y));
    v[i + 8] = vreinterpretq_u8_u64(vtrn2q_u64(x, y));
  }
}

// Table row = index along `axis`; rows of `inner` elements share one row, so the loop walks
// (outer, channel) blocks with one table pointer each (no division per element). int8 tables
// and inner >= 16: TBL/TBX, the last vector of a block overlapping the previous one (outputs
// never alias inputs, so it rewrites the same values).
class LutRowsNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    int64_t ax = ctx.attrs().get_int("axis", -1);
    if (ax < 0) ax += x.rank();
    int64_t inner = 1;
    for (int d = static_cast<int>(ax) + 1; d < x.rank(); ++d) inner *= x.dim(d);
    const int64_t c = x.dim(ax), blocks = x.size() / inner;  // blocks = outer * c
    const int8_t* q = x.data<int8_t>();
    const bool f32 = ctx.in(1).dtype() == DType::kF32;
    ctx.pool->parallel_for(blocks, std::max<int64_t>(1, kSerial / std::max<int64_t>(inner, 1)), [&](int64_t b0, int64_t b1) {
      const uint8x16_t k80 = vdupq_n_u8(0x80);
      for (int64_t b = b0; b < b1; ++b) {
        const int64_t row = b % c;
        const int8_t* qb = q + b * inner;
        if (f32) {
          const float* t = ctx.in(1).data<float>() + row * 256 + 128;
          float* y = ctx.out(0).data<float>() + b * inner;
          for (int64_t i = 0; i < inner; ++i) y[i] = t[qb[i]];
        } else if (inner >= 16) {
          const auto* t = reinterpret_cast<const uint8_t*>(ctx.in(1).data<int8_t>() + row * 256);
          const uint8x16x4_t t0 = vld1q_u8_x4(t), t1 = vld1q_u8_x4(t + 64), t2 = vld1q_u8_x4(t + 128), t3 = vld1q_u8_x4(t + 192);
          auto* y = reinterpret_cast<uint8_t*>(ctx.out(0).data<int8_t>() + b * inner);
          const auto* qu = reinterpret_cast<const uint8_t*>(qb);
          const uint8x16_t k64 = vdupq_n_u8(64);
          for (int64_t i = 0;; i += 16) {
            if (i + 16 > inner) i = inner - 16;
            uint8x16_t u = veorq_u8(vld1q_u8(qu + i), k80);
            uint8x16_t r = vqtbl4q_u8(t0, u);
            u = vsubq_u8(u, k64);
            r = vqtbx4q_u8(r, t1, u);
            u = vsubq_u8(u, k64);
            r = vqtbx4q_u8(r, t2, u);
            u = vsubq_u8(u, k64);
            vst1q_u8(y + i, vqtbx4q_u8(r, t3, u));
            if (i + 16 >= inner) break;
          }
        } else {
          const int8_t* t = ctx.in(1).data<int8_t>() + row * 256 + 128;
          int8_t* y = ctx.out(0).data<int8_t>() + b * inner;
          for (int64_t i = 0; i < inner; ++i) y[i] = t[qb[i]];
        }
      }
    });
  }
};

// Per-channel tables on the last axis: rows of C. Blocks of 16 rows x 16 channels are transposed
// so that each vector holds one channel of 16 rows, looked up in that channel's table with
// TBL/TBX over its four 64-byte quarters, and transposed back. The last block of rows may be partial (copied through a
// zero-filled 16x16 buffer, only its valid rows stored back); when C is not a multiple of 16 the last
// channel block overlaps the previous one (same thread; outputs never alias inputs, so it rewrites
// the same values). C < 16: scalar lookups.
//
// One 16x16 block: rows src + i*ss (16 bytes each) -> dst + i*ds, channel j uses table tab + 256*j.
inline __attribute__((always_inline)) void lut_block16(const uint8_t* __restrict src, int64_t ss, uint8_t* __restrict dst, int64_t ds,
                                                       const uint8_t* __restrict tab) {
  const uint8x16_t k80 = vdupq_n_u8(0x80), k64 = vdupq_n_u8(64);
  uint8x16_t v[16];
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; ++i) v[i] = vld1q_u8(src + i * ss);
  transpose16x16(v);
  // 256-entry lookup of each channel vector (u = q ^ 0x80: TBL on the first quarter, TBX with u - 64k on
  // the others, out-of-range indices keep the lane); inline asm so that the 16 consecutive tables are read with
  // post-incremented 64-byte loads (no address arithmetic, no spills of the transposed block)
  const uint8_t* t = tab;
#pragma clang loop unroll(full)
  for (int j = 0; j < 16; ++j) {
    uint8x16_t u = veorq_u8(v[j], k80), r;
    asm("ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[t]], #64\n\t"
        "ld1 {v28.16b, v29.16b, v30.16b, v31.16b}, [%[t]], #64\n\t"
        "tbl %[r].16b, {v24.16b, v25.16b, v26.16b, v27.16b}, %[u].16b\n\t"
        "sub %[u].16b, %[u].16b, %[k].16b\n\t"
        "tbx %[r].16b, {v28.16b, v29.16b, v30.16b, v31.16b}, %[u].16b\n\t"
        "ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [%[t]], #64\n\t"
        "ld1 {v28.16b, v29.16b, v30.16b, v31.16b}, [%[t]], #64\n\t"
        "sub %[u].16b, %[u].16b, %[k].16b\n\t"
        "tbx %[r].16b, {v24.16b, v25.16b, v26.16b, v27.16b}, %[u].16b\n\t"
        "sub %[u].16b, %[u].16b, %[k].16b\n\t"
        "tbx %[r].16b, {v28.16b, v29.16b, v30.16b, v31.16b}, %[u].16b\n\t"
        : [r] "=&w"(r), [u] "+w"(u), [t] "+r"(t)
        : [k] "w"(k64)
        : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
    v[j] = r;
  }
  transpose16x16(v);
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; ++i) vst1q_u8(dst + i * ds, v[i]);
}

void lut_channel_rows(const uint8_t* __restrict q, uint8_t* __restrict y, int64_t c, int64_t nr, const uint8_t* __restrict tab) {
  for (int64_t ch = 0;; ch += 16) {
    if (ch + 16 > c) ch = c - 16;
    if (nr == 16) {
      lut_block16(q + ch, c, y + ch, c, tab + ch * 256);
    } else {
      alignas(16) uint8_t tmp[16 * 16] = {};
      for (int64_t i = 0; i < nr; ++i) std::memcpy(tmp + 16 * i, q + i * c + ch, 16);
      lut_block16(tmp, 16, tmp, 16, tab + ch * 256);
      for (int64_t i = 0; i < nr; ++i) std::memcpy(y + i * c + ch, tmp + 16 * i, 16);
    }
    if (ch + 16 >= c) break;
  }
}

class LutChannelsNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t c = x.dim(-1), rows = x.size() / c;
    const int8_t* q = x.data<int8_t>();
    const int8_t* tab = ctx.in(1).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    if (c < 16) {
      ctx.pool->parallel_for(rows, std::max<int64_t>(1, kSerial / std::max<int64_t>(c, 1)), [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          const int8_t* qr = q + r * c;
          int8_t* yr = y + r * c;
          const int8_t* t = tab + 128;
          for (int64_t ch = 0; ch < c; ++ch, t += 256) yr[ch] = t[qr[ch]];
        }
      });
      return;
    }
    const auto* tu = reinterpret_cast<const uint8_t*>(tab);
    const auto* qu = reinterpret_cast<const uint8_t*>(q);
    auto* yu = reinterpret_cast<uint8_t*>(y);
    const int64_t nblk = (rows + 15) / 16;
    ctx.pool->parallel_for(nblk, std::max<int64_t>(1, kSerial / (16 * c)), [&](int64_t b0, int64_t b1) {
      for (int64_t b = b0; b < b1; ++b)
        lut_channel_rows(qu + b * 16 * c, yu + b * 16 * c, c, std::min<int64_t>(16, rows - b * 16), tu);
    });
  }
};
bool lut_rows_supported(const KernelContext& ctx) {
  int64_t ax = ctx.attrs().get_int("axis", -1);
  if (ax < 0) ax += ctx.in(0).rank();
  return ctx.in(1).dim(0) > 1 && ax != ctx.in(0).rank() - 1;
}
bool lut_channels_supported(const KernelContext& ctx) {
  int64_t ax = ctx.attrs().get_int("axis", -1);
  if (ax < 0) ax += ctx.in(0).rank();
  return ctx.in(1).dim(0) > 1 && ax == ctx.in(0).rank() - 1 && ctx.in(1).dtype() == DType::kI8;
}

// ------------------------------------------------------------- I8Transpose
// Output-order walk with incremental input offsets; runs along an unchanged last axis are memcpy'd.
class TransposeNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    auto p = ctx.attrs().get_ints("perm", {});
    r_ = x.rank();
    std::vector<int64_t> xs(r_, 1);
    for (int d = r_ - 2; d >= 0; --d) xs[d] = xs[d + 1] * x.dim(d + 1);
    ostride_.resize(r_);
    odim_.resize(r_);
    for (int d = 0; d < r_; ++d) {
      ostride_[d] = xs[p[d]];
      odim_[d] = ctx.out(0).dim(d);
    }
    run_ = (p[r_ - 1] == r_ - 1) ? odim_[r_ - 1] : 1;  // contiguous run length
  }
  void run(const KernelContext& ctx) override {
    const int8_t* x = ctx.in(0).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int nd = run_ > 1 ? r_ - 1 : r_;  // dims walked element-wise
    const int64_t outer = ctx.out(0).size() / (run_ > 1 ? run_ : 1);
    const int64_t d0 = odim_[0], rest = outer / std::max<int64_t>(d0, 1);
    ctx.pool->parallel_for(d0, std::max<int64_t>(1, kSerial / std::max<int64_t>(rest * run_, 1)), [&](int64_t a0, int64_t a1) {
      std::vector<int64_t> idx(nd, 0);
      for (int64_t a = a0; a < a1; ++a) {
        std::fill(idx.begin(), idx.end(), 0);
        int64_t off = a * ostride_[0];
        int8_t* yo = y + a * rest * (run_ > 1 ? run_ : 1);
        for (int64_t e = 0; e < rest; ++e) {
          if (run_ > 1) std::memcpy(yo + e * run_, x + off, static_cast<size_t>(run_));
          else yo[e] = x[off];
          for (int d = nd - 1; d >= 1; --d) {  // increment the multi-index over dims 1..nd-1
            off += ostride_[d];
            if (++idx[d] < odim_[d]) break;
            off -= ostride_[d] * odim_[d];
            idx[d] = 0;
          }
        }
      }
    });
  }

 private:
  int r_ = 0;
  int64_t run_ = 1;
  std::vector<int64_t> ostride_, odim_;
};

// ------------------------------------------------- I8ReduceMean (last axis)
// Rows of len >= 16: one row at a time (pairwise widening adds). Short rows, len in {2, 4, 8}: 64 bytes
// = 64/len rows per step, row sums by SADDLP + ADDP levels in int16 (|sum| <= 8*128), then
// t = sum - len*x_zero and the reference epilogue (requantized in vector lanes; rounding division
// per output).
class MeanLastNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Json& a = ctx.attrs();
    const int64_t len = x.dim(-1), rows = x.size() / len;
    const int64_t zx = a.get_int("x_zero", 0), zy = a.get_int("y_zero", 0);
    const bool rq = a.contains("mult");
    const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), s = static_cast<int32_t>(a.get_int("shift", 0));
    const int8_t* xd = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    if (len < 16) return run_short(ctx, len, rows, zx, zy, rq, m, s, xd, y);
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, kSerial / std::max<int64_t>(len, 1)), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int8_t* xr = xd + r * len;
        int32x4_t acc = vdupq_n_s32(0);
        int64_t i = 0;
        for (; i + 16 <= len; i += 16) {
          const int8x16_t v = vld1q_s8(xr + i);
          acc = vpadalq_s16(acc, vpaddlq_s8(v));
        }
        int64_t sum = vaddvq_s32(acc);
        for (; i < len; ++i) sum += xr[i];
        const int64_t t = sum - len * zx;
        y[r] = rq ? clamp8(zy + q8::requant(t, m, s)) : clamp8(zx + q8::rdiv(t, len));
      }
    });
  }

 private:
  static void run_short(const KernelContext& ctx, int64_t len, int64_t rows, int64_t zx, int64_t zy, bool rq, int32_t m, int32_t s,
                        const int8_t* xd, int8_t* y) {
    const int64_t per = 64 / len;  // rows per 64-byte step
    const int64_t steps = rows / per;
    auto scalar_row = [&](int64_t r) {
      int64_t sum = 0;
      for (int64_t i = 0; i < len; ++i) sum += xd[r * len + i];
      const int64_t t = sum - len * zx;
      y[r] = rq ? clamp8(zy + q8::requant(t, m, s)) : clamp8(zx + q8::rdiv(t, len));
    };
    ctx.pool->parallel_for(steps, std::max<int64_t>(1, kSerial / 64), [&](int64_t b0, int64_t b1) {
      const Rq vr(m, s);
      const int32x4_t vt = vdupq_n_s32(static_cast<int32_t>(len * zx)), vzy = vdupq_n_s32(static_cast<int32_t>(zy));
      for (int64_t b = b0; b < b1; ++b) {
        const int8_t* xp = xd + b * 64;
        int16x8_t v[4];
        for (int k = 0; k < 4; ++k) v[k] = vpaddlq_s8(vld1q_s8(xp + 16 * k));  // sums of 2
        int nv = 4;
        for (int64_t w = 2; w < len; w *= 2) {  // sums of 4, 8: ADDP halves the vector count
          for (int k = 0; k < nv / 2; ++k) v[k] = vpaddq_s16(v[2 * k], v[2 * k + 1]);
          nv /= 2;
        }
        // nv vectors of 8 row sums (int16) = per rows, in row order
        int8_t* yp = y + b * per;
        for (int k = 0; k < nv; ++k) {
          const int32x4_t lo = vsubq_s32(vmovl_s16(vget_low_s16(v[k])), vt), hi = vsubq_s32(vmovl_high_s16(v[k]), vt);
          if (rq) {
            const int32x4_t a = vqaddq_s32(vr(lo), vzy), c = vqaddq_s32(vr(hi), vzy);
            const int16x8_t n16 = vqmovn_high_s32(vqmovn_s32(a), c);
            vst1_s8(yp + 8 * k, vqmovn_s16(n16));
          } else {
            int32_t t[8];
            vst1q_s32(t, lo);
            vst1q_s32(t + 4, hi);
            for (int j = 0; j < 8; ++j) yp[8 * k + j] = clamp8(zx + q8::rdiv(t[j], len));
          }
        }
      }
    });
    for (int64_t r = steps * per; r < rows; ++r) scalar_row(r);
  }
};
// last axis; rows of at least 16 elements, or of 2, 4 or 8 (64-byte steps); |len*x_zero| within int32 lanes
bool mean_last_supported(const KernelContext& ctx) {
  if (ctx.node->op != "I8ReduceMean") return false;
  int64_t ax = ctx.attrs().get_int("axis", -1);
  if (ax < 0) ax += ctx.in(0).rank();
  const int64_t len = ctx.in(0).dim(-1);
  if (ax != ctx.in(0).rank() - 1) return false;
  if (len == 2 || len == 4 || len == 8) return std::abs(ctx.attrs().get_int("x_zero", 0)) <= (int64_t(1) << 26);
  return len >= 16 && len <= (int64_t(1) << 24);
}
// --------------------------------- I8GlobalAvgPool / I8ReduceMean (middle axis)
// [outer][len][inner] -> [outer][inner], inner >= 16: per 16 channels, sums of len rows in int16 lanes
// (SADDW; at most 256 rows of |x| <= 128 per partial sum, |sum| <= 32768 with -32768 the only extreme,
// so int16 is exact) flushed into int32, then the reference epilogue on t = sum - len*x_zero:
// clamp8(y_zero + requant(t)) (vector) or clamp8(x_zero + rdiv(t, len)) (scalar, outputs only).
class MeanMidNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Json& a = ctx.attrs();
    int64_t outer, len, inner;
    geometry(ctx, outer, len, inner);
    const int64_t zx = a.get_int("x_zero", 0), zy = a.get_int("y_zero", 0);
    const bool rq = a.contains("mult");
    const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), s = static_cast<int32_t>(a.get_int("shift", 0));
    const int8_t* xd = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t nblk = (inner + 15) / 16;
    ctx.pool->parallel_for(outer * nblk, std::max<int64_t>(1, kSerial / std::max<int64_t>(16 * len, 1)), [&](int64_t b0, int64_t b1) {
      const Rq vr(m, s);
      const int32x4_t vt = vdupq_n_s32(static_cast<int32_t>(len * zx)), vzy = vdupq_n_s32(static_cast<int32_t>(zy));
      const int16x8_t vlo = vdupq_n_s16(-128), vhi = vdupq_n_s16(127);
      for (int64_t b = b0; b < b1; ++b) {
        const int64_t o = b / nblk;
        const int64_t i = std::min((b % nblk) * 16, inner - 16);  // a partial last block reads the previous channels again
        const int8_t* xp = xd + o * len * inner + i;
        int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
        for (int64_t l0 = 0; l0 < len; l0 += 256) {
          const int64_t l1 = std::min(len, l0 + 256);
          int16x8_t lo16 = vdupq_n_s16(0), hi16 = vdupq_n_s16(0);
          for (int64_t l = l0; l < l1; ++l) {
            const int8x16_t v = vld1q_s8(xp + l * inner);
            lo16 = vaddw_s8(lo16, vget_low_s8(v));
            hi16 = vaddw_high_s8(hi16, v);
          }
          acc[0] = vaddw_s16(acc[0], vget_low_s16(lo16));
          acc[1] = vaddw_high_s16(acc[1], lo16);
          acc[2] = vaddw_s16(acc[2], vget_low_s16(hi16));
          acc[3] = vaddw_high_s16(acc[3], hi16);
        }
        int8_t out[16];
        if (rq) {
          int32x4_t r[4];
          for (int k = 0; k < 4; ++k) r[k] = vqaddq_s32(vr(vsubq_s32(acc[k], vt)), vzy);
          vst1q_s8(out, narrow16(r[0], r[1], r[2], r[3], vlo, vhi));
        } else {
          int32_t sums[16];
          for (int k = 0; k < 4; ++k) vst1q_s32(sums + 4 * k, acc[k]);
          for (int k = 0; k < 16; ++k) out[k] = clamp8(zx + q8::rdiv(static_cast<int64_t>(sums[k]) - len * zx, len));
        }
        // a last block that overlaps the previous one stores only its new channels (no shared bytes)
        const int64_t first = b % nblk == nblk - 1 ? (b % nblk) * 16 - i : 0;
        std::memcpy(y + o * inner + i + first, out + first, static_cast<size_t>(16 - first));
      }
    });
  }
  static void geometry(const KernelContext& ctx, int64_t& outer, int64_t& len, int64_t& inner) {
    const Tensor& x = ctx.in(0);
    if (ctx.node->op == "I8GlobalAvgPool") {
      outer = x.dim(0);
      len = x.dim(1) * x.dim(2);
      inner = x.dim(3);
      return;
    }
    int64_t ax = ctx.attrs().get_int("axis", -1);
    if (ax < 0) ax += x.rank();
    outer = inner = 1;
    for (int d = 0; d < ax; ++d) outer *= x.dim(d);
    for (int d = static_cast<int>(ax) + 1; d < x.rank(); ++d) inner *= x.dim(d);
    len = x.dim(ax);
  }
};
// inner >= 16 (ReduceMean on the last axis: i8reducemean.neon); len*(128 + |x_zero|) < 2^31 so that the
// int32 sum, len*x_zero and t are exact
bool mean_mid_supported(const KernelContext& ctx) {
  if (ctx.node->op == "I8GlobalAvgPool" && ctx.in(0).rank() != 4) return false;
  int64_t outer, len, inner;
  MeanMidNeon::geometry(ctx, outer, len, inner);
  const int64_t zx = ctx.attrs().get_int("x_zero", 0);
  return inner >= 16 && len >= 1 && len * (128 + std::abs(zx)) < (int64_t(1) << 31);
}

// ------------------------------------------------------------------ I8Pad
// Output filled with pad_q, then every row of the last dimension copied to its place (odometer over the
// leading dimensions; no per-element index arithmetic). Non-negative pads only.
class PadNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& yt = ctx.out(0);
    auto p = ctx.attrs().get_ints("pads", {});
    const int8_t v = static_cast<int8_t>(ctx.attrs().get_int("pad_q", 0));
    int8_t* y = ctx.out(0).data<int8_t>();
    const int8_t* xd = x.data<int8_t>();
    std::memset(y, static_cast<uint8_t>(v), static_cast<size_t>(yt.size()));
    const int r = x.rank();
    const int64_t row = x.dim(r - 1);
    if (row == 0 || x.size() == 0) return;
    // output strides
    std::vector<int64_t> ys(static_cast<size_t>(r), 1), idx(static_cast<size_t>(r), 0);
    for (int d = r - 2; d >= 0; --d) ys[d] = ys[d + 1] * yt.dim(d + 1);
    int64_t off = 0;
    for (int d = 0; d < r; ++d) off += p[2 * d] * ys[d];
    const int64_t rows = x.size() / row;
    for (int64_t i = 0; i < rows; ++i) {
      int64_t o = off;
      for (int d = 0; d < r - 1; ++d) o += idx[d] * ys[d];
      std::memcpy(y + o, xd + i * row, static_cast<size_t>(row));
      for (int d = r - 2; d >= 0; --d) {
        if (++idx[d] < x.dim(d)) break;
        idx[d] = 0;
      }
    }
  }
};
bool pad_neon_supported(const KernelContext& ctx) {
  for (auto v : ctx.attrs().get_ints("pads", {}))
    if (v < 0) return false;
  return ctx.in(0).rank() >= 1;
}

// ------------------------------------------------------------ I8Concat
class ConcatNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& y = ctx.out(0);
    const int r = y.rank();
    int64_t ax = ctx.attrs().get_int("axis", -1);
    if (ax < 0) ax += r;
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < ax; ++d) outer *= y.dim(d);
    for (int d = static_cast<int>(ax) + 1; d < r; ++d) inner *= y.dim(d);
    auto zs = ctx.attrs().get_ints("zeros", {}), ms = ctx.attrs().get_ints("mults", {}), ss = ctx.attrs().get_ints("shifts", {});
    const int32_t zy = static_cast<int32_t>(ctx.attrs().get_int("y_zero", 0));
    int8_t* yd = ctx.out(0).data<int8_t>();
    const int64_t ystride = y.dim(ax) * inner;
    std::vector<int64_t> offs(ctx.inputs.size());
    int64_t off = 0;
    for (size_t k = 0; k < ctx.inputs.size(); ++k) {
      offs[k] = off;
      off += ctx.in(k).dim(ax) * inner;
    }
    ctx.pool->parallel_for(outer, std::max<int64_t>(1, kSerial / std::max<int64_t>(ystride, 1)), [&](int64_t o0, int64_t o1) {
      const int16x8_t vlo = vdupq_n_s16(-128), vhi = vdupq_n_s16(127);
      const int32x4_t vzy = vdupq_n_s32(zy);
      for (int64_t o = o0; o < o1; ++o)
        for (size_t k = 0; k < ctx.inputs.size(); ++k) {
          const int64_t chunk = ctx.in(k).dim(ax) * inner;
          const int8_t* src = ctx.in(k).data<int8_t>() + o * chunk;
          int8_t* dst = yd + o * ystride + offs[k];
          if (ms[k] == 0) {
            std::memcpy(dst, src, static_cast<size_t>(chunk));
            continue;
          }
          const int32_t m = static_cast<int32_t>(ms[k]), sh = static_cast<int32_t>(ss[k]), z = static_cast<int32_t>(zs[k]);
          const Rq rq(m, sh);
          const int32x4_t vz = vdupq_n_s32(z);
          int64_t i = 0;
          if (z >= -32768 && z <= 32767)
            for (; i + 16 <= chunk; i += 16) {
              int32x4_t w[4];
              widen16(vld1q_s8(src + i), w);
              for (int t = 0; t < 4; ++t) w[t] = vqaddq_s32(rq(vsubq_s32(w[t], vz)), vzy);
              vst1q_s8(dst + i, narrow16(w[0], w[1], w[2], w[3], vlo, vhi));
            }
          for (; i < chunk; ++i) dst[i] = clamp8(zy + q8::requant(src[i] - zs[k], m, sh));
        }
    });
  }
};

// --------------------------------------------------------- I8MatMul (SDOT)
// Per batch entry: A = rows of a (M x K) and W = columns of b ([N][K]) packed in the int8_neon tile
// layout at run time; raw products sum_k a*b by the SDOT micro-kernel (|.| <= K*128*128 < 2^24 for
// K <= 1024: exact in fp32), then acc = raw - za*colsum_b[j] - zb*rowsum_a[i] + K*za*zb.
class MatMulSdot : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    kc_ = (g_.k + kC - 1) / kC;
    pb_ = (g_.m + 7) / 8;
    ob_ = (g_.n + 7) / 8;
    one_.assign(static_cast<size_t>(ob_ * 8), 0.f);
    for (int64_t j = 0; j < g_.n; ++j) one_[j] = 1.f;
    zero_.assign(static_cast<size_t>(ob_ * 8), 0.f);
  }
  void run(const KernelContext& ctx) override {
    const Json& at = ctx.attrs();
    const int64_t za = at.get_int("a_zero", 0), zb = at.get_int("b_zero", 0);
    const bool f32 = at.get_string("output", "i8") == "f32";
    const float sc = static_cast<float>(at.get_number("scale", 1.0));
    const int32_t m = static_cast<int32_t>(at.get_int("mult", 0)), s = static_cast<int32_t>(at.get_int("shift", 0));
    const int64_t zy = at.get_int("y_zero", 0);
    const int lo = static_cast<int>(at.get_int("qmin", -128)), hi = static_cast<int>(at.get_int("qmax", 127));
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    const int64_t M = g_.m, N = g_.n, K = g_.k;
    ctx.pool->parallel_for(g_.batch, 1, [&](int64_t b0, int64_t b1) {
      std::vector<int8_t> ap(static_cast<size_t>(pb_ * kc_ * 8 * kC)), wp(static_cast<size_t>(ob_ * kc_ * 8 * kC));
      std::vector<float> acc(static_cast<size_t>(M * N));
      std::vector<int32_t> rs(static_cast<size_t>(M)), cs(static_cast<size_t>(N));
      // vector epilogue only where every int32 term is exact: |v| <= K (128^2 + 2 * 128 |z| + |z|^2) < 2^31
      const bool vec = std::abs(za) <= 256 && std::abs(zb) <= 256;
      const int32_t kzz = vec ? static_cast<int32_t>(K * za * zb) : 0;
      for (int64_t bt = b0; bt < b1; ++bt) {
        const int8_t* a = qa + bt * M * K;
        const int8_t* b = qb + bt * K * N;
        std::fill(ap.begin(), ap.end(), 0);
        std::fill(wp.begin(), wp.end(), 0);
        pack_rows(a, M, ap.data(), rs.data());
        if (g_.transpose_b) pack_rows(b, N, wp.data(), cs.data());
        else pack_cols(b, wp.data(), cs.data());
        int8_neon::GemmArgs ga;
        ga.a = ap.data();
        ga.w = wp.data();
        ga.scale = one_.data();
        ga.bias = zero_.data();
        ga.y = acc.data();
        ga.rows = M;
        ga.cols = N;
        ga.kc = kc_;
        ga.act = 0;
        for (int64_t t = 0; t < pb_; ++t) int8_neon::gemm_tiles_sdot(ga, t, 0, ob_);
        // v = acc - za cs[j] - zb rs[i] + K za zb: int32 is exact (|acc| <= K 128^2 <= 2^24 for K <= 1024, the sums
        // <= K 128 |z|), so the scalar int64 sum, sat32 (identity) and requant equal the vector int32 ones
        // (acc is an integer-valued float: the conversion is exact).
        const Rq rq(m, s);
        const int32x4_t vzy = vdupq_n_s32(static_cast<int32_t>(zy)), vza = vdupq_n_s32(static_cast<int32_t>(za));
        const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
        const float32x4_t vsc = vdupq_n_f32(sc);
        for (int64_t i = 0; i < M; ++i) {
          const int32_t ri = kzz - static_cast<int32_t>(zb) * rs[i];
          const int32x4_t vri = vdupq_n_s32(ri);
          const float* ar = acc.data() + i * N;
          const int64_t at0 = (bt * M + i) * N;
          int64_t j = 0;
          auto val = [&](int64_t jj) {
            return vaddq_s32(vmlsq_s32(vcvtq_s32_f32(vld1q_f32(ar + jj)), vza, vld1q_s32(cs.data() + jj)), vri);
          };
          if (!vec) {
          } else if (f32) {
            float* yo = ctx.out(0).data<float>() + at0;
            for (; j + 4 <= N; j += 4) vst1q_f32(yo + j, vmulq_f32(vcvtq_f32_s32(val(j)), vsc));
          } else {
            int8_t* yo = ctx.out(0).data<int8_t>() + at0;
            for (; j + 16 <= N; j += 16) {
              int32x4_t r[4];
              for (int k = 0; k < 4; ++k) r[k] = vqaddq_s32(rq(val(j + 4 * k)), vzy);
              vst1q_s8(yo + j, narrow16(r[0], r[1], r[2], r[3], vlo, vhi));
            }
          }
          for (; j < N; ++j) {
            const int64_t v = static_cast<int64_t>(ar[j]) - za * cs[j] - zb * rs[i] + K * za * zb;
            const int64_t at_ = at0 + j;
            if (f32) ctx.out(0).data<float>()[at_] = static_cast<float>(v) * sc;
            else ctx.out(0).data<int8_t>()[at_] = clamp8(zy + q8::requant(q8::sat32(v), m, s), lo, hi);
          }
        }
      }
    });
  }

 private:
  // Rows of a [R][K] row-major int8 matrix into the SDOT layout [R/8][kc][8][kC] (K padding left 0) + row sums.
  void pack_rows(const int8_t* x, int64_t R, int8_t* dst, int32_t* sums) const {
    const int64_t K = g_.k, kfull = K / kC;
    for (int64_t i = 0; i < R; ++i) {
      const int8_t* xr = x + i * K;
      int8_t* d = dst + ((i / 8) * kc_ * 8 + i % 8) * kC;
      for (int64_t c = 0; c < kfull; ++c) std::memcpy(d + c * 8 * kC, xr + c * kC, kC);
      for (int64_t k = kfull * kC; k < K; ++k) d[kfull * 8 * kC + k % kC] = xr[k];
      int32x4_t sv = vdupq_n_s32(0);
      int64_t k = 0;
      for (; k + 16 <= K; k += 16) sv = vpadalq_s16(sv, vpaddlq_s8(vld1q_s8(xr + k)));
      int32_t sum = vaddvq_s32(sv);
      for (; k < K; ++k) sum += xr[k];
      sums[i] = sum;
    }
  }
  // Columns of b [K][N] (not transposed) into the same layout: blocks of 4 k x 8 columns transposed with two zip
  // levels (bytes, then 16-bit pairs), column sums in int32.
  void pack_cols(const int8_t* b, int8_t* dst, int32_t* sums) const {
    const int64_t K = g_.k, N = g_.n, k4 = K / kC * kC, n8 = N / 8 * 8;
    for (int64_t j0 = 0; j0 < n8; j0 += 8) {
      int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
      int8_t* d = dst + (j0 / 8) * kc_ * 8 * kC;
      for (int64_t k = 0; k < k4; k += kC) {
        const int8x8_t r0 = vld1_s8(b + k * N + j0), r1 = vld1_s8(b + (k + 1) * N + j0);
        const int8x8_t r2 = vld1_s8(b + (k + 2) * N + j0), r3 = vld1_s8(b + (k + 3) * N + j0);
        const int16x8_t z01 = vreinterpretq_s16_s8(vcombine_s8(vzip1_s8(r0, r1), vzip2_s8(r0, r1)));
        const int16x8_t z23 = vreinterpretq_s16_s8(vcombine_s8(vzip1_s8(r2, r3), vzip2_s8(r2, r3)));
        vst1q_s8(d + (k / kC) * 8 * kC, vreinterpretq_s8_s16(vzip1q_s16(z01, z23)));
        vst1q_s8(d + (k / kC) * 8 * kC + 16, vreinterpretq_s8_s16(vzip2q_s16(z01, z23)));
        const int16x8_t t = vaddq_s16(vaddl_s8(r0, r1), vaddl_s8(r2, r3));
        s0 = vaddw_s16(s0, vget_low_s16(t));
        s1 = vaddw_high_s16(s1, t);
      }
      vst1q_s32(sums + j0, s0);
      vst1q_s32(sums + j0 + 4, s1);
      for (int64_t k = k4; k < K; ++k)
        for (int64_t j = j0; j < j0 + 8; ++j) {
          d[((k / kC) * 8 + j % 8) * kC + k % kC] = b[k * N + j];
          sums[j] += b[k * N + j];
        }
    }
    for (int64_t j = n8; j < N; ++j) {
      int32_t sum = 0;
      for (int64_t k = 0; k < K; ++k) {
        const int8_t v = b[k * N + j];
        dst[(((j / 8) * kc_ + k / kC) * 8 + j % 8) * kC + k % kC] = v;
        sum += v;
      }
      sums[j] = sum;
    }
  }

  MatMulGeom g_;
  int64_t kc_ = 0, pb_ = 0, ob_ = 0;
  std::vector<float> one_, zero_;
};
bool matmul_sdot_supported(const KernelContext& ctx) {
  auto g = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
  return g.k <= 1024;
}

// ---------------------------------------------------------------- I8Mul
// Same shapes, or b broadcast as [N, 1, 1, C] / [1, ..., C] rows (ChannelGate): per row of C channels.
class MulNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Shape& a = ctx.in(0).shape();
    const Shape& b = ctx.in(1).shape();
    c_ = a.back();
    rows_ = ctx.in(0).size() / c_;
    if (a == b) {
      brow_period_ = 0;  // same shape
    } else {
      // b = [N, 1, ..., 1, C] against a = [N, X, ..., C]: row r of a uses row r / (rows per sample) of b
      brow_period_ = rows_ / (ctx.in(1).size() / c_);
    }
  }
  void run(const KernelContext& ctx) override {
    const Json& at = ctx.attrs();
    const int32_t za = static_cast<int32_t>(at.get_int("a_zero", 0)), zb = static_cast<int32_t>(at.get_int("b_zero", 0));
    const int32_t zy = static_cast<int32_t>(at.get_int("y_zero", 0));
    const int32_t m = static_cast<int32_t>(at.get_int("mult", 0)), sh = static_cast<int32_t>(at.get_int("shift", 0));
    const int lo = static_cast<int>(at.get_int("qmin", -128)), hi = static_cast<int>(at.get_int("qmax", 127));
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t c = c_;
    ctx.pool->parallel_for(rows_, std::max<int64_t>(1, kSerial / std::max<int64_t>(c, 1)), [&](int64_t r0, int64_t r1) {
      const Rq rq(m, sh);
      const int32x4_t vza = vdupq_n_s32(za), vzb = vdupq_n_s32(zb), vzy = vdupq_n_s32(zy);
      const int16x8_t vlo = vdupq_n_s16(static_cast<int16_t>(lo)), vhi = vdupq_n_s16(static_cast<int16_t>(hi));
      for (int64_t r = r0; r < r1; ++r) {
        const int8_t* ar = qa + r * c;
        const int8_t* br = qb + (brow_period_ ? (r / brow_period_) : r) * c;
        int8_t* yr = y + r * c;
        int64_t i = 0;
        for (; i + 16 <= c; i += 16) {
          int32x4_t wa[4], wb[4];
          widen16(vld1q_s8(ar + i), wa);
          widen16(vld1q_s8(br + i), wb);
          for (int k = 0; k < 4; ++k) wa[k] = vqaddq_s32(rq(vmulq_s32(vsubq_s32(wa[k], vza), vsubq_s32(wb[k], vzb))), vzy);
          vst1q_s8(yr + i, narrow16(wa[0], wa[1], wa[2], wa[3], vlo, vhi));
        }
        for (; i < c; ++i) yr[i] = clamp8(zy + q8::requant((ar[i] - za) * static_cast<int64_t>(br[i] - zb), m, sh), lo, hi);
      }
    });
  }

 private:
  int64_t c_ = 1, rows_ = 0, brow_period_ = 0;
};
// products of (q - z) fit int32 when |q - z| < 2^15; b same shape or one row of C per sample
bool mul_neon_supported(const KernelContext& ctx) {
  const Json& at = ctx.attrs();
  auto ok = [](int64_t z) { return z > -32000 && z < 32000; };
  if (!ok(at.get_int("a_zero", 0)) || !ok(at.get_int("b_zero", 0))) return false;
  const Shape& a = ctx.in(0).shape();
  const Shape& b = ctx.in(1).shape();
  if (a == b) return true;
  if (b.size() != a.size() || b.back() != a.back() || b[0] != a[0]) return false;
  for (size_t d = 1; d + 1 < b.size(); ++d)
    if (b[d] != 1) return false;
  return true;
}

// ------------------------------------------------------ I8MaxPool2D / I8AvgPool2D
// Per output pixel, 16 channels at a time over the in-bounds window. Average: sum of (q - z) in int32,
// then rdiv (half away from zero) by the count: |sum| + count/2 and count are integers < 2^24, so the fp32
// quotient truncated is the exact integer quotient.
class PoolNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { avg_ = ctx.node->op == "I8AvgPool2D"; }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    auto g = pool_geometry(ctx.attrs(), x.shape());
    const int32_t z = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    const bool inc = ctx.attrs().get_int("count_include_pad", 0) != 0;
    const int8_t* xd = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t pix = g.n * g.oh * g.ow;
    ctx.pool->parallel_for(pix, std::max<int64_t>(1, kSerial / std::max<int64_t>(g.c * g.kh * g.kw, 1)), [&](int64_t p0, int64_t p1) {
      const int16x8_t vlo = vdupq_n_s16(-128), vhi = vdupq_n_s16(127);
      for (int64_t p = p0; p < p1; ++p) {
        const int64_t n = p / (g.oh * g.ow), oy = (p / g.ow) % g.oh, ox = p % g.ow;
        const int64_t y0 = oy * g.sh - g.pt, x0 = ox * g.sw - g.pl;
        const int64_t ys = std::max<int64_t>(y0, 0), ye = std::min(y0 + g.kh, g.ih);
        const int64_t xs = std::max<int64_t>(x0, 0), xe = std::min(x0 + g.kw, g.iw);
        const int64_t cnt = inc ? g.kh * g.kw : (ye - ys) * (xe - xs);
        int8_t* yp = y + p * g.c;
        int64_t c0 = 0;
        // whole windows, channels in blocks of 16: max 3x3 (9 loads, 8 SMAX), average 2x2 (int16: |sum of q - z| <= 1020;
        // rdiv by 4 half away from zero = sign * ((|a| + 2) >> 2))
        const bool whole = ye - ys == g.kh && xe - xs == g.kw && g.c % 16 == 0;
        if (whole && !avg_ && g.kh == 3 && g.kw == 3) {
          const int64_t C = g.c, rs = g.iw * C;
          const int8_t* b = xd + ((n * g.ih + ys) * g.iw + xs) * C;
          for (; c0 < C; c0 += 16) {
            const int8_t* q = b + c0;
            const int8x16_t r0 = vmaxq_s8(vmaxq_s8(vld1q_s8(q), vld1q_s8(q + C)), vld1q_s8(q + 2 * C));
            const int8x16_t r1 = vmaxq_s8(vmaxq_s8(vld1q_s8(q + rs), vld1q_s8(q + rs + C)), vld1q_s8(q + rs + 2 * C));
            const int8x16_t r2 = vmaxq_s8(vmaxq_s8(vld1q_s8(q + 2 * rs), vld1q_s8(q + 2 * rs + C)), vld1q_s8(q + 2 * rs + 2 * C));
            vst1q_s8(yp + c0, vmaxq_s8(vmaxq_s8(r0, r1), r2));
          }
          continue;
        }
        if (whole && avg_ && g.kh == 2 && g.kw == 2) {
          const int64_t C = g.c, rs = g.iw * C;
          const int8_t* b = xd + ((n * g.ih + ys) * g.iw + xs) * C;
          const int16x8_t z4 = vdupq_n_s16(static_cast<int16_t>(4 * z)), zz = vdupq_n_s16(static_cast<int16_t>(z)), two = vdupq_n_s16(2);
          auto fin = [&](int16x8_t a) {
            a = vsubq_s16(a, z4);
            const int16x8_t m = vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(vaddq_s16(vabsq_s16(a), two)), 2));
            return vaddq_s16(vbslq_s16(vcltzq_s16(a), vnegq_s16(m), m), zz);
          };
          for (; c0 < C; c0 += 16) {
            const int8_t* q = b + c0;
            const int8x16_t a0 = vld1q_s8(q), a1 = vld1q_s8(q + C), a2 = vld1q_s8(q + rs), a3 = vld1q_s8(q + rs + C);
            const int16x8_t lo = vaddw_s8(vaddw_s8(vaddl_s8(vget_low_s8(a0), vget_low_s8(a1)), vget_low_s8(a2)), vget_low_s8(a3));
            const int16x8_t hi = vaddw_high_s8(vaddw_high_s8(vaddl_high_s8(a0, a1), a2), a3);
            vst1q_s8(yp + c0, vcombine_s8(vqmovn_s16(fin(lo)), vqmovn_s16(fin(hi))));
          }
          continue;
        }
        for (; c0 + 16 <= g.c; c0 += 16) {
          if (!avg_) {
            int8x16_t mx = vdupq_n_s8(-128);
            for (int64_t iy = ys; iy < ye; ++iy)
              for (int64_t ix = xs; ix < xe; ++ix) mx = vmaxq_s8(mx, vld1q_s8(xd + ((n * g.ih + iy) * g.iw + ix) * g.c + c0));
            vst1q_s8(yp + c0, mx);
            continue;
          }
          int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
          for (int64_t iy = ys; iy < ye; ++iy)
            for (int64_t ix = xs; ix < xe; ++ix) {
              int32x4_t w[4];
              widen16(vld1q_s8(xd + ((n * g.ih + iy) * g.iw + ix) * g.c + c0), w);
              for (int k = 0; k < 4; ++k) acc[k] = vaddq_s32(acc[k], w[k]);
            }
          const int32x4_t vz = vdupq_n_s32(z), vcz = vdupq_n_s32(static_cast<int32_t>((ye - ys) * (xe - xs)) * z);
          const float32x4_t fc = vdupq_n_f32(static_cast<float>(std::max<int64_t>(cnt, 1)));
          const int32x4_t half = vdupq_n_s32(static_cast<int32_t>(std::max<int64_t>(cnt, 1) / 2));
          int32x4_t o[4];
          for (int k = 0; k < 4; ++k) {
            const int32x4_t a = vsubq_s32(acc[k], vcz);  // sum of (q - z) over the in-bounds taps
            const int32x4_t mag = vaddq_s32(vabsq_s32(a), half);
            const int32x4_t qv = vcvtq_s32_f32(vdivq_f32(vcvtq_f32_s32(mag), fc));  // truncation = exact floor
            const int32x4_t sgn = vbslq_s32(vcltzq_s32(a), vnegq_s32(qv), qv);
            o[k] = vaddq_s32(sgn, vz);
          }
          vst1q_s8(yp + c0, narrow16(o[0], o[1], o[2], o[3], vlo, vhi));
        }
        for (; c0 < g.c; ++c0) {
          int64_t acc = 0;
          int mx = -128;
          for (int64_t iy = ys; iy < ye; ++iy)
            for (int64_t ix = xs; ix < xe; ++ix) {
              const int v = xd[((n * g.ih + iy) * g.iw + ix) * g.c + c0];
              acc += v - z;
              mx = std::max(mx, v);
            }
          yp[c0] = avg_ ? clamp8(z + q8::rdiv(acc, std::max<int64_t>(cnt, 1))) : static_cast<int8_t>(mx);
        }
      }
    });
  }

 private:
  bool avg_ = false;
};

bool pool_neon_supported(const KernelContext& ctx) { return ctx.in(0).dtype() == DType::kI8; }  // bits: i8maxpool2d.bits

// -------------------------------------------------- I8Conv2D depthwise (groups == C, multiplier 1)
// Per output pixel, 16 channels at a time: acc = sum_taps (x - x_zero) * w in int32 lanes (padding taps
// read pad_q), then the per-channel requantization of the reference epilogue.
class DepthwiseNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = conv_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    // weights transposed to [kh][kw][C]
    const int8_t* w = ctx.in(1).data<int8_t>();
    wt_.assign(static_cast<size_t>(g_.kh * g_.kw * g_.oc + 16), 0);
    for (int64_t o = 0; o < g_.oc; ++o)
      for (int64_t t = 0; t < g_.kh * g_.kw; ++t) wt_[t * g_.oc + o] = w[o * g_.kh * g_.kw + t];
  }
  void run(const KernelContext& ctx) override {
    const int8_t* x = ctx.in(0).data<int8_t>();
    const int32_t xz = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    const int32_t pz = static_cast<int32_t>(ctx.attrs().get_int("pad_q", xz)) - xz;
    int8_full::GemmEpilogue ep(ctx);
    const Conv2DGeom& g = g_;
    const int64_t pix = g.n * g.oh * g.ow, C = g.oc;
    ctx.pool->parallel_for(pix, std::max<int64_t>(1, kSerial / std::max<int64_t>(C * g.kh * g.kw, 1)), [&](int64_t p0, int64_t p1) {
      std::vector<int64_t> accs(static_cast<size_t>(C));
      for (int64_t p = p0; p < p1; ++p) {
        const int64_t n = p / (g.oh * g.ow), oy = (p / g.ow) % g.oh, ox = p % g.ow;
        int64_t c0 = 0;
        for (; c0 + 16 <= C; c0 += 16) {
          int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
          for (int64_t ky = 0; ky < g.kh; ++ky)
            for (int64_t kx = 0; kx < g.kw; ++kx) {
              const int64_t iy = oy * g.sh - g.pt + ky * g.dh, ix = ox * g.sw - g.pl + kx * g.dw;
              const bool in = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
              int32x4_t xv[4], wv[4];
              if (in) {
                widen16(vld1q_s8(x + ((n * g.ih + iy) * g.iw + ix) * C + c0), xv);
                for (int k = 0; k < 4; ++k) xv[k] = vsubq_s32(xv[k], vdupq_n_s32(xz));
              } else {
                for (int k = 0; k < 4; ++k) xv[k] = vdupq_n_s32(pz);
              }
              widen16(vld1q_s8(wt_.data() + (ky * g.kw + kx) * C + c0), wv);
              for (int k = 0; k < 4; ++k) acc[k] = vmlaq_s32(acc[k], xv[k], wv[k]);
            }
          int32_t tmp[16];
          for (int k = 0; k < 4; ++k) vst1q_s32(tmp + 4 * k, acc[k]);
          for (int j = 0; j < 16; ++j) accs[c0 + j] = tmp[j];
        }
        for (; c0 < C; ++c0) {
          int64_t a = 0;
          for (int64_t ky = 0; ky < g.kh; ++ky)
            for (int64_t kx = 0; kx < g.kw; ++kx) {
              const int64_t iy = oy * g.sh - g.pt + ky * g.dh, ix = ox * g.sw - g.pl + kx * g.dw;
              const bool in = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
              a += static_cast<int64_t>(in ? x[((n * g.ih + iy) * g.iw + ix) * C + c0] - xz : pz) * wt_[(ky * g.kw + kx) * C + c0];
            }
          accs[c0] = a;
        }
        for (int64_t o = 0; o < C; ++o) ep.store(ctx, p * C + o, o, accs[o]);
      }
    });
  }

 private:
  Conv2DGeom g_{};
  std::vector<int8_t> wt_;
};
bool depthwise_neon_supported(const KernelContext& ctx) {
  const int64_t groups = ctx.attrs().get_int("groups", 1);
  const Tensor& w = ctx.in(1);
  // |x - z| * |w| * taps < 2^31: zero points within int8 (|x - z| <= 255), |w| <= 127, taps <= 4096
  const int64_t xz = ctx.attrs().get_int("x_zero", 0);
  if (ctx.attrs().get_string("output", "i8") == "l8" || ctx.has_in(5)) return false;
  return groups > 1 && w.dim(0) == groups && w.dim(3) == 1 && ctx.in(0).dim(3) == groups && w.dim(1) * w.dim(2) <= 4096 &&
         xz >= -128 && xz <= 127;
}

}  // namespace

void register_int8_full_neon_kernels(OpRegistry& reg) {
  auto add = [&](const char* op, const char* name, uint32_t feat, bool (*sup)(const KernelContext&), std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, name, "int8_full_neon", 100, feat, sup, create});
  };
  add("I8Quantize", "i8quantize.neon", kFeatNeon, nullptr, make_kernel<QuantizeNeon>);
  add("I8Lut", "i8lut.neon", kFeatNeon, lut_neon_supported, make_kernel<LutNeon>);
  add("I8Add", "i8add.neon", kFeatNeon, add_neon_supported, make_kernel<AddNeon>);
  add("I8AffineSum", "i8affinesum.neon", kFeatNeon, affine_sum_neon_supported, make_kernel<AffineSumNeon>);
  add("I8LayerNorm", "i8layernorm.neon", kFeatNeon, nullptr, make_kernel<LayerNormNeon>);
  add("I8Softmax", "i8softmax.neon", kFeatNeon, softmax_neon_supported, make_kernel<SoftmaxNeon>);
  add("I8BDense", "i8bdense.neon", kFeatNeon, binary_layer_neon_supported, make_kernel<BinaryLayerNeon>);
  add("I8BConv2D", "i8bconv2d.neon", kFeatNeon, binary_layer_neon_supported, make_kernel<BinaryLayerNeon>);
  add("I8BConv2DRes", "i8bconv2dres.neon", kFeatNeon, binary_layer_neon_supported, make_kernel<BinaryLayerNeon>);
  add("I8BMatMul", "i8bmatmul.neon", kFeatNeon, bmatmul_i8_neon_supported, make_kernel<BMatMulI8Neon>);
  add("I8Conv2D", "i8conv2d.neon_sdot", kFeatNeon | kFeatDotProd, gemm_sdot_supported, make_kernel<GemmSdot>);
  add("I8Dense", "i8dense.neon_sdot", kFeatNeon | kFeatDotProd, gemm_sdot_supported, make_kernel<GemmSdot>);
  add("I8MatMul", "i8matmul.neon_sdot", kFeatNeon | kFeatDotProd, matmul_sdot_supported, make_kernel<MatMulSdot>);
  add("I8Lut", "i8lut.neon_channels", kFeatNeon, lut_channels_supported, make_kernel<LutChannelsNeon>);
  add("I8Lut", "i8lut.neon_rows", kFeatNeon, lut_rows_supported, make_kernel<LutRowsNeon>);
  add("I8Transpose", "i8transpose.neon", kFeatNeon, nullptr, make_kernel<TransposeNeon>);
  add("I8ReduceMean", "i8reducemean.neon", kFeatNeon, mean_last_supported, make_kernel<MeanLastNeon>);
  add("I8ReduceMean", "i8reducemean.neon_mid", kFeatNeon, mean_mid_supported, make_kernel<MeanMidNeon>);
  add("I8GlobalAvgPool", "i8globalavgpool.neon", kFeatNeon, mean_mid_supported, make_kernel<MeanMidNeon>);
  add("I8Pad", "i8pad.neon", kFeatNeon, pad_neon_supported, make_kernel<PadNeon>);
  add("I8Concat", "i8concat.neon", kFeatNeon, nullptr, make_kernel<ConcatNeon>);
  add("I8Mul", "i8mul.neon", kFeatNeon, mul_neon_supported, make_kernel<MulNeon>);
  add("I8MaxPool2D", "i8maxpool2d.neon", kFeatNeon, pool_neon_supported, make_kernel<PoolNeon>);
  add("I8AvgPool2D", "i8avgpool2d.neon", kFeatNeon, nullptr, make_kernel<PoolNeon>);
  add("I8Conv2D", "i8conv2d.neon_depthwise", kFeatNeon, depthwise_neon_supported, make_kernel<DepthwiseNeon>);
}

#else

void register_int8_full_neon_kernels(OpRegistry&) {}

#endif

}  // namespace bitino

// Kernels of the int8_full ops for accuracy-carrying glue (2026-11-13; docs/int8.md "int8_full ops for accuracy-carrying
// glue"; numpy oracle python/bitino/int8_ops.py, contracts in runtime/ops/op_defs_int8_glue.cc):
//  * I8ChannelShift (ShiftViT): reference (literal) and NEON (per output pixel, one copy or fill per channel segment);
//  * I8BinarizeDyn (INSTA-BNN instance-aware threshold): reference (int64 per element) and NEON (int32 channel sums of
//    the raw codes, the exact int64 threshold per (sample, channel) as an int16 code threshold, then int16 compares and
//    the bit packing of I8BConv2DRes);
//  * I8SimAM: reference only (its per-element cost in NEON would be several I8Lut passes; docs/int8.md).
// The NEON kernels are bit-identical to the reference ones (tests/cpp/test_int8_glue.cc).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "kernels/int8_full/q8_math.h"
#include "kernels/neon_a32.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__aarch64__) || BF_NEON_A32
#include <arm_neon.h>
#endif

namespace bitino {
namespace int8_glue {

// floor(sqrt(v)) for 0 <= v < 2^62 (math.isqrt)
inline int64_t isqrt64(int64_t v) {
  if (v <= 0) return 0;
  uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(v)));
  const uint64_t u = static_cast<uint64_t>(v);
  while (r * r > u) --r;
  while ((r + 1) * (r + 1) <= u) ++r;
  return static_cast<int64_t>(r);
}

// int8_ops.quantize_multiplier_c
inline void quantize_multiplier_c(double real, int32_t* m, int32_t* e) {
  *m = 0;
  *e = 0;
  if (!(real > 0.0)) return;
  int ex = 0;
  const double fr = std::frexp(real, &ex);
  int64_t mm = static_cast<int64_t>(std::floor(fr * 2147483648.0 + 0.5));
  if (mm == (int64_t(1) << 31)) {
    mm /= 2;
    ++ex;
  }
  if (ex < -62) return;
  *m = static_cast<int32_t>(mm);
  *e = std::min(ex, 30);
}

// Per (sample, channel) S1 = sum (q - z), S2 = sum (q - z)^2 over the H * W pixels (exact int64).
inline void channel_stats(const int8_t* x, int64_t hw, int64_t c, int64_t z, int64_t ch, int64_t* s1, int64_t* s2) {
  int64_t a = 0, b = 0;
  for (int64_t j = 0; j < hw; ++j) {
    const int64_t v = int64_t(x[j * c + ch]) - z;
    a += v;
    b += v * v;
  }
  *s1 = a;
  *s2 = b;
}

struct BinDynParams {
  const int32_t *t0 = nullptr, *a = nullptr, *b = nullptr, *ko = nullptr;
  const int8_t* o = nullptr;
  int64_t xz = 0, oz = 0, F = 0, G = 0;
  explicit BinDynParams(const KernelContext& ctx) {
    t0 = ctx.in(1).data<int32_t>();
    a = ctx.in(2).data<int32_t>();
    b = ctx.in(3).data<int32_t>();
    if (ctx.has_in(4)) {
      o = ctx.in(4).data<int8_t>();
      ko = ctx.in(5).data<int32_t>();
    }
    const Json& at = ctx.attrs();
    xz = at.get_int("x_zero", 0);
    oz = at.get_int("o_zero", 0);
    F = at.get_int("shift", 0);
    G = at.get_int("sqrt_shift", 0);
  }
  // acc of (n, ch) from the exact statistics
  int64_t acc(int64_t n, int64_t c, int64_t ch, int64_t hw, int64_t s1, int64_t s2) const {
    const int64_t v = hw * s2 - s1 * s1;
    int64_t r = isqrt64(v << (2 * G));
    int64_t ac = int64_t(t0[ch]) + int64_t(a[ch]) * s1 + int64_t(b[ch]) * r;
    if (o) ac += int64_t(ko[ch]) * (int64_t(o[n * c + ch]) - oz);
    return ac;
  }
  // (q - xz) * D >= acc  <=>  q >= T with T = ceil(acc / D) + xz, clamped to [-129, 128] (same result on int8 codes)
  int16_t code_thr(int64_t ac, int64_t hw) const {
    const int64_t D = hw << F;
    const int64_t t = (ac >= 0 ? (ac + D - 1) / D : -((-ac) / D)) + xz;
    return static_cast<int16_t>(std::min<int64_t>(128, std::max<int64_t>(-129, t)));
  }
};

}  // namespace int8_glue

namespace {

using int8_glue::BinDynParams;

// ------------------------------------------------------------------ reference
class ChannelShiftRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t ns = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3);
    const int8_t* q = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int8_t fill = static_cast<int8_t>(ctx.attrs().get_int("fill_q", 0));
    std::memcpy(y, q, static_cast<size_t>(x.size()));
    const std::vector<int64_t> g = ctx.attrs().get_ints("groups", {});
    for (size_t k = 0; k < g.size(); k += 4)
      for (int64_t n = 0; n < ns; ++n)
        for (int64_t i = 0; i < h; ++i)
          for (int64_t j = 0; j < w; ++j)
            for (int64_t ch = g[k]; ch < g[k + 1]; ++ch) {
              const int64_t si = i - g[k + 2], sj = j - g[k + 3];
              const bool in = si >= 0 && si < h && sj >= 0 && sj < w;
              y[((n * h + i) * w + j) * c + ch] = in ? q[((n * h + si) * w + sj) * c + ch] : fill;
            }
  }
};

class BinarizeDynRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const BinDynParams p(ctx);
    const Tensor& x = ctx.in(0);
    const int64_t ns = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3), words = bit_words(c);
    const int8_t* q = x.data<int8_t>();
    uint32_t* y = ctx.out(0).data<uint32_t>();
    std::fill_n(y, ns * hw * words, 0u);
    const int64_t D = hw << p.F;
    for (int64_t n = 0; n < ns; ++n)
      for (int64_t ch = 0; ch < c; ++ch) {
        int64_t s1 = 0, s2 = 0;
        int8_glue::channel_stats(q + n * hw * c, hw, c, p.xz, ch, &s1, &s2);
        const int64_t ac = p.acc(n, c, ch, hw, s1, s2);
        for (int64_t j = 0; j < hw; ++j) {
          const int64_t r = n * hw + j;
          if (!((int64_t(q[r * c + ch]) - p.xz) * D >= ac)) y[r * words + (ch >> 5)] |= 1u << (ch & 31);
        }
      }
  }
};

class SimAMRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t ns = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3);
    const int8_t* q = x.data<int8_t>();
    const int8_t* tab = ctx.in(1).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const Json& a = ctx.attrs();
    const int64_t xz = a.get_int("x_zero", 0), L = a.get_int("lam_q", 1), gz = a.get_int("g_zero", -128), yz = a.get_int("y_zero", 0);
    const int32_t mult = static_cast<int32_t>(a.get_int("mult", 0)), shift = static_cast<int32_t>(a.get_int("shift", 0));
    const int lo = static_cast<int>(a.get_int("qmin", -128)), hi = static_cast<int>(a.get_int("qmax", 127));
    for (int64_t n = 0; n < ns; ++n)
      for (int64_t ch = 0; ch < c; ++ch) {
        const int8_t* xb = q + n * hw * c;
        int64_t s1 = 0, s2 = 0;
        int8_glue::channel_stats(xb, hw, c, xz, ch, &s1, &s2);
        const int64_t v = hw * s2 - s1 * s1;
        const int64_t mu4 = q8::rdiv(16 * s1, hw);
        int32_t m = 0, e = 0;
        int8_glue::quantize_multiplier_c(static_cast<double>(hw * hw) / static_cast<double>(64 * (v + L)), &m, &e);
        for (int64_t j = 0; j < hw; ++j) {
          const int64_t qq = int64_t(xb[j * c + ch]) - xz;
          const int64_t dd = 16 * qq - mu4;
          const int64_t idx = std::min<int64_t>(255, std::max<int64_t>(0, q8::requant(dd * dd, m, e)));
          const int64_t gg = int64_t(tab[idx]) - gz;
          y[(n * hw + j) * c + ch] = q8::clamp8(yz + q8::requant(qq * gg, mult, shift), lo, hi);
        }
      }
  }
};

#if defined(__aarch64__) || BF_NEON_A32
// ------------------------------------------------------------------ NEON
inline void copy_bytes(int8_t* d, const int8_t* s, int64_t n) {
  for (; n >= 16; n -= 16, d += 16, s += 16) vst1q_s8(d, vld1q_s8(s));
  if (n >= 8) {
    vst1_s8(d, vld1_s8(s));
    n -= 8, d += 8, s += 8;
  }
  for (; n > 0; --n) *d++ = *s++;
}
inline void fill_bytes(int8_t* d, int8_t v, int64_t n) {
  const int8x16_t f = vdupq_n_s8(v);
  for (; n >= 16; n -= 16, d += 16) vst1q_s8(d, f);
  if (n >= 8) {
    vst1_s8(d, vget_low_s8(f));
    n -= 8, d += 8;
  }
  for (; n > 0; --n) *d++ = v;
}

// I8ChannelShift: the channel axis as segments (c0, c1, dy, dx) (unshifted ranges with 0, 0); per output pixel one copy
// from the source pixel or one fill per segment (vector loads / stores without alignment requirements).
class ChannelShiftNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const int64_t c = ctx.in(0).dim(3);
    const std::vector<int64_t> g = ctx.attrs().get_ints("groups", {});
    std::vector<std::array<int64_t, 4>> gs;
    for (size_t k = 0; k < g.size(); k += 4) gs.push_back({g[k], g[k + 1], g[k + 2], g[k + 3]});
    std::sort(gs.begin(), gs.end());
    seg_.clear();
    int64_t at = 0;
    for (const auto& s : gs) {
      if (s[0] > at) seg_.push_back({at, s[0], 0, 0});
      seg_.push_back(s);
      at = s[1];
    }
    if (at < c) seg_.push_back({at, c, 0, 0});
  }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t ns = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3);
    const int8_t* q = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int8_t fill = static_cast<int8_t>(ctx.attrs().get_int("fill_q", 0));
    const auto& seg = seg_;
    ctx.pool->parallel_for(ns * h, std::max<int64_t>(1, (64 * 1024) / std::max<int64_t>(w * c, 1)), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t n = r / h, i = r % h;
        for (int64_t j = 0; j < w; ++j) {
          int8_t* d = y + (r * w + j) * c;
          for (const auto& s : seg) {
            const int64_t si = i - s[2], sj = j - s[3];
            if (si >= 0 && si < h && sj >= 0 && sj < w) copy_bytes(d + s[0], q + ((n * h + si) * w + sj) * c + s[0], s[1] - s[0]);
            else fill_bytes(d + s[0], fill, s[1] - s[0]);
          }
        }
      }
    });
  }

 private:
  std::vector<std::array<int64_t, 4>> seg_;
};

// I8BinarizeDyn: per sample, (1) the raw-code channel sums sum q, sum q^2 in int32 lanes (exact: |q| <= 128, H W < 2^17),
// 16 channels per pass over the pixels; (2) per channel the exact int64 acc and the code threshold T (BinDynParams::
// code_thr); (3) per 16 channels and pixel: q >= T as two int16 compares, packed into 2 bytes of the bit row as
// I8BConv2DRes does (bit 1 iff !(q >= T)); channel tails scalar.
class BinarizeDynNeon : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const BinDynParams p(ctx);
    const Tensor& x = ctx.in(0);
    const int64_t ns = x.dim(0), hw = x.dim(1) * x.dim(2), c = x.dim(3), words = bit_words(c);
    const int8_t* q = x.data<int8_t>();
    uint32_t* yw = ctx.out(0).data<uint32_t>();
    const int64_t cf = c / 16 * 16;
    std::vector<int16_t> thr(static_cast<size_t>(c + 16));
    for (int64_t n = 0; n < ns; ++n) {
      const int8_t* xb = q + n * hw * c;
      // (1) + (2)
      for (int64_t u = 0; u < cf; u += 16) {
        int32x4_t s1[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
        int32x4_t s2[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
        for (int64_t j = 0; j < hw; ++j) {
          const int8x16_t v = vld1q_s8(xb + j * c + u);
          const int16x8_t l = vmovl_s8(vget_low_s8(v)), hh = vmovl_high_s8(v);
          const int16x8_t ql = vmull_s8(vget_low_s8(v), vget_low_s8(v)), qh = vmull_s8(vget_high_s8(v), vget_high_s8(v));
          s1[0] = vaddw_s16(s1[0], vget_low_s16(l));
          s1[1] = vaddw_high_s16(s1[1], l);
          s1[2] = vaddw_s16(s1[2], vget_low_s16(hh));
          s1[3] = vaddw_high_s16(s1[3], hh);
          // q^2 <= 16384: as uint16 lanes (the int16 view of 16384 is fine; widened unsigned)
          s2[0] = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(s2[0]), vget_low_u16(vreinterpretq_u16_s16(ql))));
          s2[1] = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(s2[1]), vget_high_u16(vreinterpretq_u16_s16(ql))));
          s2[2] = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(s2[2]), vget_low_u16(vreinterpretq_u16_s16(qh))));
          s2[3] = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(s2[3]), vget_high_u16(vreinterpretq_u16_s16(qh))));
        }
        int32_t a1[16], a2[16];
        for (int k = 0; k < 4; ++k) {
          vst1q_s32(a1 + 4 * k, s1[k]);
          vst1q_s32(a2 + 4 * k, s2[k]);
        }
        for (int k = 0; k < 16; ++k) {
          const int64_t r1 = a1[k], r2 = static_cast<int64_t>(static_cast<uint32_t>(a2[k]));
          const int64_t S1 = r1 - hw * p.xz, S2 = r2 - 2 * p.xz * r1 + hw * p.xz * p.xz;
          thr[u + k] = p.code_thr(p.acc(n, c, u + k, hw, S1, S2), hw);
        }
      }
      for (int64_t ch = cf; ch < c; ++ch) {
        int64_t S1 = 0, S2 = 0;
        int8_glue::channel_stats(xb, hw, c, p.xz, ch, &S1, &S2);
        thr[ch] = p.code_thr(p.acc(n, c, ch, hw, S1, S2), hw);
      }
      // (3)
      uint32_t* yb = yw + n * hw * words;
      ctx.pool->parallel_for(hw, std::max<int64_t>(1, (64 * 1024) / std::max<int64_t>(c, 1)), [&](int64_t j0, int64_t j1) {
        static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        const uint8x16_t bw = vld1q_u8(kW);
        for (int64_t u = 0; u < cf; u += 16) {
          const bool top0 = (u & 31) == 0 && u + 16 >= c;
          const int16x8_t ta = vld1q_s16(thr.data() + u), tb = vld1q_s16(thr.data() + u + 8);
          for (int64_t j = j0; j < j1; ++j) {
            const int8x16_t v = vld1q_s8(xb + j * c + u);
            const uint16x8_t ga = vcgeq_s16(vmovl_s8(vget_low_s8(v)), ta), gb = vcgeq_s16(vmovl_high_s8(v), tb);
            uint8x16_t sb = vbicq_u8(bw, vcombine_u8(vmovn_u16(ga), vmovn_u16(gb)));
            sb = vpaddq_u8(sb, sb);
            sb = vpaddq_u8(sb, sb);
            sb = vpaddq_u8(sb, sb);
            uint8_t* d = reinterpret_cast<uint8_t*>(yb + j * words) + u / 8;
            if (top0) vst1q_lane_u32(reinterpret_cast<uint32_t*>(d), vreinterpretq_u32_u16(vmovl_u16(vget_low_u16(vreinterpretq_u16_u8(sb)))), 0);
            else vst1q_lane_u16(reinterpret_cast<uint16_t*>(d), vreinterpretq_u16_u8(sb), 0);
          }
        }
        if (cf < c)
          for (int64_t j = j0; j < j1; ++j)
            for (int64_t ch = cf; ch < c; ++ch) {
              uint32_t* wd = yb + j * words + (ch >> 5);
              if ((ch & 31) == 0 || ch == cf) *wd &= (ch & 31) ? ((1u << (ch & 31)) - 1u) : 0u;  // clear the word's rest
              if (!(int16_t(xb[j * c + ch]) >= thr[ch])) *wd |= 1u << (ch & 31);
            }
      });
    }
  }
};
// the int32 channel sums: H W * 16384 < 2^31
bool binarize_dyn_neon_supported(const KernelContext& ctx) { return ctx.in(0).dim(1) * ctx.in(0).dim(2) < (int64_t(1) << 17); }
#endif

}  // namespace

void register_int8_glue_kernels(OpRegistry& reg) {
  auto ref = [&](const char* op, const char* name, std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, name, "int8_full_reference", 0, 0, nullptr, create});
  };
  ref("I8ChannelShift", "i8channelshift.reference", make_kernel<ChannelShiftRef>);
  ref("I8BinarizeDyn", "i8binarizedyn.reference", make_kernel<BinarizeDynRef>);
  ref("I8SimAM", "i8simam.reference", make_kernel<SimAMRef>);
#if defined(__aarch64__) || BF_NEON_A32
  reg.add_kernel({"I8ChannelShift", "i8channelshift.neon", "int8_full_neon", 100, kFeatNeon, nullptr, make_kernel<ChannelShiftNeon>});
  reg.add_kernel({"I8BinarizeDyn", "i8binarizedyn.neon", "int8_full_neon", 100, kFeatNeon, binarize_dyn_neon_supported,
                  make_kernel<BinarizeDynNeon>});
#endif
}

}  // namespace bitino

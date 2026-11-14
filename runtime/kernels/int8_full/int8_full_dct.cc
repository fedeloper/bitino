// JPEG-DCT network input ops (2026-11-14; the research notes (not published) section 15; numpy oracle
// python/bitino/int8_ops.py _dct_yuv420 / _bconv_bp):
//
// I8DctYuv420: on-device preprocessing, a YUV 4:2:0 frame -> the selected quantized 8x8 DCT coefficients.
//   in(0) Y  i8 [N, H, W, 1]       (Y - 128, the level-shifted luma; H, W multiples of 16)
//   in(1) C  i8 [N, H/2, W/2, 2]   (Cb - 128, Cr - 128 interleaved, NV12-style)
//   attributes y_sel / c_sel (natural indices v * 8 + u, v the vertical frequency), qy / qc (64 quantizer steps, natural
//   order), shifts (one per output channel), output i8 [N, H/8, W/8, |y_sel| + 2 |c_sel|]: Y channels, Cb, Cr; a chroma
//   block (14 x 14 grid for 224 px) is written to the 2 x 2 output pixels it covers.
//   Per 8x8 block f[y][x] (exact integers; T the Q13 table kT below, T[k][x] = round(4096 C(k) cos((2x+1) k pi / 16))):
//     R[y][u] = (sum_x T[u][x] f[y][x] + 2^10) >> 11          (Q2, only the u the selection needs)
//     F[v][u] = sum_y T[v][y] R[y][u]                          (Q15: 2^15 x the JPEG DCT)
//     code    = clamp(sgn(F) floor((|F| + D / 2) / D), -128, 127),  D = Q[v * 8 + u] 2^(15 + shift)
//   (the JPEG DCT of the standard and libjpeg's quantization rounding; the fixed-point transform differs from libjpeg's
//   jfdctint by rounding only: tools/s80_dct.py measures the agreement)
//   Kernels: reference (direct) and a fast portable one (separable passes over the needed frequencies only).
//
// I8BConvBP: a binary 1x1 convolution on integer inputs, bit-serial ("C": the binary first layer on the coefficients'
//   bit-planes). x i8 [N, H, W, C] (value q - x_zero), w bit [O, 1, 1, C] (+-1), m1 / b i32 [O], bit_thr i32 [O]?:
//     acc = sum_c w[o][c] (q_c - x_zero)
//         = sum_{b=0..7} 2^b (2 popcount(P_b & W+_o) - popcount(P_b)) - (128 + x_zero) sum_c w[o][c]
//       with P_b the bit-plane b of the unsigned codes q + 128 and W+_o the +1 weights (the exact binary-weight x integer
//       input product, computed with AND / popcount per plane);
//     v = acc m1[o] + b[o];  y = clamp(rshr(v, shift) + y_zero, qmin, qmax);  bit = -1 iff !(v >= bit_thr[o]) (output 1)
//   Kernels: reference (direct sum) and NEON (arm64 / armv7: per pixel the 8 bit-planes, then 4 output channels per
//   VCNT of the AND with their +1 masks).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "kernels/bitpack.h"
#include "kernels/int8_full/q8_math.h"
#include "kernels/neon_a32.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

#if defined(__aarch64__) || BF_NEON_A32
#include <arm_neon.h>
#endif

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

constexpr int32_t kT[8][8] = {{2896, 2896, 2896, 2896, 2896, 2896, 2896, 2896},
                              {4017, 3406, 2276, 799, -799, -2276, -3406, -4017},
                              {3784, 1567, -1567, -3784, -3784, -1567, 1567, 3784},
                              {3406, -799, -4017, -2276, 2276, 4017, 799, -3406},
                              {2896, -2896, -2896, 2896, 2896, -2896, -2896, 2896},
                              {2276, -4017, 799, 3406, -3406, -799, 4017, -2276},
                              {1567, -3784, 3784, -1567, -1567, 3784, -3784, 1567},
                              {799, -2276, 3406, -4017, 4017, -3406, 2276, -799}};

bool has(const Ins& ins, size_t i) { return i < ins.size() && ins[i]; }

// ------------------------------------------------------------------------------------------------ I8DctYuv420
struct DctCfg {
  std::vector<int> ysel, csel;
  std::vector<int64_t> qy, qc, shifts;
  explicit DctCfg(const Json& a) {
    for (int64_t v : a.get_ints("y_sel", {})) ysel.push_back(static_cast<int>(v));
    for (int64_t v : a.get_ints("c_sel", {})) csel.push_back(static_cast<int>(v));
    qy = a.get_ints("qy", {});
    qc = a.get_ints("qc", {});
    shifts = a.get_ints("shifts", {});
  }
  int64_t channels() const { return static_cast<int64_t>(ysel.size() + 2 * csel.size()); }
};

void infer_dct(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(has(ins, 0) && has(ins, 1), "I8DctYuv420 needs Y and C");
  const auto& y = *ins[0];
  const auto& c = *ins[1];
  BF_CHECK(y.dtype == DType::kI8 && y.shape.size() == 4 && y.shape[3] == 1, "I8DctYuv420: Y must be i8 [N, H, W, 1]");
  BF_CHECK(y.shape[1] % 16 == 0 && y.shape[2] % 16 == 0, "I8DctYuv420: H, W must be multiples of 16");
  BF_CHECK(c.dtype == DType::kI8 && c.shape == Shape({y.shape[0], y.shape[1] / 2, y.shape[2] / 2, 2}), "I8DctYuv420: C must be i8 [N, H/2, W/2, 2]");
  DctCfg cfg(n.attrs);
  BF_CHECK(cfg.qy.size() == 64 && cfg.qc.size() == 64, "I8DctYuv420: qy / qc must have 64 entries");
  for (int64_t q : cfg.qy) BF_CHECK(q >= 1 && q <= 255, "I8DctYuv420: quantizer steps in 1..255");
  for (int64_t q : cfg.qc) BF_CHECK(q >= 1 && q <= 255, "I8DctYuv420: quantizer steps in 1..255");
  for (int v : cfg.ysel) BF_CHECK(v >= 0 && v < 64, "I8DctYuv420: y_sel in 0..63");
  for (int v : cfg.csel) BF_CHECK(v >= 0 && v < 64, "I8DctYuv420: c_sel in 0..63");
  BF_CHECK(static_cast<int64_t>(cfg.shifts.size()) == cfg.channels(), "I8DctYuv420: one shift per output channel");
  for (int64_t s : cfg.shifts) BF_CHECK(s >= 0 && s <= 8, "I8DctYuv420: shifts in 0..8");
  BF_CHECK(!outs.empty() && outs[0], "missing output");
  outs[0]->dtype = DType::kI8;
  outs[0]->shape = {y.shape[0], y.shape[1] / 8, y.shape[2] / 8, cfg.channels()};
}

inline int8_t dct_code(int64_t F, int64_t q, int64_t shift) {
  const int64_t D = q << (15 + shift);
  const int64_t a = F < 0 ? -F : F;
  int64_t r = (a + D / 2) / D;
  if (F < 0) r = -r;
  return static_cast<int8_t>(std::min<int64_t>(127, std::max<int64_t>(-128, r)));
}

// one block: plane p (row stride ps, channel stride cs), the coefficients sel -> out[k * ostride] (reference: direct)
void dct_block_ref(const int8_t* p, int64_t ps, int64_t cs, const std::vector<int>& sel, const std::vector<int64_t>& q,
                   const int64_t* shifts, int8_t* out) {
  for (size_t k = 0; k < sel.size(); ++k) {
    const int v = sel[k] / 8, u = sel[k] % 8;
    int64_t F = 0;
    for (int yy = 0; yy < 8; ++yy) {
      int64_t R = 0;
      for (int xx = 0; xx < 8; ++xx) R += int64_t(kT[u][xx]) * p[yy * ps + xx * cs];
      R = (R + (1 << 10)) >> 11;
      F += int64_t(kT[v][yy]) * R;
    }
    out[k] = dct_code(F, q[sel[k]], shifts[k]);
  }
}

// fast form: the row pass once per needed u (int32), then the column pass per selected coefficient
void dct_block_fast(const int8_t* p, int64_t ps, int64_t cs, const std::vector<int>& sel, const std::vector<int>& us,
                    const std::vector<int64_t>& q, const int64_t* shifts, int8_t* out) {
  int32_t R[8][8];   // R[u][y]
  for (int u : us)
    for (int yy = 0; yy < 8; ++yy) {
      const int8_t* r = p + yy * ps;
      int32_t s = 0;
      for (int xx = 0; xx < 8; ++xx) s += kT[u][xx] * int32_t(r[xx * cs]);
      R[u][yy] = (s + (1 << 10)) >> 11;
    }
  for (size_t k = 0; k < sel.size(); ++k) {
    const int v = sel[k] / 8, u = sel[k] % 8;
    int32_t F = 0;
    for (int yy = 0; yy < 8; ++yy) F += kT[v][yy] * R[u][yy];
    out[k] = dct_code(F, q[sel[k]], shifts[k]);
  }
}

template <bool kFast>
class DctYuv420 : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    cfg_ = std::make_unique<DctCfg>(ctx.attrs());
    auto needed = [](const std::vector<int>& sel) {
      std::vector<int> u;
      for (int s : sel)
        if (std::find(u.begin(), u.end(), s % 8) == u.end()) u.push_back(s % 8);
      return u;
    };
    uy_ = needed(cfg_->ysel);
    uc_ = needed(cfg_->csel);
  }
  void run(const KernelContext& ctx) override {
    const DctCfg& c = *cfg_;
    const Tensor& y = ctx.in(0);
    const Tensor& ch = ctx.in(1);
    const int64_t n = y.dim(0), h = y.dim(1), w = y.dim(2), bh = h / 8, bw = w / 8, oc = c.channels();
    const int64_t ny = static_cast<int64_t>(c.ysel.size()), nc = static_cast<int64_t>(c.csel.size());
    int8_t* out = ctx.out(0).data<int8_t>();
    const int8_t* yp = y.data<int8_t>();
    const int8_t* cp = ch.data<int8_t>();
    std::vector<int8_t> tmp(static_cast<size_t>(std::max<int64_t>(ny, nc)));
    for (int64_t s = 0; s < n; ++s) {
      for (int64_t by = 0; by < bh; ++by)
        for (int64_t bx = 0; bx < bw; ++bx) {
          const int8_t* p = yp + ((s * h + by * 8) * w + bx * 8);
          int8_t* o = out + ((s * bh + by) * bw + bx) * oc;
          if (kFast) dct_block_fast(p, w, 1, c.ysel, uy_, c.qy, c.shifts.data(), o);
          else dct_block_ref(p, w, 1, c.ysel, c.qy, c.shifts.data(), o);
        }
      const int64_t ch_h = h / 2, ch_w = w / 2;
      for (int64_t cy = 0; cy < bh / 2; ++cy)
        for (int64_t cx = 0; cx < bw / 2; ++cx)
          for (int comp = 0; comp < 2; ++comp) {
            const int8_t* p = cp + ((s * ch_h + cy * 8) * ch_w + cx * 8) * 2 + comp;
            const int64_t* sh = c.shifts.data() + ny + comp * nc;
            if (kFast) dct_block_fast(p, ch_w * 2, 2, c.csel, uc_, c.qc, sh, tmp.data());
            else dct_block_ref(p, ch_w * 2, 2, c.csel, c.qc, sh, tmp.data());
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                int8_t* o = out + ((s * bh + 2 * cy + dy) * bw + 2 * cx + dx) * oc + ny + comp * nc;
                std::memcpy(o, tmp.data(), static_cast<size_t>(nc));
              }
          }
    }
  }

 private:
  std::unique_ptr<DctCfg> cfg_;
  std::vector<int> uy_, uc_;
};

// ------------------------------------------------------------------------------------------------ I8BConvBP
void infer_bconv_bp(const Node& n, const Ins& ins, Outs& outs) {
  BF_CHECK(has(ins, 0) && has(ins, 1) && has(ins, 2) && has(ins, 3), "I8BConvBP needs x, w, m1, b");
  const auto& x = *ins[0];
  const auto& w = *ins[1];
  BF_CHECK(x.dtype == DType::kI8 && x.shape.size() == 4, "I8BConvBP: x must be i8 [N, H, W, C]");
  BF_CHECK(w.dtype == DType::kBit && w.shape.size() == 4 && w.shape[1] == 1 && w.shape[2] == 1 && w.shape[3] == x.shape[3],
           "I8BConvBP: w must be bit [O, 1, 1, C]");
  const int64_t o = w.shape[0];
  for (size_t i : {size_t(2), size_t(3)})
    BF_CHECK(ins[i]->dtype == DType::kI32 && num_elements(ins[i]->shape) == o, "I8BConvBP: m1 / b must be i32 [O]");
  const int64_t sh = n.attrs.get_int("shift", 0), z = n.attrs.get_int("x_zero", 0);
  BF_CHECK(sh >= 0 && sh <= 30 && z >= -128 && z <= 127, "I8BConvBP shift / x_zero out of range");
  const int64_t qmin = n.attrs.get_int("qmin", -128), qmax = n.attrs.get_int("qmax", 127);
  BF_CHECK(qmin >= -128 && qmin <= qmax && qmax <= 127, "I8BConvBP qmin / qmax out of range");
  BF_CHECK(!outs.empty() && outs[0], "missing output");
  outs[0]->dtype = DType::kI8;
  outs[0]->shape = {x.shape[0], x.shape[1], x.shape[2], o};
  if (has(ins, 4)) {
    BF_CHECK(ins[4]->dtype == DType::kI32 && num_elements(ins[4]->shape) == o, "I8BConvBP: bit_thr must be i32 [O]");
    BF_CHECK(outs.size() == 2 && outs[1], "I8BConvBP with bit_thr needs 2 outputs");
    outs[1]->dtype = DType::kBit;
    outs[1]->shape = outs[0]->shape;
  } else {
    BF_CHECK(outs.size() == 1, "I8BConvBP without bit_thr has 1 output");
  }
}

struct BpParams {
  int64_t c = 0, o = 0;
  int32_t xz = 0, yz = 0, qmin = -128, qmax = 127;
  int shift = 0;
  const int32_t *m1 = nullptr, *b = nullptr, *thr = nullptr;
  std::vector<int8_t> w;   // +-1 [O][C]
  explicit BpParams(const KernelContext& ctx) {
    const Json& a = ctx.attrs();
    c = ctx.in(0).dim(3);
    o = ctx.in(1).dim(0);
    xz = static_cast<int32_t>(a.get_int("x_zero", 0));
    yz = static_cast<int32_t>(a.get_int("y_zero", 0));
    qmin = static_cast<int32_t>(a.get_int("qmin", -128));
    qmax = static_cast<int32_t>(a.get_int("qmax", 127));
    shift = static_cast<int>(a.get_int("shift", 0));
    m1 = ctx.in(2).data<int32_t>();
    b = ctx.in(3).data<int32_t>();
    thr = ctx.has_in(4) ? ctx.in(4).data<int32_t>() : nullptr;
    const uint32_t* wb = ctx.in(1).data<uint32_t>();
    const int64_t words = bit_words(c);
    w.resize(static_cast<size_t>(o * c));
    for (int64_t k = 0; k < o; ++k)
      for (int64_t j = 0; j < c; ++j) w[k * c + j] = static_cast<int8_t>(bit_value(wb + k * words, j));
    int64_t mm = 0, mb = 0;
    for (int64_t k = 0; k < o; ++k) {
      mm = std::max<int64_t>(mm, std::llabs(m1[k]));
      mb = std::max<int64_t>(mb, std::llabs(b[k]));
    }
    BF_CHECK(255 * c * mm + mb < (int64_t(1) << 30), "I8BConvBP: constants exceed the int32 bound");
  }
  int8_t y(int64_t v) const { return q8::clamp8(q8::rshr(v, shift) + yz, qmin, qmax); }
};

class BConvBPRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    BpParams p(ctx);
    const int8_t* x = ctx.in(0).data<int8_t>();
    const int64_t pix = ctx.in(0).size() / p.c, ow = bit_words(p.o);
    int8_t* y = ctx.out(0).data<int8_t>();
    uint32_t* bits = p.thr ? ctx.out(1).data<uint32_t>() : nullptr;
    if (bits) std::fill_n(bits, pix * ow, 0u);
    for (int64_t q = 0; q < pix; ++q)
      for (int64_t k = 0; k < p.o; ++k) {
        int64_t acc = 0;
        for (int64_t j = 0; j < p.c; ++j) acc += int64_t(p.w[k * p.c + j]) * (int64_t(x[q * p.c + j]) - p.xz);
        const int64_t v = acc * p.m1[k] + p.b[k];
        y[q * p.o + k] = p.y(v);
        if (bits && !(v >= p.thr[k])) bits[q * ow + (k >> 5)] |= 1u << (k & 31);
      }
  }
};

#if defined(__aarch64__) || BF_NEON_A32
// bit-serial NEON kernel: per pixel the 8 bit-planes of q + 128 (C <= 32 per word), then per 4 output channels
// sum_b 2^b (2 popcount(P_b & W+) - popcount(P_b)) with VCNT + pairwise widening adds
class BConvBPNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    p_ = std::make_unique<BpParams>(ctx);
    words_ = bit_words(p_->c);
    o4_ = (p_->o + 3) / 4 * 4;
    wpos_.assign(static_cast<size_t>(o4_ * words_), 0u);   // [word][o4]
    off_.assign(static_cast<size_t>(o4_), 0);
    m1_.assign(static_cast<size_t>(o4_), 0);
    b_.assign(static_cast<size_t>(o4_), 0);
    for (int64_t k = 0; k < p_->o; ++k) {
      int32_t sw = 0;
      for (int64_t j = 0; j < p_->c; ++j) {
        if (p_->w[k * p_->c + j] > 0) wpos_[(j >> 5) * o4_ + k] |= 1u << (j & 31);
        sw += p_->w[k * p_->c + j];
      }
      off_[k] = (128 + p_->xz) * sw;
      m1_[k] = p_->m1[k];
      b_[k] = p_->b[k];
    }
  }
  void run(const KernelContext& ctx) override {
    const BpParams& p = *p_;
    const int8_t* x = ctx.in(0).data<int8_t>();
    const int64_t pix = ctx.in(0).size() / p.c, ow = bit_words(p.o);
    int8_t* y = ctx.out(0).data<int8_t>();
    uint32_t* bits = p.thr ? ctx.out(1).data<uint32_t>() : nullptr;
    std::vector<uint32_t> planes(static_cast<size_t>(8 * words_));
    const int32x4_t sh = vdupq_n_s32(-p.shift), yz = vdupq_n_s32(p.yz);
    for (int64_t q = 0; q < pix; ++q) {
      std::fill(planes.begin(), planes.end(), 0u);
      const int8_t* xr = x + q * p.c;
      for (int64_t j = 0; j < p.c; ++j) {
        const uint32_t u = static_cast<uint8_t>(xr[j]) ^ 0x80u, bit = 1u << (j & 31);
        uint32_t* pw = planes.data() + (j >> 5);
        for (int b = 0; b < 8; ++b)
          if (u & (1u << b)) pw[b * words_] |= bit;
      }
      int32_t pc[8];
      for (int b = 0; b < 8; ++b) {
        int32_t s = 0;
        for (int64_t wd = 0; wd < words_; ++wd) s += __builtin_popcount(planes[b * words_ + wd]);
        pc[b] = s;
      }
      int32_t vbuf[4];
      if (bits) std::fill_n(bits + q * ow, ow, 0u);
      for (int64_t k = 0; k < o4_; k += 4) {
        int32x4_t acc = vnegq_s32(vld1q_s32(off_.data() + k));
        for (int b = 0; b < 8; ++b) {
          uint32x4_t cnt = vdupq_n_u32(0);
          for (int64_t wd = 0; wd < words_; ++wd) {
            const uint32x4_t a = vandq_u32(vdupq_n_u32(planes[b * words_ + wd]), vld1q_u32(wpos_.data() + wd * o4_ + k));
            cnt = vaddq_u32(cnt, vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(a)))));
          }
          const int32x4_t d = vsubq_s32(vshlq_n_s32(vreinterpretq_s32_u32(cnt), 1), vdupq_n_s32(pc[b]));
          acc = vaddq_s32(acc, vshlq_s32(d, vdupq_n_s32(b)));
        }
        const int32x4_t v = vmlaq_s32(vld1q_s32(b_.data() + k), acc, vld1q_s32(m1_.data() + k));
        const int32x4_t r = vaddq_s32(vrshlq_s32(v, sh), yz);
        vst1q_s32(vbuf, v);
        int32_t rb[4];
        vst1q_s32(rb, r);
        const int64_t nk = std::min<int64_t>(4, p.o - k);
        for (int64_t i = 0; i < nk; ++i) {
          y[q * p.o + k + i] = static_cast<int8_t>(std::min(p.qmax, std::max(p.qmin, rb[i])));
          if (bits && !(vbuf[i] >= p.thr[k + i])) bits[q * ow + ((k + i) >> 5)] |= 1u << ((k + i) & 31);
        }
      }
    }
  }

 private:
  std::unique_ptr<BpParams> p_;
  int64_t words_ = 1, o4_ = 0;
  std::vector<uint32_t> wpos_;
  std::vector<int32_t> off_, m1_, b_;
};
#endif

}  // namespace

void register_int8_dct(OpRegistry& reg) {
  reg.add_op({"I8DctYuv420", infer_dct});
  reg.add_op({"I8BConvBP", infer_bconv_bp});
  reg.add_kernel({"I8DctYuv420", "i8dctyuv420.reference", "int8_full_reference", 0, 0, nullptr, make_kernel<DctYuv420<false>>});
  reg.add_kernel({"I8DctYuv420", "i8dctyuv420.fast", "int8_full_opt", 50, 0, nullptr, make_kernel<DctYuv420<true>>});
  reg.add_kernel({"I8BConvBP", "i8bconvbp.reference", "int8_full_reference", 0, 0, nullptr, make_kernel<BConvBPRef>});
#if defined(__aarch64__) || BF_NEON_A32
  reg.add_kernel({"I8BConvBP", "i8bconvbp.neon", "int8_full_neon", 100, kFeatNeon, nullptr, make_kernel<BConvBPNeon>});
#endif
}

}  // namespace bitino

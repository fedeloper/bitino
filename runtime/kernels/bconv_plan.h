// Shared preparation for bit-packed BConv2D / BDense kernels.
//
// Math (all binary values in {-1,+1}, K = number of real bits):
//   a·b = K - 2·popcount(a XOR b)
// Patch rows and weight rows are padded with zero bits up to a multiple of
// 4 words (128 bits). Zero padding bits are equal in both operands, so they
// never contribute to popcount and the formula holds with the *real* K.
//
// Spatial padding: out-of-bounds taps are filled with bit 0 (+1). Their true
// contribution is pad_value·S[o][tap] where S[o][tap] = Σ_c w[o,tap,c], so
//   dot_true = dot_computed + (pad_value - 1)·Σ_{oob taps} S[o][tap].
//
// Grouped convolution (groups > 1) and the patch-row layout of every packed
// kernel are described by BConvGroupLayout below: output channel o belongs to
// group o / ocg and only sees that group's kic = ic / groups input channels.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "bitino/op.h"
#include "kernels/bit_output.h"
#include "kernels/bitpack.h"
#include "ops/ops_common.h"

namespace bitino {

// Patch-row layout shared by the packed BConv2D kernels (BConvPlan users,
// bconv2d.neon_opt, bconv2d.avx2), including grouped convolution.
//
// The patch row of output pixel p for group gi is the concatenation over taps
// t = ky*kw + kx of the group's kic input bits at that tap, stored in "units"
// of 16 or 32 bits (ut units per tap, k_units = taps*ut units per row):
//  * 32-bit units read directly from the packed input [pix][bit_words(ic)] at
//    unit offset gi*ut, when groups == 1 (ut = bit_words(ic); the padding bits
//    of the last word are zero in input and weights alike) or kic % 32 == 0.
//  * 16-bit units read directly from the packed input when kic % 32 == 16
//    (words are little-endian, so halfword h of a pixel holds channels
//    [16h, 16h + 16)). Taps are then packed densely: kic = 16 with a 3x3 kernel
//    needs 144 bits = 4.5 words per row instead of 9.
//  * otherwise (groups > 1, kic % 16 != 0) the input is first repacked per
//    group into [pix][groups][bit_words(kic)] words with zero padding bits
//    (`regroup`), then read in 32-bit units.
// Weight rows use the same layout, so a·b = k_bits - 2·popcount(a XOR b) holds
// with k_bits = taps*kic; the row tail beyond k_units must be zero.
struct BConvGroupLayout {
  int64_t groups = 1, ocg = 0, kic = 0, taps = 0;
  int64_t unit_bytes = 4;  // 2 or 4
  int64_t ut = 0;          // units per tap
  int64_t k_units = 0;     // units per patch row
  bool regroup = false;    // source = per-group repacked input
  int64_t src_words = 0;   // words per pixel of the source buffer
  int64_t k_bits = 0;      // real bits per patch row = taps * kic
  int64_t k_words = 0;     // 32-bit words per patch row (unpadded)

  void init(const Conv2DGeom& g) {
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "16-bit units assume little-endian words");
    groups = g.groups;
    ocg = g.oc / g.groups;
    kic = g.kic;
    taps = g.kh * g.kw;
    regroup = false;
    if (groups == 1 && kic % 32 != 16) {
      unit_bytes = 4;
      ut = bit_words(kic);
    } else if (kic % 32 == 0) {
      unit_bytes = 4;
      ut = kic / 32;
    } else if (kic % 16 == 0) {
      unit_bytes = 2;
      ut = kic / 16;
    } else {
      unit_bytes = 4;
      ut = bit_words(kic);
      regroup = true;
    }
    src_words = regroup ? groups * ut : bit_words(g.ic);
    k_units = taps * ut;
    k_bits = taps * kic;
    k_words = (k_units * unit_bytes + 3) / 4;
  }

  // Byte stride between pixels / offset of group gi within a pixel of the source buffer.
  int64_t src_pixel_bytes() const { return src_words * 4; }
  int64_t group_offset_bytes(int64_t gi) const { return gi * ut * unit_bytes; }

  // Writes the dense weight rows of all oc = groups*ocg output channels:
  // dst[o * dst_stride + k] for k < dst_stride words (the tail beyond k_units
  // is zero). `w` is the OHWI_bit filter [oc][taps][bit_words(kic)].
  void pack_weight_rows(const uint32_t* w, int64_t oc, uint32_t* dst, int64_t dst_stride) const {
    const int64_t tw = bit_words(kic);
    for (int64_t o = 0; o < oc; ++o) {
      auto* d = reinterpret_cast<uint8_t*>(dst + o * dst_stride);
      std::memset(d, 0, static_cast<size_t>(dst_stride * 4));
      for (int64_t t = 0; t < taps; ++t)
        std::memcpy(d + t * ut * unit_bytes, w + (o * taps + t) * tw, static_cast<size_t>(ut * unit_bytes));
    }
  }

  // S[o][t] = Σ_c w[o, t, c] over the kic channels of tap t.
  int32_t tap_sum(const uint32_t* w, int64_t o, int64_t t) const {
    const int64_t tw = bit_words(kic);
    int ones = 0;
    for (int64_t q = 0; q < tw; ++q) ones += popcount32(w[(o * taps + t) * tw + q]);
    return static_cast<int32_t>(kic - 2 * ones);
  }
};

// Repacks one bit row of `groups*kic` channels into groups x bit_words(kic)
// words with zero padding bits (BConvGroupLayout::regroup, bit input).
inline void regroup_bit_row(const uint32_t* src, int64_t groups, int64_t kic, uint32_t* dst) {
  const int64_t tw = bit_words(kic);
  for (int64_t gi = 0; gi < groups; ++gi) {
    for (int64_t q = 0; q < tw; ++q) {
      const int64_t b0 = gi * kic + q * 32;               // first source bit of this word
      const int64_t nb = kic - q * 32 < 32 ? kic - q * 32 : 32;
      const int64_t wi = b0 >> 5, sh = b0 & 31;
      uint64_t v = src[wi];
      if (sh + nb > 32) v |= uint64_t(src[wi + 1]) << 32;
      v >>= sh;
      dst[gi * tw + q] = static_cast<uint32_t>(nb == 32 ? v : v & ((uint64_t(1) << nb) - 1));
    }
  }
}

// Small byte copies (im2col taps are a few to a few hundred bytes, always a
// multiple of 2): fixed-size memcpy chunks compile to vector/GPR moves.
inline void copy_small(uint8_t* d, const uint8_t* s, int64_t n) {
  for (; n >= 16; n -= 16, d += 16, s += 16) std::memcpy(d, s, 16);
  if (n >= 8) {
    std::memcpy(d, s, 8);
    d += 8;
    s += 8;
    n -= 8;
  }
  if (n >= 4) {
    std::memcpy(d, s, 4);
    d += 4;
    s += 4;
    n -= 4;
  }
  if (n >= 2) std::memcpy(d, s, 2);
}

// Writes the patch row of output pixel (n, oy, ox), group gi, to `row`
// (k_units units; bytes beyond them are not touched) from the source buffer
// `src` (see BConvGroupLayout). Out-of-bounds taps are zero bits (+1).
// Returns the out-of-bounds tap mask (bit t for tap t < 64; 0 = interior).
// Interior pixels whose taps are contiguous in the source (groups == 1 or
// one group spanning whole pixels, dilation_w == 1) copy each kernel row in
// one run.
inline uint64_t build_patch_row_any(const Conv2DGeom& g, const BConvGroupLayout& L, const uint8_t* src, int64_t gi, int64_t n,
                                    int64_t oy, int64_t ox, uint8_t* row) {
  const int64_t tb = L.ut * L.unit_bytes;  // bytes per tap
  const int64_t pb = L.src_pixel_bytes();
  const int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl;
  const uint8_t* img = src + n * g.ih * g.iw * pb + L.group_offset_bytes(gi);
  if (iy0 >= 0 && ix0 >= 0 && iy0 + g.dh * (g.kh - 1) < g.ih && ix0 + g.dw * (g.kw - 1) < g.iw) {
    const bool runs = g.dw == 1 && tb == pb;
    for (int64_t ky = 0; ky < g.kh; ++ky) {
      const uint8_t* s = img + ((iy0 + ky * g.dh) * g.iw + ix0) * pb;
      uint8_t* d = row + ky * g.kw * tb;
      if (runs) {
        copy_small(d, s, g.kw * tb);
      } else {
        for (int64_t kx = 0; kx < g.kw; ++kx) copy_small(d + kx * tb, s + kx * g.dw * pb, tb);
      }
    }
    return 0;
  }
  uint64_t m = 0;
  int64_t t = 0;
  for (int64_t ky = 0; ky < g.kh; ++ky) {
    const int64_t iy = iy0 + ky * g.dh;
    for (int64_t kx = 0; kx < g.kw; ++kx, ++t) {
      const int64_t ix = ix0 + kx * g.dw;
      uint8_t* d = row + t * tb;
      if (iy < 0 || iy >= g.ih || ix < 0 || ix >= g.iw) {
        if (t < 64) m |= uint64_t(1) << t;
        std::memset(d, 0, static_cast<size_t>(tb));
      } else {
        copy_small(d, img + (iy * g.iw + ix) * pb, tb);
      }
    }
  }
  return m;
}

// Copies a run of n bytes (n >= 2, even: whole 16-bit units) with 16-, 8-, 4-
// or 2-byte moves; the last move overlaps the previous one instead of a
// byte-size tail (it rewrites bytes it already wrote, with the same values).
inline __attribute__((always_inline)) void copy_run(uint8_t* d, const uint8_t* s, int64_t n) {
  if (n >= 16) {
    for (int64_t i = 0; i + 16 < n; i += 16) std::memcpy(d + i, s + i, 16);
    std::memcpy(d + n - 16, s + n - 16, 16);
  } else if (n >= 8) {
    std::memcpy(d, s, 8);
    std::memcpy(d + n - 8, s + n - 8, 8);
  } else if (n >= 4) {
    std::memcpy(d, s, 4);
    std::memcpy(d + n - 4, s + n - 4, 4);
  } else {
    std::memcpy(d, s, 2);
  }
}

// Inline fast paths for interior pixels: grouped convolutions with 16 / 32
// channels per group and a 3x3 kernel (one 2- or 4-byte load / store per tap),
// and taps contiguous in the source (groups == 1, dilation_w == 1: one run of
// kw taps per kernel row); everything else through build_patch_row_any.
inline __attribute__((always_inline)) uint64_t build_patch_row(const Conv2DGeom& g, const BConvGroupLayout& L, const uint8_t* src,
                                                               int64_t gi, int64_t n, int64_t oy, int64_t ox, uint8_t* row) {
  const int64_t tb = L.ut * L.unit_bytes, pb = L.src_pixel_bytes();
  const int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl;
  if (tb == pb && g.dw == 1 && iy0 >= 0 && ix0 >= 0 && iy0 + g.dh * (g.kh - 1) < g.ih && ix0 + g.kw - 1 < g.iw) {
    const uint8_t* s = src + ((n * g.ih + iy0) * g.iw + ix0) * pb + L.group_offset_bytes(gi);
    const int64_t run = g.kw * tb, rs = g.dh * g.iw * pb;
    for (int64_t ky = 0; ky < g.kh; ++ky, s += rs, row += run) copy_run(row, s, run);
    return 0;
  }
  if ((tb == 2 || tb == 4) && g.kh == 3 && g.kw == 3 && tb != pb && iy0 >= 0 && ix0 >= 0 && iy0 + 2 * g.dh < g.ih &&
      ix0 + 2 * g.dw < g.iw) {
    const uint8_t* s0 = src + ((n * g.ih + iy0) * g.iw + ix0) * pb + L.group_offset_bytes(gi);
    const int64_t rs = g.dh * g.iw * pb, cs = g.dw * pb;
    if (tb == 2) {
      for (int ky = 0; ky < 3; ++ky, s0 += rs)
        for (int kx = 0; kx < 3; ++kx, row += 2) std::memcpy(row, s0 + kx * cs, 2);
    } else {
      for (int ky = 0; ky < 3; ++ky, s0 += rs)
        for (int kx = 0; kx < 3; ++kx, row += 4) std::memcpy(row, s0 + kx * cs, 4);
    }
    return 0;
  }
  return build_patch_row_any(g, L, src, gi, n, oy, ox, row);
}

struct BConvPlan {
  Conv2DGeom g;
  BConvGroupLayout L;
  Activation act = Activation::kNone;
  BinarizeMode mode = BinarizeMode::kGe;
  int pad_value = 0;
  bool bit_out = false;
  bool bit_in = false;
  int64_t k_words = 0;      // words per patch row before padding (L.k_words)
  int64_t k_words_pad = 0;  // padded to a multiple of 4
  int64_t k_bits = 0;       // real bits per patch row = kh*kw*kic
  int64_t out_words = 0;
  int64_t pixels = 0;       // n*oh*ow
  std::shared_ptr<void> wbuf;     // [oc][k_words_pad] uint32, im2col order (group of o = o / L.ocg)
  std::vector<int32_t> tap_sum;   // [oc][kh*kw] S[o][tap]
  std::shared_ptr<void> xbits;    // packed (or per-group repacked) input [n*ih*iw][L.src_words]
  std::shared_ptr<void> patches;  // [groups][n*oh*ow][k_words_pad]

  const uint32_t* weights() const { return static_cast<const uint32_t*>(wbuf.get()); }
  size_t prepacked_bytes() const { return static_cast<size_t>(g.oc * k_words_pad * 4 + tap_sum.size() * 4); }
  // Patch row of output pixel q for the group of output channel o.
  const uint32_t* patch_row(const uint32_t* pt, int64_t q, int64_t o) const {
    return pt + ((o / L.ocg) * pixels + q) * k_words_pad;
  }

  void build(const KernelContext& ctx) {
    const Json& a = ctx.attrs();
    g = conv_geometry(a, ctx.in(0).shape(), ctx.in(1).shape());
    L.init(g);
    act = parse_activation(a);
    mode = parse_binarize(a);
    pad_value = static_cast<int>(a.get_number("pad_value", 0.0));
    bout.parse(ctx);
    bit_out = bout.active;
    bit_in = ctx.in(0).dtype() == DType::kBit;
    k_words = L.k_words;
    k_words_pad = (k_words + 3) / 4 * 4;
    k_bits = L.k_bits;
    out_words = bit_words(g.oc);
    pixels = g.n * g.oh * g.ow;

    const uint32_t* w = ctx.in(1).data<uint32_t>();  // [oc][kh][kw][bit_words(kic)]
    wbuf = aligned_alloc_shared(static_cast<size_t>(g.oc * k_words_pad * 4));
    L.pack_weight_rows(w, g.oc, static_cast<uint32_t*>(wbuf.get()), k_words_pad);
    const int64_t taps = L.taps;
    tap_sum.assign(static_cast<size_t>(g.oc * taps), 0);
    for (int64_t o = 0; o < g.oc; ++o)
      for (int64_t t = 0; t < taps; ++t) tap_sum[o * taps + t] = L.tap_sum(w, o, t);
    if (!bit_in || L.regroup) xbits = aligned_alloc_shared(static_cast<size_t>(g.n * g.ih * g.iw * L.src_words * 4));
    patches = aligned_alloc_shared(static_cast<size_t>(g.groups * pixels * k_words_pad * 4));
  }

  // Packs f32 input (if needed) and builds patch rows. Returns patch matrix
  // [groups][pixels][k_words_pad] (index rows with patch_row()).
  const uint32_t* prepare_patches(const KernelContext& ctx, const float* thr) {
    const uint32_t* xb;
    const int64_t pix = g.n * g.ih * g.iw, sw = L.src_words;
    if (bit_in && !L.regroup) {
      xb = ctx.in(0).data<uint32_t>();
    } else {
      auto* dst = static_cast<uint32_t*>(xbits.get());
      const BinarizeMode m = mode;
      const BConvGroupLayout& lay = L;
      const int64_t ic = g.ic;
      const bool bin = bit_in;
      const uint32_t* xin = bin ? ctx.in(0).data<uint32_t>() : nullptr;
      const float* x = bin ? nullptr : ctx.in(0).data<float>();
      ctx.pool->parallel_for(pix, 64, [&](int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; ++p) {
          if (bin) {
            regroup_bit_row(xin + p * bit_words(ic), lay.groups, lay.kic, dst + p * sw);
          } else if (!lay.regroup) {
            pack_row_f32(x + p * ic, ic, dst + p * sw, m, thr);
          } else {
            for (int64_t gi = 0; gi < lay.groups; ++gi)
              pack_row_f32(x + p * ic + gi * lay.kic, lay.kic, dst + p * sw + gi * lay.ut, m, thr ? thr + gi * lay.kic : nullptr);
          }
        }
      });
      xb = dst;
    }
    auto* pt = static_cast<uint32_t*>(patches.get());
    const Conv2DGeom gg = g;
    const BConvGroupLayout& lay = L;
    const int64_t kwp = k_words_pad, px = pixels;
    const auto* src = reinterpret_cast<const uint8_t*>(xb);
    const int64_t used = lay.k_units * lay.unit_bytes;
    ctx.pool->parallel_for(gg.n * gg.oh, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        int64_t n = r / gg.oh, oy = r % gg.oh;
        for (int64_t gi = 0; gi < gg.groups; ++gi)
          for (int64_t ox = 0; ox < gg.ow; ++ox) {
            auto* row = reinterpret_cast<uint8_t*>(pt + (gi * px + r * gg.ow + ox) * kwp);
            build_patch_row(gg, lay, src, gi, n, oy, ox, row);
            std::memset(row + used, 0, static_cast<size_t>(kwp * 4 - used));
          }
      }
    });
    return pt;
  }

  // Spatial-padding correction for output pixel `opix` and channel o.
  int32_t pad_correction(int64_t opix, int64_t o) const {
    if (pad_value == 1) return 0;
    int64_t ox = opix % g.ow, oy = (opix / g.ow) % g.oh;
    int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl;
    if (iy0 >= 0 && ix0 >= 0 && iy0 + g.dh * (g.kh - 1) < g.ih && ix0 + g.dw * (g.kw - 1) < g.iw) return 0;
    int32_t s = 0;
    for (int64_t ky = 0; ky < g.kh; ++ky) {
      int64_t iy = iy0 + ky * g.dh;
      for (int64_t kx = 0; kx < g.kw; ++kx) {
        int64_t ix = ix0 + kx * g.dw;
        if (iy < 0 || iy >= g.ih || ix < 0 || ix >= g.iw) s += tap_sum[o * g.kh * g.kw + ky * g.kw + kx];
      }
    }
    return (pad_value - 1) * s;
  }

  bool pixel_is_interior(int64_t opix) const {
    int64_t ox = opix % g.ow, oy = (opix / g.ow) % g.oh;
    int64_t iy0 = oy * g.sh - g.pt, ix0 = ox * g.sw - g.pl;
    return iy0 >= 0 && ix0 >= 0 && iy0 + g.dh * (g.kh - 1) < g.ih && ix0 + g.dw * (g.kw - 1) < g.iw;
  }

  // Applies post_mul/post_add/activation and stores one output value.
  inline void store(const KernelContext& ctx, int64_t opix, int64_t o, int32_t dot, const float* mul, const float* add) const {
    float v = static_cast<float>(dot);
    if (mul) v *= mul[o];
    if (add) v += add[o];
    v = apply_activation(act, v);
    if (bit_out) {
      store_bit(ctx, opix, o, v);
    } else {
      ctx.out(0).data<float>()[opix * g.oc + o] = v;
    }
  }

  // Bit output (kernels/bit_output.h): the bit of the final value v, or v
  // itself into float_rows when the kernel still has to run an epilogue on it
  // (it then binarizes those rows with pack_output_rows()).
  void store_bit(const KernelContext& ctx, int64_t opix, int64_t o, float v) const {
    if (float_rows) {
      float_rows[opix * g.oc + o] = v;
      return;
    }
    const float* t = bout.threshold(ctx);
    uint32_t* row = ctx.out(0).data<uint32_t>() + opix * out_words;
    uint32_t bit = 1u << (o & 31);
    if (output_bit(v, bout.mode, t ? t[o] : 0.f)) row[o >> 5] |= bit;
    else row[o >> 5] &= ~bit;
    // The output buffer may hold stale arena data: clear the padding bits
    // of the last word (owned by the same pixel, hence the same thread).
    if (o == g.oc - 1 && (g.oc & 31)) row[o >> 5] &= (1u << (g.oc & 31)) - 1;
  }

  BitOutput bout;
  float* float_rows = nullptr;  // [pixels][oc] float output before binarization (bit output + epilogue)
};

// Packed kernels without groups support (bconv2d.neon).
inline bool bconv_packed_supported(const KernelContext& ctx) {
  const Json& a = ctx.attrs();
  if (a.get_int("groups", 1) != 1) return false;
  if (a.get_string("input_binarize", "ge") == "sign_zero") return false;
  return true;
}

// Packed kernels that handle groups >= 1 through BConvGroupLayout.
inline bool bconv_packed_grouped_supported(const KernelContext& ctx) {
  return ctx.attrs().get_string("input_binarize", "ge") != "sign_zero";
}

}  // namespace bitino

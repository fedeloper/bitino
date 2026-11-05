// Reference ("oracle") kernels of the full-integer I8* operators, tier
// int8_full_reference. Portable C++, literal semantics of docs/int8.md
// ("Full-integer execution: int8_full"), bit-identical to the numpy oracle
// python/bitino/int8_ops.py. Integer arithmetic in int64 where the contract
// says int64; fp32 epilogues one IEEE operation at a time (the build uses
// -ffp-contract=off). The binary layers pack their binarized int8 inputs into
// bit-planes and use 64-bit popcounts so that whole models run at usable
// speed on the host; the arithmetic is exact either way.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "kernels/bitpack.h"
#include "kernels/int8_full/int8_full.h"
#include "kernels/int8_full/q8_math.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"

namespace bitino {
namespace {

using q8::clamp8;

int qmin_of(const KernelContext& ctx) { return static_cast<int>(ctx.attrs().get_int("qmin", -128)); }
int qmax_of(const KernelContext& ctx) { return static_cast<int>(ctx.attrs().get_int("qmax", 127)); }
bool f32_out(const KernelContext& ctx) { return ctx.attrs().get_string("output", "i8") == "f32"; }

// ------------------------------------------------------------- (De)Quantize
class QuantizeRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const float s = static_cast<float>(ctx.attrs().get_number("scale", 1));
    const float z = static_cast<float>(ctx.attrs().get_int("zero", 0));
    const float* x = ctx.in(0).data<float>();
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t i = 0, n = ctx.in(0).size(); i < n; ++i) y[i] = int8_full::quantize1(x[i], s, z);
  }
};

class DequantizeRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const float s = static_cast<float>(ctx.attrs().get_number("scale", 1));
    const int32_t z = static_cast<int32_t>(ctx.attrs().get_int("zero", 0));
    const int8_t* x = ctx.in(0).data<int8_t>();
    float* y = ctx.out(0).data<float>();
    for (int64_t i = 0, n = ctx.in(0).size(); i < n; ++i) y[i] = q8::deq(x[i], s, z);
  }
};

// --------------------------------------------------------------------- Lut
class LutRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& t = ctx.in(1);
    // table row = index along `axis` (default last): (i / inner) % C
    int64_t ax = ctx.attrs().get_int("axis", -1);
    if (ax < 0) ax += x.rank();
    int64_t inner = 1;
    for (int d = static_cast<int>(ax) + 1; d < x.rank(); ++d) inner *= x.dim(d);
    const int64_t c = x.dim(ax), n = x.size(), rows = t.dim(0);
    auto row = [&](int64_t i) { return rows == 1 ? 0 : (i / inner) % c; };
    const int8_t* q = x.data<int8_t>();
    if (t.dtype() == DType::kF32) {
      const float* tab = t.data<float>();
      float* y = ctx.out(0).data<float>();
      for (int64_t i = 0; i < n; ++i) y[i] = tab[row(i) * 256 + q[i] + 128];
      return;
    }
    const int8_t* tab = t.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t i = 0; i < n; ++i) y[i] = tab[row(i) * 256 + q[i] + 128];
  }
};

// ----------------------------------------------------------------- Add / Mul
// b is broadcast to a's shape (right-aligned); index of b for each a element.
std::vector<int64_t> bcast_index(const Shape& a, const Shape& b) {
  const int64_t n = num_elements(a);
  std::vector<int64_t> idx(static_cast<size_t>(n));
  if (a == b) {
    for (int64_t i = 0; i < n; ++i) idx[i] = i;
    return idx;
  }
  const int ra = static_cast<int>(a.size()), rb = static_cast<int>(b.size());
  std::vector<int64_t> bs(ra, 1), stride(ra, 0);
  for (int i = 0; i < rb; ++i) bs[ra - rb + i] = b[i];
  int64_t st = 1;
  for (int d = ra - 1; d >= 0; --d) {
    stride[d] = bs[d] == 1 ? 0 : st;
    st *= bs[d];
  }
  for (int64_t i = 0; i < n; ++i) {
    int64_t rem = i, off = 0;
    for (int d = ra - 1; d >= 0; --d) {
      off += (rem % a[d]) * stride[d];
      rem /= a[d];
    }
    idx[i] = off;
  }
  return idx;
}

class AddMulRef : public Kernel {
 public:
  explicit AddMulRef(bool add) : add_(add) {}
  void prepare(const KernelContext& ctx) override { bidx_ = bcast_index(ctx.in(0).shape(), ctx.in(1).shape()); }
  void run(const KernelContext& ctx) override {
    const Json& a = ctx.attrs();
    const int64_t za = a.get_int("a_zero", 0), zb = a.get_int("b_zero", 0), zy = a.get_int("y_zero", 0);
    const int lo = qmin_of(ctx), hi = qmax_of(ctx);
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t n = ctx.in(0).size();
    if (add_) {
      const int ls = static_cast<int>(a.get_int("left_shift", 20));
      const int32_t am = static_cast<int32_t>(a.get_int("a_mult", 0)), as = static_cast<int32_t>(a.get_int("a_shift", 0));
      const int32_t bm = static_cast<int32_t>(a.get_int("b_mult", 0)), bs = static_cast<int32_t>(a.get_int("b_shift", 0));
      const int32_t om = static_cast<int32_t>(a.get_int("out_mult", 0)), os = static_cast<int32_t>(a.get_int("out_shift", 0));
      for (int64_t i = 0; i < n; ++i) {
        int64_t xa = q8::requant((qa[i] - za) * (int64_t(1) << ls), am, as);
        int64_t xb = q8::requant((qb[bidx_[i]] - zb) * (int64_t(1) << ls), bm, bs);
        y[i] = clamp8(zy + q8::requant(xa + xb, om, os), lo, hi);
      }
    } else {
      const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), s = static_cast<int32_t>(a.get_int("shift", 0));
      for (int64_t i = 0; i < n; ++i) y[i] = clamp8(zy + q8::requant((qa[i] - za) * (qb[bidx_[i]] - zb), m, s), lo, hi);
    }
  }

 private:
  bool add_;
  std::vector<int64_t> bidx_;
};
class AddRef : public AddMulRef {
 public:
  AddRef() : AddMulRef(true) {}
};
class MulRef : public AddMulRef {
 public:
  MulRef() : AddMulRef(false) {}
};

// --------------------------------------------------------------- AffineSum
// acc = sum_i (q_i - z_i) * K[i][c] + B[c] (int64; the quantizer keeps it within int32),
// y = clamp(y_zero + requant(acc, mult, shift)).
class AffineSumRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const size_t nin = ctx.inputs.size() - 2;
    const Tensor& k = ctx.in(nin);
    const int32_t* K = k.data<int32_t>();
    const int32_t* B = ctx.in(nin + 1).data<int32_t>();
    const int64_t kc = k.dim(1), c = ctx.in(0).dim(-1), n = ctx.in(0).size();
    auto zs = ctx.attrs().get_ints("zeros", {});
    const int32_t m = static_cast<int32_t>(ctx.attrs().get_int("mult", 0)), s = static_cast<int32_t>(ctx.attrs().get_int("shift", 0));
    const int64_t zy = ctx.attrs().get_int("y_zero", 0);
    const int lo = qmin_of(ctx), hi = qmax_of(ctx);
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t e = 0; e < n; ++e) {
      const int64_t ch = kc == 1 ? 0 : e % c;
      int64_t acc = B[ch];
      for (size_t i = 0; i < nin; ++i) acc += (ctx.in(i).data<int8_t>()[e] - zs[i]) * static_cast<int64_t>(K[i * kc + ch]);
      y[e] = clamp8(zy + q8::requant(acc, m, s), lo, hi);
    }
  }
};

// ------------------------------------------------- Conv2D / Dense / MatMul
// GEMM epilogue shared by the reference kernels (int8_full.h).
class Conv2DRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& w = ctx.in(1);
    auto g = conv_geometry(ctx.attrs(), x.shape(), w.shape());
    const int32_t xz = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    const int32_t pz = static_cast<int32_t>(ctx.attrs().get_int("pad_q", xz)) - xz;
    const int8_t* xd = x.data<int8_t>();
    const int8_t* wd = w.data<int8_t>();
    int8_full::GemmEpilogue ep(ctx);
    const int64_t ocg = g.oc / g.groups;
    // bit output: bit -1 iff !(acc >= bit_thr[o])
    const int32_t* bthr = ctx.has_in(5) ? ctx.in(5).data<int32_t>() : nullptr;
    const int64_t bw = bit_words(g.oc);
    if (bthr) std::fill_n(ctx.out(1).data<uint32_t>(), g.n * g.oh * g.ow * bw, 0u);
    for (int64_t n = 0; n < g.n; ++n)
      for (int64_t oy = 0; oy < g.oh; ++oy)
        for (int64_t ox = 0; ox < g.ow; ++ox) {
          const int64_t p = (n * g.oh + oy) * g.ow + ox;
          for (int64_t o = 0; o < g.oc; ++o) {
            const int64_t gi = o / ocg;
            int64_t acc = 0;
            for (int64_t ky = 0; ky < g.kh; ++ky)
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                const int64_t iy = oy * g.sh - g.pt + ky * g.dh, ix = ox * g.sw - g.pl + kx * g.dw;
                const bool in = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
                const int8_t* wr = wd + ((o * g.kh + ky) * g.kw + kx) * g.kic;
                const int8_t* xr = in ? xd + ((n * g.ih + iy) * g.iw + ix) * g.ic + gi * g.kic : nullptr;
                for (int64_t c = 0; c < g.kic; ++c) acc += static_cast<int64_t>(in ? xr[c] - xz : pz) * wr[c];
              }
            ep.store(ctx, p * g.oc + o, o, acc);
            if (bthr && !(acc >= bthr[o])) ctx.out(1).data<uint32_t>()[p * bw + (o >> 5)] |= 1u << (o & 31);
          }
        }
  }
};

class DenseRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& w = ctx.in(1);
    const int64_t k = w.dim(1), o = w.dim(0), rows = x.size() / k;
    const int32_t xz = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    const int8_t* xd = x.data<int8_t>();
    const int8_t* wd = w.data<int8_t>();
    int8_full::GemmEpilogue ep(ctx);
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t j = 0; j < o; ++j) {
        int64_t acc = 0;
        for (int64_t i = 0; i < k; ++i) acc += static_cast<int64_t>(xd[r * k + i] - xz) * wd[j * k + i];
        ep.store(ctx, r * o + j, j, acc);
      }
  }
};

class MatMulRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    auto g = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    const Json& a = ctx.attrs();
    const int64_t za = a.get_int("a_zero", 0), zb = a.get_int("b_zero", 0);
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    const bool f32 = f32_out(ctx);
    const float sc = static_cast<float>(a.get_number("scale", 1.0));
    const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), s = static_cast<int32_t>(a.get_int("shift", 0));
    const int64_t zy = a.get_int("y_zero", 0);
    const int lo = qmin_of(ctx), hi = qmax_of(ctx);
    for (int64_t bt = 0; bt < g.batch; ++bt)
      for (int64_t i = 0; i < g.m; ++i)
        for (int64_t j = 0; j < g.n; ++j) {
          int64_t acc = 0;
          for (int64_t k = 0; k < g.k; ++k) {
            const int64_t bv = g.transpose_b ? qb[(bt * g.n + j) * g.k + k] : qb[(bt * g.k + k) * g.n + j];
            acc += (qa[(bt * g.m + i) * g.k + k] - za) * (bv - zb);
          }
          const int64_t at = (bt * g.m + i) * g.n + j;
          if (f32) ctx.out(0).data<float>()[at] = static_cast<float>(acc) * sc;
          else ctx.out(0).data<int8_t>()[at] = clamp8(zy + q8::requant(q8::sat32(acc), m, s), lo, hi);
        }
  }
};

// ---------------------------------------------------------------- LayerNorm
class LayerNormRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t d = x.dim(-1), rows = x.size() / d;
    const int32_t xz = static_cast<int32_t>(ctx.attrs().get_int("x_zero", 0));
    const double xs = ctx.attrs().get_number("x_scale", 1.0), eps = ctx.attrs().get_number("eps", 1e-5);
    const float* G = ctx.in(1).data<float>();
    const float* B = ctx.in(2).data<float>();
    const int lo = qmin_of(ctx), hi = qmax_of(ctx);
    const int8_t* q = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t r = 0; r < rows; ++r) {
      const int8_t* qr = q + r * d;
      int64_t s1 = 0, s2 = 0;
      for (int64_t i = 0; i < d; ++i) {
        const int64_t t = qr[i] - xz;
        s1 += t;
        s2 += t * t;
      }
      const float rf = int8_full::layernorm_row_factor(d, s1, s2, xs, eps);
      for (int64_t i = 0; i < d; ++i) {
        const float u = static_cast<float>(d * (qr[i] - xz) - s1);
        const float v = u * rf;
        const float w = v * G[i];
        const float zz = w + B[i];
        y[r * d + i] = q8::round_f32(zz, lo, hi);
      }
    }
  }
};

// ------------------------------------------------------------------ Softmax
class SoftmaxRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t n = x.dim(-1), rows = x.size() / n;
    const int32_t* tab = ctx.in(1).data<int32_t>();
    const bool f32 = f32_out(ctx);
    const double num = f32 ? 1.0 : ctx.attrs().get_number("inv_scale", 1.0);
    const float zy = static_cast<float>(ctx.attrs().get_int("y_zero", 0));
    const float lo = static_cast<float>(qmin_of(ctx)), hi = static_cast<float>(qmax_of(ctx));
    const int8_t* q = x.data<int8_t>();
    for (int64_t r = 0; r < rows; ++r) {
      const int8_t* qr = q + r * n;
      int m = -128;
      for (int64_t i = 0; i < n; ++i) m = std::max<int>(m, qr[i]);
      int64_t s = 0;
      for (int64_t i = 0; i < n; ++i) s += tab[m - qr[i]];
      const float rr = static_cast<float>(num / static_cast<double>(s));
      for (int64_t i = 0; i < n; ++i) {
        const float v = static_cast<float>(tab[m - qr[i]]) * rr;
        if (f32) {
          ctx.out(0).data<float>()[r * n + i] = v;
        } else {
          const float t = std::nearbyint(v) + zy;
          ctx.out(0).data<int8_t>()[r * n + i] = static_cast<int8_t>(std::min(std::max(t, lo), hi));
        }
      }
    }
  }
};

// ---------------------------------------------------------- data movement
class ConcatRef : public Kernel {
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
    const int64_t zy = ctx.attrs().get_int("y_zero", 0);
    int8_t* yd = ctx.out(0).data<int8_t>();
    const int64_t ystride = y.dim(ax) * inner;
    int64_t off = 0;
    for (size_t k = 0; k < ctx.inputs.size(); ++k) {
      const Tensor& x = ctx.in(k);
      const int64_t chunk = x.dim(ax) * inner;
      const int8_t* xd = x.data<int8_t>();
      for (int64_t o = 0; o < outer; ++o) {
        int8_t* dst = yd + o * ystride + off;
        const int8_t* src = xd + o * chunk;
        if (ms[k] == 0) std::memcpy(dst, src, static_cast<size_t>(chunk));
        else
          for (int64_t i = 0; i < chunk; ++i)
            dst[i] = clamp8(zy + q8::requant(src[i] - zs[k], static_cast<int32_t>(ms[k]), static_cast<int32_t>(ss[k])));
      }
      off += chunk;
    }
  }
};

class TransposeRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    auto p = ctx.attrs().get_ints("perm", {});
    const int r = x.rank();
    std::vector<int64_t> xs(r, 1);
    for (int d = r - 2; d >= 0; --d) xs[d] = xs[d + 1] * x.dim(d + 1);
    const Tensor& y = ctx.out(0);
    const int8_t* xd = x.data<int8_t>();
    int8_t* yd = ctx.out(0).data<int8_t>();
    std::vector<int64_t> idx(r, 0);
    for (int64_t i = 0, n = y.size(); i < n; ++i) {
      int64_t off = 0;
      for (int d = 0; d < r; ++d) off += idx[d] * xs[p[d]];
      yd[i] = xd[off];
      for (int d = r - 1; d >= 0; --d) {
        if (++idx[d] < y.dim(d)) break;
        idx[d] = 0;
      }
    }
  }
};

class GatherRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& t = ctx.in(0);
    const Tensor& idx = ctx.in(1);
    const int64_t v = t.dim(0), d = t.dim(1), n = idx.size();
    const int8_t fill = static_cast<int8_t>(ctx.attrs().get_int("fill_q", 0));
    const int32_t* ids = idx.data<int32_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    const int64_t words = bit_words(d);
    for (int64_t i = 0; i < n; ++i) {
      int8_t* yr = y + i * d;
      const int64_t r = ids[i];
      if (r < 0 || r >= v) {
        std::fill_n(yr, d, fill);
        continue;
      }
      if (t.dtype() == DType::kI8) {
        std::memcpy(yr, t.data<int8_t>() + r * d, static_cast<size_t>(d));
        continue;
      }
      const uint32_t* row = t.data<uint32_t>() + r * words;
      const uint32_t* zrow = ctx.has_in(3) ? ctx.in(3).data<uint32_t>() + r * words : nullptr;
      const int8_t qp = ctx.in(2).data<int8_t>()[2 * r], qn = ctx.in(2).data<int8_t>()[2 * r + 1];
      for (int64_t c = 0; c < d; ++c) yr[c] = (zrow && bit_value(zrow, c) < 0) ? fill : (bit_value(row, c) > 0 ? qp : qn);
    }
  }
};

class PoolRef : public Kernel {
 public:
  explicit PoolRef(bool avg) : avg_(avg) {}
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    auto g = pool_geometry(ctx.attrs(), x.shape());
    const int64_t z = ctx.attrs().get_int("x_zero", 0);
    const bool inc = ctx.attrs().get_int("count_include_pad", 0) != 0;
    if (x.dtype() == DType::kBit) {  // I8MaxPool2D of a bit tensor: +1 iff some element of the window is +1
      const int64_t bw = bit_words(g.c);
      const uint32_t* xb = x.data<uint32_t>();
      uint32_t* yb = ctx.out(0).data<uint32_t>();
      std::fill_n(yb, g.n * g.oh * g.ow * bw, 0u);
      for (int64_t n = 0; n < g.n; ++n)
        for (int64_t oy = 0; oy < g.oh; ++oy)
          for (int64_t ox = 0; ox < g.ow; ++ox)
            for (int64_t c = 0; c < g.c; ++c) {
              int mx = -1;
              for (int64_t iy = std::max<int64_t>(oy * g.sh - g.pt, 0); iy < std::min(oy * g.sh - g.pt + g.kh, g.ih); ++iy)
                for (int64_t ix = std::max<int64_t>(ox * g.sw - g.pl, 0); ix < std::min(ox * g.sw - g.pl + g.kw, g.iw); ++ix)
                  mx = std::max(mx, bit_value(xb + ((n * g.ih + iy) * g.iw + ix) * bw, c));
              if (mx < 0) yb[((n * g.oh + oy) * g.ow + ox) * bw + (c >> 5)] |= 1u << (c & 31);
            }
      return;
    }
    const int8_t* xd = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t n = 0; n < g.n; ++n)
      for (int64_t oy = 0; oy < g.oh; ++oy)
        for (int64_t ox = 0; ox < g.ow; ++ox) {
          const int64_t y0 = oy * g.sh - g.pt, x0 = ox * g.sw - g.pl;
          const int64_t ys = std::max<int64_t>(y0, 0), ye = std::min(y0 + g.kh, g.ih);
          const int64_t xs = std::max<int64_t>(x0, 0), xe = std::min(x0 + g.kw, g.iw);
          for (int64_t c = 0; c < g.c; ++c) {
            int64_t acc = 0;
            int mx = -128;
            for (int64_t iy = ys; iy < ye; ++iy)
              for (int64_t ix = xs; ix < xe; ++ix) {
                const int v = xd[((n * g.ih + iy) * g.iw + ix) * g.c + c];
                acc += v - z;
                mx = std::max(mx, v);
              }
            const int64_t at = ((n * g.oh + oy) * g.ow + ox) * g.c + c;
            if (!avg_) {
              y[at] = static_cast<int8_t>(mx);
            } else {
              const int64_t cnt = inc ? g.kh * g.kw : (ye - ys) * (xe - xs);
              y[at] = clamp8(z + q8::rdiv(acc, std::max<int64_t>(cnt, 1)));
            }
          }
        }
  }

 private:
  bool avg_;
};
class MaxPoolRef : public PoolRef {
 public:
  MaxPoolRef() : PoolRef(false) {}
};
class AvgPoolRef : public PoolRef {
 public:
  AvgPoolRef() : PoolRef(true) {}
};

// GlobalAvgPool (NHWC, over H*W) and ReduceMean (one axis): requantizing mean.
class MeanRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Json& a = ctx.attrs();
    int64_t outer, len, inner;
    if (ctx.node->op == "I8GlobalAvgPool") {
      outer = x.dim(0);
      len = x.dim(1) * x.dim(2);
      inner = x.dim(3);
    } else {
      int64_t ax = a.get_int("axis", -1);
      if (ax < 0) ax += x.rank();
      outer = inner = 1;
      for (int d = 0; d < ax; ++d) outer *= x.dim(d);
      for (int d = static_cast<int>(ax) + 1; d < x.rank(); ++d) inner *= x.dim(d);
      len = x.dim(ax);
    }
    const int64_t zx = a.get_int("x_zero", 0), zy = a.get_int("y_zero", 0);
    const bool rq = a.contains("mult");
    const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), s = static_cast<int32_t>(a.get_int("shift", 0));
    const int8_t* xd = x.data<int8_t>();
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t o = 0; o < outer; ++o)
      for (int64_t i = 0; i < inner; ++i) {
        int64_t acc = 0;
        for (int64_t l = 0; l < len; ++l) acc += xd[(o * len + l) * inner + i] - zx;
        y[o * inner + i] = rq ? clamp8(zy + q8::requant(acc, m, s)) : clamp8(zx + q8::rdiv(acc, len));
      }
  }
};

class PadRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& y = ctx.out(0);
    auto p = ctx.attrs().get_ints("pads", {});
    const int8_t v = static_cast<int8_t>(ctx.attrs().get_int("pad_q", 0));
    int8_t* yd = ctx.out(0).data<int8_t>();
    std::fill_n(yd, y.size(), v);
    const int r = x.rank();
    for (int64_t i = 0, n = x.size(); i < n; ++i) {
      int64_t rem = i, off = 0, mul = 1;
      std::vector<int64_t> idx(r);
      for (int d = r - 1; d >= 0; --d) {
        idx[d] = rem % x.dim(d);
        rem /= x.dim(d);
      }
      for (int d = r - 1; d >= 0; --d) {
        off += (idx[d] + p[2 * d]) * mul;
        mul *= y.dim(d);
      }
      yd[off] = x.data<int8_t>()[i];
    }
  }
};

// ------------------------------------------------------- binary operators
// I8BDense / I8BConv2D: binarize the int8 input with integer thresholds,
// exact accumulation, fp32 epilogue v = acc*post_mul + post_add, then
// round to int8 (or store v for output f32).
class BDenseRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { bin_.prepare(ctx, ctx.in(1).dim(1)); }
  void run(const KernelContext& ctx) override {
    const Tensor& w = ctx.in(1);
    const int64_t k = w.dim(1), o = w.dim(0), rows = ctx.in(0).size() / k;
    std::vector<int8_t> xb(static_cast<size_t>(k));
    const uint32_t* wz = ctx.has_in(5) ? ctx.in(5).data<uint32_t>() : nullptr;
    const int64_t words = bit_words(k);
    for (int64_t r = 0; r < rows; ++r) {
      bin_.row(ctx.in(0).data<int8_t>() + r * k, k, xb.data());
      for (int64_t j = 0; j < o; ++j) {
        const uint32_t* wr = w.data<uint32_t>() + j * words;
        int64_t acc = 0;
        for (int64_t i = 0; i < k; ++i) {
          if (wz && bit_value(wz + j * words, i) < 0) continue;
          acc += xb[i] * bit_value(wr, i);
        }
        int8_full::binary_store(ctx, r * o + j, j, acc);
      }
    }
  }

 private:
  int8_full::BinarizeI8 bin_;
};

// I8BConv2D and I8BConv2DRes (res_ = true: residual + PReLU epilogue, l8 output, optional bit output).
// x: int8 codes binarized with integer thresholds, or a bit tensor taken as is.
class BConv2DRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    bit_in_ = ctx.in(0).dtype() == DType::kBit;
    if (!bit_in_) bin_.prepare(ctx, ctx.in(0).dim(3));
  }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Tensor& w = ctx.in(1);
    auto g = conv_geometry(ctx.attrs(), x.shape(), w.shape());
    const int pv = static_cast<int>(ctx.attrs().get_int("pad_value", 0));
    const int64_t np = g.n * g.ih * g.iw;
    std::vector<int8_t> xb(static_cast<size_t>(np * g.ic));
    if (bit_in_) {
      const int64_t xw = bit_words(g.ic);
      for (int64_t p = 0; p < np; ++p)
        for (int64_t c = 0; c < g.ic; ++c) xb[p * g.ic + c] = static_cast<int8_t>(bit_value(x.data<uint32_t>() + p * xw, c));
    } else {
      for (int64_t p = 0; p < np; ++p) bin_.row(x.data<int8_t>() + p * g.ic, g.ic, xb.data() + p * g.ic);
    }
    const bool res = ctx.node->op == "I8BConv2DRes";
    std::unique_ptr<int8_full::ResEpilogue> ep;
    if (res) {
      ep = std::make_unique<int8_full::ResEpilogue>(ctx);
      ep->bind_gate(ctx);
    }
    const float* bthr = res && ctx.has_in(8) ? ctx.in(8).data<float>() : nullptr;
    const bool bgt = ctx.attrs().get_string("bit_binarize", "ge") == "gt";
    const int64_t ow_words = bit_words(g.oc);
    if (bthr) std::fill_n(ctx.out(1).data<uint32_t>(), g.n * g.oh * g.ow * ow_words, 0u);
    const int64_t words = bit_words(g.kic), ocg = g.oc / g.groups;
    for (int64_t n = 0; n < g.n; ++n)
      for (int64_t oy = 0; oy < g.oh; ++oy)
        for (int64_t ox = 0; ox < g.ow; ++ox)
          for (int64_t o = 0; o < g.oc; ++o) {
            const int64_t gi = o / ocg;
            int64_t acc = 0;
            for (int64_t ky = 0; ky < g.kh; ++ky)
              for (int64_t kx = 0; kx < g.kw; ++kx) {
                const int64_t iy = oy * g.sh - g.pt + ky * g.dh, ix = ox * g.sw - g.pl + kx * g.dw;
                const bool in = iy >= 0 && iy < g.ih && ix >= 0 && ix < g.iw;
                const uint32_t* wr = w.data<uint32_t>() + ((o * g.kh + ky) * g.kw + kx) * words;
                for (int64_t c = 0; c < g.kic; ++c) {
                  const int b = in ? xb[((n * g.ih + iy) * g.iw + ix) * g.ic + gi * g.kic + c] : pv;
                  acc += b * bit_value(wr, c);
                }
              }
            const int64_t pix = (n * g.oh + oy) * g.ow + ox, at = pix * g.oc + o;
            if (!res) {
              int8_full::binary_store(ctx, at, o, acc);
              continue;
            }
            const float v = ep->pre(int8_full::binary_value(ctx, o, acc), n, o, g.oc);
            float s;  // u (bit_src "sum") or w (bit_src "out")
            ctx.out(0).data<int8_t>()[at] = ep->finish(v, o, ctx.in(5).data<int8_t>()[at], &s);
            if (bthr) {
              const bool pos = bgt ? s > bthr[o] : s >= bthr[o];
              if (!pos) ctx.out(1).data<uint32_t>()[pix * ow_words + (o >> 5)] |= 1u << (o & 31);
            }
          }
  }

 private:
  bool bit_in_ = false;
  int8_full::BinarizeI8 bin_;
};

class BMatMulRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    auto g = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    int8_full::BMatMulParams p(ctx);
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    std::vector<int8_t> A(static_cast<size_t>(g.m * g.k)), B(static_cast<size_t>(g.k * g.n));
    std::vector<int64_t> rs(static_cast<size_t>(g.m)), cs(static_cast<size_t>(g.n));
    for (int64_t bt = 0; bt < g.batch; ++bt) {
      for (int64_t i = 0; i < g.m * g.k; ++i) A[i] = static_cast<int8_t>(p.bin_a(qa[bt * g.m * g.k + i]));
      for (int64_t k = 0; k < g.k; ++k)
        for (int64_t j = 0; j < g.n; ++j)
          B[k * g.n + j] = static_cast<int8_t>(p.bin_b(g.transpose_b ? qb[(bt * g.n + j) * g.k + k] : qb[(bt * g.k + k) * g.n + j]));
      for (int64_t i = 0; i < g.m; ++i) {
        rs[i] = 0;
        for (int64_t k = 0; k < g.k; ++k) rs[i] += A[i * g.k + k];
      }
      for (int64_t j = 0; j < g.n; ++j) {
        cs[j] = 0;
        for (int64_t k = 0; k < g.k; ++k) cs[j] += B[k * g.n + j];
      }
      for (int64_t i = 0; i < g.m; ++i)
        for (int64_t j = 0; j < g.n; ++j) {
          int64_t ab = 0;
          if (p.c_ab)
            for (int64_t k = 0; k < g.k; ++k) ab += A[i * g.k + k] * B[k * g.n + j];
          p.store(ctx, (bt * g.m + i) * g.n + j, j, p.combine(ab, rs[i], cs[j]));
        }
    }
  }
};

}  // namespace

void register_int8_full_reference_kernels(OpRegistry& reg) {
  auto add = [&](const char* op, const char* name, std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, name, "int8_full_reference", 0, 0, nullptr, create});
  };
  add("I8Quantize", "i8quantize.reference", make_kernel<QuantizeRef>);
  add("I8Dequantize", "i8dequantize.reference", make_kernel<DequantizeRef>);
  add("I8Lut", "i8lut.reference", make_kernel<LutRef>);
  add("I8Add", "i8add.reference", make_kernel<AddRef>);
  add("I8Mul", "i8mul.reference", make_kernel<MulRef>);
  add("I8AffineSum", "i8affinesum.reference", make_kernel<AffineSumRef>);
  add("I8Conv2D", "i8conv2d.reference", make_kernel<Conv2DRef>);
  add("I8Dense", "i8dense.reference", make_kernel<DenseRef>);
  add("I8MatMul", "i8matmul.reference", make_kernel<MatMulRef>);
  add("I8LayerNorm", "i8layernorm.reference", make_kernel<LayerNormRef>);
  add("I8Softmax", "i8softmax.reference", make_kernel<SoftmaxRef>);
  add("I8Concat", "i8concat.reference", make_kernel<ConcatRef>);
  add("I8Transpose", "i8transpose.reference", make_kernel<TransposeRef>);
  add("I8Gather", "i8gather.reference", make_kernel<GatherRef>);
  add("I8MaxPool2D", "i8maxpool2d.reference", make_kernel<MaxPoolRef>);
  add("I8AvgPool2D", "i8avgpool2d.reference", make_kernel<AvgPoolRef>);
  add("I8GlobalAvgPool", "i8globalavgpool.reference", make_kernel<MeanRef>);
  add("I8ReduceMean", "i8reducemean.reference", make_kernel<MeanRef>);
  add("I8Pad", "i8pad.reference", make_kernel<PadRef>);
  add("I8BDense", "i8bdense.reference", make_kernel<BDenseRef>);
  add("I8BConv2D", "i8bconv2d.reference", make_kernel<BConv2DRef>);
  add("I8BConv2DRes", "i8bconv2dres.reference", make_kernel<BConv2DRef>);
  add("I8BMatMul", "i8bmatmul.reference", make_kernel<BMatMulRef>);
}

}  // namespace bitino

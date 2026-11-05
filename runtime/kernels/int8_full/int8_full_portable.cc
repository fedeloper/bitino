// Portable optimized kernels of the int8_full binary operators (tier int8_full_opt), used where the NEON
// kernels are not available (x86-64 hosts, for fast evaluation runs). Same scheme as the NEON wrappers
// (int8_full_neon.cc): binarize the int8 input with the integer thresholds of q8_math.h (bits for ge/gt,
// {-1,0,+1} floats for sign_zero and BMatMul operands), run the best float-path binary kernel the policy
// would pick for that operator on this CPU (bdense.avx2, bconv2d.avx2, bmatmul.avx2, ..., or binary_scalar),
// then apply the int8_full epilogue. Bit-identical to int8_full_reference (tests/cpp/test_int8_full.cc).
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "bitino/session.h"
#include "kernels/int8_full/int8_full.h"
#include "kernels/int8_full/q8_math.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"

namespace bitino {
namespace {

struct Sub {
  Node node;
  std::vector<Tensor> owned;
  KernelContext ctx;
  std::unique_ptr<Kernel> k;
};

// Highest-priority kernel of `op` among the exact binary tiers whose CPU features are present and that
// accepts the sub-node (never an int8_full or float_opt kernel).
std::unique_ptr<Kernel> pick_binary_kernel(const std::string& op, const KernelContext& sub) {
  const KernelDef* best = nullptr;
  for (const KernelDef* k : OpRegistry::global().kernels_for(op)) {
    if (k->tier != "binary_avx2" && k->tier != "binary_scalar" && k->tier != "neon" && k->tier != "reference") continue;
    if (!sub.cpu->has(k->required_features)) continue;
    if (k->supports && !k->supports(sub)) continue;
    if (!best || k->priority > best->priority) best = k;
  }
  BF_CHECK(best, "no binary kernel for the int8_full sub-node ", op);
  return best->create();
}

struct Thr {
  q8::QBin mode = q8::QBin::kGe;
  std::vector<int> tge, tgt;  // size 1 or C
};

Thr thresholds(const std::string& m, const q8::Deq& d, const float* thr, int64_t c) {
  Thr t;
  t.mode = m == "gt" ? q8::QBin::kGt : (m == "sign_zero" ? q8::QBin::kSignZero : (m == "gt0" ? q8::QBin::kGt0 : q8::QBin::kGe));
  const int64_t n = thr ? c : 1;
  for (int64_t i = 0; i < n; ++i) {
    const float v = thr ? thr[i] : 0.f;
    t.tge.push_back(q8::qthr_ge(v, d));
    t.tgt.push_back(q8::qthr_gt(v, d));
  }
  return t;
}

class BinaryLayerPortable : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    conv_ = ctx.node->op != "I8BDense";
    res_ = ctx.node->op == "I8BConv2DRes";
    bit_in_ = ctx.in(0).dtype() == DType::kBit;
    c_ = ctx.in(0).dim(-1);
    pix_ = ctx.in(0).size() / c_;
    const std::string mode = ctx.attrs().get_string("input_binarize", "ge");
    ternary_ = mode == "sign_zero" && !bit_in_;
    if (!bit_in_) thr_ = thresholds(mode, int8_full::deq_of(ctx.attrs(), "x"), ctx.has_in(4) ? ctx.in(4).data<float>() : nullptr, c_);
    if (res_) ep_ = std::make_unique<int8_full::ResEpilogue>(ctx);
    sub_.node.op = conv_ ? "BConv2D" : "BDense";
    sub_.node.name = ctx.node->name + "/binary";
    Json attrs = Json::Object{};
    for (const auto& kv : ctx.attrs().as_object())
      if (kv.first != "x_scale" && kv.first != "x_zero" && kv.first != "qmin" && kv.first != "qmax" && kv.first != "output" &&
          kv.first != "input_binarize" && kv.first.rfind("x_", 0) != 0 && kv.first.rfind("r_", 0) != 0 && kv.first.rfind("y_", 0) != 0 &&
          kv.first != "bit_binarize")
        attrs[kv.first] = kv.second;
    attrs["input_binarize"] = ternary_ ? "sign_zero" : "ge";
    sub_.node.attrs = attrs;
    sub_.owned.reserve(2);
    sub_.owned.push_back(ternary_ ? Tensor::allocate(DType::kF32, ctx.in(0).shape()) : Tensor::allocate(DType::kBit, ctx.in(0).shape()));
    sub_.owned.push_back(Tensor::allocate(DType::kF32, ctx.out(0).shape()));
    sub_.ctx.node = &sub_.node;
    sub_.ctx.pool = ctx.pool;
    sub_.ctx.cpu = ctx.cpu;
    sub_.ctx.inputs = {bit_in_ ? ctx.inputs[0] : &sub_.owned[0], ctx.inputs[1], ctx.inputs[2], ctx.inputs[3], nullptr};
    if (!res_ && ctx.has_in(5)) sub_.ctx.inputs.push_back(ctx.inputs[5]);
    sub_.ctx.outputs = {&sub_.owned[1]};
    sub_.k = pick_binary_kernel(sub_.node.op, sub_.ctx);
    sub_.k->prepare(sub_.ctx);
  }
  size_t prepacked_bytes() const override { return sub_.k ? sub_.k->prepacked_bytes() : 0; }

  void run(const KernelContext& ctx) override {
    if (bit_in_) sub_.ctx.inputs[0] = ctx.inputs[0];
    else pack(ctx);
    sub_.k->run(sub_.ctx);
    const float* v = sub_.owned[1].data<float>();
    const int64_t n = ctx.out(0).size();
    if (res_) {
      const int64_t oc = ctx.out(0).dim(-1), pix = n / oc, ow = bit_words(oc);
      const int8_t* r = ctx.in(5).data<int8_t>();
      int8_t* y = ctx.out(0).data<int8_t>();
      const float* bt = ctx.has_in(8) ? ctx.in(8).data<float>() : nullptr;
      const bool gt = ctx.attrs().get_string("bit_binarize", "ge") == "gt";
      uint32_t* yb = bt ? ctx.out(1).data<uint32_t>() : nullptr;
      const bool pre = ep_->pre_on();
      ep_->bind_gate(ctx);
      const int64_t ppi = pix / ctx.out(0).dim(0);  // pixels per sample (gate index)
      ctx.pool->parallel_for(pix, std::max<int64_t>(1, 16384 / std::max<int64_t>(oc, 1)), [&](int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; ++p) {
          if (yb) std::memset(yb + p * ow, 0, sizeof(uint32_t) * ow);
          for (int64_t o = 0; o < oc; ++o) {
            const int64_t i = p * oc + o;
            float s;  // u or w (bit_src)
            y[i] = ep_->finish(pre ? ep_->pre(v[i], p / ppi, o, oc) : v[i], o, r[i], &s);
            if (yb) {
              if (!(gt ? s > bt[o] : s >= bt[o])) yb[p * ow + (o >> 5)] |= 1u << (o & 31);
            }
          }
        }
      });
      return;
    }
    if (ctx.out(0).dtype() == DType::kF32) {
      std::memcpy(ctx.out(0).data<float>(), v, sizeof(float) * n);
      return;
    }
    const int lo = static_cast<int>(ctx.attrs().get_int("qmin", -128)), hi = static_cast<int>(ctx.attrs().get_int("qmax", 127));
    int8_t* y = ctx.out(0).data<int8_t>();
    for (int64_t i = 0; i < n; ++i) y[i] = q8::round_f32(v[i], lo, hi);
  }

 private:
  void pack(const KernelContext& ctx) {
    const int8_t* q = ctx.in(0).data<int8_t>();
    const bool pc = thr_.tge.size() > 1;
    const int64_t words = bit_words(c_);
    ctx.pool->parallel_for(pix_, std::max<int64_t>(1, 65536 / std::max<int64_t>(c_, 1)), [&](int64_t p0, int64_t p1) {
      for (int64_t p = p0; p < p1; ++p) {
        const int8_t* qr = q + p * c_;
        if (ternary_) {
          float* xf = sub_.owned[0].data<float>() + p * c_;
          for (int64_t i = 0; i < c_; ++i) {
            const size_t j = pc ? static_cast<size_t>(i) : 0;
            xf[i] = static_cast<float>(q8::qbin(thr_.mode, qr[i], thr_.tge[j], thr_.tgt[j]));
          }
        } else {
          uint32_t* xb = sub_.owned[0].data<uint32_t>() + p * words;
          std::memset(xb, 0, sizeof(uint32_t) * words);
          for (int64_t i = 0; i < c_; ++i) {
            const size_t j = pc ? static_cast<size_t>(i) : 0;
            if (q8::qbin(thr_.mode, qr[i], thr_.tge[j], thr_.tgt[j]) < 0) xb[i >> 5] |= 1u << (i & 31);
          }
        }
      }
    });
  }

  bool conv_ = false, ternary_ = false, res_ = false, bit_in_ = false;
  std::unique_ptr<int8_full::ResEpilogue> ep_;
  int64_t c_ = 0, pix_ = 0;
  Thr thr_;
  Sub sub_;
};

class BMatMulPortable : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    p_.reset(new int8_full::BMatMulParams(ctx));
    af_ = Tensor::allocate(DType::kF32, ctx.in(0).shape());
    bf_ = Tensor::allocate(DType::kF32, ctx.in(1).shape());
    acc_ = Tensor::allocate(DType::kF32, ctx.out(0).shape());
    if (p_->c_ab != 0) {
      sub_.node.op = "BMatMul";
      sub_.node.name = ctx.node->name + "/binary";
      sub_.node.attrs = Json::Object{{"a_binarize", "sign_zero"}, {"b_binarize", "sign_zero"},
                                     {"transpose_b", static_cast<int64_t>(g_.transpose_b ? 1 : 0)}, {"scale", 1.0}};
      sub_.ctx.node = &sub_.node;
      sub_.ctx.pool = ctx.pool;
      sub_.ctx.cpu = ctx.cpu;
      sub_.ctx.inputs = {&af_, &bf_};
      sub_.ctx.outputs = {&acc_};
      sub_.k = pick_binary_kernel("BMatMul", sub_.ctx);
      sub_.k->prepare(sub_.ctx);
    }
  }
  void run(const KernelContext& ctx) override {
    const int8_full::BMatMulParams& p = *p_;
    const int64_t m = g_.m, n = g_.n, k = g_.k;
    const int8_t* qa = ctx.in(0).data<int8_t>();
    const int8_t* qb = ctx.in(1).data<int8_t>();
    float* af = af_.data<float>();
    float* bf = bf_.data<float>();
    std::vector<int64_t> rs(static_cast<size_t>(g_.batch * m), 0), cs(static_cast<size_t>(g_.batch * n), 0);
    for (int64_t i = 0, na = ctx.in(0).size(); i < na; ++i) {
      af[i] = static_cast<float>(p.bin_a(qa[i]));
      rs[i / k] += static_cast<int64_t>(af[i]);
    }
    for (int64_t i = 0, nb = ctx.in(1).size(); i < nb; ++i) {
      bf[i] = static_cast<float>(p.bin_b(qb[i]));
      const int64_t bt = i / (k * n), r = i % (k * n);
      cs[bt * n + (g_.transpose_b ? r / k : r % n)] += static_cast<int64_t>(bf[i]);
    }
    if (p.c_ab != 0) sub_.k->run(sub_.ctx);
    const float* ab = acc_.data<float>();
    for (int64_t r = 0; r < g_.batch * m; ++r)
      for (int64_t j = 0; j < n; ++j) {
        const int64_t abv = p.c_ab != 0 ? static_cast<int64_t>(ab[r * n + j]) : 0;
        p.store(ctx, r * n + j, j, p.combine(abv, rs[r], cs[(r / m) * n + j]));
      }
  }

 private:
  MatMulGeom g_;
  std::unique_ptr<int8_full::BMatMulParams> p_;
  Tensor af_, bf_, acc_;
  Sub sub_;
};

// I8MaxPool2D of a bit tensor (word-parallel; any CPU): an output bit is -1 iff every bit of the window is -1, so the
// output word is the AND of the window's words (no empty windows: padding < kernel, op_defs).
class MaxPoolBits : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const Pool2DGeom g = pool_geometry(ctx.attrs(), x.shape());
    const int64_t bw = bit_words(g.c);
    const uint32_t* xb = x.data<uint32_t>();
    uint32_t* yb = ctx.out(0).data<uint32_t>();
    const uint32_t last = g.c % 32 ? (1u << (g.c % 32)) - 1u : 0xffffffffu;  // padding bits of the last word stay 0
    const int64_t rows = g.n * g.oh;  // one task = one output row
    ctx.pool->parallel_for(rows, std::max<int64_t>(1, 4096 / (g.ow * bw * g.kh * g.kw)), [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t n = r / g.oh, y0 = (r % g.oh) * g.sh - g.pt;
        const int64_t ys = std::max<int64_t>(y0, 0), ye = std::min(y0 + g.kh, g.ih);
        const uint32_t* img = xb + n * g.ih * g.iw * bw;
        uint32_t* yp = yb + r * g.ow * bw;
        for (int64_t ox = 0; ox < g.ow; ++ox, yp += bw) {
          const int64_t x0 = ox * g.sw - g.pl;
          const int64_t xs = std::max<int64_t>(x0, 0), xe = std::min(x0 + g.kw, g.iw);
          for (int64_t k = 0; k < bw; ++k) {
            uint32_t v = k == bw - 1 ? last : 0xffffffffu;
            for (int64_t iy = ys; iy < ye; ++iy)
              for (int64_t ix = xs; ix < xe; ++ix) v &= img[(iy * g.iw + ix) * bw + k];
            yp[k] = v;
          }
        }
      }
    });
  }
};
bool maxpool_bits_supported(const KernelContext& ctx) { return ctx.in(0).dtype() == DType::kBit; }

}  // namespace

void register_int8_full_portable_kernels(OpRegistry& reg) {
  reg.add_kernel({"I8MaxPool2D", "i8maxpool2d.bits", "int8_full_opt", 50, 0, maxpool_bits_supported, make_kernel<MaxPoolBits>});
  reg.add_kernel({"I8BDense", "i8bdense.portable", "int8_full_opt", 50, 0, nullptr, make_kernel<BinaryLayerPortable>});
  reg.add_kernel({"I8BConv2D", "i8bconv2d.portable", "int8_full_opt", 50, 0, nullptr, make_kernel<BinaryLayerPortable>});
  reg.add_kernel({"I8BConv2DRes", "i8bconv2dres.portable", "int8_full_opt", 50, 0, nullptr, make_kernel<BinaryLayerPortable>});
  reg.add_kernel({"I8BMatMul", "i8bmatmul.portable", "int8_full_opt", 50, 0, nullptr, make_kernel<BMatMulPortable>});
}

}  // namespace bitino

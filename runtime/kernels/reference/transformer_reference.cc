// Reference ("oracle") kernels for the transformer operators. Literal loops
// following docs/operators.md § Transformer operators; the numpy executor
// (python/bitino/executor.py) mirrors the float operation order exactly so
// the two are bit-identical:
//   * LayerNorm: double, sequential sums in index order, ((x-mean)/sqrt(var+eps))*gamma+beta;
//   * MatMul: double accumulation in k order starting at 0, then (float)(acc*scale);
//   * BMatMul: exact int32 accumulation of binarized operands, then (float)acc * (float)scale.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/bitpack.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"
#include "ops/transformer_common.h"

namespace bitino {
namespace {

// ------------------------------------------------------------------- Gather
class GatherRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const Tensor& table = ctx.in(0);
    const Tensor& idx = ctx.in(1);
    const int64_t v = table.dim(0), d = table.dim(1), n = idx.size();
    const int32_t* ids = idx.data<int32_t>();
    const float* scale = ctx.has_in(2) ? ctx.in(2).data<float>() : nullptr;
    const uint32_t* zmask = ctx.has_in(3) ? ctx.in(3).data<uint32_t>() : nullptr;
    float* y = ctx.out(0).data<float>();
    const int64_t words = bit_words(d);
    for (int64_t i = 0; i < n; ++i) {
      float* yr = y + i * d;
      int64_t r = ids[i];
      if (r < 0 || r >= v) {  // documented: out-of-range ids give a zero row (hot paths never throw)
        std::fill_n(yr, d, 0.f);
        continue;
      }
      if (table.dtype() == DType::kF32) {
        std::copy_n(table.data<float>() + r * d, d, yr);
        continue;
      }
      const uint32_t* row = table.data<uint32_t>() + r * words;
      const uint32_t* zrow = zmask ? zmask + r * words : nullptr;
      float s = scale ? scale[r] : 1.f;
      for (int64_t c = 0; c < d; ++c) {
        if (zrow && bit_value(zrow, c) < 0) yr[c] = 0.f;
        else yr[c] = static_cast<float>(bit_value(row, c)) * s;
      }
    }
  }
};

// ---------------------------------------------------------------- LayerNorm
class LayerNormRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override { eps_ = ctx.attrs().get_number("eps", 1e-5); }
  void run(const KernelContext& ctx) override {
    const Tensor& x = ctx.in(0);
    const int64_t d = x.dim(-1), rows = x.size() / d;
    const float* g = ctx.has_in(1) ? ctx.in(1).data<float>() : nullptr;
    const float* b = ctx.has_in(2) ? ctx.in(2).data<float>() : nullptr;
    const float* xp = x.data<float>();
    float* y = ctx.out(0).data<float>();
    const double eps = eps_;
    ctx.pool->parallel_for(rows, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float* xr = xp + r * d;
        double s = 0.0;
        for (int64_t i = 0; i < d; ++i) s += static_cast<double>(xr[i]);
        const double mean = s / static_cast<double>(d);
        double v = 0.0;
        for (int64_t i = 0; i < d; ++i) {
          double t = static_cast<double>(xr[i]) - mean;
          v += t * t;
        }
        const double sd = std::sqrt(v / static_cast<double>(d) + eps);
        for (int64_t i = 0; i < d; ++i) {
          double t = (static_cast<double>(xr[i]) - mean) / sd;
          if (g) t = t * static_cast<double>(g[i]);
          if (b) t = t + static_cast<double>(b[i]);
          y[r * d + i] = static_cast<float>(t);
        }
      }
    });
  }

 private:
  double eps_ = 1e-5;
};

// ------------------------------------------------------------------- MatMul
class MatMulRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    scale_ = ctx.attrs().get_number("scale", 1.0);
  }
  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    const double scale = scale_;
    ctx.pool->parallel_for(g.batch * g.m, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        int64_t bt = r / g.m;
        const float* ar = a + r * g.k;
        const float* bb = b + bt * g.k * g.n;
        for (int64_t j = 0; j < g.n; ++j) {
          double acc = 0.0;
          for (int64_t k = 0; k < g.k; ++k) {
            float bv = g.transpose_b ? bb[j * g.k + k] : bb[k * g.n + j];
            acc += static_cast<double>(ar[k]) * static_cast<double>(bv);
          }
          y[r * g.n + j] = static_cast<float>(acc * scale);
        }
      }
    });
  }

 private:
  MatMulGeom g_;
  double scale_ = 1.0;
};

// ------------------------------------------------------------------ BMatMul
// Binarizes a to int8 [batch, M, K] and b to int8 [batch, N, K] (transposed
// to K-contiguous rows), then exact int32 dot products.
class BMatMulRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    ma_ = parse_bmode(ctx.attrs().get_string("a_binarize", "sign_zero"));
    mb_ = parse_bmode(ctx.attrs().get_string("b_binarize", "sign_zero"));
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    aq_.resize(static_cast<size_t>(g_.batch * g_.m * g_.k));
    bq_.resize(static_cast<size_t>(g_.batch * g_.n * g_.k));
  }
  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    for (int64_t i = 0; i < g.batch * g.m * g.k; ++i) aq_[i] = static_cast<int8_t>(bmode_apply(ma_, a[i]));
    for (int64_t bt = 0; bt < g.batch; ++bt)
      for (int64_t j = 0; j < g.n; ++j)
        for (int64_t k = 0; k < g.k; ++k) {
          float v = g.transpose_b ? b[(bt * g.n + j) * g.k + k] : b[(bt * g.k + k) * g.n + j];
          bq_[(bt * g.n + j) * g.k + k] = static_cast<int8_t>(bmode_apply(mb_, v));
        }
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(g.batch * g.m, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        int64_t bt = r / g.m;
        const int8_t* ar = &aq_[r * g.k];
        for (int64_t j = 0; j < g.n; ++j) {
          const int8_t* br = &bq_[(bt * g.n + j) * g.k];
          int32_t acc = 0;
          for (int64_t k = 0; k < g.k; ++k) acc += ar[k] * br[k];
          y[r * g.n + j] = static_cast<float>(acc) * scale_;
        }
      }
    });
  }

 private:
  MatMulGeom g_;
  BMode ma_ = BMode::kSignZero, mb_ = BMode::kSignZero;
  float scale_ = 1.f;
  std::vector<int8_t> aq_, bq_;
};

// ------------------------------------------------------------- AddBroadcast
class AddBroadcastRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    const Shape& as = ctx.in(0).shape();
    const Shape& bs = ctx.in(1).shape();
    size_t r = as.size(), off = r - bs.size();
    dims_ = as;
    bstride_.assign(r, 0);
    int64_t st = 1;
    for (size_t d = r; d-- > 0;) {
      int64_t bd = d >= off ? bs[d - off] : 1;
      bstride_[d] = bd == 1 ? 0 : st;
      st *= bd;
    }
    idx_.assign(r, 0);
  }
  void run(const KernelContext& ctx) override {
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.in(0).size();
    const size_t r = dims_.size();
    std::vector<int64_t>& idx = idx_;
    std::fill(idx.begin(), idx.end(), 0);
    int64_t boff = 0;
    for (int64_t i = 0; i < n; ++i) {
      y[i] = a[i] + b[boff];
      for (size_t d = r; d-- > 0;) {  // increment the multi-index, keep boff in sync
        if (++idx[d] < dims_[d]) {
          boff += bstride_[d];
          break;
        }
        boff -= bstride_[d] * (dims_[d] - 1);
        idx[d] = 0;
      }
    }
  }

 private:
  Shape dims_;
  std::vector<int64_t> bstride_, idx_;
};

// ---------------------------------------------------------------- Transpose
class TransposeRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    perm_ = ctx.attrs().get_ints("perm", {});
    const Shape& xs = ctx.in(0).shape();
    size_t r = xs.size();
    std::vector<int64_t> xstride(r, 1);
    for (size_t d = r - 1; d-- > 0;) xstride[d] = xstride[d + 1] * xs[d + 1];
    out_dims_.resize(r);
    src_stride_.resize(r);
    for (size_t i = 0; i < r; ++i) {
      out_dims_[i] = xs[perm_[i]];
      src_stride_[i] = xstride[perm_[i]];
    }
    idx_.assign(r, 0);
  }
  void run(const KernelContext& ctx) override {
    const float* x = ctx.in(0).data<float>();
    float* y = ctx.out(0).data<float>();
    const int64_t n = ctx.in(0).size();
    const size_t r = out_dims_.size();
    std::vector<int64_t>& idx = idx_;
    std::fill(idx.begin(), idx.end(), 0);
    int64_t src = 0;
    for (int64_t i = 0; i < n; ++i) {
      y[i] = x[src];
      for (size_t d = r; d-- > 0;) {
        if (++idx[d] < out_dims_[d]) {
          src += src_stride_[d];
          break;
        }
        src -= src_stride_[d] * (out_dims_[d] - 1);
        idx[d] = 0;
      }
    }
  }

 private:
  std::vector<int64_t> perm_;
  Shape out_dims_;
  std::vector<int64_t> src_stride_, idx_;
};

}  // namespace

void register_transformer_reference_kernels(OpRegistry& reg) {
  auto add = [&](const char* op, const char* name, std::unique_ptr<Kernel> (*create)()) {
    reg.add_kernel({op, name, "reference", 0, 0, nullptr, create});
  };
  add("Gather", "gather.reference", make_kernel<GatherRef>);
  add("LayerNorm", "layernorm.reference", make_kernel<LayerNormRef>);
  add("MatMul", "matmul.reference", make_kernel<MatMulRef>);
  add("BMatMul", "bmatmul.reference", make_kernel<BMatMulRef>);
  add("AddBroadcast", "add_broadcast.reference", make_kernel<AddBroadcastRef>);
  add("Transpose", "transpose.reference", make_kernel<TransposeRef>);
}

}  // namespace bitino

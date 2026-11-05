// INT8 MatMul (activation x activation), opt-in tiers like the INT8
// Conv2D/Dense kernels (docs/int8.md § MatMul):
//   matmul.int8_reference      tier int8_reference, portable oracle (priority 30)
//   matmul.int8_neon_sdot/i8mm tier int8_neon (dotprod / +i8mm, 140 / 150)
//   matmul.int8_avx2[_vnni]    tier int8_avx2 (avx2 / +avxvnni, 140 / 150)
//
// Contract (every kernel bit-exact with int8_reference, kernels/int8_quant.h):
//   sa = int8_scale(max |a|) over the whole a tensor, qa = clamp(rne(a / sa), -127, 127)
//   sb = int8_scale(max |b|) over the whole b tensor, qb likewise
//   acc[bt][i][j] = sum_k qa[bt][i][k] * qb[bt][k][j]      exact int32 (K <= kInt8MaxK)
//   c = (sa * sb) * scale                                   fp32, in that order
//   y = float(acc) * c + 0.0f                               fp32, one rounding each, no FMA
// (the INT8 Conv2D/Dense epilogue with a zero bias). Only the float MatMul
// op is quantized; BMatMul stays binary.
//
// The NEON and AVX2 kernels reuse the Conv2D/Dense GEMM micro-kernels: per
// batch entry, A = quantized rows of a, W = quantized columns of b packed into
// the micro-kernel layout in thread scratch (padding zeroed once, never
// written), then one GEMM over the entry. Work items are batch entries.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "kernels/int8_quant.h"
#include "ops/op_defs.h"
#include "ops/transformer_common.h"

#if defined(__x86_64__)
#include "kernels/int8_avx2/int8_avx2.h"
#endif
#if defined(__aarch64__)
#include <arm_neon.h>

#include "kernels/int8_neon/int8_gemm.h"
#endif

namespace bitino {
namespace {

bool matmul_int8_supported(const KernelContext& ctx) {
  if (!int8_allowed(ctx)) return false;
  if (ctx.in(0).dtype() != DType::kF32 || ctx.in(1).dtype() != DType::kF32) return false;
  return ctx.in(0).dim(-1) > 0 && ctx.in(0).dim(-1) <= kInt8MaxK;
}

float combined_scale(float sa, float sb, float scale) { return (sa * sb) * scale; }

// Per-thread scratch slots (one per pool thread; acquire never blocks for long).
class Slots {
 public:
  void init(int n, size_t bytes) {
    n_ = std::max(n, 1);
    stride_ = (bytes + 63) / 64 * 64;
    buf_ = aligned_alloc_shared(stride_ * n_);
    busy_.reset(new std::atomic<bool>[n_]);
    for (int i = 0; i < n_; ++i) busy_[i].store(false);
  }
  int acquire() {
    for (;;)
      for (int i = 0; i < n_; ++i) {
        bool e = false;
        if (busy_[i].compare_exchange_strong(e, true, std::memory_order_acquire)) return i;
      }
  }
  void release(int i) { busy_[i].store(false, std::memory_order_release); }
  uint8_t* get(int i) const { return static_cast<uint8_t*>(buf_.get()) + stride_ * i; }

 private:
  int n_ = 1;
  size_t stride_ = 0;
  std::shared_ptr<void> buf_;
  std::unique_ptr<std::atomic<bool>[]> busy_;
};

// ------------------------------------------------------------- reference
class Int8MatMulRef : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    qa_.resize(static_cast<size_t>(ctx.in(0).size()));
    qb_.resize(static_cast<size_t>(ctx.in(1).size()));
  }
  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float sa = quantize(ctx.in(0).data<float>(), ctx.in(0).size(), qa_.data());
    const float sb = quantize(ctx.in(1).data<float>(), ctx.in(1).size(), qb_.data());
    const float c = combined_scale(sa, sb, scale_);
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(g.batch * g.m, 1, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t bt = r / g.m;
        const int8_t* ar = &qa_[r * g.k];
        const int8_t* bb = &qb_[bt * g.k * g.n];
        for (int64_t j = 0; j < g.n; ++j) {
          int32_t acc = 0;
          for (int64_t k = 0; k < g.k; ++k) acc += ar[k] * (g.transpose_b ? bb[j * g.k + k] : bb[k * g.n + j]);
          float v = static_cast<float>(acc) * c;
          y[r * g.n + j] = v + 0.f;
        }
      }
    });
  }

 private:
  static float quantize(const float* x, int64_t n, int8_t* q) {
    float m = 0.f;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    const float s = int8_scale(m);
    for (int64_t i = 0; i < n; ++i) q[i] = int8_quantize(x[i], s);
    return s;
  }
  MatMulGeom g_;
  float scale_ = 1.f;
  std::vector<int8_t> qa_, qb_;
};

#if defined(__x86_64__)
// ------------------------------------------------------------------ AVX2
// A rows: Variant encoding (u8 = q + 128 for VNNI, int16 for MADD), quantized
// in one pass over the whole a tensor (rows back to back, see lda_). b is
// quantized to int16 once per run (exact int8 values), then per batch entry
// packed into W panels [ceil(N/16)][kc][16][4 bytes] (+ VNNI correction).
template <bool kVnni>
class Int8MatMulAvx2 : public Kernel {
 public:
  using T = typename std::conditional<kVnni, uint8_t, int16_t>::type;
  static constexpr int64_t E = kVnni ? 4 : 2;  // k per 4-byte chunk

  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    kc_ = (g_.k + E - 1) / E;
    // A rows are packed back to back (lda = K elements): the micro-kernel reads
    // 4*kc bytes per row, i.e. up to 3 (VNNI) / 1 (MADD) elements of the next
    // row as K padding, which only ever meet zero weights (W's K padding is 0).
    lda_ = g_.k * static_cast<int64_t>(sizeof(T));
    panels_ = (g_.n + int8_avx2::kNR - 1) / int8_avx2::kNR;
    abuf_ = aligned_alloc_shared(static_cast<size_t>(g_.batch * g_.m * lda_ + 64));
    qb_.reset(new int16_t[static_cast<size_t>(ctx.in(1).size()) + 16]);
    const size_t wbytes = static_cast<size_t>(panels_ * kc_ * 64);
    const size_t cbytes = static_cast<size_t>(panels_ * int8_avx2::kNR * 4);
    slots_.init(ctx.pool->num_threads(), wbytes + cbytes + static_cast<size_t>(g_.n) * sizeof(int64_t));
    cbytes_ = cbytes;
    wbytes_ = wbytes;
    scale_v_.assign(static_cast<size_t>(panels_ * int8_avx2::kNR), 0.f);
    bias_v_.assign(static_cast<size_t>(panels_ * int8_avx2::kNR), 0.f);
    const int64_t macs = g_.batch * g_.m * g_.n * g_.k;
    grain_ = ctx.pool->num_threads() == 1 || macs < (int64_t(1) << 20) ? std::max<int64_t>(g_.batch, 1) : 1;
  }

  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    const int64_t na = ctx.in(0).size(), nb = ctx.in(1).size();
    const float sa = int8_scale(int8_avx2::maxabs(a, na));
    const float sb = int8_scale(int8_avx2::maxabs(b, nb));
    int8_avx2::quantize_s16(b, nb, sb, qb_.get());
    const float c = combined_scale(sa, sb, scale_);
    for (int64_t j = 0; j < g.n; ++j) scale_v_[j] = c;
    uint8_t* abuf = static_cast<uint8_t*>(abuf_.get());
    if (kVnni) int8_avx2::quantize_u8(a, na, sa, abuf);  // whole tensor at once (rows back to back)
    else int8_avx2::quantize_s16(a, na, sa, reinterpret_cast<int16_t*>(abuf));
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(g.batch, grain_, [&](int64_t b0, int64_t b1) {
      const int s = slots_.acquire();
      uint8_t* w = slots_.get(s);
      int32_t* corr = reinterpret_cast<int32_t*>(w + wbytes_);
      int64_t* colsum = reinterpret_cast<int64_t*>(w + wbytes_ + cbytes_);
      for (int64_t bt = b0; bt < b1; ++bt) {
        pack_w(qb_.get() + bt * g.k * g.n, w, corr, colsum);
        int8_avx2::GemmArgs ga;
        ga.a = abuf + bt * g.m * lda_;
        ga.lda = lda_;
        ga.w = reinterpret_cast<const int8_t*>(w);
        ga.corr = kVnni ? corr : nullptr;
        ga.scale = scale_v_.data();
        ga.bias = bias_v_.data();
        ga.y = y + bt * g.m * g.n;
        ga.cols = g.n;
        ga.kc = kc_;
        ga.act = 0;
        if (kVnni) int8_avx2::gemm_vnni(ga, 0, g.m, 0, panels_);
        else int8_avx2::gemm_madd(ga, 0, g.m, 0, panels_);
      }
      slots_.release(s);
    });
  }

 private:
  // Column j of b (qb[k][j], or qb[j][k] with transpose_b) -> panel j / 16,
  // lane j % 16; corr[j] = 128 * sum_k qb (mod 2^32, VNNI). The non-transposed
  // case uses the AVX2 transposing packer.
  void pack_w(const int16_t* qb, uint8_t* w, int32_t* corr, int64_t* sum) const {
    const MatMulGeom& g = g_;
    if (!g.transpose_b) {  // vector 4x16 / 2x16 transposes (int8_gemm_avx2.cc)
      int8_avx2::pack_w_kn(qb, g.k, g.n, g.n, kc_, kVnni, w, corr);
      return;
    }
    // transposed b: column j is row j of qb (contiguous in k)
    (void)sum;
    for (int64_t j = 0; j < g.n; ++j) {
      uint8_t* panel = w + (j / int8_avx2::kNR) * kc_ * 64 + (j % int8_avx2::kNR) * 4;
      int64_t sum = 0;
      for (int64_t k = 0; k < g.k; ++k) {
        const int16_t q = qb[j * g.k + k];
        const int64_t ch = k / E, t = k % E;
        if (kVnni) panel[ch * 64 + t] = static_cast<uint8_t>(static_cast<int8_t>(q));
        else std::memcpy(panel + ch * 64 + t * 2, &q, 2);
        sum += q;
      }
      corr[j] = static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(128 * sum)));
    }
  }

  MatMulGeom g_;
  float scale_ = 1.f;
  int64_t kc_ = 0, lda_ = 0, panels_ = 0, grain_ = 1;
  size_t wbytes_ = 0, cbytes_ = 0;
  std::shared_ptr<void> abuf_;
  std::unique_ptr<int16_t[]> qb_;
  Slots slots_;
  std::vector<float> scale_v_, bias_v_;
};
#endif

#if defined(__aarch64__)
// dst[j * k + kk] = src[kk * n + j] (int8), 8x8 blocks with VTRN.
void transpose_s8(const int8_t* src, int64_t k, int64_t n, int8_t* dst) {
  int64_t k0 = 0;
  for (; k0 + 8 <= k; k0 += 8) {
    int64_t j0 = 0;
    for (; j0 + 8 <= n; j0 += 8) {
      int8x8_t r[8];
      for (int i = 0; i < 8; ++i) r[i] = vld1_s8(src + (k0 + i) * n + j0);
      const int8x8x2_t t01 = vtrn_s8(r[0], r[1]), t23 = vtrn_s8(r[2], r[3]), t45 = vtrn_s8(r[4], r[5]), t67 = vtrn_s8(r[6], r[7]);
      const int16x4x2_t u02 = vtrn_s16(vreinterpret_s16_s8(t01.val[0]), vreinterpret_s16_s8(t23.val[0]));
      const int16x4x2_t u13 = vtrn_s16(vreinterpret_s16_s8(t01.val[1]), vreinterpret_s16_s8(t23.val[1]));
      const int16x4x2_t u46 = vtrn_s16(vreinterpret_s16_s8(t45.val[0]), vreinterpret_s16_s8(t67.val[0]));
      const int16x4x2_t u57 = vtrn_s16(vreinterpret_s16_s8(t45.val[1]), vreinterpret_s16_s8(t67.val[1]));
      const int32x2x2_t v04 = vtrn_s32(vreinterpret_s32_s16(u02.val[0]), vreinterpret_s32_s16(u46.val[0]));
      const int32x2x2_t v26 = vtrn_s32(vreinterpret_s32_s16(u02.val[1]), vreinterpret_s32_s16(u46.val[1]));
      const int32x2x2_t v15 = vtrn_s32(vreinterpret_s32_s16(u13.val[0]), vreinterpret_s32_s16(u57.val[0]));
      const int32x2x2_t v37 = vtrn_s32(vreinterpret_s32_s16(u13.val[1]), vreinterpret_s32_s16(u57.val[1]));
      const int32x2_t col[8] = {v04.val[0], v15.val[0], v26.val[0], v37.val[0], v04.val[1], v15.val[1], v26.val[1], v37.val[1]};
      for (int j = 0; j < 8; ++j) vst1_s8(dst + (j0 + j) * k + k0, vreinterpret_s8_s32(col[j]));
    }
    for (; j0 < n; ++j0)
      for (int i = 0; i < 8; ++i) dst[j0 * k + k0 + i] = src[(k0 + i) * n + j0];
  }
  for (; k0 < k; ++k0)
    for (int64_t j = 0; j < n; ++j) dst[j * k + k0] = src[k0 * n + j];
}

// ------------------------------------------------------------------ NEON
// Both operands quantized per tensor (int8_neon::quantize_tensor), then per
// batch entry packed into the tiled layouts of int8_gemm.h:
//   A [ceil(M/8)][kc][8][C] from the rows of qa, W [ceil(N/8)][kc][8][C] from
//   the columns of qb (rows of qb^T; non-transposed b is transposed first).
template <int kChunk>
class Int8MatMulNeon : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    g_ = matmul_geometry(ctx.attrs(), ctx.in(0).shape(), ctx.in(1).shape());
    scale_ = static_cast<float>(ctx.attrs().get_number("scale", 1.0));
    kc_ = (g_.k + kChunk - 1) / kChunk;
    pb_ = (g_.m + int8_neon::kTile - 1) / int8_neon::kTile;
    ob_ = (g_.n + int8_neon::kTile - 1) / int8_neon::kTile;
    na_ = ctx.in(0).size();
    nb_ = ctx.in(1).size();
    qa_.reset(new int8_t[static_cast<size_t>(na_)]);
    qb_.reset(new int8_t[static_cast<size_t>(nb_)]);
    pa_ = int8_neon::quant_parts(na_);
    pbp_ = int8_neon::quant_parts(nb_);
    partial_.assign(static_cast<size_t>(std::max(pa_, pbp_)), 0.f);
    abytes_ = static_cast<size_t>(pb_ * kc_ * int8_neon::kTile * kChunk);
    wbytes_ = static_cast<size_t>(ob_ * kc_ * int8_neon::kTile * kChunk);
    tbytes_ = g_.transpose_b ? 0 : static_cast<size_t>(g_.n * g_.k);
    slots_.init(ctx.pool->num_threads(), abytes_ + wbytes_ + tbytes_);
    scale_v_.assign(static_cast<size_t>(ob_ * int8_neon::kTile), 0.f);
    bias_v_.assign(static_cast<size_t>(ob_ * int8_neon::kTile), 0.f);
    const int64_t macs = g_.batch * g_.m * g_.n * g_.k;
    grain_ = ctx.pool->num_threads() == 1 || macs < (int64_t(1) << 20) ? std::max<int64_t>(g_.batch, 1) : 1;
  }

  void run(const KernelContext& ctx) override {
    const MatMulGeom g = g_;
    const float sa = int8_neon::quantize_tensor(ctx.in(0).data<float>(), na_, qa_.get(), partial_.data(), pa_, *ctx.pool);
    const float sb = int8_neon::quantize_tensor(ctx.in(1).data<float>(), nb_, qb_.get(), partial_.data(), pbp_, *ctx.pool);
    const float c = combined_scale(sa, sb, scale_);
    for (int64_t j = 0; j < g.n; ++j) scale_v_[j] = c;
    float* y = ctx.out(0).data<float>();
    auto tiles = kChunk == 8 ? int8_neon::gemm_tiles_i8mm : int8_neon::gemm_tiles_sdot;
    ctx.pool->parallel_for(g.batch, grain_, [&](int64_t b0, int64_t b1) {
      const int s = slots_.acquire();
      int8_t* ap = reinterpret_cast<int8_t*>(slots_.get(s));
      int8_t* wp = ap + abytes_;
      int8_t* tp = wp + wbytes_;
      for (int64_t bt = b0; bt < b1; ++bt) {
        pack_rows(qa_.get() + bt * g.m * g.k, g.m, ap);
        const int8_t* qb = qb_.get() + bt * g.k * g.n;
        if (!g.transpose_b) {  // [K][N] -> [N][K]
          transpose_s8(qb, g.k, g.n, tp);
          qb = tp;
        }
        pack_rows(qb, g.n, wp);
        int8_neon::GemmArgs ga;
        ga.a = ap;
        ga.w = wp;
        ga.scale = scale_v_.data();
        ga.bias = bias_v_.data();
        ga.y = y + bt * g.m * g.n;
        ga.rows = g.m;
        ga.cols = g.n;
        ga.kc = kc_;
        ga.act = 0;
        for (int64_t t = 0; t < pb_; ++t) tiles(ga, t, 0, ob_);
      }
      slots_.release(s);
    });
  }

 private:
  // rows x K int8 (row-major) -> [ceil(rows/8)][kc][8][C]; padding stays 0.
  void pack_rows(const int8_t* src, int64_t rows, int8_t* dst) const {
    const int64_t k = g_.k, full = k / kChunk;
    for (int64_t r = 0; r < rows; ++r) {
      int8_t* tile = dst + (r / int8_neon::kTile) * kc_ * int8_neon::kTile * kChunk + (r % int8_neon::kTile) * kChunk;
      const int8_t* s = src + r * k;
      for (int64_t c = 0; c < full; ++c) std::memcpy(tile + c * int8_neon::kTile * kChunk, s + c * kChunk, kChunk);
      if (full < kc_) std::memcpy(tile + full * int8_neon::kTile * kChunk, s + full * kChunk, static_cast<size_t>(k - full * kChunk));
    }
  }

  MatMulGeom g_;
  float scale_ = 1.f;
  int64_t kc_ = 0, pb_ = 0, ob_ = 0, na_ = 0, nb_ = 0, pa_ = 1, pbp_ = 1, grain_ = 1;
  size_t abytes_ = 0, wbytes_ = 0, tbytes_ = 0;
  std::unique_ptr<int8_t[]> qa_, qb_;
  std::vector<float> partial_;
  Slots slots_;
  std::vector<float> scale_v_, bias_v_;
};
#endif

}  // namespace

void register_int8_matmul_kernels(OpRegistry& reg) {
  reg.add_kernel({"MatMul", "matmul.int8_reference", "int8_reference", 30, 0, matmul_int8_supported, make_kernel<Int8MatMulRef>});
#if defined(__x86_64__)
  reg.add_kernel({"MatMul", "matmul.int8_avx2", "int8_avx2", 140, kFeatAvx2, matmul_int8_supported, make_kernel<Int8MatMulAvx2<false>>});
  reg.add_kernel({"MatMul", "matmul.int8_avx2_vnni", "int8_avx2", 150, kFeatAvx2 | kFeatAvxVnni, matmul_int8_supported,
                  make_kernel<Int8MatMulAvx2<true>>});
#endif
#if defined(__aarch64__)
  const uint32_t sdot = kFeatNeon | kFeatDotProd, i8mm = sdot | kFeatI8mm;
  reg.add_kernel({"MatMul", "matmul.int8_neon_sdot", "int8_neon", 140, sdot, matmul_int8_supported, make_kernel<Int8MatMulNeon<4>>});
  reg.add_kernel({"MatMul", "matmul.int8_neon_i8mm", "int8_neon", 150, i8mm, matmul_int8_supported, make_kernel<Int8MatMulNeon<8>>});
#endif
}

}  // namespace bitino

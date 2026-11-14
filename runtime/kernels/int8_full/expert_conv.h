// Expert binary convolution (EBN, Bulat et al., arXiv 2010.03558; the research notes (not published), 2026-11-14):
// I8BConv2DResIX = I8BConv2DResI with E weight sets. in(1) holds the E sets stacked along the output channels
// ([E * O, KH, KW, I] bits), in(16) the per-sample expert index (i32 [N], from I8ArgMax of a binary gate), attribute
// "experts" E. Every other input is the base op's (O-sized constants shared by the experts). Semantics
// (int8_ops._bconv_res_ix): the base op with the weight rows [e * O, (e + 1) * O) of the sample's expert e.
// Kernels: E instances of the base op's kernel, each prepared with its weight set (E x the prepacked binary weights);
// a run reads the expert index and runs that instance: one convolution's time plus a pointer select. Samples of one
// batch must share the expert (batch 1 in practice); otherwise the run is refused.
#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "bitino/op.h"
#include "bitino/tensor.h"

namespace bitino {
namespace int8_full {

template <class Base>
class ExpertConv : public Kernel {
 public:
  void prepare(const KernelContext& ctx) override {
    e_ = ctx.attrs().get_int("experts", 1);
    BF_CHECK(e_ >= 1, "I8BConv2DResIX: experts must be >= 1");
    base_node_ = *ctx.node;
    base_node_.op = "I8BConv2DResI";
    const Tensor& w = ctx.in(1);
    BF_CHECK(w.dtype() == DType::kBit && w.dim(0) % e_ == 0, "I8BConv2DResIX: weights must be bit [E * O, KH, KW, I]");
    Shape s = w.shape();
    s[0] /= e_;
    const size_t bytes = w.bytes() / static_cast<size_t>(e_);
    BF_CHECK(bytes * static_cast<size_t>(e_) == w.bytes(), "I8BConv2DResIX: weight sets are not whole rows");
    subs_.resize(static_cast<size_t>(e_));
    for (int64_t k = 0; k < e_; ++k) {
      Sub& sb = subs_[static_cast<size_t>(k)];
      sb.w = Tensor::allocate(DType::kBit, s);
      Tensor& wt = sb.w;
      std::memcpy(wt.data<uint8_t>(), w.data<uint8_t>() + static_cast<size_t>(k) * bytes, bytes);
      sb.ctx = ctx;
      sb.ctx.node = &base_node_;
      if (sb.ctx.inputs.size() > 16) sb.ctx.inputs.resize(16);
      sb.ctx.inputs[1] = &sb.w;
      sb.k = std::make_unique<Base>();
      sb.k->prepare(sb.ctx);
      const auto rel = sb.k->released_inputs();   // the base kernel prepacked the weights: drop the copy (memory: E x prepacked)
      if (std::find(rel.begin(), rel.end(), 1) != rel.end()) {
        sb.w = Tensor();
        sb.ctx.inputs[1] = nullptr;
      }
    }
  }
  void run(const KernelContext& ctx) override {
    const Tensor& ix = ctx.in(16);
    const int32_t* e = ix.data<int32_t>();
    for (int64_t i = 1; i < ix.size(); ++i) BF_CHECK(e[i] == e[0], "I8BConv2DResIX: the samples of a batch use different experts");
    BF_CHECK(e[0] >= 0 && e[0] < e_, "I8BConv2DResIX: expert index out of range");
    Sub& sb = subs_[static_cast<size_t>(e[0])];
    for (size_t i = 0; i < sb.ctx.inputs.size(); ++i)
      if (i != 1) sb.ctx.inputs[i] = ctx.inputs[i];
    for (size_t i = 0; i < ctx.outputs.size(); ++i) sb.ctx.outputs[i] = ctx.outputs[i];
    sb.k->run(sb.ctx);
  }
  size_t prepacked_bytes() const override {
    size_t b = 0;
    for (const auto& s : subs_) b += s.k->prepacked_bytes() + (s.ctx.inputs[1] ? s.w.bytes() : 0);
    return b;
  }
  std::vector<int> released_inputs() const override { return {1}; }

 private:
  struct Sub {
    Tensor w;
    KernelContext ctx;
    std::unique_ptr<Kernel> k;
  };
  int64_t e_ = 1;
  Node base_node_;
  std::vector<Sub> subs_;
};

}  // namespace int8_full
}  // namespace bitino

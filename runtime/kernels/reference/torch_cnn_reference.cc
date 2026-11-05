// Reference kernel for Mul (docs/operators_torch_cnn.md): one fp32 product per element,
// mirrored exactly by python/bitino/executor.py.
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

class MulRef : public Kernel {
 public:
  void run(const KernelContext& ctx) override {
    const float* a = ctx.in(0).data<float>();
    const float* b = ctx.in(1).data<float>();
    float* y = ctx.out(0).data<float>();
    ctx.pool->parallel_for(ctx.in(0).size(), 4096, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) y[i] = a[i] * b[i];
    });
  }
};

}  // namespace

void register_torch_cnn_kernels(OpRegistry& reg) {
  reg.add_kernel({"Mul", "mul.reference", "reference", 0, 0, nullptr, make_kernel<MulRef>});
}

}  // namespace bitino

// Registration of the float_opt kernel family (listed in runtime/ops/registry.cc).
#include "kernels/float_opt/float_opt.h"

#include "ops/op_defs.h"

namespace bitino {

void register_float_opt_kernels(OpRegistry& reg) {
  register_float_opt_gemm_kernels(reg);
  register_float_opt_depthwise_kernels(reg);
  register_float_opt_elementwise_kernels(reg);
}

}  // namespace bitino

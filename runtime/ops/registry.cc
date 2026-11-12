#include "ops/op_defs.h"

#include "bitino/error.h"

namespace bitino {

OpRegistry& OpRegistry::global() {
  static OpRegistry* reg = [] {
    auto* r = new OpRegistry();
    register_builtin_ops(*r);
    return r;
  }();
  return *reg;
}

void OpRegistry::add_op(OpDef def) {
  if (find_op(def.name)) fail("operator '", def.name, "' registered twice");
  ops_.push_back(std::move(def));
}

void OpRegistry::add_kernel(KernelDef def) {
  if (!find_op(def.op)) fail("kernel '", def.name, "' registered for unknown op '", def.op, "'");
  for (const auto& k : kernels_)
    if (k.name == def.name) fail("kernel '", def.name, "' registered twice");
  kernels_.push_back(std::move(def));
}

const OpDef* OpRegistry::find_op(const std::string& name) const {
  for (const auto& o : ops_)
    if (o.name == name) return &o;
  return nullptr;
}

std::vector<const KernelDef*> OpRegistry::kernels_for(const std::string& op) const {
  std::vector<const KernelDef*> out;
  for (const auto& k : kernels_)
    if (k.op == op) out.push_back(&k);
  return out;
}

void register_builtin_ops(OpRegistry& reg) {
  register_op_defs(reg);
  register_torch_cnn_ops(reg);
  register_larq_ops(reg);
  register_reference_kernels(reg);
  register_larq_kernels(reg);
  register_binary_scalar_kernels(reg);
  register_binary_neon_kernels(reg);
  register_binary_neon_opt_kernels(reg);
  register_binary_avx2_kernels(reg);
  register_int8_kernels(reg);
  register_int8_neon_kernels(reg);
  register_int8_avx2_kernels(reg);
  register_float_opt_kernels(reg);
  register_transformer_op_defs(reg);
  register_transformer_reference_kernels(reg);
  register_ternary_kernels(reg);
  register_bhvit_op_defs(reg);
  register_bhvit_reference_kernels(reg);
  register_torch_cnn_kernels(reg);
  register_fused_ops(reg);
  register_fused_reference_kernels(reg);
  register_float_opt_fused_kernels(reg);
  register_float_opt_transformer_kernels(reg);  // after the transformer / BHViT OpDefs
  register_bmatmul_kernels(reg);
  register_int8_matmul_kernels(reg);
  register_int8_full_op_defs(reg);
  register_int8_full_reference_kernels(reg);
  register_int8_full_neon_kernels(reg);
  register_int8_full_portable_kernels(reg);
  register_image_feature_ops(reg);
  register_image_feature_reference_kernels(reg);
  register_image_feature_opt_kernels(reg);
}

}  // namespace bitino

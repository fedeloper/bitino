// Operator and kernel extension interfaces.
//
// An OpDef defines the *semantics* of an operator: it validates attributes
// and infers output dtypes/shapes. A KernelDef is one *implementation* of an
// op. Several kernels may implement one op; the selector picks the
// highest-priority kernel whose tier is allowed by the session policy, whose
// CPU features are present, and whose supports() predicate accepts the node.
//
// To add an operator: write an OpDef + at least a "reference" tier kernel and
// list them in runtime/ops/registry.cc. See docs/architecture.md.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "bitino/cpu_features.h"
#include "bitino/model.h"
#include "bitino/thread_pool.h"

namespace bitino {

struct OpDef {
  std::string name;
  // Sets dtype/shape of `outs` from `ins` and node.attrs. Throws on invalid
  // attributes or shapes. Optional inputs are nullptr.
  void (*infer)(const Node& node, const std::vector<const TensorInfo*>& ins, std::vector<TensorInfo*>& outs);
};

// Execution context handed to kernels. Tensor pointers are stable for the
// lifetime of the session.
struct KernelContext {
  const Node* node = nullptr;
  std::vector<const Tensor*> inputs;  // nullptr for omitted optional inputs
  std::vector<Tensor*> outputs;
  ThreadPool* pool = nullptr;
  const CpuInfo* cpu = nullptr;

  const Json& attrs() const { return node->attrs; }
  const Tensor& in(size_t i) const { return *inputs[i]; }
  bool has_in(size_t i) const { return i < inputs.size() && inputs[i] != nullptr; }
  Tensor& out(size_t i) const { return *outputs[i]; }
};

class Kernel {
 public:
  virtual ~Kernel() = default;
  // Called once per session after shapes are known: prepack weights, size
  // scratch buffers. Not timed as inference.
  virtual void prepare(const KernelContext& ctx) { (void)ctx; }
  virtual void run(const KernelContext& ctx) = 0;
  // Bytes of kernel-owned prepacked constant data (for memory reports).
  virtual size_t prepacked_bytes() const { return 0; }
  // Inputs this kernel reads only in prepare() (constants it prepacked). Session drops the pages of a mapped model's
  // constant when every reader lists it (SessionOptions::release_constants); a page read again faults it back in
  // from the file, so a wrong entry costs memory, never correctness.
  virtual std::vector<int> released_inputs() const { return {}; }
};

struct KernelDef {
  std::string op;
  std::string name;  // unique, e.g. "bconv2d.neon"
  std::string tier;  // "reference", "packed_naive", "binary_scalar", "neon", "int8_reference", "int8_neon", ...
  int priority = 0;
  uint32_t required_features = 0;
  bool (*supports)(const KernelContext& ctx) = nullptr;  // nullptr = always
  std::unique_ptr<Kernel> (*create)() = nullptr;
};

class OpRegistry {
 public:
  static OpRegistry& global();  // populated by register_builtin_ops() on first use
  void add_op(OpDef def);
  void add_kernel(KernelDef def);
  const OpDef* find_op(const std::string& name) const;
  std::vector<const KernelDef*> kernels_for(const std::string& op) const;
  const std::vector<KernelDef>& all_kernels() const { return kernels_; }

 private:
  std::vector<OpDef> ops_;
  std::vector<KernelDef> kernels_;
};

// Registers every built-in op and kernel (explicit list, no static-init magic).
void register_builtin_ops(OpRegistry& reg);

template <typename K>
std::unique_ptr<Kernel> make_kernel() {
  return std::make_unique<K>();
}

}  // namespace bitino

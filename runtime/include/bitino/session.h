// Inference session: shape inference, kernel selection, memory allocation,
// and execution of a loaded Model.
//
// Ownership: a Session keeps the Model alive (shared_ptr). Input/output
// tensors are owned by the session and remain valid until it is destroyed.
// A Session must not be run concurrently from several threads; create one
// session per thread instead (the Model can be shared).
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "bitino/model.h"
#include "bitino/op.h"
#include "bitino/thread_pool.h"

namespace bitino {

struct KernelPolicy {
  // Tiers the selector may use. Empty = all exact tiers: the lossy "int8_*"
  // tiers (quantized Conv2D/Dense) are opt-in and only used when listed here
  // (or forced).
  std::set<std::string> allowed_tiers;
  // Force a kernel by name for an op (op name -> kernel name). Fails at
  // session creation if the kernel cannot run on this node/CPU.
  std::map<std::string, std::string> force;

  static KernelPolicy all() { return {}; }
  static KernelPolicy tiers(std::set<std::string> t) {
    KernelPolicy p;
    p.allowed_tiers = std::move(t);
    return p;
  }
};

struct SessionOptions {
  int num_threads = 1;
  std::vector<int> affinity;  // optional CPU ids (see ThreadPool)
  int spin_us = 0;            // thread-pool spin-wait budget before blocking (see ThreadPool)
  // Spin without a time limit between the parallel regions of one run() (ThreadPool::set_hot): no OS wake-up per
  // node; the workers block again after the run (spin_us). Only with num_threads > 1. On by default since
  // 2026-11-04 (S22: ReActNet-18 log2 at 4 threads 10.3-10.9 ms instead of 17-19 ms when the cores wake slowly);
  // off: every parallel region pays the OS wake-up of the workers (docs/threads.md).
  bool spin_run = true;
  // Drop the pages of constants that every kernel reading them prepacked (weights) when the model is file-mapped
  // (Model::LoadOptions::map_file): resident memory holds the packed copy only (Session::released_bytes()).
  bool release_constants = true;
  KernelPolicy policy;
  bool profile = false;       // per-node timing (adds clock reads between nodes)
  bool reuse_memory = true;   // liveness-based activation buffer sharing
  // Plan-time rewrites (src/graph_rewrite.h): fold bias-only ChannelAffine into
  // the threshold of the binary op it feeds, and fuse trees of elementwise ops
  // into one FusedElementwise node. Bit-exact; applied only when the float_opt
  // tier is allowed and reuse_memory is on (with reuse_memory off every
  // intermediate tensor stays inspectable).
  bool fuse = true;
  // With fuse: also move fused trees into the epilogue of the BConv2D producing
  // an operand (off: such trees stay separate FusedElementwise nodes).
  bool host_epilogues = true;
  // With fuse: a binary layer whose (epilogue-fused) float output only feeds the
  // ge / gt binarization of the next binary layer writes that binarization as
  // bit-packed output, and the next layer reads the bits (bit-exact).
  bool bit_outputs = true;
};

struct NodeProfile {
  std::string name, op, kernel;
  double total_ms = 0;
  int64_t calls = 0;
  int64_t out_elems = 0;  // elements of output 0 (per-element costs in profiles)
};

class Session {
 public:
  Session(std::shared_ptr<const Model> model, SessionOptions opts);
  ~Session();

  size_t num_inputs() const { return model_->graph().inputs.size(); }
  size_t num_outputs() const { return model_->graph().outputs.size(); }
  Tensor& input(size_t i);
  const Tensor& output(size_t i) const;
  // Debug access to any tensor by name (constants, inputs, intermediates).
  // Intermediates hold the value from the last run() only when the session
  // was created with reuse_memory = false (otherwise buffers are shared).
  const Tensor& tensor(const std::string& name) const;
  std::vector<std::string> tensor_names() const;
  const std::string& input_name(size_t i) const;
  const std::string& output_name(size_t i) const;

  void run();

  // Diagnostics.
  const std::vector<NodeProfile>& profile() const { return profile_; }
  void reset_profile();
  std::map<std::string, std::string> kernel_choices() const;  // node name -> kernel name
  double prepare_ms() const { return prepare_ms_; }
  size_t activation_bytes() const { return activation_bytes_; }
  // Plan-time rewrites applied (see SessionOptions::fuse) and Reshape nodes
  // whose output aliases the input buffer (zero-copy).
  int fused_nodes() const { return fused_nodes_; }
  int absorbed_nodes() const { return absorbed_nodes_; }
  int absorbed_luts() const { return absorbed_luts_; }  // I8Lut nodes folded into an I8BConv2DRes (out_lut)
  // int8_full rewrites (src/graph_rewrite_i8.h): composed / folded / removed I8Lut nodes, twin nodes created, nodes merged
  const std::vector<int>& i8_rewrites() const { return i8_rewrites_; }
  int folded_thresholds() const { return folded_thresholds_; }
  int hosted_epilogues() const { return hosted_epilogues_; }
  int bit_outputs() const { return bit_outputs_; }
  int shared_binarizations() const { return shared_binarizations_; }  // PackBits nodes (one packing, several BDense)
  int aliased_reshapes() const { return aliased_reshapes_; }
  size_t prepacked_bytes() const;
  // Bytes of model constants whose pages were dropped after prepacking (SessionOptions::release_constants).
  size_t released_bytes() const { return released_bytes_; }
  int num_threads() const { return pool_->num_threads(); }
  const ThreadPool& pool() const { return *pool_; }

 private:
  struct Step {
    const Node* node;
    const KernelDef* def;
    std::unique_ptr<Kernel> kernel;
    KernelContext ctx;
  };

  std::shared_ptr<const Model> model_;
  SessionOptions opts_;
  std::vector<Node> nodes_;        // the session's (possibly rewritten) copy of the graph's nodes
  std::unique_ptr<ThreadPool> pool_;
  std::vector<TensorInfo> infos_;  // per tensor id, shapes resolved
  std::vector<Tensor> values_;     // per tensor id
  std::vector<std::shared_ptr<void>> arenas_;
  std::vector<Step> steps_;
  std::vector<NodeProfile> profile_;
  double prepare_ms_ = 0;
  size_t activation_bytes_ = 0;
  size_t released_bytes_ = 0;
  int absorbed_luts_ = 0;
  std::vector<int> i8_rewrites_ = std::vector<int>(6, 0);
  int fused_nodes_ = 0, absorbed_nodes_ = 0, folded_thresholds_ = 0, aliased_reshapes_ = 0, hosted_epilogues_ = 0, bit_outputs_ = 0;
  int shared_binarizations_ = 0;
};

}  // namespace bitino

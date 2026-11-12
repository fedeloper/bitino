#include "bitino/session.h"

#include <algorithm>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "bitino/error.h"
#include "bitino/util.h"
#include "src/graph_rewrite.h"
#include "src/graph_rewrite_i8.h"

namespace bitino {
namespace {

// Lossy tiers approximate an op (int8_*: dynamic-range quantization of the
// float Conv2D/Dense, see kernels/int8_quant.h). An empty policy never picks
// them; they must be listed in allowed_tiers or forced. The int8_full_* tiers
// are not lossy: they implement the I8* operators of a quantized model exactly
// (docs/int8.md) and are the only kernels of those operators.
bool is_opt_in_tier(const std::string& tier) {
  return tier.compare(0, 5, "int8_") == 0 && tier.compare(0, 10, "int8_full_") != 0;
}

const KernelDef* select_kernel(const OpRegistry& reg, const Node& node, const KernelContext& ctx,
                               const KernelPolicy& policy, const CpuInfo& cpu) {
  auto forced = policy.force.find(node.op);
  std::vector<std::string> rejected;
  const KernelDef* best = nullptr;
  for (const KernelDef* k : reg.kernels_for(node.op)) {
    if (forced != policy.force.end() && k->name != forced->second) continue;
    bool tier_ok = forced != policy.force.end() ||
                   (policy.allowed_tiers.empty() ? !is_opt_in_tier(k->tier) : policy.allowed_tiers.count(k->tier) > 0);
    if (!tier_ok) continue;
    if (!cpu.has(k->required_features)) {
      rejected.push_back(k->name + " (missing CPU features)");
      continue;
    }
    if (k->supports && !k->supports(ctx)) {
      rejected.push_back(k->name + " (unsupported shape/attrs)");
      continue;
    }
    if (!best || k->priority > best->priority) best = k;
  }
  if (!best) {
    std::string why;
    for (const auto& r : rejected) why += " " + r + ";";
    fail("no kernel for node '", node.name, "' (op ", node.op, ")", forced != policy.force.end() ? " with forced kernel " + forced->second : "",
         " under the current policy.", why.empty() ? "" : " Rejected:" + why);
  }
  return best;
}

}  // namespace

Session::Session(std::shared_ptr<const Model> model, SessionOptions opts) : model_(std::move(model)), opts_(std::move(opts)) {
  double t0 = now_ms();
  const Graph& g = model_->graph();
  OpRegistry& reg = OpRegistry::global();
  pool_ = std::make_unique<ThreadPool>(opts_.num_threads, opts_.affinity, opts_.spin_us);

  // 1. Shape inference.
  infos_ = g.tensors;
  nodes_ = g.nodes;
  for (const Node& node : nodes_) {
    const OpDef* def = reg.find_op(node.op);
    if (!def) fail("unsupported operator '", node.op, "' in node '", node.name, "'");
    std::vector<const TensorInfo*> ins;
    for (int id : node.inputs) ins.push_back(id >= 0 ? &infos_[id] : nullptr);
    std::vector<TensorInfo*> outs;
    for (int id : node.outputs) outs.push_back(&infos_[id]);
    try {
      def->infer(node, ins, outs);
    } catch (const Error& e) {
      fail("shape inference failed for node '", node.name, "' (", node.op, "): ", e.what());
    }
  }

  // 1b. Plan-time rewrites (bit-exact; src/graph_rewrite.h) on the session's
  // copy of the nodes, only when optimized float kernels are allowed.
  const bool float_opt_allowed = opts_.policy.allowed_tiers.empty() || opts_.policy.allowed_tiers.count("float_opt") > 0;
  if (opts_.fuse && opts_.reuse_memory && float_opt_allowed) {
    RewriteOptions ro;
    ro.host_epilogues = opts_.host_epilogues;
    ro.host_bconv2d = opts_.policy.force.count("BConv2D") == 0;  // a forced kernel may lack epilogue support
    ro.host_bdense = opts_.policy.force.count("BDense") == 0;
    ro.bit_outputs = opts_.bit_outputs;
    ro.bit_bconv2d = ro.host_bconv2d;  // a forced kernel may lack bit-output / bit-input support
    ro.bit_bdense = ro.host_bdense;
    RewriteStats st = rewrite_graph(nodes_, infos_, g.outputs, ro);
    fused_nodes_ = st.fused_nodes;
    hosted_epilogues_ = st.hosted_epilogues;
    absorbed_nodes_ = st.absorbed_nodes;
    folded_thresholds_ = st.thresholds_folded;
    bit_outputs_ = st.bit_outputs;
    shared_binarizations_ = st.shared_binarizations;
    for (const Node& node : nodes_) {
      if (node.op != "FusedElementwise" && node.op != "BConv2D" && node.op != "BDense") continue;
      std::vector<const TensorInfo*> ins;
      for (int id : node.inputs) ins.push_back(id >= 0 ? &infos_[id] : nullptr);
      TensorInfo check = infos_[node.outputs[0]];
      std::vector<TensorInfo*> outs{&check};
      reg.find_op(node.op)->infer(node, ins, outs);
      BF_CHECK(check.shape == infos_[node.outputs[0]].shape && check.dtype == infos_[node.outputs[0]].dtype, "rewritten node '",
               node.name, "' changed the output shape or dtype");
    }
  }

  // 1c'. int8_full rewrites (bit-exact, independent of the tiers; src/graph_rewrite_i8.h): LUT composition and
  // threshold folding through LUTs here, twin merging after 1c.
  auto check_rewritten = [&](const char* what) {
    for (const Node& node : nodes_) {
      if (node.op != "I8LutN" && node.op != "I8AddN" && node.op != "I8Conv2DN" && node.op != "I8Lut" && node.op != "I8BConv2D" && node.op != "I8BDense") continue;
      std::vector<const TensorInfo*> ins;
      for (int id : node.inputs) ins.push_back(id >= 0 ? &infos_[id] : nullptr);
      std::vector<TensorInfo> check;
      for (int id : node.outputs) check.push_back(infos_[id]);
      std::vector<TensorInfo*> outs;
      for (auto& c : check) outs.push_back(&c);
      reg.find_op(node.op)->infer(node, ins, outs);
      for (size_t i = 0; i < check.size(); ++i)
        BF_CHECK(check[i].shape == infos_[node.outputs[i]].shape && check[i].dtype == infos_[node.outputs[i]].dtype, what, ": rewritten node '",
                 node.name, "' changed an output shape or dtype");
    }
  };
  if (opts_.fuse) {
    I8RewriteStats st = rewrite_int8_luts(nodes_, infos_, g.outputs);
    i8_rewrites_[0] += st.composed_luts + st.collapsed_ops;
    i8_rewrites_[1] += st.folded_luts;
    i8_rewrites_[2] += st.removed_luts;
    if (st.composed_luts || st.folded_luts || st.collapsed_ops) check_rewritten("int8 LUT rewrite");
  }

  // 1c. int8_full rewrite (bit-exact, independent of the tiers): an I8Lut with one table row whose input is output 0
  // of an I8BConv2DRes, read by nothing else and not a graph output, becomes that node's out_lut (input 9): the
  // kernel writes the remapped codes and the separate pass (and its parallel region) disappears.
  if (opts_.fuse) {
    std::vector<int> consumers(infos_.size(), 0);
    std::vector<int> producer(infos_.size(), -1);
    for (size_t s = 0; s < nodes_.size(); ++s) {
      for (int id : nodes_[s].inputs)
        if (id >= 0) ++consumers[id];
      for (int id : nodes_[s].outputs) producer[id] = static_cast<int>(s);
    }
    std::vector<bool> is_out(infos_.size(), false);
    for (int id : g.outputs) is_out[id] = true;
    std::vector<bool> drop(nodes_.size(), false);
    for (size_t s = 0; s < nodes_.size(); ++s) {
      const Node& l = nodes_[s];
      if (l.op != "I8Lut" || l.inputs.size() != 2 || l.inputs[0] < 0 || l.inputs[1] < 0) continue;
      const int x = l.inputs[0], t = l.inputs[1];
      const TensorInfo& ti = infos_[t];
      if (!ti.is_constant || ti.dtype != DType::kI8 || ti.shape != Shape({1, 256})) continue;
      if (consumers[x] != 1 || is_out[x] || producer[x] < 0) continue;
      Node& p = nodes_[producer[x]];
      if (p.op != "I8BConv2DRes" || p.outputs.empty() || p.outputs[0] != x || (p.inputs.size() > 9 && p.inputs[9] >= 0)) continue;
      if (infos_[l.outputs[0]].dtype != DType::kI8 || infos_[l.outputs[0]].shape != infos_[x].shape) continue;
      if (p.inputs.size() < 10) p.inputs.resize(10, -1);
      p.inputs[9] = t;
      p.outputs[0] = l.outputs[0];
      drop[s] = true;
      ++absorbed_luts_;
    }
    if (absorbed_luts_ > 0) {
      std::vector<Node> kept;
      for (size_t s = 0; s < nodes_.size(); ++s)
        if (!drop[s]) kept.push_back(std::move(nodes_[s]));
      nodes_ = std::move(kept);
    }
  }

  if (opts_.fuse) {
    I8RewriteStats st = rewrite_int8_twins(nodes_, infos_, g.outputs);
    i8_rewrites_[3] += st.lut_twins;
    i8_rewrites_[4] += st.add_twins + st.conv_twins;
    i8_rewrites_[5] += st.merged_nodes;
    if (st.lut_twins || st.add_twins || st.conv_twins) check_rewritten("int8 twin rewrite");
  }

  // 2. Activation memory: constants are views into the model; graph inputs and
  // outputs get dedicated buffers; intermediates share buffers by liveness.
  // The output of a Reshape aliases its input's buffer (zero-copy) unless it
  // is a graph output; the shared buffer lives until the last use of any of
  // its views.
  values_.resize(infos_.size());
  std::vector<int> last_use(infos_.size(), -1);
  for (size_t s = 0; s < nodes_.size(); ++s)
    for (int id : nodes_[s].inputs)
      if (id >= 0) last_use[id] = static_cast<int>(s);
  std::vector<bool> pinned(infos_.size(), false);
  for (int id : g.inputs) pinned[id] = true;
  for (int id : g.outputs) pinned[id] = true;
  std::vector<int> alias_root(infos_.size(), -1);
  for (const Node& node : nodes_) {
    if (node.op != "Reshape" || node.inputs.empty() || node.inputs[0] < 0) continue;
    const int in = node.inputs[0], out = node.outputs[0];
    if (pinned[out] || infos_[in].dtype == DType::kBit || infos_[in].dtype != infos_[out].dtype ||
        storage_bytes(infos_[in].dtype, infos_[in].shape) != storage_bytes(infos_[out].dtype, infos_[out].shape))
      continue;
    alias_root[out] = alias_root[in] >= 0 ? alias_root[in] : in;
    ++aliased_reshapes_;
  }
  std::vector<int> group_last_use = last_use;
  for (size_t id = 0; id < infos_.size(); ++id)
    if (alias_root[id] >= 0) group_last_use[alias_root[id]] = std::max(group_last_use[alias_root[id]], last_use[id]);

  for (size_t id = 0; id < infos_.size(); ++id) {
    const TensorInfo& t = infos_[id];
    if (t.is_constant) values_[id] = t.constant;
    else if (pinned[id]) {
      values_[id] = Tensor::allocate(t.dtype, t.shape);
      activation_bytes_ += storage_bytes(t.dtype, t.shape);
    }
  }

  struct Slot {
    size_t bytes = 0;
    int busy_until = -1;  // step index of the occupant's last use
    std::vector<int> tensors;
  };
  std::vector<Slot> slots;
  std::vector<int> slot_of(infos_.size(), -1);
  for (size_t s = 0; s < nodes_.size(); ++s) {
    for (int id : nodes_[s].outputs) {
      if (pinned[id] || alias_root[id] >= 0) continue;
      size_t need = storage_bytes(infos_[id].dtype, infos_[id].shape);
      int lu = std::max(group_last_use[id], static_cast<int>(s));
      int chosen = -1;
      if (opts_.reuse_memory) {
        // A slot is free if its occupant died strictly before this step, so a
        // node's outputs never alias its own inputs.
        for (size_t k = 0; k < slots.size(); ++k) {
          if (slots[k].busy_until >= static_cast<int>(s)) continue;
          if (chosen < 0) { chosen = static_cast<int>(k); continue; }
          const Slot& c = slots[chosen];
          bool fits_k = slots[k].bytes >= need, fits_c = c.bytes >= need;
          if ((fits_k && !fits_c) || (fits_k && fits_c && slots[k].bytes < c.bytes) ||
              (!fits_k && !fits_c && slots[k].bytes > c.bytes))
            chosen = static_cast<int>(k);
        }
      }
      if (chosen < 0) {
        slots.push_back(Slot{});
        chosen = static_cast<int>(slots.size()) - 1;
      }
      Slot& slot = slots[chosen];
      slot.bytes = std::max(slot.bytes, need);
      slot.busy_until = lu;
      slot.tensors.push_back(id);
      slot_of[id] = chosen;
    }
  }
  for (Slot& slot : slots) {
    auto buf = aligned_alloc_shared(slot.bytes);
    activation_bytes_ += slot.bytes;
    for (int id : slot.tensors) values_[id] = Tensor(infos_[id].dtype, infos_[id].shape, buf.get(), buf);
    arenas_.push_back(std::move(buf));
  }
  for (const Node& node : nodes_)  // topological order: a view's root is set before the view
    for (int id : node.outputs)
      if (alias_root[id] >= 0) values_[id] = values_[alias_root[id]].reshaped(infos_[id].shape);

  // 3. Kernel selection and preparation.
  const CpuInfo& cpu = cpu_info();
  steps_.reserve(nodes_.size());
  for (const Node& node : nodes_) {
    Step step;
    step.node = &node;
    step.ctx.node = &node;
    step.ctx.pool = pool_.get();
    step.ctx.cpu = &cpu;
    for (int id : node.inputs) step.ctx.inputs.push_back(id >= 0 ? &values_[id] : nullptr);
    for (int id : node.outputs) step.ctx.outputs.push_back(&values_[id]);
    step.def = select_kernel(reg, node, step.ctx, opts_.policy, cpu);
    step.kernel = step.def->create();
    step.kernel->prepare(step.ctx);
    steps_.push_back(std::move(step));
  }
  // 4. Constants whose every reader prepacked them (Kernel::released_inputs): drop their pages from a file-mapped
  // model (MADV_DONTNEED on a private file mapping: a later read faults the original bytes back in).
#if defined(__unix__) || defined(__APPLE__)
  if (opts_.release_constants && model_->mapped()) {
    std::vector<int> released(infos_.size(), 0), kept(infos_.size(), 0);
    for (const Step& st : steps_) {
      const std::vector<int> rel = st.kernel->released_inputs();
      for (size_t k = 0; k < st.node->inputs.size(); ++k) {
        const int id = st.node->inputs[k];
        if (id < 0 || !infos_[id].is_constant) continue;
        if (std::find(rel.begin(), rel.end(), static_cast<int>(k)) != rel.end()) ++released[id];
        else ++kept[id];
      }
    }
    const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    for (size_t id = 0; id < infos_.size(); ++id) {
      if (!released[id] || kept[id]) continue;
      const Tensor& t = infos_[id].constant;
      const uintptr_t b = reinterpret_cast<uintptr_t>(t.raw()), e = b + storage_bytes(t.dtype(), t.shape());
      const uintptr_t pb = (b + page - 1) / page * page, pe = e / page * page;  // whole pages inside the tensor
      if (pe > pb && madvise(reinterpret_cast<void*>(pb), pe - pb, MADV_DONTNEED) == 0) released_bytes_ += pe - pb;
    }
  }
#endif
  profile_.resize(steps_.size());
  for (size_t i = 0; i < steps_.size(); ++i) {
    profile_[i] = NodeProfile{steps_[i].node->name, steps_[i].node->op, steps_[i].def->name, 0, 0, 0};
    if (!steps_[i].node->outputs.empty() && steps_[i].node->outputs[0] >= 0)
      profile_[i].out_elems = num_elements(infos_[steps_[i].node->outputs[0]].shape);
  }
  prepare_ms_ = now_ms() - t0;
}

Session::~Session() = default;

Tensor& Session::input(size_t i) {
  BF_CHECK(i < num_inputs(), "input index ", i);
  return values_[model_->graph().inputs[i]];
}
const Tensor& Session::output(size_t i) const {
  BF_CHECK(i < num_outputs(), "output index ", i);
  return values_[model_->graph().outputs[i]];
}
const std::string& Session::input_name(size_t i) const { return infos_[model_->graph().inputs[i]].name; }
const std::string& Session::output_name(size_t i) const { return infos_[model_->graph().outputs[i]].name; }

void Session::run() {
  if (!opts_.affinity.empty()) pool_->begin_run();
  const bool hot = opts_.spin_run && pool_->num_threads() > 1;
  if (hot) pool_->set_hot(true);
  struct Cool {
    ThreadPool* p;
    bool on;
    ~Cool() {
      if (on) p->set_hot(false);
    }
  } cool{pool_.get(), hot};
  if (!opts_.profile) {
    for (Step& s : steps_) s.kernel->run(s.ctx);
    return;
  }
  for (size_t i = 0; i < steps_.size(); ++i) {
    double t0 = now_ms();
    steps_[i].kernel->run(steps_[i].ctx);
    profile_[i].total_ms += now_ms() - t0;
    profile_[i].calls += 1;
  }
}

void Session::reset_profile() {
  for (auto& p : profile_) {
    p.total_ms = 0;
    p.calls = 0;
  }
}

std::map<std::string, std::string> Session::kernel_choices() const {
  std::map<std::string, std::string> out;
  for (const Step& s : steps_) out[s.node->name] = s.def->name;
  return out;
}

const Tensor& Session::tensor(const std::string& name) const {
  int id = model_->graph().find(name);
  BF_CHECK(id >= 0, "unknown tensor '", name, "'");
  return values_[id];
}

std::vector<std::string> Session::tensor_names() const {
  std::vector<std::string> out;
  // the graph's tensors only (plan-time rewrites may add internal ones, e.g. PackBits outputs)
  for (size_t i = 0; i < model_->graph().tensors.size(); ++i) out.push_back(infos_[i].name);
  return out;
}

size_t Session::prepacked_bytes() const {
  size_t n = 0;
  for (const Step& s : steps_) n += s.kernel->prepacked_bytes();
  return n;
}

}  // namespace bitino

#include "src/graph_rewrite.h"

#include <algorithm>
#include <string>

#include "bitino/error.h"
#include "kernels/float_opt/eltwise_program.h"
#include "ops/fused_elementwise.h"

namespace bitino {
namespace {

constexpr int kMaxFusedDepth = 8;    // stack slots of a fused program
constexpr int kMaxFusedInputs = 24;  // distinct input tensors of a fused node

struct Uses {
  std::vector<int> count;     // number of (node, input slot) uses + graph outputs
  std::vector<int> producer;  // node index, -1 for constants / graph inputs
  std::vector<bool> is_output;
};

Uses compute_uses(const std::vector<Node>& nodes, size_t ntensors, const std::vector<int>& outputs) {
  Uses u;
  u.count.assign(ntensors, 0);
  u.producer.assign(ntensors, -1);
  u.is_output.assign(ntensors, false);
  for (size_t n = 0; n < nodes.size(); ++n) {
    for (int id : nodes[n].inputs)
      if (id >= 0) ++u.count[id];
    for (int id : nodes[n].outputs) u.producer[id] = static_cast<int>(n);
  }
  for (int id : outputs) {
    ++u.count[id];
    u.is_output[id] = true;
  }
  return u;
}

int in_at(const Node& n, size_t i) { return i < n.inputs.size() ? n.inputs[i] : -1; }

// ------------------------------------------------------- threshold folding
bool binarizes_f32_input(const Node& c, int t) {
  if (c.op != "BConv2D" && c.op != "BDense") return false;
  if (in_at(c, 0) != t || in_at(c, 4) >= 0) return false;
  for (size_t i = 1; i < c.inputs.size(); ++i)
    if (c.inputs[i] == t) return false;
  const std::string m = c.attrs.get_string("input_binarize", "ge");
  return m == "ge" || m == "gt" || m == "sign_zero";
}

int fold_thresholds(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs) {
  int folded = 0;
  Uses u = compute_uses(nodes, infos.size(), outputs);
  std::vector<bool> removed(nodes.size(), false);
  for (size_t n = 0; n < nodes.size(); ++n) {
    Node& ca = nodes[n];
    if (ca.op != "ChannelAffine" || in_at(ca, 1) >= 0) continue;
    const int bias = in_at(ca, 2), x = in_at(ca, 0), t = ca.outputs[0];
    if (bias < 0 || !infos[bias].is_constant || u.is_output[t] || u.count[t] == 0) continue;
    std::vector<size_t> cons;
    bool ok = true;
    for (size_t c = n + 1; c < nodes.size() && ok; ++c) {
      bool uses_t = false;
      for (int id : nodes[c].inputs) uses_t |= id == t;
      if (!uses_t) continue;
      if (removed[c] || !binarizes_f32_input(nodes[c], t)) ok = false;
      else cons.push_back(c);
    }
    if (!ok || static_cast<int>(cons.size()) != u.count[t]) continue;
    // threshold = -bias (negation is exact)
    TensorInfo thr;
    thr.name = ca.name + "/folded_threshold";
    thr.dtype = DType::kF32;
    thr.shape = infos[bias].shape;
    thr.is_constant = true;
    thr.constant = Tensor::allocate(DType::kF32, thr.shape);
    const float* b = infos[bias].constant.data<float>();
    float* d = thr.constant.data<float>();
    for (int64_t i = 0; i < num_elements(thr.shape); ++i) d[i] = -b[i];
    const int thr_id = static_cast<int>(infos.size());
    infos.push_back(std::move(thr));
    for (size_t c : cons) {
      Node& bn = nodes[c];
      if (bn.inputs.size() < 5) bn.inputs.resize(5, -1);
      bn.inputs[0] = x;
      bn.inputs[4] = thr_id;
    }
    removed[n] = true;
    ++folded;
  }
  std::vector<Node> kept;
  for (size_t n = 0; n < nodes.size(); ++n)
    if (!removed[n]) kept.push_back(std::move(nodes[n]));
  nodes.swap(kept);
  return folded;
}

// ------------------------------------------------------- elementwise fusion
// Elementwise operators the fused kernels evaluate exactly; positions of the
// operands that have the output's shape ("tensor operands").
bool fusable(const Node& n, const std::vector<TensorInfo>& infos) {
  if (n.outputs.size() != 1 || infos[n.outputs[0]].dtype != DType::kF32) return false;
  if (n.op == "ChannelAffine" || n.op == "Add" || n.op == "Mul" || n.op == "ChannelGate") return true;
  if (n.op == "Activation") {
    const std::string t = n.attrs.get_string("type", "");
    return t == "relu" || t == "relu6" || t == "hardtanh" || t == "prelu" || t == "leaky_relu" || t == "sign" || t == "sign_zero";
  }
  return false;
}

int tensor_operands(const Node& n) { return (n.op == "Add" || n.op == "Mul") ? 2 : 1; }

class Fuser {
 public:
  Fuser(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs, const RewriteOptions& opts)
      : nodes_(nodes),
        infos_(infos),
        u_(compute_uses(nodes, infos.size(), outputs)),
        cut_(infos.size(), false),
        host_bconv_(opts.host_epilogues && opts.host_bconv2d),
        host_bdense_(opts.host_epilogues && opts.host_bdense) {}

  RewriteStats run() {
    RewriteStats st;
    const size_t nn = nodes_.size();
    std::vector<bool> removed(nn, false);
    for (size_t n = nn; n-- > 0;) {  // roots first (consumers come later in topological order)
      if (removed[n] || !fusable(nodes_[n], infos_) || absorbable_into_consumer(n)) continue;
      // Build the tree; while it exceeds the limits, cut off its largest
      // proper subtree (that subtree's root is materialized and becomes a root
      // itself, visited later in this loop) and rebuild.
      bool ok = false;
      for (;;) {
        prog_.clear();
        inputs_.clear();
        absorbed_.clear();
        subtree_.clear();
        ok_ = true;
        emit(n);
        if (!ok_) break;
        int depth = kMaxFusedDepth + 1;
        try {
          depth = fe_stack_depth(prog_);
        } catch (const Error&) {
          break;
        }
        if (depth <= kMaxFusedDepth && static_cast<int>(inputs_.size()) <= kMaxFusedInputs) {
          ok = true;
          break;
        }
        size_t best = 0;
        int best_size = 0;
        for (const auto& st2 : subtree_)
          if (st2.first != n && st2.second > best_size) {
            best = st2.first;
            best_size = st2.second;
          }
        if (best_size == 0) break;
        cut_[nodes_[best].outputs[0]] = true;
      }
      if (!ok) continue;
      const int host = host_bconv_ || host_bdense_ ? find_host(removed) : -1;
      if (host >= 0) {
        host_tree(static_cast<size_t>(host), n, removed);
        st.hosted_epilogues += 1;
        st.absorbed_nodes += static_cast<int>(absorbed_.size());
        continue;
      }
      if (absorbed_.size() < 2) continue;
      Node f;
      f.name = nodes_[n].name + "/fused" + std::to_string(absorbed_.size());
      f.op = "FusedElementwise";
      f.inputs = inputs_;
      f.outputs = nodes_[n].outputs;
      Json attrs = Json::Object{};
      attrs["program"] = fe_program_json(prog_);
      Json::Array osh;
      for (int64_t d : infos_[nodes_[n].outputs[0]].shape) osh.push_back(Json(d));
      attrs["out_shape"] = Json(osh);
      Json::Array names;
      for (size_t a : absorbed_) names.push_back(Json(nodes_[a].name));
      attrs["fused_nodes"] = Json(names);
      f.attrs = attrs;
      for (size_t a : absorbed_) removed[a] = true;
      nodes_[n] = std::move(f);
      removed[n] = false;
      st.fused_nodes += 1;
      st.absorbed_nodes += static_cast<int>(absorbed_.size());
    }
    std::vector<Node> kept;
    for (size_t n = 0; n < nn; ++n)
      if (!removed[n]) kept.push_back(std::move(nodes_[n]));
    nodes_.swap(kept);
    return st;
  }

 private:
  // The output of node p feeds exactly one tensor-operand slot of a fusable
  // consumer (and nothing else), so p belongs to the consumer's tree.
  bool absorbable_into_consumer(size_t p) const {
    const int t = nodes_[p].outputs[0];
    if (u_.is_output[t] || u_.count[t] != 1 || cut_[t]) return false;
    for (size_t c = p + 1; c < nodes_.size(); ++c) {
      const Node& cn = nodes_[c];
      for (size_t i = 0; i < cn.inputs.size(); ++i) {
        if (cn.inputs[i] != t) continue;
        return fusable(cn, infos_) && static_cast<int>(i) < tensor_operands(cn);
      }
    }
    return false;
  }

  // Number of fixed inputs of a node that can host an epilogue (the program's
  // operands are appended after them), 0 if it cannot: BConv2D with f32
  // output; BDense with a binary (ge / gt) input and no w_zero (sign_zero
  // BDense goes to the ternary kernels, which have no epilogue support).
  int host_base(const Node& pn) const {
    if (pn.attrs.contains("epilogue")) return 0;
    if (pn.op == "BConv2D" && host_bconv_) {
      return pn.attrs.get_string("output", "f32") == "f32" && pn.inputs.size() <= 5 ? 5 : 0;
    }
    if (pn.op == "BDense" && host_bdense_) {
      const std::string m = pn.attrs.get_string("input_binarize", "ge");
      return (m == "ge" || m == "gt") && in_at(pn, 5) < 0 && pn.inputs.size() <= 6 ? 6 : 0;
    }
    return 0;
  }

  // A BConv2D / BDense producing one tree operand (f32 output used only by the
  // tree) that can run the tree as its epilogue: every other operand must
  // already exist when it runs (produced by an earlier node, or constant /
  // input). Returns its node index or -1 (the latest such producer wins).
  int find_host(const std::vector<bool>& removed) const {
    int best = -1;
    for (int t : inputs_) {
      const int p = u_.producer[t];
      if (p < 0 || removed[p]) continue;
      const Node& pn = nodes_[p];
      if (host_base(pn) == 0) continue;
      if (u_.count[t] != 1 || u_.is_output[t]) continue;
      bool avail = true;
      for (int o : inputs_)
        if (o != t && u_.producer[o] >= p) avail = false;
      if (avail && p > best) best = p;
    }
    return best;
  }

  // Turns BConv2D / BDense node `h` into "node + epilogue" producing the
  // tree's output; the tree's nodes (root n included) are removed.
  void host_tree(size_t h, size_t n, std::vector<bool>& removed) {
    Node hn = nodes_[h];
    const int conv_out = hn.outputs[0];
    const int base = host_base(hn);
    hn.inputs.resize(static_cast<size_t>(base), -1);
    std::vector<FeInstr> prog = prog_;
    for (FeInstr& in : prog) {
      if (in.op == FeOp::kLoad) {
        in.a = inputs_[in.a] == conv_out ? kEltwiseInPlace : in.a + base;
      } else if (in.op == FeOp::kAffine) {
        if (in.a >= 0) in.a += base;
        if (in.b >= 0) in.b += base;
      } else if (in.op == FeOp::kPrelu || in.op == FeOp::kGate || in.op == FeOp::kLoadTile) {
        in.a += base;
      } else if (in.op == FeOp::kLoadGather) {
        in.a += base;
        in.b += base;
      }
    }
    for (int t : inputs_) hn.inputs.push_back(t == conv_out ? -1 : t);
    hn.outputs = nodes_[n].outputs;
    hn.name = nodes_[h].name + "+epilogue" + std::to_string(absorbed_.size());
    Json attrs = hn.attrs;
    attrs["epilogue"] = fe_program_json(prog);
    Json::Array names;
    for (size_t a : absorbed_) names.push_back(Json(nodes_[a].name));
    attrs["epilogue_nodes"] = Json(names);
    hn.attrs = attrs;
    for (size_t a : absorbed_) removed[a] = true;
    nodes_[h] = std::move(hn);
  }

  bool single_use(int t) const { return t >= 0 && u_.count[t] == 1 && !u_.is_output[t]; }

  // Operand t = [Reshape](Gather(table f32 [V, D], ids i32)) used only by the
  // tree: the fused program gathers the rows itself (load-gather); the Gather
  // and the Reshape are absorbed, the table stays an input.
  bool emit_gather_load(int t) {
    if (!single_use(t)) return false;
    int p = u_.producer[t];
    if (p < 0) return false;
    std::vector<size_t> nodes;
    if (nodes_[p].op == "Reshape") {
      const int t2 = in_at(nodes_[p], 0);
      if (!single_use(t2) || u_.producer[t2] < 0) return false;
      nodes.push_back(static_cast<size_t>(p));
      p = u_.producer[t2];
    }
    const Node& g = nodes_[p];
    if (g.op != "Gather" || in_at(g, 2) >= 0 || in_at(g, 3) >= 0) return false;
    const int table = in_at(g, 0), ids = in_at(g, 1);
    if (table < 0 || ids < 0) return false;
    const TensorInfo& ti = infos_[table];
    if (ti.dtype != DType::kF32 || ti.shape.size() != 2 || infos_[ids].dtype != DType::kI32) return false;
    const int64_t d = ti.shape[1], c = infos_[t].shape.back();
    if (d <= 0 || c % d != 0) return false;
    nodes.push_back(static_cast<size_t>(p));
    FeInstr ld;
    ld.op = FeOp::kLoadGather;
    ld.a = input_index(table);
    ld.b = input_index(ids);
    ld.act = static_cast<int32_t>(d);
    prog_.push_back(ld);
    for (size_t q : nodes) absorbed_.push_back(q);
    return true;
  }

  // Operand t = Concat(x, x, ..., x) along the last axis, used only by the
  // tree: the fused program repeats x's rows itself (load-tile).
  bool emit_tile_load(int t) {
    if (!single_use(t)) return false;
    const int p = u_.producer[t];
    if (p < 0 || nodes_[p].op != "Concat" || nodes_[p].inputs.empty()) return false;
    const Node& cn = nodes_[p];
    const int x = cn.inputs[0];
    const int64_t rank = static_cast<int64_t>(infos_[t].shape.size());
    int64_t axis = cn.attrs.get_int("axis", -1);
    if (axis < 0) axis += rank;
    if (x < 0 || axis != rank - 1 || infos_[x].dtype != DType::kF32) return false;
    for (int i : cn.inputs)
      if (i != x) return false;
    const int64_t c0 = infos_[x].shape.back();
    if (c0 <= 0 || c0 * static_cast<int64_t>(cn.inputs.size()) != infos_[t].shape.back()) return false;
    FeInstr ld;
    ld.op = FeOp::kLoadTile;
    ld.a = input_index(x);
    ld.act = static_cast<int32_t>(c0);
    prog_.push_back(ld);
    absorbed_.push_back(static_cast<size_t>(p));
    return true;
  }

  int input_index(int tensor) {
    for (size_t i = 0; i < inputs_.size(); ++i)
      if (inputs_[i] == tensor) return static_cast<int>(i);
    inputs_.push_back(tensor);
    return static_cast<int>(inputs_.size()) - 1;
  }

  void emit(size_t n) {
    const Node& node = nodes_[n];
    const size_t first = absorbed_.size();
    absorbed_.push_back(n);
    for (int i = 0; i < tensor_operands(node); ++i) {
      const int t = in_at(node, i);
      if (t < 0) {
        ok_ = false;
        return;
      }
      const int p = u_.producer[t];
      if (p >= 0 && fusable(nodes_[p], infos_) && absorbable_into_consumer(p)) {
        emit(static_cast<size_t>(p));
      } else if (emit_gather_load(t) || emit_tile_load(t)) {
      } else {
        FeInstr ld;
        ld.op = FeOp::kLoad;
        ld.a = input_index(t);
        prog_.push_back(ld);
      }
      if (!ok_) return;
    }
    FeInstr in;
    if (node.op == "ChannelAffine") {
      in.op = FeOp::kAffine;
      in.a = in_at(node, 1) >= 0 ? input_index(node.inputs[1]) : -1;
      in.b = in_at(node, 2) >= 0 ? input_index(node.inputs[2]) : -1;
    } else if (node.op == "Add") {
      in.op = FeOp::kAdd;
      const std::string a = node.attrs.get_string("activation", "none");
      in.act = a == "relu" ? 1 : a == "relu6" ? 2 : 0;
      if (a != "none" && a != "relu" && a != "relu6") ok_ = false;
    } else if (node.op == "Mul") {
      in.op = FeOp::kMul;
    } else if (node.op == "ChannelGate") {
      in.op = FeOp::kGate;
      in.a = input_index(in_at(node, 1));
      in.b = node.attrs.get_string("gate", "sigmoid") == "sigmoid" ? 1 : 0;
    } else {  // Activation
      const std::string t = node.attrs.get_string("type", "");
      if (t == "relu") {
        in.op = FeOp::kRelu;
      } else if (t == "relu6") {
        in.op = FeOp::kClamp;
        in.lo = 0.f;
        in.hi = 6.f;
      } else if (t == "hardtanh") {
        in.op = FeOp::kClamp;
        in.lo = static_cast<float>(node.attrs.get_number("min", -1.0));
        in.hi = static_cast<float>(node.attrs.get_number("max", 1.0));
      } else if (t == "leaky_relu") {
        in.op = FeOp::kLeaky;
        in.alpha = static_cast<float>(node.attrs.get_number("alpha", 0.01));
      } else if (t == "prelu") {
        in.op = FeOp::kPrelu;
        in.a = input_index(in_at(node, 1));
      } else if (t == "sign") {
        in.op = FeOp::kSign;
      } else {
        in.op = FeOp::kSignZero;
      }
    }
    prog_.push_back(in);
    subtree_.push_back({n, static_cast<int>(absorbed_.size() - first)});
  }

  std::vector<Node>& nodes_;
  std::vector<TensorInfo>& infos_;
  Uses u_;
  std::vector<FeInstr> prog_;
  std::vector<int> inputs_;
  std::vector<size_t> absorbed_;
  std::vector<std::pair<size_t, int>> subtree_;  // (absorbed node, size of its subtree)
  std::vector<bool> cut_;                        // tensors that must stay materialized
  bool host_bconv_ = true, host_bdense_ = true;
  bool ok_ = true;
};

// ------------------------------------------------------- bit-packed outputs
// Binary binarization modes a single bit can carry (sign_zero needs three values).
bool one_bit_mode(const std::string& m) { return m == "ge" || m == "gt"; }

// Number of fixed inputs of a node that can store bit output, 0 if it cannot:
// BConv2D with f32 output; BDense with a binary (ge / gt) input, no w_zero and
// f32 output (sign_zero BDense goes to the ternary kernels, which only write
// floats). Epilogue operands, if any, follow the fixed inputs.
size_t bit_producer_base(const Node& p, const RewriteOptions& opts) {
  if (p.outputs.size() != 1 || p.attrs.get_string("output", "f32") != "f32") return 0;
  if (p.op == "BConv2D" && opts.bit_bconv2d) return 5;
  if (p.op == "BDense" && opts.bit_bdense) return one_bit_mode(p.attrs.get_string("input_binarize", "ge")) && in_at(p, 5) < 0 ? 6 : 0;
  return 0;
}

// The consumer binarizes input 0 = t with one bit and uses t nowhere else.
bool bit_consumer(const Node& c, int t, const RewriteOptions& opts) {
  if (!((c.op == "BConv2D" && opts.bit_bconv2d) || (c.op == "BDense" && opts.bit_bdense))) return false;
  if (in_at(c, 0) != t || !one_bit_mode(c.attrs.get_string("input_binarize", "ge"))) return false;
  for (size_t i = 1; i < c.inputs.size(); ++i)
    if (c.inputs[i] == t) return false;
  return true;
}

int bit_outputs(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs, const RewriteOptions& opts) {
  int count = 0;
  const Uses u = compute_uses(nodes, infos.size(), outputs);
  for (size_t p = 0; p < nodes.size(); ++p) {
    Node& pn = nodes[p];
    const size_t base = bit_producer_base(pn, opts);
    if (base == 0) continue;
    const int y = pn.outputs[0];
    if (infos[y].dtype != DType::kF32 || u.is_output[y] || u.count[y] != 1) continue;
    size_t c = p + 1;
    for (; c < nodes.size(); ++c) {
      bool uses = false;
      for (int id : nodes[c].inputs) uses |= id == y;
      if (uses) break;
    }
    if (c == nodes.size() || !bit_consumer(nodes[c], y, opts)) continue;
    Node& cn = nodes[c];
    // The consumer's threshold becomes the producer's output threshold: it must exist when the producer runs.
    const int thr = in_at(cn, 4);
    if (thr >= 0 && !infos[thr].is_constant && u.producer[thr] >= static_cast<int>(p)) continue;
    if (thr >= 0 && (infos[thr].dtype != DType::kF32 || infos[thr].shape != Shape{infos[y].shape.back()})) continue;
    Json attrs = pn.attrs;
    attrs["output"] = "bit";
    attrs["output_binarize"] = cn.attrs.get_string("input_binarize", "ge");
    if (thr >= 0) {
      if (!pn.attrs.contains("epilogue") && pn.inputs.size() < base) pn.inputs.resize(base, -1);
      attrs["output_threshold_input"] = Json(static_cast<int64_t>(pn.inputs.size()));
      pn.inputs.push_back(thr);
      cn.inputs[4] = -1;
    }
    pn.attrs = attrs;
    infos[y].dtype = DType::kBit;
    ++count;
  }
  return count;
}

}  // namespace

RewriteStats rewrite_graph(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& graph_outputs,
                           const RewriteOptions& opts) {
  RewriteStats st;
  if (opts.fold_thresholds) st.thresholds_folded = fold_thresholds(nodes, infos, graph_outputs);
  if (opts.fuse_elementwise) {
    RewriteStats f = Fuser(nodes, infos, graph_outputs, opts).run();
    st.fused_nodes = f.fused_nodes;
    st.absorbed_nodes = f.absorbed_nodes;
    st.hosted_epilogues = f.hosted_epilogues;
  }
  if (opts.bit_outputs) st.bit_outputs = bit_outputs(nodes, infos, graph_outputs, opts);
  return st;
}

}  // namespace bitino

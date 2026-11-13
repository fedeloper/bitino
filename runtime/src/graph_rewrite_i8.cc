#include "src/graph_rewrite_i8.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>

#include "bitino/error.h"
#include "kernels/int8_full/q8_math.h"

namespace bitino {
namespace {

struct UseInfo {
  std::vector<int> count;     // (node, slot) uses + graph outputs
  std::vector<int> producer;  // node index or -1
  std::vector<bool> is_output;
};

UseInfo uses_of(const std::vector<Node>& nodes, size_t ntensors, const std::vector<int>& outputs) {
  UseInfo u;
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

std::string attr_dump(const Json& a, const char* k) { return a.contains(k) ? a[k].dump() : std::string(); }

int add_constant(std::vector<TensorInfo>& infos, const std::string& name, DType dt, const Shape& shape) {
  TensorInfo t;
  t.name = name;
  t.dtype = dt;
  t.shape = shape;
  t.is_constant = true;
  t.constant = Tensor::allocate(dt, shape);
  infos.push_back(std::move(t));
  return static_cast<int>(infos.size()) - 1;
}

void drop_removed(std::vector<Node>& nodes, const std::vector<bool>& removed) {
  std::vector<Node> kept;
  kept.reserve(nodes.size());
  for (size_t n = 0; n < nodes.size(); ++n)
    if (!removed[n]) kept.push_back(std::move(nodes[n]));
  nodes.swap(kept);
}

// An I8Lut with an int8 table (rows 1 or along `axis`): its normalized axis, or -1 if not a plain int8 LUT.
struct LutInfo {
  int table = -1;
  int64_t rows = 0, axis = 0;
};
bool lut_info(const Node& n, const std::vector<TensorInfo>& infos, LutInfo* li) {
  if (n.op != "I8Lut" || n.inputs.size() != 2 || n.outputs.size() != 1 || n.inputs[0] < 0 || n.inputs[1] < 0) return false;
  const TensorInfo& t = infos[n.inputs[1]];
  const TensorInfo& x = infos[n.inputs[0]];
  if (!t.is_constant || t.shape.size() != 2 || t.shape[1] != 256 || x.dtype != DType::kI8 || x.shape.empty()) return false;
  int64_t ax = n.attrs.get_int("axis", -1);
  if (ax < 0) ax += static_cast<int64_t>(x.shape.size());
  li->table = n.inputs[1];
  li->rows = t.shape[0];
  li->axis = ax;
  return true;
}

// ------------------------------------------------------------- compose_luts
int compose_luts(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs) {
  int composed = 0;
  for (bool changed = true; changed;) {
    changed = false;
    UseInfo u = uses_of(nodes, infos.size(), outputs);
    std::vector<bool> removed(nodes.size(), false);
    for (size_t n2 = 0; n2 < nodes.size(); ++n2) {
      LutInfo l2, l1;
      if (removed[n2] || !lut_info(nodes[n2], infos, &l2)) continue;
      const int t = nodes[n2].inputs[0];
      const int p = u.producer[t];
      if (p < 0 || removed[p] || u.count[t] != 1 || u.is_output[t] || !lut_info(nodes[p], infos, &l1)) continue;
      if (infos[l1.table].dtype != DType::kI8) continue;  // the inner table must produce int8 codes
      if (l1.rows > 1 && l2.rows > 1 && l1.axis != l2.axis) continue;
      const int64_t rows = std::max(l1.rows, l2.rows);
      const int64_t axis = l1.rows > 1 ? l1.axis : l2.axis;
      const DType dt = infos[l2.table].dtype;
      const int tid = add_constant(infos, nodes[n2].name + "/composed_lut", dt, Shape{rows, 256});
      const int8_t* t1 = infos[l1.table].constant.data<int8_t>();
      for (int64_t r = 0; r < rows; ++r) {
        const int8_t* r1 = t1 + (l1.rows > 1 ? r : 0) * 256;
        const int64_t r2 = l2.rows > 1 ? r : 0;
        for (int q = 0; q < 256; ++q) {
          const int idx = static_cast<int>(r2 * 256 + r1[q] + 128);
          if (dt == DType::kF32) infos[tid].constant.data<float>()[r * 256 + q] = infos[l2.table].constant.data<float>()[idx];
          else infos[tid].constant.data<int8_t>()[r * 256 + q] = infos[l2.table].constant.data<int8_t>()[idx];
        }
      }
      Node& n = nodes[n2];
      n.inputs = {nodes[p].inputs[0], tid};
      Json attrs = n.attrs;
      attrs["axis"] = axis;
      n.attrs = attrs;
      n.name = nodes[p].name + "+" + n.name;
      removed[p] = true;
      ++composed;
      changed = true;
    }
    drop_removed(nodes, removed);
  }
  return composed;
}

// --------------------------------------------------------- collapse_unary
// An I8Mul / I8Add / I8AffineSum whose operands are all x or I8Lut(x) of one tensor x (same shapes, no broadcast) is a
// function of x alone: its table T[r][q] is the operator's integer formula (as the reference kernels evaluate it) on
// the operand codes of q, for every code q (rows: one, or per channel of the last axis when a LUT or the AffineSum
// coefficients are per channel). Exact: the same integer function on all 256 codes.
int64_t i8_mul_ref(const Json& a, int qa, int qb) {
  const int64_t za = a.get_int("a_zero", 0), zb = a.get_int("b_zero", 0), zy = a.get_int("y_zero", 0);
  const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), sh = static_cast<int32_t>(a.get_int("shift", 0));
  return q8::clamp8(zy + q8::requant((qa - za) * (qb - zb), m, sh), static_cast<int>(a.get_int("qmin", -128)),
                    static_cast<int>(a.get_int("qmax", 127)));
}
int64_t i8_add_ref(const Json& a, int qa, int qb) {
  const int64_t za = a.get_int("a_zero", 0), zb = a.get_int("b_zero", 0), zy = a.get_int("y_zero", 0);
  const int ls = static_cast<int>(a.get_int("left_shift", 20));
  const int32_t am = static_cast<int32_t>(a.get_int("a_mult", 0)), as = static_cast<int32_t>(a.get_int("a_shift", 0));
  const int32_t bm = static_cast<int32_t>(a.get_int("b_mult", 0)), bs = static_cast<int32_t>(a.get_int("b_shift", 0));
  const int32_t om = static_cast<int32_t>(a.get_int("out_mult", 0)), os = static_cast<int32_t>(a.get_int("out_shift", 0));
  const int64_t xa = q8::requant((qa - za) * (int64_t(1) << ls), am, as);
  const int64_t xb = q8::requant((qb - zb) * (int64_t(1) << ls), bm, bs);
  return q8::clamp8(zy + q8::requant(xa + xb, om, os), static_cast<int>(a.get_int("qmin", -128)), static_cast<int>(a.get_int("qmax", 127)));
}

int collapse_unary(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs) {
  int collapsed = 0;
  UseInfo u = uses_of(nodes, infos.size(), outputs);
  for (size_t n = 0; n < nodes.size(); ++n) {
    Node& nd = nodes[n];
    const bool aff = nd.op == "I8AffineSum";
    if ((nd.op != "I8Mul" && nd.op != "I8Add" && !aff) || nd.outputs.size() != 1) continue;
    const size_t nin = aff ? nd.inputs.size() - 2 : 2;
    if (nd.inputs.size() < nin || nin == 0) continue;
    const Shape& ys = infos[nd.outputs[0]].shape;
    if (ys.empty() || infos[nd.outputs[0]].dtype != DType::kI8) continue;
    const int64_t c = ys.back();
    int root = -1;
    std::vector<int> tab(nin, -1);  // operand LUT table (or -1: the root itself)
    std::vector<int64_t> trows(nin, 1);
    bool ok = true;
    for (size_t i = 0; i < nin && ok; ++i) {
      const int t = nd.inputs[i];
      if (t < 0 || infos[t].shape != ys) {
        ok = false;
        break;
      }
      int r = t;
      const int p = u.producer[t];
      LutInfo li;
      if (p >= 0 && lut_info(nodes[p], infos, &li) && infos[li.table].dtype == DType::kI8 &&
          (li.rows == 1 || li.axis == static_cast<int64_t>(ys.size()) - 1) && infos[nodes[p].inputs[0]].shape == ys) {
        r = nodes[p].inputs[0];
        tab[i] = li.table;
        trows[i] = li.rows;
      }
      if (root < 0) root = r;
      ok = r == root;
    }
    if (!ok || root < 0 || nin > 64) continue;
    int64_t kc = 1;
    const int32_t* K = nullptr;
    const int32_t* B = nullptr;
    if (aff) {
      const int kt = nd.inputs[nin], bt = nd.inputs[nin + 1];
      if (kt < 0 || bt < 0 || !infos[kt].is_constant || !infos[bt].is_constant) continue;
      kc = infos[kt].shape[1];
      K = infos[kt].constant.data<int32_t>();
      B = infos[bt].constant.data<int32_t>();
    }
    int64_t rows = kc > 1 ? c : 1;
    for (size_t i = 0; i < nin; ++i) rows = std::max(rows, trows[i] > 1 ? c : int64_t(1));
    const int tid = add_constant(infos, nd.name + "/unary_lut", DType::kI8, Shape{rows, 256});
    int8_t* out = infos[tid].constant.data<int8_t>();
    const auto zs = aff ? nd.attrs.get_ints("zeros", {}) : std::vector<int64_t>{};
    for (int64_t r = 0; r < rows; ++r) {
      for (int q = -128; q <= 127; ++q) {
        int ops[64];
        for (size_t i = 0; i < nin; ++i) ops[i] = tab[i] < 0 ? q : infos[tab[i]].constant.data<int8_t>()[(trows[i] > 1 ? r : 0) * 256 + q + 128];
        int64_t y;
        if (aff) {
          const int64_t ch = kc > 1 ? r : 0;
          int64_t acc = B[ch];
          for (size_t i = 0; i < nin; ++i) acc += (ops[i] - zs[i]) * static_cast<int64_t>(K[i * kc + ch]);
          y = q8::clamp8(nd.attrs.get_int("y_zero", 0) + q8::requant(acc, static_cast<int32_t>(nd.attrs.get_int("mult", 0)),
                                                                   static_cast<int32_t>(nd.attrs.get_int("shift", 0))),
                         static_cast<int>(nd.attrs.get_int("qmin", -128)), static_cast<int>(nd.attrs.get_int("qmax", 127)));
        } else if (nd.op == "I8Mul") {
          y = i8_mul_ref(nd.attrs, ops[0], ops[1]);
        } else {
          y = i8_add_ref(nd.attrs, ops[0], ops[1]);
        }
        out[r * 256 + q + 128] = static_cast<int8_t>(y);
      }
    }
    Node l;
    l.op = "I8Lut";
    l.name = nd.name + "/unary";
    l.inputs = {root, tid};
    l.outputs = nd.outputs;
    l.attrs = Json::Object{{"collapsed", nd.op}};
    nodes[n] = std::move(l);
    ++collapsed;
  }
  return collapsed;
}

// I8Lut / I8LutN nodes whose outputs are all unused (e.g. after collapse_unary or threshold folding)
int remove_dead_luts(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs) {
  int removed_total = 0;
  for (bool changed = true; changed;) {
    changed = false;
    UseInfo u = uses_of(nodes, infos.size(), outputs);
    std::vector<bool> removed(nodes.size(), false);
    for (size_t n = 0; n < nodes.size(); ++n) {
      if (nodes[n].op != "I8Lut") continue;
      bool dead = true;
      for (int o : nodes[n].outputs) dead = dead && u.count[o] == 0;
      if (dead) {
        removed[n] = true;
        ++removed_total;
        changed = true;
      }
    }
    drop_removed(nodes, removed);
  }
  return removed_total;
}

// ------------------------------------------------------ fold_lut_thresholds
bool binarizes_input0(const Node& c, int y) {
  // I8BConv2DRes (int8_full v2 / log2 blocks) binarizes its input 0 like I8BConv2D (y must not also be its residual)
  if ((c.op != "I8BConv2D" && c.op != "I8BDense" && c.op != "I8BConv2DRes" && c.op != "I8BConv2DResI") || in_at(c, 0) != y)
    return false;
  for (size_t i = 1; i < c.inputs.size(); ++i)
    if (c.inputs[i] == y) return false;
  const std::string m = c.attrs.get_string("input_binarize", "ge");
  return m == "ge" || m == "gt";
}

void fold_lut_thresholds(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs, I8RewriteStats* st) {
  UseInfo u = uses_of(nodes, infos.size(), outputs);
  std::vector<bool> removed(nodes.size(), false);
  // interval folds: I8PackBits nodes to insert before node index k (shared per (x, bounds))
  std::map<size_t, std::vector<Node>> insert_before;
  std::map<std::pair<int, std::string>, int> packed;  // (x, bounds bytes) -> bit tensor
  for (size_t l = 0; l < nodes.size(); ++l) {
    LutInfo li;
    if (!lut_info(nodes[l], infos, &li) || infos[li.table].dtype != DType::kI8) continue;
    const int x = nodes[l].inputs[0], y = nodes[l].outputs[0];
    const Shape ys = infos[y].shape;  // a copy: infos grows below
    if (li.rows > 1 && li.axis != static_cast<int64_t>(ys.size()) - 1) continue;  // per-channel along the binarized axis only
    const int64_t c = ys.back();
    const int8_t* tab = infos[li.table].constant.data<int8_t>();
    int remaining = u.count[y];
    for (size_t k = l + 1; k < nodes.size(); ++k) {
      Node& cn = nodes[k];
      if (!binarizes_input0(cn, y)) continue;
      const bool gt = cn.attrs.get_string("input_binarize", "ge") == "gt";
      q8::Deq d;
      d.l8 = cn.attrs.get_string("x_code", "linear") == "l8";
      d.s = static_cast<float>(cn.attrs.get_number("x_scale", 1.0));
      d.z = static_cast<int32_t>(cn.attrs.get_int("x_zero", 0));
      d.e = static_cast<int>(cn.attrs.get_int("x_exp", 0));
      const int thr_in = in_at(cn, 4);
      if (thr_in >= 0 && (!infos[thr_in].is_constant || infos[thr_in].dtype != DType::kF32)) continue;
      const int64_t nthr = thr_in >= 0 ? num_elements(infos[thr_in].shape) : 1;
      if (nthr != 1 && nthr != c) continue;
      std::vector<float> tx(static_cast<size_t>(c));
      std::vector<int8_t> lo(static_cast<size_t>(c)), hi(static_cast<size_t>(c));
      bool upper = true, interval = true;
      for (int64_t ch = 0; ch < c && (upper || interval); ++ch) {
        const float t = thr_in >= 0 ? infos[thr_in].constant.data<float>()[nthr == 1 ? 0 : ch] : 0.f;
        const int ty = gt ? q8::qthr_gt(t, d) : q8::qthr_ge(t, d);  // bit +1 <=> y >= ty
        const int8_t* row = tab + (li.rows > 1 ? ch : 0) * 256;
        int first = 128;  // min{x : L[x] >= ty}
        for (int q = -128; q <= 127; ++q)
          if (row[q + 128] >= ty) {
            first = q;
            break;
          }
        for (int q = first; q <= 127 && upper; ++q) upper = row[q + 128] >= ty;  // upper set
        tx[ch] = static_cast<float>(first);
        // the -1 codes {x : L[x] < ty} as one interval [a, b] (empty: a = 127, b = -128)
        int a = 128, b = -129;
        for (int q = -128; q <= 127; ++q)
          if (row[q + 128] < ty) {
            a = std::min(a, q);
            b = q;
          }
        for (int q = a; q <= b && interval; ++q) interval = row[q + 128] < ty;
        lo[ch] = static_cast<int8_t>(a > b ? 127 : a);
        hi[ch] = static_cast<int8_t>(a > b ? -128 : b);
      }
      if (upper) {
        const int tid = add_constant(infos, cn.name + "/thr_through_lut", DType::kF32, Shape{c});
        std::copy(tx.begin(), tx.end(), infos[tid].constant.data<float>());
        if (cn.inputs.size() < 5) cn.inputs.resize(5, -1);
        cn.inputs[0] = x;
        cn.inputs[4] = tid;
        Json attrs = cn.attrs;
        attrs["input_binarize"] = "ge";
        attrs["x_code"] = "linear";
        attrs["x_scale"] = 1.0;
        attrs["x_zero"] = 0;
        attrs["thresholds_through"] = nodes[l].name;
        cn.attrs = attrs;
        --remaining;
        ++st->folded_luts;
        continue;
      }
      if (!interval) continue;
      // interval fold (2026-11-13; e.g. V-shaped PReLU LUTs of BHViT, MeliusNet, ReActNet-18 variants): the consumer reads
      // the bits of I8PackBits(x) (bit -1 iff lo[c] <= x <= hi[c]), the same binarization of y = L(x)
      std::string key(reinterpret_cast<const char*>(lo.data()), static_cast<size_t>(c));
      key.append(reinterpret_cast<const char*>(hi.data()), static_cast<size_t>(c));
      auto it = packed.find({x, key});
      int bt;
      if (it != packed.end()) {
        bt = it->second;
      } else {
        const int bid = add_constant(infos, cn.name + "/interval_bounds", DType::kI8, Shape{2, c});
        std::copy(lo.begin(), lo.end(), infos[bid].constant.data<int8_t>());
        std::copy(hi.begin(), hi.end(), infos[bid].constant.data<int8_t>() + c);
        TensorInfo bi;
        bi.name = infos[y].name + ":bits_through_lut" + std::to_string(packed.size());
        bi.dtype = DType::kBit;
        bi.shape = ys;
        infos.push_back(std::move(bi));
        bt = static_cast<int>(infos.size()) - 1;
        Node pb;
        pb.op = "I8PackBits";
        pb.name = nodes[l].name + "/interval_bits";
        pb.inputs = {x, bid};
        pb.outputs = {bt};
        insert_before[k].push_back(std::move(pb));
        packed[{x, key}] = bt;
      }
      cn.inputs[0] = bt;
      if (cn.inputs.size() > 4) cn.inputs[4] = -1;
      Json attrs = cn.attrs;
      attrs["input_binarize"] = "ge";
      attrs["bits_through"] = nodes[l].name;
      cn.attrs = attrs;
      --remaining;
      ++st->interval_folds;
    }
    if (remaining == 0) {
      removed[l] = true;
      ++st->removed_luts;
    }
  }
  std::vector<Node> kept;
  kept.reserve(nodes.size() + insert_before.size());
  for (size_t n = 0; n < nodes.size(); ++n) {
    auto it = insert_before.find(n);
    if (it != insert_before.end())
      for (Node& pb : it->second) kept.push_back(std::move(pb));
    if (!removed[n]) kept.push_back(std::move(nodes[n]));
  }
  nodes.swap(kept);
}

// -------------------------------------------------------------- merge_twins
void merge_twins(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, I8RewriteStats* st) {
  std::vector<bool> removed(nodes.size(), false);
  // I8Lut twins: same input, int8 tables, one table or per-channel tables on the last axis
  std::map<int, std::vector<size_t>> luts;
  for (size_t n = 0; n < nodes.size(); ++n) {
    LutInfo li;
    if (!lut_info(nodes[n], infos, &li) || infos[li.table].dtype != DType::kI8) continue;
    const int64_t rank = static_cast<int64_t>(infos[nodes[n].inputs[0]].shape.size());
    if (li.rows > 1 && li.axis != rank - 1) continue;
    luts[nodes[n].inputs[0]].push_back(n);
  }
  for (auto& kv : luts) {
    if (kv.second.size() < 2) continue;
    const size_t first = kv.second.front();
    Node m;
    m.op = "I8LutN";
    m.name = nodes[first].name + "/twins" + std::to_string(kv.second.size());
    m.inputs.push_back(kv.first);
    Json::Array names;
    for (size_t n : kv.second) {
      m.inputs.push_back(nodes[n].inputs[1]);
      m.outputs.push_back(nodes[n].outputs[0]);
      names.push_back(Json(nodes[n].name));
      removed[n] = true;
    }
    m.attrs = Json::Object{{"merged_nodes", Json(names)}};
    nodes[first] = std::move(m);
    removed[first] = false;
    ++st->lut_twins;
    st->merged_nodes += static_cast<int>(kv.second.size());
  }
  // I8Add twins: same a, b and the same input parameters (only the output stage differs)
  static const char* kIn[] = {"a_zero", "b_zero", "left_shift", "a_mult", "a_shift", "b_mult", "b_shift"};
  std::map<std::vector<int64_t>, std::vector<size_t>> adds;
  for (size_t n = 0; n < nodes.size(); ++n) {
    const Node& a = nodes[n];
    if (removed[n] || a.op != "I8Add" || a.inputs.size() != 2 || a.outputs.size() != 1 || a.inputs[0] < 0 || a.inputs[1] < 0) continue;
    std::vector<int64_t> key = {a.inputs[0], a.inputs[1]};
    for (const char* k : kIn) key.push_back(a.attrs.get_int(k, std::string(k) == "left_shift" ? 20 : 0));
    adds[key].push_back(n);
  }
  for (auto& kv : adds) {
    if (kv.second.size() < 2) continue;
    const size_t first = kv.second.front();
    Node m;
    m.op = "I8AddN";
    m.name = nodes[first].name + "/twins" + std::to_string(kv.second.size());
    m.inputs = nodes[first].inputs;
    Json attrs = Json::Object{};
    for (const char* k : kIn) attrs[k] = nodes[first].attrs.get_int(k, std::string(k) == "left_shift" ? 20 : 0);
    Json::Array om, os, zy, lo, hi, names;
    for (size_t n : kv.second) {
      const Json& a = nodes[n].attrs;
      om.push_back(Json(a.get_int("out_mult", 0)));
      os.push_back(Json(a.get_int("out_shift", 0)));
      zy.push_back(Json(a.get_int("y_zero", 0)));
      lo.push_back(Json(a.get_int("qmin", -128)));
      hi.push_back(Json(a.get_int("qmax", 127)));
      names.push_back(Json(nodes[n].name));
      m.outputs.push_back(nodes[n].outputs[0]);
      removed[n] = true;
    }
    attrs["out_mults"] = Json(om);
    attrs["out_shifts"] = Json(os);
    attrs["y_zeros"] = Json(zy);
    attrs["qmins"] = Json(lo);
    attrs["qmaxs"] = Json(hi);
    attrs["merged_nodes"] = Json(names);
    m.attrs = attrs;
    nodes[first] = std::move(m);
    removed[first] = false;
    ++st->add_twins;
    st->merged_nodes += static_cast<int>(kv.second.size());
  }
  // I8LayerNorm twins (2026-11-13): same x, x_zero, x_scale, eps (the row statistics and the factor r), 2-4 of them ->
  // one I8LayerNormN (statistics, u and fp32(u) * r once; per output G, B and the clamp)
  std::map<std::vector<std::string>, std::vector<size_t>> lns;
  for (size_t n = 0; n < nodes.size(); ++n) {
    const Node& l = nodes[n];
    if (removed[n] || l.op != "I8LayerNorm" || l.inputs.size() != 3 || l.outputs.size() != 1 || l.inputs[0] < 0) continue;
    lns[{std::to_string(l.inputs[0]), attr_dump(l.attrs, "x_zero"), attr_dump(l.attrs, "x_scale"), attr_dump(l.attrs, "eps")}].push_back(n);
  }
  for (auto& kv : lns) {
    if (kv.second.size() < 2) continue;
    for (size_t g0 = 0; g0 + 1 < kv.second.size(); g0 += 4) {
      std::vector<size_t> grp(kv.second.begin() + g0, kv.second.begin() + std::min(kv.second.size(), g0 + 4));
      if (grp.size() < 2) continue;
      Node m;
      m.op = "I8LayerNormN";
      m.name = nodes[grp.front()].name + "/twins" + std::to_string(grp.size());
      m.inputs.push_back(nodes[grp.front()].inputs[0]);
      Json attrs = Json::Object{};
      for (const char* key : {"x_zero", "x_scale", "eps"})
        if (nodes[grp.front()].attrs.contains(key)) attrs[key] = nodes[grp.front()].attrs[key];
      Json::Array lo, hi, names;
      for (size_t n : grp) {
        const Node& l = nodes[n];
        m.inputs.push_back(l.inputs[1]);
        m.inputs.push_back(l.inputs[2]);
        m.outputs.push_back(l.outputs[0]);
        lo.push_back(Json(l.attrs.get_int("qmin", -128)));
        hi.push_back(Json(l.attrs.get_int("qmax", 127)));
        names.push_back(Json(l.name));
        removed[n] = true;
      }
      attrs["qmins"] = Json(lo);
      attrs["qmaxs"] = Json(hi);
      attrs["merged_nodes"] = Json(names);
      m.attrs = attrs;
      nodes[grp.front()] = std::move(m);
      removed[grp.front()] = false;
      ++st->ln_twins;
      st->merged_nodes += static_cast<int>(grp.size());
    }
  }
  // I8Conv2D twins: same x, byte-identical weights and bias, same geometry (groups 1), int8 output, no bit output -> one
  // I8Conv2DN (the accumulators computed once, one requantization per twin)
  auto same_const = [&](int a, int b) {
    if (a < 0 || b < 0) return a == b;
    const TensorInfo& ta = infos[a];
    const TensorInfo& tb = infos[b];
    if (!ta.is_constant || !tb.is_constant || ta.dtype != tb.dtype || ta.shape != tb.shape) return false;
    return a == b || std::memcmp(ta.constant.raw(), tb.constant.raw(), ta.constant.bytes()) == 0;
  };
  static const char* kGeo[] = {"strides", "pads", "dilations", "groups", "x_zero", "pad_q", "output"};
  auto conv_ok = [&](const Node& c) {
    return c.op == "I8Conv2D" && c.outputs.size() == 1 && c.inputs.size() >= 5 && in_at(c, 5) < 0 && c.inputs[0] >= 0 && c.inputs[1] >= 0 &&
           c.inputs[3] >= 0 && c.inputs[4] >= 0 && c.attrs.get_string("output", "i8") == "i8" &&
           c.attrs.get_int("groups", 1) == 1;
  };
  std::map<int, std::vector<size_t>> convs;
  for (size_t n = 0; n < nodes.size(); ++n)
    if (!removed[n] && conv_ok(nodes[n])) convs[nodes[n].inputs[0]].push_back(n);
  for (auto& kv : convs) {
    std::vector<bool> used(kv.second.size(), false);
    for (size_t a = 0; a < kv.second.size(); ++a) {
      if (used[a]) continue;
      const Node& na = nodes[kv.second[a]];
      std::vector<size_t> grp = {kv.second[a]};
      for (size_t b = a + 1; b < kv.second.size(); ++b) {
        if (used[b]) continue;
        const Node& nb = nodes[kv.second[b]];
        bool same = same_const(na.inputs[1], nb.inputs[1]) && same_const(in_at(na, 2), in_at(nb, 2));
        for (const char* k : kGeo) same = same && attr_dump(na.attrs, k) == attr_dump(nb.attrs, k);
        if (same) {
          grp.push_back(kv.second[b]);
          used[b] = true;
        }
      }
      if (grp.size() < 2) continue;
      Node m;
      m.op = "I8Conv2DN";
      m.name = na.name + "/twins" + std::to_string(grp.size());
      m.inputs = {na.inputs[0], na.inputs[1], in_at(na, 2)};
      Json attrs = Json::Object{};
      for (const char* k : kGeo)
        if (na.attrs.contains(k)) attrs[k] = na.attrs[k];
      Json::Array zy, lo, hi, names;
      for (size_t n : grp) {
        const Node& c = nodes[n];
        m.inputs.push_back(c.inputs[3]);
        m.inputs.push_back(c.inputs[4]);
        m.outputs.push_back(c.outputs[0]);
        zy.push_back(Json(c.attrs.get_int("y_zero", 0)));
        lo.push_back(Json(c.attrs.get_int("qmin", -128)));
        hi.push_back(Json(c.attrs.get_int("qmax", 127)));
        names.push_back(Json(c.name));
        removed[n] = true;
      }
      attrs["y_zeros"] = Json(zy);
      attrs["qmins"] = Json(lo);
      attrs["qmaxs"] = Json(hi);
      attrs["merged_nodes"] = Json(names);
      m.attrs = attrs;
      nodes[grp.front()] = std::move(m);
      removed[grp.front()] = false;
      ++st->conv_twins;
      st->merged_nodes += static_cast<int>(grp.size());
    }
  }
  drop_removed(nodes, removed);
}

// ------------------------------------------------------------- bit outputs
// The per-channel integer thresholds T[c] (+1 iff y[c] >= T[c]) of the binary layers reading y, when every reader
// binarizes y (input 0, ge / gt) with the same thresholds and y is no graph output; empty otherwise.
std::vector<int> reader_thresholds(const std::vector<Node>& nodes, const std::vector<TensorInfo>& infos, const UseInfo& u, int y) {
  if (y < 0 || u.is_output[y] || infos[y].dtype != DType::kI8 || infos[y].shape.empty()) return {};
  const int64_t c = infos[y].shape.back();
  std::vector<int> ty;
  int slots = 0;
  for (const Node& cn : nodes) {
    int uses = 0;
    for (int id : cn.inputs) uses += id == y;
    if (uses == 0) continue;
    if (!binarizes_input0(cn, y)) return {};
    const bool gt = cn.attrs.get_string("input_binarize", "ge") == "gt";
    q8::Deq d;
    d.l8 = cn.attrs.get_string("x_code", "linear") == "l8";
    d.s = static_cast<float>(cn.attrs.get_number("x_scale", 1.0));
    d.z = static_cast<int32_t>(cn.attrs.get_int("x_zero", 0));
    d.e = static_cast<int>(cn.attrs.get_int("x_exp", 0));
    const int thr_in = in_at(cn, 4);
    if (thr_in >= 0 && (!infos[thr_in].is_constant || infos[thr_in].dtype != DType::kF32)) return {};
    const int64_t nthr = thr_in >= 0 ? num_elements(infos[thr_in].shape) : 1;
    if (nthr != 1 && nthr != c) return {};
    std::vector<int> t(static_cast<size_t>(c));
    for (int64_t ch = 0; ch < c; ++ch) {
      const float v = thr_in >= 0 ? infos[thr_in].constant.data<float>()[nthr == 1 ? 0 : ch] : 0.f;
      t[ch] = gt ? q8::qthr_gt(v, d) : q8::qthr_ge(v, d);
    }
    if (ty.empty()) ty = t;
    else if (ty != t) return {};
    ++slots;
  }
  if (slots == 0 || slots != u.count[y]) return {};
  return ty;
}

// y(acc) = clamp(zy + requant(acc, m, s), lo, hi) is non-decreasing in acc for m > 0 (sat32, mulhi and rshr are):
// y >= T <=> acc >= A = min{acc : y(acc) >= T}, i.e. bit -1 <=> acc <= A - 1. Returned as an int32 bound U for
// accumulators with |acc| <= 2^31 - 1 (callers check it): A - 1 above int32 is "always" (U = INT32_MAX), below is
// "never" (U = INT32_MIN, unreachable).
int32_t acc_bound(int t, int64_t zy, int32_t m, int32_t s, int lo, int hi) {
  auto y = [&](int64_t a) { return q8::clamp8(zy + q8::requant(a, m, s), lo, hi); };
  int64_t l = -(int64_t(1) << 40), h = int64_t(1) << 40;
  int64_t a;
  if (y(h) < t) a = h + 1;
  else if (y(l) >= t) a = l;
  else {
    while (h - l > 1) {  // y(l) < t <= y(h)
      const int64_t mid = l + (h - l) / 2;
      if (y(mid) >= t) h = mid;
      else l = mid;
    }
    a = h;
  }
  return static_cast<int32_t>(std::min<int64_t>(std::max<int64_t>(a - 1, INT32_MIN), INT32_MAX));
}

// The readers of a tensor turned into bits read them as an already binarized input.
void point_readers_to_bits(std::vector<Node>& nodes, int y, const std::string& from) {
  for (Node& cn : nodes) {
    if (in_at(cn, 0) != y) continue;
    if (cn.inputs.size() > 4) cn.inputs[4] = -1;
    Json attrs = cn.attrs;
    attrs["input_binarize"] = "ge";
    attrs["bits_from"] = from;
    cn.attrs = attrs;
  }
}

int add_bounds(std::vector<TensorInfo>& infos, const std::string& name, const std::vector<int32_t>& u) {
  const bool same = std::all_of(u.begin(), u.end(), [&](int32_t v) { return v == u[0]; });
  const int64_t n = same ? 1 : static_cast<int64_t>(u.size());
  const int id = add_constant(infos, name, DType::kI32, Shape{n});
  std::copy_n(u.data(), n, infos[id].constant.data<int32_t>());
  return id;
}

// I8Add / I8AddN outputs and I8AffineSum nodes whose output only feeds binarizers (e.g. RBNN's mean centering, the
// binarize-aware twins of Bi-Real / QuickNet / ReActNet) produce the bits themselves: bit -1 iff the integer value
// before the output requantization (I8Add: xa + xb; I8AffineSum: B + sum (q - z) K) is <= U[c], with U from
// acc_bound. Exact: y >= T <=> acc >= A for the monotone requantization (multiplier > 0), for every reachable acc.
void bit_outputs(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& outputs, I8RewriteStats* st) {
  UseInfo u = uses_of(nodes, infos.size(), outputs);
  for (size_t n = 0; n < nodes.size(); ++n) {
    Node& nd = nodes[n];
    if (nd.op == "I8Add" || nd.op == "I8AddN") {
      const bool twin = nd.op == "I8AddN";
      if (nd.inputs.size() != 2 || nd.inputs[0] < 0 || nd.inputs[1] < 0) continue;
      const Json& a = nd.attrs;
      // |xa|, |xb| < 255 * 2^22 / 2 (input multipliers < 2^31, no left shifts): |xa + xb| < 2^31
      if (a.get_int("left_shift", 20) > 22 || a.get_int("a_shift", 0) > 0 || a.get_int("b_shift", 0) > 0 || a.get_int("a_mult", 0) <= 0 ||
          a.get_int("b_mult", 0) <= 0)
        continue;
      auto in8 = [](int64_t z) { return z >= -128 && z <= 127; };
      if (!in8(a.get_int("a_zero", 0)) || !in8(a.get_int("b_zero", 0))) continue;
      const Shape as = infos[nd.inputs[0]].shape;  // copies: add_bounds grows infos
      const Shape bs = infos[nd.inputs[1]].shape;
      const int64_t c = as.empty() ? 0 : as.back();
      // same shapes, a scalar b, or (twins) a broadcast row of b
      const bool row = bs.size() >= 1 && num_elements(bs) == c && bs.back() == c;
      if (!(as == bs || num_elements(bs) == 1 || (twin && row)) || c == 0) continue;
      const size_t k = nd.outputs.size();
      std::vector<int64_t> om, os, zy, lo, hi;
      if (twin) {
        om = a.get_ints("out_mults", {});
        os = a.get_ints("out_shifts", {});
        zy = a.get_ints("y_zeros", {});
        lo = a.get_ints("qmins", {});
        hi = a.get_ints("qmaxs", {});
      } else {
        om = {a.get_int("out_mult", 0)};
        os = {a.get_int("out_shift", 0)};
        zy = {a.get_int("y_zero", 0)};
        lo = {a.get_int("qmin", -128)};
        hi = {a.get_int("qmax", 127)};
      }
      Json::Array bit_outs;
      std::vector<int> bounds;
      for (size_t j = 0; j < k; ++j) {
        const int y = nd.outputs[j];
        if (om[j] <= 0 || infos[y].shape != as) continue;
        const std::vector<int> ty = reader_thresholds(nodes, infos, u, y);
        if (ty.empty()) continue;
        std::vector<int32_t> ub(ty.size());
        for (size_t ch = 0; ch < ty.size(); ++ch)
          ub[ch] = acc_bound(ty[ch], zy[j], static_cast<int32_t>(om[j]), static_cast<int32_t>(os[j]), static_cast<int>(lo[j]), static_cast<int>(hi[j]));
        bit_outs.push_back(Json(static_cast<int64_t>(j)));
        bounds.push_back(add_bounds(infos, nd.name + "/bit_bound" + std::to_string(j), ub));
        infos[y].dtype = DType::kBit;
        point_readers_to_bits(nodes, y, nd.name);
        ++st->bit_outputs;
      }
      if (bit_outs.empty()) continue;
      Json attrs = nd.attrs;
      if (twin) {
        attrs["bit_outputs"] = Json(bit_outs);
      } else {
        nd.op = "I8AddB";
      }
      nd.attrs = attrs;
      for (int b : bounds) nd.inputs.push_back(b);
      continue;
    }
    // I8Conv2DN twins (x, w, bias, mult_0, shift_0, mult_1, shift_1) whose output 1 (or 0) only feeds binarizers ->
    // I8Conv2D with the other output and the bit output of bit_thr (input 5): bit -1 iff acc < bit_thr[o], acc =
    // sum (x - x_zero) w before the bias. y_bin = clamp(zy + requant(sat32(acc + bias), m, s)) >= T <=> sat32(acc + bias)
    // >= A <=> acc >= A - bias for |acc| <= taps * 255 * 128 < 2^30 (sat32 is monotone; thresholds clamped to int32).
    if (nd.op == "I8Conv2DN" && nd.outputs.size() == 2 && nd.inputs.size() == 7 && nd.inputs[0] >= 0 && nd.inputs[1] >= 0) {
      const int w = nd.inputs[1];
      const Shape ws = infos[w].shape;  // [O, kh, kw, kic]
      if (ws.size() != 4 || num_elements(ws) / ws[0] * 255 * 128 >= (int64_t(1) << 30)) continue;
      const auto zys = nd.attrs.get_ints("y_zeros", {}), los = nd.attrs.get_ints("qmins", {}), his = nd.attrs.get_ints("qmaxs", {});
      int jb = -1;
      std::vector<int> ty;
      for (int j = 1; j >= 0 && jb < 0; --j) {
        ty = reader_thresholds(nodes, infos, u, nd.outputs[j]);
        if (!ty.empty()) jb = j;
      }
      if (jb < 0) continue;
      const int mt = nd.inputs[3 + 2 * jb], stt = nd.inputs[4 + 2 * jb], bias = nd.inputs[2];
      if (mt < 0 || stt < 0 || !infos[mt].is_constant || !infos[stt].is_constant || (bias >= 0 && !infos[bias].is_constant)) continue;
      const int64_t oc = ws[0];
      if (static_cast<int64_t>(ty.size()) != oc) continue;
      bool pos = true;
      for (int64_t o = 0; o < oc; ++o) pos = pos && infos[mt].constant.data<int32_t>()[o] > 0;
      if (!pos) continue;
      std::vector<int32_t> thr(static_cast<size_t>(oc));
      for (int64_t o = 0; o < oc; ++o) {
        const int32_t ub = acc_bound(ty[o], zys[jb], infos[mt].constant.data<int32_t>()[o], infos[stt].constant.data<int32_t>()[o],
                                     static_cast<int>(los[jb]), static_cast<int>(his[jb]));
        const int64_t b = bias >= 0 ? infos[bias].constant.data<int32_t>()[o] : 0;
        // bit -1 iff sat32(acc + b) <= ub <=> acc + b <= ub (ub < INT32_MAX; ub == INT32_MAX: always) <=> acc < ub - b + 1
        const int64_t t = ub == INT32_MAX ? (int64_t(1) << 40) : static_cast<int64_t>(ub) - b + 1;
        thr[o] = static_cast<int32_t>(std::min<int64_t>(std::max<int64_t>(t, INT32_MIN), INT32_MAX));
      }
      const int jm = 1 - jb;
      const int tid = add_constant(infos, nd.name + "/bit_thr", DType::kI32, Shape{oc});
      std::copy(thr.begin(), thr.end(), infos[tid].constant.data<int32_t>());
      const int ybits = nd.outputs[jb];
      Node c;
      c.op = "I8Conv2D";
      c.name = nd.name + "/bits";
      c.inputs = {nd.inputs[0], w, bias, nd.inputs[3 + 2 * jm], nd.inputs[4 + 2 * jm], tid};
      c.outputs = {nd.outputs[jm], ybits};
      Json attrs = Json::Object{};
      for (const char* key : {"strides", "pads", "dilations", "groups", "x_zero", "pad_q"})
        if (nd.attrs.contains(key)) attrs[key] = nd.attrs[key];
      attrs["y_zero"] = zys[jm];
      attrs["qmin"] = los[jm];
      attrs["qmax"] = his[jm];
      attrs["merged_nodes"] = nd.attrs.contains("merged_nodes") ? nd.attrs["merged_nodes"] : Json(Json::Array{});
      c.attrs = attrs;
      nodes[n] = std::move(c);
      infos[ybits].dtype = DType::kBit;
      point_readers_to_bits(nodes, ybits, nodes[n].name);
      ++st->bit_outputs;
      continue;
    }
    if (nd.op == "I8AffineSum" && nd.outputs.size() == 1 && nd.inputs.size() >= 3) {
      const size_t nin = nd.inputs.size() - 2;
      const int kt = nd.inputs[nin], bt = nd.inputs[nin + 1];
      if (kt < 0 || bt < 0 || !infos[kt].is_constant || !infos[bt].is_constant) continue;
      const int y = nd.outputs[0];
      const Json& a = nd.attrs;
      const int32_t m = static_cast<int32_t>(a.get_int("mult", 0)), s = static_cast<int32_t>(a.get_int("shift", 0));
      if (m <= 0) continue;
      const auto zs = a.get_ints("zeros", {});
      if (zs.size() != nin) continue;
      const int64_t c = infos[y].shape.empty() ? 0 : infos[y].shape.back();
      const int64_t kc = infos[kt].shape[1];
      const int32_t* K = infos[kt].constant.data<int32_t>();
      const int32_t* B = infos[bt].constant.data<int32_t>();
      // int32-exact accumulators: |B| + sum |K| (128 + |z|) < 2^31 per channel (|acc| <= 2^31 - 1)
      bool fits = c > 0;
      for (int64_t ch = 0; ch < c && fits; ++ch) {
        int64_t bound = std::abs(static_cast<int64_t>(B[kc == 1 ? 0 : ch]));
        for (size_t i = 0; i < nin; ++i) bound += std::abs(static_cast<int64_t>(K[i * kc + (kc == 1 ? 0 : ch)])) * (128 + std::abs(zs[i]));
        fits = bound < (int64_t(1) << 31);
      }
      if (!fits) continue;
      const std::vector<int> ty = reader_thresholds(nodes, infos, u, y);
      if (ty.empty()) continue;
      std::vector<int32_t> ub(ty.size());
      for (size_t ch = 0; ch < ty.size(); ++ch)
        ub[ch] = acc_bound(ty[ch], a.get_int("y_zero", 0), m, s, static_cast<int>(a.get_int("qmin", -128)), static_cast<int>(a.get_int("qmax", 127)));
      nd.op = "I8AffineSumB";
      nd.inputs.push_back(add_bounds(infos, nd.name + "/bit_bound", ub));
      infos[y].dtype = DType::kBit;
      point_readers_to_bits(nodes, y, nd.name);
      ++st->bit_outputs;
    }
  }
}

}  // namespace

I8RewriteStats rewrite_int8_luts(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& graph_outputs) {
  I8RewriteStats st;
  st.collapsed_ops = collapse_unary(nodes, infos, graph_outputs);
  if (st.collapsed_ops) st.removed_luts += remove_dead_luts(nodes, infos, graph_outputs);
  st.composed_luts = compose_luts(nodes, infos, graph_outputs);
  fold_lut_thresholds(nodes, infos, graph_outputs, &st);
  return st;
}

I8RewriteStats rewrite_int8_twins(std::vector<Node>& nodes, std::vector<TensorInfo>& infos, const std::vector<int>& graph_outputs) {
  I8RewriteStats st;
  merge_twins(nodes, infos, &st);
  bit_outputs(nodes, infos, graph_outputs, &st);
  return st;
}

}  // namespace bitino

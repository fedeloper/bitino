// bitino_microbench: single-operator benchmarks on synthetic data.
//
//   bitino_microbench --op bconv|conv|dwconv|dense|bdense --h 56 --w 56 --ic 64 --oc 64
//                       [--k 3] [--stride 1] [--pad same|valid] [--pad-value 0|1|-1] [--groups G (conv)]
//                       [--tiers t1,t2 | --force kernel] [--threads N] [--warmup W] [--iterations I]
//                       [--bit-out]
//   bitino_microbench --suite quicknet|birealnet [--tiers ...] [--threads N]
//
// Prints one JSON object per benchmarked configuration (median/mean/min ms,
// effective GOPS where one binary MAC or one float MAC counts as 2 ops).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "bitino/bitino.h"
#include "bitino/util.h"
#include "tools/model_builder.h"

using namespace bitino;

namespace {

struct Cfg {
  std::string op = "bconv";
  int64_t h = 56, w = 56, ic = 64, oc = 64, k = 3, stride = 1, groups = 1;
  std::string pad = "same";
  int pad_value = 0;
  bool bit_out = false;
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t b = 0;
  while (b <= s.size()) {
    size_t e = s.find(sep, b);
    if (e == std::string::npos) e = s.size();
    if (e > b) out.push_back(s.substr(b, e - b));
    b = e + 1;
  }
  return out;
}

Json ints(std::initializer_list<int64_t> v) {
  Json::Array a;
  for (auto x : v) a.push_back(Json(x));
  return Json(a);
}

std::shared_ptr<Model> build(const Cfg& c) {
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  bftest::ModelBuilder mb;
  int64_t p = c.pad == "same" ? c.k / 2 : 0;
  Json attrs = Json::Object{{"strides", ints({c.stride, c.stride})}, {"pads", ints({p, p, p, p})}};
  std::vector<float> mul(c.oc), add(c.oc), bias(c.oc);
  for (auto& v : mul) v = 0.05f + std::fabs(nd(rng)) * 0.01f;
  for (auto& v : add) v = nd(rng);
  for (auto& v : bias) v = nd(rng);
  if (c.op == "bconv" || c.op == "conv" || c.op == "dwconv") {
    mb.input("x", DType::kF32, {1, c.h, c.w, c.ic});
    if (c.op == "bconv") {
      std::vector<int8_t> wv(static_cast<size_t>(c.oc * c.k * c.k * c.ic));
      for (auto& v : wv) v = (rng() & 1) ? 1 : -1;
      auto wp = bftest::pack_pm1(wv, c.oc * c.k * c.k, c.ic);
      mb.constant("w", DType::kBit, {c.oc, c.k, c.k, c.ic}, wp.data());
      mb.constant("mul", DType::kF32, {c.oc}, mul.data());
      mb.constant("add", DType::kF32, {c.oc}, add.data());
      attrs["pad_value"] = Json(c.pad_value);
      attrs["output"] = c.bit_out ? "bit" : "f32";
      mb.node("BConv2D", {"x", "w", "mul", "add", ""}, {"y"}, attrs);
    } else {
      int64_t kic = c.op == "dwconv" ? 1 : c.ic / c.groups;
      int64_t oc = c.op == "dwconv" ? c.ic : c.oc;
      std::vector<float> wv(static_cast<size_t>(oc * c.k * c.k * kic));
      for (auto& v : wv) v = nd(rng) * 0.1f;
      mb.constant("w", DType::kF32, {oc, c.k, c.k, kic}, wv.data());
      std::vector<float> b(static_cast<size_t>(oc));
      mb.constant("b", DType::kF32, {oc}, b.data());
      if (c.op == "dwconv") attrs["groups"] = Json(c.ic);
      else if (c.groups > 1) attrs["groups"] = Json(c.groups);
      mb.node("Conv2D", {"x", "w", "b"}, {"y"}, attrs);
    }
  } else if (c.op == "dense" || c.op == "bdense") {
    mb.input("x", DType::kF32, {c.h, c.ic});  // h = rows
    if (c.op == "dense") {
      std::vector<float> wv(static_cast<size_t>(c.oc * c.ic));
      for (auto& v : wv) v = nd(rng) * 0.1f;
      mb.constant("w", DType::kF32, {c.oc, c.ic}, wv.data());
      mb.constant("b", DType::kF32, {c.oc}, bias.data());
      mb.node("Dense", {"x", "w", "b"}, {"y"});
    } else {
      std::vector<int8_t> wv(static_cast<size_t>(c.oc * c.ic));
      for (auto& v : wv) v = (rng() & 1) ? 1 : -1;
      auto wp = bftest::pack_pm1(wv, c.oc, c.ic);
      mb.constant("w", DType::kBit, {c.oc, c.ic}, wp.data());
      mb.constant("mul", DType::kF32, {c.oc}, mul.data());
      mb.constant("add", DType::kF32, {c.oc}, add.data());
      mb.node("BDense", {"x", "w", "mul", "add", ""}, {"y"});
    }
  } else {
    fail("unknown --op ", c.op);
  }
  mb.output("y");
  return mb.build();
}

double macs(const Cfg& c) {
  int64_t p = c.pad == "same" ? c.k / 2 : 0;
  int64_t oh = (c.h + 2 * p - c.k) / c.stride + 1, ow = (c.w + 2 * p - c.k) / c.stride + 1;
  if (c.op == "bconv" || c.op == "conv") return double(oh * ow) * c.oc * c.k * c.k * c.ic / c.groups;
  if (c.op == "dwconv") return double(oh * ow) * c.ic * c.k * c.k;
  return double(c.h) * c.oc * c.ic;
}

std::vector<Cfg> suite(const std::string& name) {
  std::vector<Cfg> s;
  auto bc = [&](int64_t h, int64_t ch, int64_t oc, int64_t k, int64_t st, int pv) {
    Cfg c; c.op = "bconv"; c.h = c.w = h; c.ic = ch; c.oc = oc; c.k = k; c.stride = st; c.pad_value = pv; s.push_back(c);
  };
  auto fc = [&](const char* op, int64_t h, int64_t ch, int64_t oc, int64_t k, int64_t st) {
    Cfg c; c.op = op; c.h = c.w = h; c.ic = ch; c.oc = oc; c.k = k; c.stride = st; c.pad = k == 1 ? "valid" : "same"; s.push_back(c);
  };
  if (name == "quicknet") {  // QuickNet(-Small) layer shapes, 224x224 input
    bc(56, 32, 32, 3, 1, 1); bc(56, 64, 64, 3, 1, 1); bc(28, 64, 64, 3, 1, 1); bc(28, 128, 128, 3, 1, 1);
    bc(14, 256, 256, 3, 1, 1); bc(7, 512, 512, 3, 1, 1);
    fc("conv", 224, 3, 8, 3, 2); fc("dwconv", 112, 8, 8, 3, 2); fc("conv", 56, 8, 32, 1, 1);
    fc("dwconv", 28, 64, 64, 3, 2); fc("conv", 14, 128, 256, 1, 1); fc("conv", 7, 256, 512, 1, 1);
    Cfg d; d.op = "dense"; d.h = 1; d.ic = 512; d.oc = 1000; s.push_back(d);
  } else if (name == "birealnet") {
    bc(56, 64, 64, 3, 1, 0); bc(56, 64, 128, 3, 2, 0); bc(28, 128, 128, 3, 1, 0); bc(14, 256, 256, 3, 1, 0); bc(7, 512, 512, 3, 1, 0);
    fc("conv", 224, 3, 64, 7, 2); fc("conv", 28, 64, 128, 1, 1);
    Cfg d; d.op = "dense"; d.h = 1; d.ic = 512; d.oc = 1000; s.push_back(d);
  } else {
    fail("unknown suite ", name);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  Cfg cfg;
  std::string suite_name, tiers, force;
  int threads = 1, warmup = 5, iters = 30;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--op") cfg.op = nx();
    else if (a == "--h") cfg.h = std::atoll(nx().c_str());
    else if (a == "--w") cfg.w = std::atoll(nx().c_str());
    else if (a == "--ic") cfg.ic = std::atoll(nx().c_str());
    else if (a == "--oc") cfg.oc = std::atoll(nx().c_str());
    else if (a == "--k") cfg.k = std::atoll(nx().c_str());
    else if (a == "--stride") cfg.stride = std::atoll(nx().c_str());
    else if (a == "--groups") cfg.groups = std::atoll(nx().c_str());
    else if (a == "--pad") cfg.pad = nx();
    else if (a == "--pad-value") cfg.pad_value = std::atoi(nx().c_str());
    else if (a == "--bit-out") cfg.bit_out = true;
    else if (a == "--suite") suite_name = nx();
    else if (a == "--tiers") tiers = nx();
    else if (a == "--force") force = nx();
    else if (a == "--threads") threads = std::atoi(nx().c_str());
    else if (a == "--warmup") warmup = std::atoi(nx().c_str());
    else if (a == "--iterations") iters = std::atoi(nx().c_str());
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (cfg.w != cfg.h && suite_name.empty() && cfg.op.find("dense") == std::string::npos) { /* rectangular allowed */ }
  std::vector<Cfg> cfgs = suite_name.empty() ? std::vector<Cfg>{cfg} : suite(suite_name);
  try {
    for (const Cfg& c : cfgs) {
      SessionOptions o;
      o.num_threads = threads;
      for (auto& t : split(tiers, ',')) o.policy.allowed_tiers.insert(t);
      if (!force.empty()) {
        std::string op = c.op == "bconv" ? "BConv2D" : c.op == "bdense" ? "BDense" : c.op == "dense" ? "Dense" : "Conv2D";
        o.policy.force[op] = force;
      }
      auto model = build(c);
      Session s(model, o);
      Tensor& x = s.input(0);
      for (int64_t k = 0; k < x.size(); ++k) x.data<float>()[k] = std::sin(0.1f * k);
      for (int i = 0; i < warmup; ++i) s.run();
      std::vector<double> lat;
      for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        s.run();
        lat.push_back(now_ms() - t0);
      }
      std::sort(lat.begin(), lat.end());
      double med = lat[lat.size() / 2], mean = 0;
      for (double v : lat) mean += v;
      mean /= lat.size();
      std::printf("{\"op\":\"%s\",\"h\":%lld,\"w\":%lld,\"ic\":%lld,\"oc\":%lld,\"k\":%lld,\"stride\":%lld,\"pad_value\":%d,"
                  "\"threads\":%d,\"kernel\":\"%s\",\"median_ms\":%.4f,\"mean_ms\":%.4f,\"min_ms\":%.4f,\"gops\":%.2f,\"cpu\":\"%s\"}\n",
                  c.op.c_str(), (long long)c.h, (long long)c.w, (long long)c.ic, (long long)c.oc, (long long)c.k, (long long)c.stride,
                  c.pad_value, threads, s.kernel_choices().begin()->second.c_str(), med, mean, lat.front(),
                  2.0 * macs(c) / (med * 1e6), cpu_info().describe().c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

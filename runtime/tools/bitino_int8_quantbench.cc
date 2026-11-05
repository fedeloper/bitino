// bitino_int8_quantbench: cost of the per-tensor activation quantization
// (max-abs reduction + quantize) done by the int8 Conv2D/Dense kernels at
// every run(), isolated from im2col and GEMM.
//
//   bitino_int8_quantbench [--suite quicknet|birealnet|all | --n ELEMS] [--threads N]
//                            [--warmup W] [--iterations I]
//
// For each input size prints one JSON line with the median ms of
//   "scalar": the int8_reference path (scalar loops, single thread)
//   "neon":   int8_neon::quantize_tensor on an N-thread pool (AArch64 only)
// and checks that both produce identical bytes and scale.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "bitino/thread_pool.h"
#include "bitino/util.h"
#include "kernels/int8_neon/int8_gemm.h"
#include "kernels/int8_quant.h"

using namespace bitino;

namespace {

struct Size {
  std::string label;
  int64_t n;
};

std::vector<Size> suite(const std::string& name) {
  // Input tensors of the float Conv2D / Dense rows of bitino_microbench.
  if (name == "quicknet")
    return {{"conv 224x224x3", 224 * 224 * 3}, {"conv 56x56x8", 56 * 56 * 8}, {"conv 14x14x128", 14 * 14 * 128},
            {"conv 7x7x256", 7 * 7 * 256}, {"dense 512", 512}};
  if (name == "birealnet") return {{"conv 224x224x3", 224 * 224 * 3}, {"conv 28x28x64", 28 * 28 * 64}, {"dense 512", 512}};
  if (name == "all")
    return {{"conv 224x224x3", 224 * 224 * 3}, {"conv 56x56x8", 56 * 56 * 8}, {"conv 28x28x64", 28 * 28 * 64},
            {"conv 14x14x128", 14 * 14 * 128}, {"conv 7x7x256", 7 * 7 * 256}, {"dense 512", 512}};
  std::fprintf(stderr, "unknown suite %s\n", name.c_str());
  std::exit(2);
}

float scalar_quantize(const float* x, int64_t n, int8_t* q) {
  float m = 0.f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
  float sx = int8_scale(m);
  for (int64_t i = 0; i < n; ++i) q[i] = int8_quantize(x[i], sx);
  return sx;
}

template <typename F>
double median_ms(int warmup, int iters, F&& f) {
  for (int i = 0; i < warmup; ++i) f();
  std::vector<double> t;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_ms();
    f();
    t.push_back(now_ms() - t0);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  std::string suite_name = "quicknet";
  int64_t n_arg = 0;
  int threads = 1, warmup = 20, iters = 200;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--suite") suite_name = nx();
    else if (a == "--n") n_arg = std::atoll(nx().c_str());
    else if (a == "--threads") threads = std::atoi(nx().c_str());
    else if (a == "--warmup") warmup = std::atoi(nx().c_str());
    else if (a == "--iterations") iters = std::atoi(nx().c_str());
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  std::vector<Size> sizes;
  if (n_arg > 0) sizes.push_back(Size{"n", n_arg});
  else sizes = suite(suite_name);
  ThreadPool pool(threads);
  for (const Size& s : sizes) {
    std::vector<float> x(static_cast<size_t>(s.n));
    for (int64_t k = 0; k < s.n; ++k) x[k] = std::sin(0.1f * k) * 3.f;
    std::vector<int8_t> q0(x.size()), q1(x.size());
    float sx0 = 0.f;
    double t_scalar = median_ms(warmup, iters, [&] { sx0 = scalar_quantize(x.data(), s.n, q0.data()); });
    double t_neon = -1;
    bool same = true;
#if defined(__aarch64__)
    int64_t parts = int8_neon::quant_parts(s.n);
    std::vector<float> partial(static_cast<size_t>(parts));
    float sx1 = 0.f;
    t_neon = median_ms(warmup, iters, [&] { sx1 = int8_neon::quantize_tensor(x.data(), s.n, q1.data(), partial.data(), parts, pool); });
    same = sx0 == sx1 && std::memcmp(q0.data(), q1.data(), q0.size()) == 0;
#endif
    std::printf("{\"tensor\":\"%s\",\"elems\":%lld,\"threads\":%d,\"scalar_ms\":%.4f,\"neon_ms\":%.4f,\"bit_exact\":%s}\n",
                s.label.c_str(), (long long)s.n, threads, t_scalar, t_neon, same ? "true" : "false");
    if (!same) return 1;
  }
  return 0;
}

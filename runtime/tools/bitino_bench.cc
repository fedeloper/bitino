// bitino_bench: native benchmark + correctness runner (host and Android).
//
//   bitino_bench --model m.bitino [--threads N] [--warmup W] [--iterations I]
//                  [--stage NAME | --tiers a,b,c] [--force op=kernel,...]
//                  [--affinity 4,5,6,7] [--json out.json] [--profile]
//                  [--inputs samples.bin --outputs outputs.bin]
//                  (multi-input models: each sample record is the concatenation of all
//                   graph inputs' raw bytes in graph input order)
//                  [--no-verify-hash] [--no-reuse-memory] [--no-fuse] [--no-epilogues] [--no-bit-outputs] [--label TEXT]
//
// Timing covers Session::run() only. Model load, session creation (kernel
// selection, weight prepacking) and correctness I/O are reported separately.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "bitino/bitino.h"
#include "bitino/stages.h"
#include "bitino/util.h"

using namespace bitino;

namespace {

#ifndef BITINO_BUILD_FLAGS
#define BITINO_BUILD_FLAGS "unknown"
#endif
#ifndef BITINO_BUILD_TYPE
#define BITINO_BUILD_TYPE "unknown"
#endif

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep))
    if (!item.empty()) out.push_back(item);
  return out;
}

std::string read_text(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

std::string run_cmd(const char* cmd) {
  std::string out;
  if (FILE* p = popen(cmd, "r")) {
    char buf[256];
    while (fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

// Reads a "VmRSS:" style field (kB) from /proc/self/status; -1 if unavailable.
int64_t proc_status_kb(const char* field) {
  std::ifstream f("/proc/self/status");
  std::string line;
  size_t n = std::strlen(field);
  while (std::getline(f, line))
    if (line.compare(0, n, field) == 0) return std::atoll(line.c_str() + n + 1);
  return -1;
}

Json cpu_freq_snapshot() {
  Json::Array out;
  for (int c = 0; c < 64; ++c) {
    std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/";
    std::string cur = read_text(base + "scaling_cur_freq");
    if (cur.empty()) {
      if (read_text("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/online").empty() && c > 0 &&
          read_text(base + "cpuinfo_max_freq").empty())
        break;
      continue;
    }
    out.push_back(Json(Json::Object{{"cpu", c}, {"cur_khz", Json(static_cast<int64_t>(std::atoll(cur.c_str())))},
                                    {"max_khz", Json(static_cast<int64_t>(std::atoll(read_text(base + "cpuinfo_max_freq").c_str())))}}));
  }
  return Json(out);
}

Json thermal_snapshot() {
  Json::Object out;
  for (int z = 0; z < 128; ++z) {
    std::string base = "/sys/class/thermal/thermal_zone" + std::to_string(z) + "/";
    std::string type = read_text(base + "type");
    if (type.empty()) continue;
    std::string temp = read_text(base + "temp");
    if (temp.empty()) continue;
    // Keep CPU/SoC-related zones only to keep reports small.
    if (type.find("cpu") == std::string::npos && type.find("soc") == std::string::npos && type != "AP" && type != "x86_pkg_temp")
      continue;
    out[type + "#" + std::to_string(z)] = Json(std::atof(temp.c_str()) / 1000.0);
  }
  return Json(out);
}

Json device_info() {
  Json::Object d;
  d["cpu_features"] = cpu_info().describe();
#if defined(__ANDROID__)
  d["model"] = run_cmd("getprop ro.product.model");
  d["soc"] = run_cmd("getprop ro.soc.model");
  d["android_release"] = run_cmd("getprop ro.build.version.release");
  d["api_level"] = run_cmd("getprop ro.build.version.sdk");
  d["build_fingerprint"] = run_cmd("getprop ro.build.fingerprint");
  d["thermal_status"] = run_cmd("dumpsys thermalservice 2>/dev/null | grep 'Thermal Status' | head -1");
#else
  d["hostname"] = run_cmd("hostname");
  d["cpu_model"] = run_cmd("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2");
#endif
  d["kernel"] = run_cmd("uname -r");
  std::string parts = run_cmd("grep 'CPU part' /proc/cpuinfo | awk '{print $4}' | tr '\\n' ' '");
  if (!parts.empty()) d["cpu_parts"] = parts;
  d["online_cpus"] = read_text("/sys/devices/system/cpu/online");
  return Json(d);
}

Json stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) {
    if (v.empty()) return 0.0;
    double idx = p * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx)), hi = static_cast<size_t>(std::ceil(idx));
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
  };
  double mean = 0;
  for (double x : v) mean += x;
  mean /= std::max<size_t>(v.size(), 1);
  double var = 0;
  for (double x : v) var += (x - mean) * (x - mean);
  double sd = v.size() > 1 ? std::sqrt(var / (v.size() - 1)) : 0.0;
  Json::Object o{{"n", Json(v.size())}, {"median_ms", pct(0.5)}, {"mean_ms", mean}, {"stddev_ms", sd},
                 {"min_ms", v.empty() ? 0.0 : v.front()}, {"max_ms", v.empty() ? 0.0 : v.back()},
                 {"throughput_per_s", pct(0.5) > 0 ? 1000.0 / pct(0.5) : 0.0}};
  if (v.size() >= 20) o["p90_ms"] = pct(0.9);
  if (v.size() >= 40) o["p95_ms"] = pct(0.95);
  return Json(o);
}

std::string iso_time() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

// Busy pre-warm of the --affinity cores before the session pins its threads. Android's core control pauses idle
// cores (OnePlus Open: the Cortex-X3 at low load) and sched_setaffinity to a paused core fails; a model of ~0.02 ms
// never loads the phone long enough for the core to be woken, so every repeat stayed unpinned. One busy thread per
// requested CPU, unpinned, retries the pin every 0.5 ms until all of them hold their core or max_ms passed; the
// session is created right after, while the cores are awake.
struct Prewarm {
  double ms = 0;
  int cpus = 0, pinned = 0;
};

Prewarm prewarm_cores(std::vector<int> cpus, double max_ms) {
  Prewarm r;
#if defined(__linux__)
  std::sort(cpus.begin(), cpus.end());
  cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
  r.cpus = static_cast<int>(cpus.size());
  std::atomic<int> ok{0};
  std::atomic<bool> stop{false};
  const double t0 = now_ms();
  std::vector<std::thread> th;
  for (int c : cpus)
    th.emplace_back([&, c] {
      bool pinned = false;
      double last = -1;
      volatile uint64_t sink = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 2000; ++i) sink = sink + static_cast<uint64_t>(i);
        const double t = now_ms();
        if (!pinned && t - last >= 0.5) {
          last = t;
          cpu_set_t set;
          CPU_ZERO(&set);
          CPU_SET(c, &set);
          if (sched_setaffinity(0, sizeof(set), &set) == 0) {
            pinned = true;
            ok.fetch_add(1);
          }
        }
      }
    });
  while (ok.load() < r.cpus && now_ms() - t0 < max_ms) std::this_thread::sleep_for(std::chrono::microseconds(200));
  stop.store(true);
  for (auto& t : th) t.join();
  r.pinned = ok.load();
  r.ms = now_ms() - t0;
#else
  (void)cpus;
  (void)max_ms;
#endif
  return r;
}

void usage() {
  std::fprintf(stderr,
               "usage: bitino_bench --model PATH [--threads N] [--warmup W] [--iterations I]\n"
               "       [--stage reference|packed_naive|binary_scalar|neon|int8_reference|int8_neon|optimized|optimized_int8|int8_full_reference|int8_full|all]\n"
               "       [--tiers t1,t2] [--force op=kernel,...] [--affinity c0,c1,...] [--json PATH]\n"
               "       [--profile] [--inputs PATH --outputs PATH] [--no-verify-hash] [--no-reuse-memory] [--no-fuse] [--no-epilogues] [--no-bit-outputs]\n"
               "       [--label TEXT] [--list-kernels] [--spin-us N] [--spin-run | --no-spin-run] [--no-map] [--no-release-constants] [--prewarm-ms MS] [--no-auto-threads] [--tune-before-inputs] [--dump-dir DIR --dump-sample K (with --inputs, --no-reuse-memory)]\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string dump_dir;
  int dump_sample = 0;
  int spin_us = 0;
  bool spin_run = true;  // SessionOptions default; --no-spin-run turns it off
  bool map_file = true, release_constants = true;  // --no-map: read the model into a heap copy
  double prewarm_ms = 1000;
  bool auto_threads = true;  // SessionOptions::auto_threads (--no-auto-threads)
  bool tune_first = false;   // --tune-before-inputs: finish the per-node thread tuning before the correctness pass  // with --affinity: busy pre-warm of the cores before pinning (0 = off)
  std::string model_path, json_path, inputs_path, outputs_path, stage = "all", tiers_arg, force_arg, affinity_arg, label;
  int threads = 1, warmup = 10, iterations = 50;
  bool profile = false, verify_hash = true, reuse = true, list_kernels = false, fuse = true, host_epilogues = true, bit_outputs = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model_path = next();
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--warmup") warmup = std::atoi(next().c_str());
    else if (a == "--iterations") iterations = std::atoi(next().c_str());
    else if (a == "--json") json_path = next();
    else if (a == "--stage") stage = next();
    else if (a == "--tiers") tiers_arg = next();
    else if (a == "--force") force_arg = next();
    else if (a == "--affinity") affinity_arg = next();
    else if (a == "--inputs") inputs_path = next();
    else if (a == "--outputs") outputs_path = next();
    else if (a == "--label") label = next();
    else if (a == "--dump-dir") dump_dir = next();
    else if (a == "--spin-us") spin_us = std::atoi(next().c_str());
    else if (a == "--spin-run") spin_run = true;
    else if (a == "--no-spin-run") spin_run = false;
    else if (a == "--dump-sample") dump_sample = std::atoi(next().c_str());
    else if (a == "--profile") profile = true;
    else if (a == "--no-verify-hash") verify_hash = false;
    else if (a == "--no-reuse-memory") reuse = false;
    else if (a == "--no-map") map_file = false;
    else if (a == "--no-release-constants") release_constants = false;
    else if (a == "--prewarm-ms") prewarm_ms = std::atof(next().c_str());
    else if (a == "--no-auto-threads") auto_threads = false;
    else if (a == "--tune-before-inputs") tune_first = true;
    else if (a == "--no-fuse") fuse = false;
    else if (a == "--no-epilogues") host_epilogues = false;
    else if (a == "--no-bit-outputs") bit_outputs = false;
    else if (a == "--list-kernels") list_kernels = true;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 2; }
  }
  if (list_kernels) {
    for (const auto& k : OpRegistry::global().all_kernels())
      std::printf("%-16s %-32s tier=%-16s prio=%d available=%s\n", k.op.c_str(), k.name.c_str(), k.tier.c_str(), k.priority,
                  cpu_info().has(k.required_features) ? "yes" : "no");
    return 0;
  }
  if (model_path.empty() || threads < 1 || iterations < 1 || warmup < 0) {
    usage();
    return 2;
  }

  try {
    SessionOptions opts;
    opts.num_threads = threads;
    opts.profile = profile;
    opts.reuse_memory = reuse;
    opts.fuse = fuse;
    opts.host_epilogues = host_epilogues;
    opts.bit_outputs = bit_outputs;
    opts.spin_us = spin_us;
    opts.spin_run = spin_run;
    opts.auto_threads = auto_threads;
    if (!tiers_arg.empty()) {
      for (auto& t : split(tiers_arg, ',')) opts.policy.allowed_tiers.insert(t);
      stage = "custom:" + tiers_arg;
    } else {
      auto it = stage_tiers().find(stage);
      if (it == stage_tiers().end()) {
        std::fprintf(stderr, "unknown stage '%s'\n", stage.c_str());
        return 2;
      }
      opts.policy.allowed_tiers = it->second;
    }
    for (auto& f : split(force_arg, ',')) {
      auto kv = split(f, '=');
      if (kv.size() != 2) { std::fprintf(stderr, "bad --force entry '%s'\n", f.c_str()); return 2; }
      opts.policy.force[kv[0]] = kv[1];
    }
    for (auto& c : split(affinity_arg, ',')) opts.affinity.push_back(std::atoi(c.c_str()));

    int64_t rss_start = proc_status_kb("VmRSS:");
    Json freq_before = cpu_freq_snapshot();
    Json thermal_before = thermal_snapshot();
    Model::LoadOptions lo;
    lo.verify_hash = verify_hash;
    lo.map_file = map_file;
    opts.release_constants = release_constants;
    double t_load0 = now_ms();
    auto model = Model::load(model_path, lo);
    double load_ms = now_ms() - t_load0;
    int64_t rss_after_load = proc_status_kb("VmRSS:");
    BF_CHECK(dump_dir.empty() || !reuse, "--dump-dir requires --no-reuse-memory");
    Prewarm prewarm;
    if (!opts.affinity.empty() && prewarm_ms > 0) prewarm = prewarm_cores(opts.affinity, prewarm_ms);
    Session session(model, opts);
    int64_t rss_after_session = proc_status_kb("VmRSS:");

    // --tune-before-inputs: the correctness pass then runs with the tuned per-node thread counts (tests of thread-count
    // identity); the tuning runs see the synthetic input.
    int tune_runs = 0;
    while (tune_first && session.tuning_threads() && tune_runs < 100000) {
      session.run();
      ++tune_runs;
    }
    // Correctness pass: run every sample once, outside of timing.
    int64_t num_samples = 0;
    if (!inputs_path.empty()) {
      // Sample record = the raw bytes of every graph input, in graph input order
      // (e.g. BERT: input_ids i32 | token_type_ids i32 | attention_mask f32).
      // Single-input models: plain concatenation of input tensors, as before.
      auto in = read_file(inputs_path);
      size_t per = 0;
      for (size_t i = 0; i < session.num_inputs(); ++i) per += session.input(i).bytes();
      BF_CHECK(per > 0 && in.size() % per == 0, "inputs file size ", in.size(), " is not a multiple of the per-sample input size ", per);
      num_samples = static_cast<int64_t>(in.size() / per);
      std::string out;
      // A dump run whose outputs are discarded (tools/validate_model.py --explain) only
      // needs the dumped sample: skip the others (runs are independent of each other).
      const bool only_dump = !dump_dir.empty() && (outputs_path.empty() || outputs_path == "/dev/null");
      for (int64_t s = 0; s < num_samples; ++s) {
        if (only_dump && s != dump_sample) continue;
        size_t off = static_cast<size_t>(s) * per;
        for (size_t i = 0; i < session.num_inputs(); ++i) {
          std::memcpy(session.input(i).raw(), in.data() + off, session.input(i).bytes());
          off += session.input(i).bytes();
        }
        session.run();
        if (!dump_dir.empty() && s == dump_sample) {
          // All non-constant tensors of this sample (requires --no-reuse-memory).
          Json index = Json::Object{};
          int k = 0;
          for (const auto& name : session.tensor_names()) {
            const Tensor& t = session.tensor(name);
            if (t.empty() || model->graph().tensors[model->graph().find(name)].is_constant) continue;
            std::string file = "t" + std::to_string(k++) + ".bin";
            write_file(dump_dir + "/" + file, std::string(static_cast<const char*>(t.raw()), t.bytes()));
            Json::Array sh;
            for (auto d : t.shape()) sh.push_back(Json(d));
            index[name] = Json(Json::Object{{"file", file}, {"dtype", dtype_name(t.dtype())}, {"shape", Json(sh)}});
          }
          write_file(dump_dir + "/index.json", index.dump(1));
        }
        for (size_t o = 0; o < session.num_outputs(); ++o)
          out.append(static_cast<const char*>(session.output(o).raw()), session.output(o).bytes());
      }
      if (!outputs_path.empty()) write_file(outputs_path, out);
      std::fprintf(stderr, "ran %lld samples\n", static_cast<long long>(num_samples));
    } else {
      // Deterministic synthetic input for timing-only runs (non-f32 inputs, e.g.
      // i32 token ids, stay zero = a valid index).
      for (size_t i = 0; i < session.num_inputs(); ++i) {
        Tensor& t = session.input(i);
        if (t.dtype() == DType::kF32)
          for (int64_t k = 0; k < t.size(); ++k) t.data<float>()[k] = static_cast<float>(std::sin(0.37 * k));
      }
    }

    // Warmup, extended while the session still times its per-node thread counts (SessionOptions::auto_threads), so
    // the timed iterations never include tuning runs.
    int warmup_runs = 0;
    while (warmup_runs < warmup || (session.tuning_threads() && warmup_runs < warmup + 100000)) {
      session.run();
      ++warmup_runs;
    }
    session.reset_profile();
    session.pool().reset_stats();
    std::vector<double> lat;
    lat.reserve(iterations);
    const uint64_t calls0 = session.pool().calls(), disp0 = session.pool().dispatches();
    double t_all0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
      double t0 = now_ms();
      session.run();
      lat.push_back(now_ms() - t0);
    }
    double wall_ms = now_ms() - t_all0;
    const double calls_per_run = static_cast<double>(session.pool().calls() - calls0) / iterations;
    const double disp_per_run = static_cast<double>(session.pool().dispatches() - disp0) / iterations;
    int64_t rss_after_run = proc_status_kb("VmRSS:");

    Json report = Json::Object{};
    report["schema"] = "bitino.benchmark.v1";
    report["timestamp_utc"] = iso_time();
    report["label"] = label;
    report["bitino_version"] = BITINO_VERSION;
    report["build"] = Json(Json::Object{{"compiler", __VERSION__}, {"flags", BITINO_BUILD_FLAGS}, {"type", BITINO_BUILD_TYPE}});
    report["device"] = device_info();
    Json m = Json::Object{};
    m["path"] = model_path;
    m["data_sha256"] = model->data_sha256();
    m["hash_verified"] = model->hash_verified();
    m["file_bytes"] = Json(model->file_bytes());
    m["constant_bytes"] = Json(model->constant_bytes());
    m["manifest"] = model->manifest();
    Json::Array ins;
    for (size_t i = 0; i < session.num_inputs(); ++i) {
      Json::Array sh;
      for (auto d : session.input(i).shape()) sh.push_back(Json(d));
      ins.push_back(Json(Json::Object{{"name", session.input_name(i)}, {"shape", Json(sh)}, {"dtype", dtype_name(session.input(i).dtype())}}));
    }
    m["inputs"] = Json(ins);
    report["model"] = m;
    Json cfg = Json::Object{};
    cfg["stage"] = stage;
    cfg["threads"] = threads;
    cfg["pool_threads"] = session.num_threads();
    cfg["affinity"] = affinity_arg;
    cfg["affinity_errors"] = session.pool().affinity_errors();
    cfg["unpinned_threads_after"] = session.pool().unpinned_threads();  // pinning still failing after the runs (paused cores)
    if (!opts.affinity.empty())
      cfg["prewarm"] = Json(Json::Object{{"max_ms", Json(prewarm_ms)}, {"ms", Json(prewarm.ms)}, {"cpus", Json(prewarm.cpus)},
                                         {"pinned", Json(prewarm.pinned)}});
    cfg["warmup"] = warmup;
    cfg["warmup_runs"] = warmup_runs;  // warmup plus the runs the thread tuning needed beyond it
    {
      std::map<std::string, int> hist;  // "<threads>" (weighted claims) or "<threads>e" (equal chunks)
      for (size_t i = 0; i < session.node_threads().size(); ++i)
        hist[std::to_string(session.node_threads()[i]) +
             (session.node_weighted()[i] || session.node_threads()[i] == 1 ? "" : "e")]++;
      Json h = Json::Object{};
      for (const auto& kv : hist) h[kv.first] = kv.second;
      cfg["auto_threads"] = Json(Json::Object{{"enabled", Json(auto_threads)}, {"tuning_runs", Json(session.tuning_runs())},
                                              {"tuned_run_ms", Json(session.tuned_run_ms())},
                                              {"all_threads_run_ms", Json(session.all_threads_run_ms())},
                                              {"node_threads", h}});
    }
    cfg["iterations"] = iterations;
    cfg["reuse_memory"] = reuse;
    cfg["fuse"] = fuse;
    cfg["host_epilogues"] = host_epilogues;
    cfg["bit_outputs"] = bit_outputs;
    cfg["plan_rewrites"] = Json(Json::Object{{"fused_nodes", Json(session.fused_nodes())},
                                             {"absorbed_nodes", Json(session.absorbed_nodes())},
                                             {"folded_thresholds", Json(session.folded_thresholds())},
                                             {"runtime_thresholds", Json(session.runtime_thresholds())},
                                             {"hosted_epilogues", Json(session.hosted_epilogues())},
                                             {"bit_outputs", Json(session.bit_outputs())},
                                             {"shared_binarizations", Json(session.shared_binarizations())},
                                             {"aliased_reshapes", Json(session.aliased_reshapes())},
                                             {"i8_absorbed_luts", Json(session.absorbed_luts())},
                                             {"i8_composed_luts", Json(session.i8_rewrites()[0])},
                                             {"i8_folded_luts", Json(session.i8_rewrites()[1])},
                                             {"i8_interval_folds", Json(session.i8_rewrites()[6])},
                                             {"i8_removed_luts", Json(session.i8_rewrites()[2])},
                                             {"i8_lut_twins", Json(session.i8_rewrites()[3])},
                                             {"i8_add_twins", Json(session.i8_rewrites()[4])},
                                             {"i8_merged_nodes", Json(session.i8_rewrites()[5])},
                                             {"i8_bit_outputs", Json(session.i8_rewrites()[7])}});
    cfg["spin_us"] = spin_us;
    cfg["spin_run"] = spin_run;
    cfg["nodes"] = static_cast<int64_t>(session.kernel_choices().size());
    cfg["parallel_for_per_run"] = calls_per_run;      // structural: parallel regions per inference
    cfg["pool_dispatches_per_run"] = disp_per_run;    // of which handed to the workers (barriers)
    cfg["profile"] = profile;
    if (session.pool().stats_enabled()) {  // BITINO_POOL_STATS=1: per-run averages over the timed iterations
      const ThreadPool::Stats ps = session.pool().stats();
      Json::Array busy, items, chunks;
      for (size_t t = 0; t < ps.busy_us.size(); ++t) {
        busy.push_back(Json(ps.busy_us[t] / iterations));
        items.push_back(Json(static_cast<double>(ps.items[t]) / iterations));
        chunks.push_back(Json(static_cast<double>(ps.chunks[t]) / iterations));
      }
      cfg["pool_stats"] = Json(Json::Object{{"regions", Json(static_cast<double>(ps.regions) / iterations)},
                                            {"region_us", Json(ps.region_us / iterations)},
                                            {"tail_us", Json(ps.tail_us / iterations)},
                                            {"inline_regions", Json(static_cast<double>(ps.inline_regions) / iterations)},
                                            {"inline_us", Json(ps.inline_us / iterations)},
                                            {"busy_us", Json(busy)}, {"items", Json(items)}, {"chunks", Json(chunks)}});
    }
    report["config"] = cfg;
    Json kernels = Json::Object{};
    std::map<std::string, int> kernel_counts;
    for (const auto& kv : session.kernel_choices()) kernel_counts[kv.second]++;
    for (const auto& kv : kernel_counts) kernels[kv.first] = kv.second;
    report["kernels"] = kernels;
    Json timing = stats(lat);
    timing["load_ms"] = load_ms;
    timing["session_prepare_ms"] = session.prepare_ms();
    timing["timed_wall_ms"] = wall_ms;
    Json::Array raw;
    for (double v : lat) raw.push_back(Json(v));
    timing["samples_ms"] = Json(raw);
    report["latency"] = timing;
    report["memory"] = Json(Json::Object{{"rss_start_kb", Json(rss_start)}, {"rss_after_load_kb", Json(rss_after_load)},
                                         {"rss_after_session_kb", Json(rss_after_session)}, {"rss_after_run_kb", Json(rss_after_run)},
                                         {"vm_hwm_kb", Json(proc_status_kb("VmHWM:"))},
                                         {"activation_bytes", Json(session.activation_bytes())},
                                         {"prepacked_bytes", Json(session.prepacked_bytes())},
                                         {"released_constant_bytes", Json(session.released_bytes())},
                                         {"model_mapped", Json(model->mapped())}});
    report["cpu_freq_before"] = freq_before;
    report["cpu_freq_after"] = cpu_freq_snapshot();
    report["thermal_before_c"] = thermal_before;
    report["thermal_after_c"] = thermal_snapshot();
    if (num_samples) report["correctness_samples"] = Json(num_samples);
    if (profile) {
      std::map<std::string, std::pair<double, std::string>> by_op;
      Json::Array nodes;
      for (const auto& p : session.profile()) {
        double per_call = p.calls ? p.total_ms / p.calls : 0.0;
        nodes.push_back(Json(Json::Object{{"node", p.name}, {"op", p.op}, {"kernel", p.kernel}, {"ms", per_call}, {"elems", Json(p.out_elems)}}));
        by_op[p.op].first += per_call;
      }
      Json ops = Json::Object{};
      for (const auto& kv : by_op) ops[kv.first] = kv.second.first;
      report["profile"] = Json(Json::Object{{"per_op_ms", ops}, {"nodes", Json(nodes)}});
    }

    std::printf("model=%s stage=%s threads=%d median=%.3fms mean=%.3fms min=%.3fms (n=%d, warmup=%d) prepare=%.1fms\n",
                model_path.c_str(), stage.c_str(), threads, timing["median_ms"].as_number(), timing["mean_ms"].as_number(),
                timing["min_ms"].as_number(), iterations, warmup, session.prepare_ms());
    if (!json_path.empty()) write_file(json_path, report.dump(1));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bitino_bench: error: %s\n", e.what());
    return 1;
  }
}

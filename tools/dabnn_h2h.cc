// Head-to-head timing and output driver for daBNN (JDAI-CV/dabnn), linked against daBNN's own static library
// (libdabnn.a built from the unmodified daBNN sources). Not part of daBNN. Used by tools/h2h_dabnn.py, method in
// docs/comparison-dabnn.md. It mirrors what bitino_bench does around the network so the two numbers are comparable:
//   load (Net::read: mmap + daBNN's own prepare/optimize), 1 untimed run on input sample 0 whose output is written,
//   --warmup more untimed runs, --iterations timed runs (steady_clock around Net::run, one sample per run),
//   per-run statistics (mean, median, min, max, std), cpufreq of every CPU before and after the timed loop.
// Net flags are the ones daBNN's benchmark_single_model and its `run` binary use: optimize, run_fconv, strict = true.
// With --eval N the driver instead runs N samples of the input file once each (no timing) and writes N outputs
// (correctness check on the 128-image eval set).
//
//   dabnn_h2h --model m.dab --inputs in.f32 --blob 188 --outputs out.f32 --warmup 9 --iterations 50 --json r.json
//   dabnn_h2h --model m.dab --inputs eval128.f32 --blob 188 --outputs out.f32 --eval 128
// inputs: float32 NHWC [n,224,224,3] records (the layout of daBNN's input blob).
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <dabnn/net.h>

namespace {
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string read_small(const std::string &path) {
    std::ifstream f(path);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

std::string cpufreq_json() {
    std::ostringstream o;
    o << "{";
    const long n = sysconf(_SC_NPROCESSORS_CONF);
    for (long c = 0; c < n; ++c) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/";
        const std::string cur = read_small(base + "scaling_cur_freq");
        const std::string mx = read_small(base + "scaling_max_freq");
        o << (c ? "," : "") << "\"" << c << "\":{\"cur_khz\":" << (cur.empty() ? "null" : cur)
          << ",\"max_khz\":" << (mx.empty() ? "null" : mx) << "}";
    }
    o << "}";
    return o.str();
}

std::string affinity_list() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return "?";
    std::string s;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) s += (s.empty() ? "" : ",") + std::to_string(c);
    return s;
}

const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

void write_blob(const std::shared_ptr<bnn::Mat> &b, FILE *fo) {
    for (int h = 0; h < b->h; ++h)
        for (int w = 0; w < b->w; ++w) fwrite(b->point<float>(h, w), sizeof(float), b->c, fo);
}
}  // namespace

int main(int argc, char **argv) {
    const char *model = arg(argc, argv, "--model", nullptr);
    const char *inputs = arg(argc, argv, "--inputs", nullptr);
    const char *blob = arg(argc, argv, "--blob", nullptr);
    const char *outputs = arg(argc, argv, "--outputs", nullptr);
    const char *json = arg(argc, argv, "--json", nullptr);
    const char *label = arg(argc, argv, "--label", "");
    const int warmup = atoi(arg(argc, argv, "--warmup", "9"));
    const int iters = atoi(arg(argc, argv, "--iterations", "50"));
    const int eval_n = atoi(arg(argc, argv, "--eval", "0"));
    if (!model || !inputs || !blob || !outputs) {
        fprintf(stderr,
                "usage: %s --model m.dab --inputs in.f32 --blob NAME --outputs out.f32 [--warmup 9 --iterations 50 "
                "--json r.json --label L | --eval N]\n",
                argv[0]);
        return 2;
    }
    google::InitGoogleLogging(argv[0]);
    const size_t per = 224 * 224 * 3;
    std::vector<float> in(per);
    FILE *fi = fopen(inputs, "rb");
    FILE *fo = fopen(outputs, "wb");
    if (!fi || !fo) {
        fprintf(stderr, "cannot open %s or %s\n", inputs, outputs);
        return 3;
    }
    const auto t_load = Clock::now();
    auto net = bnn::Net::create();
    net->optimize = true;
    net->run_fconv = true;
    net->strict = true;
    net->read(model);
    const double load_ms = ms_since(t_load);

    if (eval_n > 0) {
        for (int s = 0; s < eval_n; ++s) {
            if (fread(in.data(), sizeof(float), per, fi) != per) return 4;
            net->run(in.data());
            write_blob(net->get_blob(blob), fo);
        }
        fclose(fi);
        fclose(fo);
        printf("DABNN_EVAL {\"n\":%d,\"load_ms\":%.3f}\n", eval_n, load_ms);
        return 0;
    }
    if (fread(in.data(), sizeof(float), per, fi) != per) return 4;
    fclose(fi);
    const auto t_first = Clock::now();
    net->run(in.data());  // correctness run (untimed in the statistics)
    const double first_ms = ms_since(t_first);
    const auto &b = net->get_blob(blob);
    write_blob(b, fo);
    fclose(fo);
    for (int i = 0; i < warmup; ++i) net->run(in.data());
    const std::string f_before = cpufreq_json();
    std::vector<double> t(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        net->run(in.data());
        t[i] = ms_since(t0);
    }
    const std::string f_after = cpufreq_json();
    std::vector<double> s = t;
    std::sort(s.begin(), s.end());
    double mean = 0;
    for (double v : t) mean += v;
    mean /= iters;
    double var = 0;
    for (double v : t) var += (v - mean) * (v - mean);
    const double sd = iters > 1 ? std::sqrt(var / (iters - 1)) : 0.0;
    const double median = iters % 2 ? s[iters / 2] : 0.5 * (s[iters / 2 - 1] + s[iters / 2]);
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(4);
    o << "{\"tool\":\"dabnn_h2h\",\"label\":\"" << label << "\",\"model\":\"" << model << "\",\"blob\":\"" << blob
      << "\",\"blob_shape_hwc\":[" << b->h << "," << b->w << "," << b->c << "],\"threads\":1,\"warmup_untimed\":"
      << warmup + 1 << ",\"affinity\":\"" << affinity_list() << "\",\"cpu_at_end\":" << sched_getcpu()
      << ",\"latency\":{\"n\":" << iters << ",\"mean_ms\":" << mean << ",\"median_ms\":" << median
      << ",\"min_ms\":" << s.front() << ",\"max_ms\":" << s.back() << ",\"stddev_ms\":" << sd
      << ",\"load_ms\":" << load_ms << ",\"first_run_ms\":" << first_ms << ",\"runs_ms\":[";
    for (int i = 0; i < iters; ++i) o << (i ? "," : "") << t[i];
    o << "]},\"cpu_freq_before\":" << f_before << ",\"cpu_freq_after\":" << f_after << "}";
    const std::string js = o.str();
    printf("DABNN_H2H %s\n", js.c_str());
    if (json) {
        FILE *fj = fopen(json, "w");
        if (fj) {
            fputs(js.c_str(), fj);
            fclose(fj);
        }
    }
    return 0;
}

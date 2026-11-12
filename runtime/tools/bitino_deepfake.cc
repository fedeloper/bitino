// bitino_deepfake: runs a converted "Faster Than Lies" deepfake detector (tools/convert_ftl_deepfake.py) on
// images or video frames and prints, per frame, the real/fake score and where the time went.
//
//   bitino_deepfake --model model.bitino [--threads 1] [--affinity 7] [--stage all|neon|...] [--warmup 3]
//                     [--repeat 1] INPUT...
//
// INPUT is a binary PPM (P6, maxval 255) frame, a directory (its *.ppm files, sorted: e.g. the output of
// `ffmpeg -i clip.mp4 -vf fps=10 frames/%05d.ppm`), or a raw float32 file of N x 224 x 224 x 3 RGB in [0, 1]
// already preprocessed exactly as the authors' test loader (`bitino deepfake` writes it from JPEG / PNG files with
// torchvision; tools/eval_sets/ftl_deepfake.py).
//
// PPM frames are resized here (short side to 252, bilinear with half-pixel centres, no antialiasing) and center-
// cropped to 224, then quantized as torchvision's ToDtype does (u8 * fl32(1/255)). That approximates the source's
// PIL/torchvision resize (antialiased when downscaling); for bit-exact source preprocessing use the .f32 route.
//
// Timing per frame: "resize" (PPM decode-free part: resize + crop + scaling, host code), "features" (the model's
// features.* nodes: ImageGrayU8, FftLogMagnitude, LocalBinaryPattern, Concat), "network" (adapter + BNext + head);
// features and network come from the session's per-node profile (SessionOptions::profile).
// Output 1 of the model is the logit of "real" (row 0 is a constant 0): P(fake) = 1 / (1 + exp(logit)).
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "bitino/bitino.h"

using namespace bitino;

namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t b = 0;
  while (b <= s.size() && !s.empty()) {
    size_t e = s.find(sep, b);
    if (e == std::string::npos) e = s.size();
    if (e > b) out.push_back(s.substr(b, e - b));
    b = e + 1;
  }
  return out;
}

const std::map<std::string, std::set<std::string>>& stage_tiers() {  // same sets as bitino_bench
  static const std::map<std::string, std::set<std::string>> m = {
      {"reference", {"reference"}},
      {"neon", {"reference", "binary_scalar", "neon", "binary_avx2"}},
      {"optimized", {"reference", "binary_scalar", "neon", "binary_avx2", "float_opt"}},
      {"all", {}},
  };
  return m;
}

struct Image {
  int w = 0, h = 0;
  std::vector<uint8_t> rgb;
};

bool read_token(FILE* f, std::string& tok) {
  tok.clear();
  int c;
  while ((c = std::fgetc(f)) != EOF) {
    if (c == '#') {
      while ((c = std::fgetc(f)) != EOF && c != '\n') {
      }
      continue;
    }
    if (std::isspace(c)) {
      if (!tok.empty()) return true;
      continue;
    }
    tok.push_back(static_cast<char>(c));
  }
  return !tok.empty();
}

bool read_ppm(const std::string& path, Image& img) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::string magic, ws, hs, ms;
  bool ok = read_token(f, magic) && magic == "P6" && read_token(f, ws) && read_token(f, hs) && read_token(f, ms) && ms == "255";
  if (ok) {
    img.w = std::atoi(ws.c_str());
    img.h = std::atoi(hs.c_str());
    ok = img.w > 0 && img.h > 0 && img.w < 65536 && img.h < 65536;
  }
  if (ok) {
    img.rgb.resize(static_cast<size_t>(img.w) * img.h * 3);
    ok = std::fread(img.rgb.data(), 1, img.rgb.size(), f) == img.rgb.size();
  }
  std::fclose(f);
  return ok;
}

// Short side -> 252 (bilinear, half-pixel centres), center crop 224, u8 * fl32(1/255) (torchvision ToDtype).
void resize_crop(const Image& img, float* out, int size = 224, int short_side = 252) {
  const double s = static_cast<double>(short_side) / std::min(img.w, img.h);
  const int rw = img.w <= img.h ? short_side : static_cast<int>(img.w * s);
  const int rh = img.h < img.w ? short_side : static_cast<int>(img.h * s);
  const int x0 = static_cast<int>(std::lround((rw - size) / 2.0)), y0 = static_cast<int>(std::lround((rh - size) / 2.0));
  const float sx = static_cast<float>(img.w) / rw, sy = static_cast<float>(img.h) / rh;
  const float inv255 = 1.f / 255.f;
  for (int y = 0; y < size; ++y) {
    float fy = (y + y0 + 0.5f) * sy - 0.5f;
    fy = std::min(std::max(fy, 0.f), static_cast<float>(img.h - 1));
    const int iy = static_cast<int>(fy), iy1 = std::min(iy + 1, img.h - 1);
    const float wy = fy - iy;
    for (int x = 0; x < size; ++x) {
      float fx = (x + x0 + 0.5f) * sx - 0.5f;
      fx = std::min(std::max(fx, 0.f), static_cast<float>(img.w - 1));
      const int ix = static_cast<int>(fx), ix1 = std::min(ix + 1, img.w - 1);
      const float wx = fx - ix;
      for (int c = 0; c < 3; ++c) {
        auto px = [&](int yy, int xx) { return static_cast<float>(img.rgb[(static_cast<size_t>(yy) * img.w + xx) * 3 + c]); };
        const float v = (px(iy, ix) * (1 - wx) + px(iy, ix1) * wx) * (1 - wy) + (px(iy1, ix) * (1 - wx) + px(iy1, ix1) * wx) * wy;
        const float q = std::min(std::max(std::nearbyint(v), 0.f), 255.f);
        out[(static_cast<size_t>(y) * size + x) * 3 + c] = q * inv255;
      }
    }
  }
}

bool is_dir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

struct Frame {
  std::string name;
  std::string ppm;          // path, or empty for a tensor from a .f32 file
  std::vector<float> data;  // preprocessed tensor (f32 inputs)
};

double median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

void usage() {
  std::fprintf(stderr,
               "usage: bitino_deepfake --model PATH [--threads N] [--affinity a,b,...] [--stage all|neon|optimized|reference]\n"
               "                         [--warmup K] [--repeat R] INPUT...   (INPUT: frame.ppm | directory of *.ppm | tensors.f32)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, stage = "all", affinity;
  int threads = 1, warmup = 3, repeat = 1;
  std::vector<std::string> inputs;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--model") model_path = next();
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--affinity") affinity = next();
    else if (a == "--stage") stage = next();
    else if (a == "--warmup") warmup = std::atoi(next().c_str());
    else if (a == "--repeat") repeat = std::atoi(next().c_str());
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 2; }
    else inputs.push_back(a);
  }
  if (model_path.empty() || inputs.empty() || threads < 1 || repeat < 1 || warmup < 0 || !stage_tiers().count(stage)) {
    usage();
    return 2;
  }
  try {
    SessionOptions opts;
    opts.num_threads = threads;
    opts.profile = true;
    opts.policy.allowed_tiers = stage_tiers().at(stage);
    for (auto& c : split(affinity, ',')) opts.affinity.push_back(std::atoi(c.c_str()));
    auto model = Model::load(model_path);
    Session session(model, opts);
    const Shape& in_shape = session.input(0).shape();
    BF_CHECK(in_shape.size() == 4 && in_shape[3] == 3 && in_shape[1] == 224 && in_shape[2] == 224,
             "expected a [1,224,224,3] RGB model input, got ", shape_str(in_shape));
    BF_CHECK(session.output(0).size() == 2, "expected the two-logit head [0, logit] of tools/convert_ftl_deepfake.py");
    const size_t per = 224 * 224 * 3;

    std::vector<Frame> frames;
    for (const auto& in : inputs) {
      if (is_dir(in)) {
        std::vector<std::string> names;
        if (DIR* d = opendir(in.c_str())) {
          while (dirent* e = readdir(d))
            if (ends_with(e->d_name, ".ppm")) names.push_back(e->d_name);
          closedir(d);
        }
        std::sort(names.begin(), names.end());
        for (auto& n : names) frames.push_back({n, in + "/" + n, {}});
      } else if (ends_with(in, ".f32")) {
        std::ifstream f(in, std::ios::binary | std::ios::ate);
        BF_CHECK(f.good(), "cannot open ", in);
        const size_t bytes = static_cast<size_t>(f.tellg());
        BF_CHECK(bytes % (per * 4) == 0, in, ": size is not a multiple of 224x224x3 floats");
        f.seekg(0);
        for (size_t k = 0; k < bytes / (per * 4); ++k) {
          Frame fr{in.substr(in.find_last_of('/') + 1) + "[" + std::to_string(k) + "]", "", std::vector<float>(per)};
          f.read(reinterpret_cast<char*>(fr.data.data()), per * 4);
          frames.push_back(std::move(fr));
        }
      } else {
        frames.push_back({in, in, {}});
      }
    }
    BF_CHECK(!frames.empty(), "no input frames");
    std::printf("model %s, stage %s, %d thread(s)%s%s, %zu frame(s)\n", model_path.c_str(), stage.c_str(), threads,
                affinity.empty() ? "" : ", affinity ", affinity.c_str(), frames.size());

    float* x = session.input(0).data<float>();
    auto load = [&](const Frame& fr) {
      if (fr.ppm.empty()) {
        std::memcpy(x, fr.data.data(), per * 4);
        return true;
      }
      Image img;
      if (!read_ppm(fr.ppm, img)) return false;
      resize_crop(img, x);
      return true;
    };
    for (int i = 0; i < warmup; ++i) {  // warm caches / clocks on the first frame
      if (!load(frames[0])) break;
      session.run();
    }
    std::vector<double> t_resize, t_feat, t_net, t_total;
    int n_fake = 0;
    std::printf("%-40s %9s %8s %-5s %9s %9s %9s %9s\n", "frame", "logit", "P(fake)", "label", "resize_ms", "feat_ms", "net_ms", "total_ms");
    for (int rep = 0; rep < repeat; ++rep) {
      for (const auto& fr : frames) {
        const double t0 = now_ms();
        if (!load(fr)) {
          std::fprintf(stderr, "%s: not a binary PPM (P6, maxval 255)\n", fr.ppm.c_str());
          continue;
        }
        const double t1 = now_ms();
        session.reset_profile();
        session.run();
        const double t2 = now_ms();
        double feat = 0, net = 0;
        for (const auto& p : session.profile()) (p.name.rfind("features.", 0) == 0 ? feat : net) += p.total_ms;
        const float logit = session.output(0).data<float>()[1];
        const double p_fake = 1.0 / (1.0 + std::exp(static_cast<double>(logit)));
        if (rep == 0) {
          n_fake += p_fake > 0.5;
          std::printf("%-40s %9.4f %8.4f %-5s %9.3f %9.3f %9.3f %9.3f\n", fr.name.c_str(), logit, p_fake, p_fake > 0.5 ? "FAKE" : "real",
                      fr.ppm.empty() ? 0.0 : t1 - t0, feat, net, (fr.ppm.empty() ? 0.0 : t1 - t0) + (t2 - t1));
        }
        t_resize.push_back(fr.ppm.empty() ? 0.0 : t1 - t0);
        t_feat.push_back(feat);
        t_net.push_back(net);
        t_total.push_back((fr.ppm.empty() ? 0.0 : t1 - t0) + (t2 - t1));
      }
    }
    const double tot = median(t_total);
    std::printf("frames flagged FAKE: %d / %zu\n", n_fake, frames.size());
    std::printf("median per frame over %zu runs: resize %.3f ms, features %.3f ms, network %.3f ms, total %.3f ms -> %.1f fps\n",
                t_total.size(), median(t_resize), median(t_feat), median(t_net), tot, 1000.0 / tot);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

#include "bitino/bitino_c.h"

#include <cstring>
#include <memory>
#include <string>

#include "bitino/bitino.h"

struct bf_model {
  std::shared_ptr<bitino::Model> model;
};
struct bf_session {
  std::unique_ptr<bitino::Session> session;
};

namespace {

thread_local std::string g_last_error;

template <typename F>
int guarded(F&& f) {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    g_last_error = e.what();
  } catch (...) {
    g_last_error = "unknown error";
  }
  return 1;
}

bitino::KernelPolicy policy_for(const char* stage) {
  using bitino::KernelPolicy;
  std::string s = stage ? stage : "";
  if (s.empty() || s == "all") return KernelPolicy::all();
  if (s == "reference") return KernelPolicy::tiers({"reference"});
  if (s == "binary_scalar") return KernelPolicy::tiers({"reference", "binary_scalar"});
  if (s == "neon") return KernelPolicy::tiers({"reference", "binary_scalar", "neon", "binary_avx2"});
  bitino::fail("unknown stage '", s, "' (use all, reference, binary_scalar, neon)");
}

}  // namespace

extern "C" {

const char* bf_version(void) { return BITINO_VERSION; }

const char* bf_cpu_features(void) {
  static const std::string s = bitino::cpu_info().describe();
  return s.c_str();
}

const char* bf_last_error(void) { return g_last_error.c_str(); }

int bf_model_load(const char* path, int verify_hash, bf_model** out) {
  return guarded([&] {
    bitino::Model::LoadOptions o;
    o.verify_hash = verify_hash != 0;
    auto m = std::make_unique<bf_model>();
    m->model = bitino::Model::load(path, o);
    *out = m.release();
  });
}

void bf_model_free(bf_model* model) { delete model; }

int bf_session_create(bf_model* model, int num_threads, const char* stage, bf_session** out) {
  return guarded([&] {
    BF_CHECK(model != nullptr, "null model");
    bitino::SessionOptions o;
    o.num_threads = num_threads;
    o.policy = policy_for(stage);
    auto s = std::make_unique<bf_session>();
    s->session = std::make_unique<bitino::Session>(model->model, o);
    *out = s.release();
  });
}

void bf_session_free(bf_session* session) { delete session; }

size_t bf_session_input_size(bf_session* s, int index) {
  return s && index >= 0 && static_cast<size_t>(index) < s->session->num_inputs() ? static_cast<size_t>(s->session->input(index).size()) : 0;
}

size_t bf_session_output_size(bf_session* s, int index) {
  return s && index >= 0 && static_cast<size_t>(index) < s->session->num_outputs() ? static_cast<size_t>(s->session->output(index).size()) : 0;
}

int bf_session_num_outputs(bf_session* s) { return s ? static_cast<int>(s->session->num_outputs()) : 0; }

int bf_session_run(bf_session* s, const float* input, size_t count) {
  return guarded([&] {
    BF_CHECK(s != nullptr, "null session");
    bitino::Tensor& in = s->session->input(0);
    BF_CHECK(in.dtype() == bitino::DType::kF32 && count == static_cast<size_t>(in.size()),
             "input expects ", in.size(), " floats, got ", count);
    std::memcpy(in.raw(), input, count * sizeof(float));
    s->session->run();
  });
}

int bf_session_get_output(bf_session* s, int out_index, float* dst, size_t count) {
  return guarded([&] {
    BF_CHECK(s != nullptr, "null session");
    const bitino::Tensor& o = s->session->output(static_cast<size_t>(out_index));
    BF_CHECK(o.dtype() == bitino::DType::kF32 && count == static_cast<size_t>(o.size()),
             "output ", out_index, " has ", o.size(), " floats, got buffer of ", count);
    std::memcpy(dst, o.raw(), count * sizeof(float));
  });
}

}  // extern "C"

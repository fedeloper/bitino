#include "bitino/bitino_c.h"

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bitino/bitino.h"
#include "bitino/stages.h"
#include "bitino/util.h"

struct bf_model {
  std::shared_ptr<bitino::Model> model;
  std::string manifest_json;
};
struct bf_session {
  std::unique_ptr<bitino::Session> session;
  std::string stage;
  std::string describe_json;
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

bitino::KernelPolicy policy_for(const std::string& s) {
  if (s.empty()) return bitino::KernelPolicy::all();
  const auto& m = bitino::stage_tiers();
  auto it = m.find(s);
  if (it == m.end()) {
    std::string names;
    for (const auto& kv : m) names += (names.empty() ? "" : ", ") + kv.first;
    bitino::fail("unknown stage '", s, "' (use one of: ", names, ")");
  }
  return bitino::KernelPolicy::tiers(it->second);
}

bool valid_input(bf_session* s, int i) { return s && i >= 0 && static_cast<size_t>(i) < s->session->num_inputs(); }
bool valid_output(bf_session* s, int i) { return s && i >= 0 && static_cast<size_t>(i) < s->session->num_outputs(); }

int dtype_code(bitino::DType t) {
  switch (t) {
    case bitino::DType::kF32: return BF_DTYPE_F32;
    case bitino::DType::kI32: return BF_DTYPE_I32;
    case bitino::DType::kI8: return BF_DTYPE_I8;
    case bitino::DType::kU8: return BF_DTYPE_U8;
    case bitino::DType::kBit: return BF_DTYPE_BIT;
  }
  return -1;
}

void fill_info(const std::string& name, const bitino::Tensor& t, bf_tensor_info* out) {
  BF_CHECK(out != nullptr, "null bf_tensor_info");
  BF_CHECK(t.rank() <= BF_MAX_RANK, "tensor ", name, " has rank ", t.rank(), " > ", BF_MAX_RANK);
  std::memset(out, 0, sizeof(*out));
  out->name = name.c_str();
  out->dtype = dtype_code(t.dtype());
  out->rank = t.rank();
  for (int d = 0; d < t.rank(); ++d) out->shape[d] = t.dim(d);
  out->bytes = static_cast<uint64_t>(t.bytes());
}

}  // namespace

extern "C" {

const char* bf_version(void) { return BITINO_VERSION; }

int bf_c_api_version(void) { return BF_C_API_VERSION; }

const char* bf_cpu_features(void) {
  static const std::string s = bitino::cpu_info().describe();
  return s.c_str();
}

const char* bf_last_error(void) { return g_last_error.c_str(); }

int bf_model_load(const char* path, int verify_hash, bf_model** out) {
  return guarded([&] {
    BF_CHECK(path != nullptr && out != nullptr, "null argument");
    bitino::Model::LoadOptions o;
    o.verify_hash = verify_hash != 0;
    auto m = std::make_unique<bf_model>();
    m->model = bitino::Model::load(path, o);
    m->manifest_json = m->model->manifest().dump();
    *out = m.release();
  });
}

int bf_model_load_bytes(const void* data, size_t size, int verify_hash, bf_model** out) {
  return guarded([&] {
    BF_CHECK(data != nullptr, "null model data");
    bitino::Model::LoadOptions o;
    o.verify_hash = verify_hash != 0;
    auto m = std::make_unique<bf_model>();
    m->model = bitino::Model::from_bytes(static_cast<const uint8_t*>(data), size, o);
    *out = m.release();
  });
}

void bf_model_free(bf_model* model) { delete model; }

const char* bf_model_manifest_json(bf_model* model) {
  if (!model) return "{}";
  if (model->manifest_json.empty()) model->manifest_json = model->model->manifest().dump();   // e.g. bf_model_load_bytes
  return model->manifest_json.c_str();
}

const char* bf_model_manifest(bf_model* model) { return bf_model_manifest_json(model); }

int bf_session_create_ex(bf_model* model, int num_threads, const char* stage, const int* affinity, int num_affinity,
                         bf_session** out) {
  return guarded([&] {
    BF_CHECK(model != nullptr, "null model");
    BF_CHECK(out != nullptr, "null output handle");
    BF_CHECK(num_affinity >= 0 && (num_affinity == 0 || affinity != nullptr), "bad affinity list");
    bitino::SessionOptions o;
    o.num_threads = num_threads;
    auto s = std::make_unique<bf_session>();
    s->stage = stage && *stage ? stage : "all";
    o.policy = policy_for(s->stage == "all" ? "" : s->stage);
    for (int i = 0; i < num_affinity; ++i) o.affinity.push_back(affinity[i]);
    s->session = std::make_unique<bitino::Session>(model->model, o);
    *out = s.release();
  });
}

int bf_session_create(bf_model* model, int num_threads, const char* stage, bf_session** out) {
  return bf_session_create_ex(model, num_threads, stage, nullptr, 0, out);
}

void bf_session_free(bf_session* session) { delete session; }

size_t bf_session_input_size(bf_session* s, int index) {
  return valid_input(s, index) ? static_cast<size_t>(s->session->input(index).size()) : 0;
}

size_t bf_session_output_size(bf_session* s, int index) {
  return valid_output(s, index) ? static_cast<size_t>(s->session->output(index).size()) : 0;
}

int bf_session_num_inputs(bf_session* s) { return s ? static_cast<int>(s->session->num_inputs()) : 0; }

int bf_session_input_shape(bf_session* s, int index, int64_t* dims, int max_dims) {
  if (!s || index < 0 || static_cast<size_t>(index) >= s->session->num_inputs()) return 0;
  const bitino::Shape& sh = s->session->input(index).shape();
  for (int i = 0; i < max_dims && i < static_cast<int>(sh.size()); ++i) dims[i] = sh[i];
  return static_cast<int>(sh.size());
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
    BF_CHECK(valid_output(s, out_index), "no output ", out_index);
    const bitino::Tensor& o = s->session->output(static_cast<size_t>(out_index));
    BF_CHECK(o.dtype() == bitino::DType::kF32 && count == static_cast<size_t>(o.size()),
             "output ", out_index, " has ", o.size(), " floats, got buffer of ", count);
    std::memcpy(dst, o.raw(), count * sizeof(float));
  });
}

int bf_session_input_info(bf_session* s, int index, bf_tensor_info* out) {
  return guarded([&] {
    BF_CHECK(valid_input(s, index), "no input ", index);
    fill_info(s->session->input_name(index), s->session->input(index), out);
  });
}

int bf_session_output_info(bf_session* s, int index, bf_tensor_info* out) {
  return guarded([&] {
    BF_CHECK(valid_output(s, index), "no output ", index);
    fill_info(s->session->output_name(index), s->session->output(index), out);
  });
}

void* bf_session_input_data(bf_session* s, int index) {
  return valid_input(s, index) ? s->session->input(index).raw() : nullptr;
}

const void* bf_session_output_data(bf_session* s, int index) {
  return valid_output(s, index) ? s->session->output(index).raw() : nullptr;
}

int bf_session_set_input(bf_session* s, int index, const void* data, size_t bytes) {
  return guarded([&] {
    BF_CHECK(valid_input(s, index), "no input ", index);
    bitino::Tensor& in = s->session->input(index);
    BF_CHECK(data != nullptr && bytes == in.bytes(), "input ", index, " expects ", in.bytes(), " bytes, got ", bytes);
    std::memcpy(in.raw(), data, bytes);
  });
}

int bf_session_invoke(bf_session* s) {
  return guarded([&] {
    BF_CHECK(s != nullptr, "null session");
    s->session->run();
  });
}

int bf_session_benchmark(bf_session* s, int warmup, int iterations, double* samples_ms) {
  return guarded([&] {
    BF_CHECK(s != nullptr, "null session");
    BF_CHECK(warmup >= 0 && iterations >= 1 && samples_ms != nullptr, "bad benchmark arguments");
    for (int i = 0; i < warmup; ++i) s->session->run();
    for (int i = 0; i < iterations; ++i) {
      const double t0 = bitino::now_ms();
      s->session->run();
      samples_ms[i] = bitino::now_ms() - t0;
    }
  });
}

const char* bf_session_describe(bf_session* s) {
  if (!s) return "{}";
  using bitino::Json;
  try {
    const bitino::Session& se = *s->session;
    std::map<std::string, int> counts;
    for (const auto& kv : se.kernel_choices()) counts[kv.second]++;
    Json kernels = Json::Object{};
    for (const auto& kv : counts) kernels[kv.first] = kv.second;
    Json o = Json::Object{};
    o["stage"] = s->stage;
    o["threads"] = se.num_threads();
    o["cpu_features"] = bitino::cpu_info().describe();
    o["nodes"] = static_cast<int64_t>(se.kernel_choices().size());
    o["kernels"] = kernels;
    o["plan_rewrites"] = Json(Json::Object{{"fused_nodes", Json(se.fused_nodes())},
                                           {"absorbed_nodes", Json(se.absorbed_nodes())},
                                           {"folded_thresholds", Json(se.folded_thresholds())},
                                             {"runtime_thresholds", Json(se.runtime_thresholds())},
                                           {"hosted_epilogues", Json(se.hosted_epilogues())},
                                           {"bit_outputs", Json(se.bit_outputs())},
                                           {"shared_binarizations", Json(se.shared_binarizations())},
                                           {"aliased_reshapes", Json(se.aliased_reshapes())}});
    o["prepare_ms"] = se.prepare_ms();
    o["activation_bytes"] = Json(se.activation_bytes());
    o["prepacked_bytes"] = Json(se.prepacked_bytes());
    o["affinity_errors"] = se.pool().affinity_errors();
    o["unpinned_threads"] = se.pool().unpinned_threads();
    s->describe_json = o.dump();
  } catch (const std::exception& e) {
    g_last_error = e.what();
    s->describe_json = "{}";
  }
  return s->describe_json.c_str();
}

}  // extern "C"

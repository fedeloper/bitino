// Core runtime tests: JSON, SHA-256, tensors, thread pool, model loading.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "bitino/bitino.h"
#include "bitino/json.h"
#include "bitino/thread_pool.h"
#include "bitino/util.h"
#include "tools/model_builder.h"
#include "testing.h"

using namespace bitino;

BF_TEST(json_roundtrip) {
  Json j = Json::parse(R"({"a": [1, 2.5, -3e2], "b": {"s": "x\"yé"}, "t": true, "n": null})");
  EXPECT_EQ(j["a"].size(), size_t(3));
  EXPECT_EQ(j["a"][1].as_number(), 2.5);
  EXPECT_EQ(j["a"][2].as_int(), int64_t(-300));
  EXPECT_EQ(j["b"]["s"].as_string(), std::string("x\"y\xc3\xa9"));
  EXPECT_TRUE(j["t"].as_bool());
  EXPECT_TRUE(j["n"].is_null());
  Json k = Json::parse(j.dump(2));
  EXPECT_EQ(k.dump(), j.dump());
  EXPECT_THROWS(Json::parse("{\"a\": }"));
  const Json& cj = j;
  EXPECT_THROWS(cj["missing"]);
}

BF_TEST(json_copy_on_write) {
  Json a = Json::Object{{"x", 1}};
  Json b = a;
  b["x"] = Json(2);
  EXPECT_EQ(a["x"].as_int(), int64_t(1));
  EXPECT_EQ(b["x"].as_int(), int64_t(2));
}

BF_TEST(sha256_known_vectors) {
  EXPECT_EQ(sha256_hex("", 0), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  EXPECT_EQ(sha256_hex("abc", 3), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";  // 56 bytes: two-block padding
  EXPECT_EQ(sha256_hex(m.data(), m.size()), std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
  std::string mil(1000000, 'a');
  EXPECT_EQ(sha256_hex(mil.data(), mil.size()), std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}

BF_TEST(tensor_storage_bit_packing) {
  EXPECT_EQ(storage_bytes(DType::kBit, {1, 4, 4, 32}), size_t(16 * 4));
  EXPECT_EQ(storage_bytes(DType::kBit, {1, 4, 4, 33}), size_t(16 * 8));
  EXPECT_EQ(storage_bytes(DType::kF32, {2, 3}), size_t(24));
  Tensor t = Tensor::allocate(DType::kF32, {4, 16});
  EXPECT_EQ(reinterpret_cast<uintptr_t>(t.raw()) % 64, uintptr_t(0));
  EXPECT_THROWS(t.reshaped({5, 13}));
}

BF_TEST(thread_pool_covers_range_exactly_once) {
  for (int spin : {0, 50, 1000})
  for (int threads : {1, 2, 3, 8}) {
    ThreadPool pool(threads, {}, spin);
    for (int64_t n : {1, 7, 100, 1001}) {
      std::vector<std::atomic<int>> hits(static_cast<size_t>(n));
      for (auto& h : hits) h = 0;
      for (int rep = 0; rep < 200; ++rep)
        pool.parallel_for(n, 3, [&](int64_t b, int64_t e) {
          for (int64_t i = b; i < e; ++i) hits[i]++;
        });
      for (auto& h : hits) EXPECT_EQ(h.load(), 200);
    }
  }
}

// Hot mode (SessionOptions::spin_run): exact coverage while hot, workers block again when it is switched off
// (a later parallel_for still completes), counters advance.
BF_TEST(thread_pool_hot_mode) {
  for (int spin : {0, 100})
    for (int threads : {2, 4}) {
      ThreadPool pool(threads, {}, spin);
      std::vector<std::atomic<int>> hits(257);
      for (auto& h : hits) h = 0;
      const uint64_t c0 = pool.calls();
      for (int round = 0; round < 3; ++round) {
        pool.set_hot(round != 1);
        for (int rep = 0; rep < 100; ++rep)
          pool.parallel_for(257, 2, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) hits[i]++;
          });
        pool.set_hot(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));  // workers may go to sleep here
      }
      for (auto& h : hits) EXPECT_EQ(h.load(), 300);
      EXPECT_EQ(pool.calls() - c0, uint64_t(300));
      EXPECT_TRUE(pool.dispatches() >= 300);
    }
}

BF_TEST(model_load_rejects_corruption) {
  bftest::ModelBuilder mb;
  float w[4] = {1, 2, 3, 4};
  mb.constant("w", DType::kF32, {2, 2}, w);
  mb.input("x", DType::kF32, {1, 2});
  mb.node("Dense", {"x", "w", ""}, {"y"});
  mb.output("y");
  auto bytes = mb.bytes();
  EXPECT_TRUE(Model::from_bytes(bytes.data(), bytes.size(), {}) != nullptr);
  auto bad = bytes;
  bad[bad.size() - 1] ^= 0xFF;  // corrupt the data section
  EXPECT_THROWS(Model::from_bytes(bad.data(), bad.size(), {}));
  Model::LoadOptions no_verify;
  no_verify.verify_hash = false;
  EXPECT_TRUE(Model::from_bytes(bad.data(), bad.size(), no_verify) != nullptr);
  auto bad_magic = bytes;
  bad_magic[0] = 'X';
  EXPECT_THROWS(Model::from_bytes(bad_magic.data(), bad_magic.size(), {}));
  EXPECT_THROWS(Model::from_bytes(bytes.data(), 40, {}));
}

BF_TEST(session_dense_and_unknown_op) {
  bftest::ModelBuilder mb;
  float w[6] = {1, 2, 3, 4, 5, 6};  // [O=2, K=3]
  float b[2] = {0.5f, -1.f};
  mb.constant("w", DType::kF32, {2, 3}, w);
  mb.constant("b", DType::kF32, {2}, b);
  mb.input("x", DType::kF32, {1, 3});
  mb.node("Dense", {"x", "w", "b"}, {"y"});
  mb.output("y");
  Session s(mb.build(), {});
  float x[3] = {1, -1, 2};
  std::memcpy(s.input(0).raw(), x, sizeof x);
  s.run();
  EXPECT_NEAR(s.output(0).data<float>()[0], 1 - 2 + 6 + 0.5, 1e-6);
  EXPECT_NEAR(s.output(0).data<float>()[1], 4 - 5 + 12 - 1, 1e-6);

  bftest::ModelBuilder bad;
  bad.input("x", DType::kF32, {1, 3});
  bad.node("NoSuchOp", {"x"}, {"y"});
  bad.output("y");
  EXPECT_THROWS(Session(bad.build(), {}));
}

BF_TEST(memory_planner_reuses_buffers_without_aliasing) {
  // Chain of Activation ops: x -> a -> b -> c -> d. With reuse, a and c can
  // share storage, but no node's output may alias its input.
  bftest::ModelBuilder mb;
  mb.input("x", DType::kF32, {1, 64});
  mb.node("Activation", {"x"}, {"a"}, Json(Json::Object{{"type", "relu"}}));
  mb.node("Activation", {"a"}, {"b"}, Json(Json::Object{{"type", "hardtanh"}}));
  mb.node("Activation", {"b"}, {"c"}, Json(Json::Object{{"type", "relu"}}));
  mb.node("Activation", {"c"}, {"d"}, Json(Json::Object{{"type", "hardtanh"}}));
  mb.output("d");
  auto model = mb.build();
  SessionOptions reuse, noreuse;
  noreuse.reuse_memory = false;
  Session s1(model, reuse), s2(model, noreuse);
  EXPECT_TRUE(s1.activation_bytes() < s2.activation_bytes());
  for (Session* s : {&s1, &s2}) {
    for (int i = 0; i < 64; ++i) s->input(0).data<float>()[i] = (i - 32) * 0.1f;
    s->run();
    for (int i = 0; i < 64; ++i) {
      float v = std::min(std::max((i - 32) * 0.1f, 0.f), 1.f);
      EXPECT_NEAR(s->output(0).data<float>()[i], v, 1e-7);
    }
  }
}

#include "bitino/bitino_c.h"

BF_TEST(c_api_roundtrip_and_errors) {
  bftest::ModelBuilder mb;
  float w[6] = {1, 2, 3, 4, 5, 6};
  mb.constant("w", DType::kF32, {2, 3}, w);
  mb.input("x", DType::kF32, {1, 3});
  mb.node("Dense", {"x", "w", ""}, {"y"});
  mb.output("y");
  auto bytes = mb.bytes();
  std::string path = "/tmp/bitino_c_api_test.bitino";
  if (const char* t = std::getenv("TMPDIR")) path = std::string(t) + "/bitino_c_api_test.bitino";
  write_file(path, std::string(bytes.begin(), bytes.end()));
  bf_model* m = nullptr;
  EXPECT_EQ(bf_model_load(path.c_str(), 1, &m), 0);
  bf_session* s = nullptr;
  EXPECT_EQ(bf_session_create(m, 2, "all", &s), 0);
  bf_model_free(m);  // session keeps the model alive
  EXPECT_EQ(bf_session_input_size(s, 0), size_t(3));
  float x[3] = {1, 1, 1}, y[2];
  EXPECT_EQ(bf_session_run(s, x, 3), 0);
  EXPECT_EQ(bf_session_get_output(s, 0, y, 2), 0);
  EXPECT_NEAR(y[0], 6, 1e-6);
  EXPECT_NEAR(y[1], 15, 1e-6);
  EXPECT_TRUE(bf_session_run(s, x, 2) != 0);  // wrong size
  EXPECT_TRUE(std::string(bf_last_error()).find("expects 3") != std::string::npos);
  bf_session_free(s);
  EXPECT_TRUE(bf_model_load("/nonexistent.bitino", 1, &m) != 0);
  EXPECT_TRUE(bf_session_create(nullptr, 1, "all", &s) != 0);
  // in-memory image (Android assets), manifest and input shape
  EXPECT_EQ(bf_model_load_bytes(bytes.data(), bytes.size(), 1, &m), 0);
  EXPECT_EQ(std::string(bf_model_manifest_json(m)).front(), '{');
  EXPECT_EQ(bf_session_create(m, 1, "all", &s), 0);
  bf_model_free(m);
  int64_t dims[4] = {0, 0, 0, 0};
  EXPECT_EQ(bf_session_input_shape(s, 0, dims, 4), 2);
  EXPECT_EQ(dims[0], 1);
  EXPECT_EQ(dims[1], 3);
  EXPECT_EQ(bf_session_input_shape(s, 1, dims, 4), 0);
  EXPECT_EQ(bf_session_run(s, x, 3), 0);
  EXPECT_EQ(bf_session_get_output(s, 0, y, 2), 0);
  EXPECT_NEAR(y[1], 15, 1e-6);
  bf_session_free(s);
  std::vector<uint8_t> bad(bytes.begin(), bytes.end());
  bad[0] ^= 1;
  EXPECT_TRUE(bf_model_load_bytes(bad.data(), bad.size(), 1, &m) != 0);
  EXPECT_TRUE(std::string(bf_last_error()).find("magic") != std::string::npos);
}

BF_TEST(c_api_v2_buffers_info_benchmark) {
  bftest::ModelBuilder mb;
  float w[6] = {1, 2, 3, 4, 5, 6};
  mb.constant("w", DType::kF32, {2, 3}, w);
  mb.input("x", DType::kF32, {1, 3});
  mb.node("Dense", {"x", "w", ""}, {"y"});
  mb.output("y");
  auto bytes = mb.bytes();
  std::string path = "/tmp/bitino_c_api_v2_test.bitino";
  if (const char* t = std::getenv("TMPDIR")) path = std::string(t) + "/bitino_c_api_v2_test.bitino";
  write_file(path, std::string(bytes.begin(), bytes.end()));
  EXPECT_EQ(bf_c_api_version(), BF_C_API_VERSION);
  bf_model* m = nullptr;
  EXPECT_EQ(bf_model_load(path.c_str(), 1, &m), 0);
  EXPECT_TRUE(std::string(bf_model_manifest(m)).front() == '{');
  bf_session* s = nullptr;
  EXPECT_TRUE(bf_session_create_ex(m, 1, "warp", nullptr, 0, &s) != 0);
  EXPECT_TRUE(std::string(bf_last_error()).find("unknown stage") != std::string::npos);
  EXPECT_EQ(bf_session_create_ex(m, 2, "neon", nullptr, 0, &s), 0);
  bf_model_free(m);
  EXPECT_EQ(bf_session_num_inputs(s), 1);
  bf_tensor_info info;
  EXPECT_EQ(bf_session_input_info(s, 0, &info), 0);
  EXPECT_EQ(std::string(info.name), std::string("x"));
  EXPECT_EQ(info.dtype, BF_DTYPE_F32);
  EXPECT_EQ(info.rank, 2);
  EXPECT_EQ(info.shape[1], 3);
  EXPECT_EQ(info.bytes, uint64_t(12));
  EXPECT_EQ(bf_session_output_info(s, 0, &info), 0);
  EXPECT_EQ(info.shape[1], 2);
  EXPECT_TRUE(bf_session_input_info(s, 1, &info) != 0);
  EXPECT_TRUE(bf_session_input_data(s, 1) == nullptr);
  float* in = static_cast<float*>(bf_session_input_data(s, 0));
  in[0] = 1, in[1] = 0, in[2] = -1;
  EXPECT_EQ(bf_session_invoke(s), 0);
  const float* out = static_cast<const float*>(bf_session_output_data(s, 0));
  EXPECT_NEAR(out[0], -2, 1e-6);
  EXPECT_NEAR(out[1], -2, 1e-6);
  float x[3] = {1, 1, 1};
  EXPECT_TRUE(bf_session_set_input(s, 0, x, 8) != 0);
  EXPECT_EQ(bf_session_set_input(s, 0, x, sizeof x), 0);
  double t[3];
  EXPECT_EQ(bf_session_benchmark(s, 1, 3, t), 0);
  EXPECT_TRUE(t[0] >= 0 && t[2] >= 0);
  out = static_cast<const float*>(bf_session_output_data(s, 0));
  EXPECT_NEAR(out[1], 15, 1e-6);
  std::string d = bf_session_describe(s);
  EXPECT_TRUE(d.find("\"stage\":\"neon\"") != std::string::npos);
  EXPECT_TRUE(d.find("\"threads\":2") != std::string::npos);
  bf_session_free(s);
}

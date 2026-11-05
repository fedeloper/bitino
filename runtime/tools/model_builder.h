// Builds .bitino images in memory (mirrors python/bitino/format.py). Used by
// C++ tests and the microbenchmark tool.
#pragma once

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bitino/bitino.h"
#include "bitino/util.h"

namespace bftest {

using bitino::DType;
using bitino::Json;
using bitino::Shape;

class ModelBuilder {
 public:
  void constant(const std::string& name, DType dt, const Shape& shape, const void* data, const std::string& layout = "") {
    size_t nbytes = bitino::storage_bytes(dt, shape);
    size_t off = (data_.size() + 63) / 64 * 64;
    data_.resize(off + nbytes, 0);
    std::memcpy(data_.data() + off, data, nbytes);
    Json::Array sh;
    for (auto d : shape) sh.push_back(Json(d));
    Json t = Json::Object{{"name", name}, {"dtype", bitino::dtype_name(dt)}, {"shape", Json(sh)},
                          {"offset", Json(off)}, {"nbytes", Json(nbytes)}, {"layout", layout}};
    tensors_.push_back(t);
  }
  void input(const std::string& name, DType dt, const Shape& shape) {
    Json::Array sh;
    for (auto d : shape) sh.push_back(Json(d));
    inputs_.push_back(Json(Json::Object{{"name", name}, {"dtype", bitino::dtype_name(dt)}, {"shape", Json(sh)}}));
  }
  void node(const std::string& op, std::vector<std::string> ins, std::vector<std::string> outs, Json attrs = Json(Json::Object{})) {
    Json::Array i, o;
    for (auto& s : ins) i.push_back(s.empty() ? Json() : Json(s));
    for (auto& s : outs) o.push_back(Json(s));
    nodes_.push_back(Json(Json::Object{{"op", op}, {"name", op + "_" + std::to_string(nodes_.size())},
                                       {"inputs", Json(i)}, {"outputs", Json(o)}, {"attrs", attrs}}));
  }
  void output(const std::string& name) { outputs_.push_back(Json(name)); }

  std::vector<uint8_t> bytes() const {
    std::vector<uint8_t> data = data_;
    data.resize((data.size() + 63) / 64 * 64);
    Json doc = Json::Object{{"format_version", 1}, {"tensors", Json(tensors_)}, {"inputs", Json(inputs_)},
                            {"nodes", Json(nodes_)}, {"outputs", Json(outputs_)},
                            {"data_sha256", bitino::sha256_hex(data.data(), data.size())},
                            {"manifest", Json(Json::Object{{"source", "test"}})}};
    std::string js = doc.dump();
    uint64_t json_off = 64, json_size = js.size();
    uint64_t data_off = (json_off + json_size + 63) / 64 * 64;
    std::vector<uint8_t> out(data_off + data.size(), 0);
    std::memcpy(out.data(), "BFMODEL1", 8);
    uint32_t ver = 1, hdr = 64;
    uint64_t data_size = data.size();
    std::memcpy(out.data() + 8, &ver, 4);
    std::memcpy(out.data() + 12, &hdr, 4);
    std::memcpy(out.data() + 16, &json_off, 8);
    std::memcpy(out.data() + 24, &json_size, 8);
    std::memcpy(out.data() + 32, &data_off, 8);
    std::memcpy(out.data() + 40, &data_size, 8);
    std::memcpy(out.data() + json_off, js.data(), js.size());
    if (!data.empty()) std::memcpy(out.data() + data_off, data.data(), data.size());
    return out;
  }

  std::shared_ptr<bitino::Model> build() const {
    auto b = bytes();
    return bitino::Model::from_bytes(b.data(), b.size(), {});
  }

 private:
  std::vector<uint8_t> data_;
  Json::Array tensors_, inputs_, nodes_, outputs_;
};

// Packs ±1 values (int8) OHWI along the last dim, bit 1 = -1.
inline std::vector<uint32_t> pack_pm1(const std::vector<int8_t>& v, int64_t rows, int64_t c) {
  int64_t words = bitino::bit_words(c);
  std::vector<uint32_t> out(static_cast<size_t>(rows * words), 0);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t j = 0; j < c; ++j)
      if (v[r * c + j] < 0) out[r * words + j / 32] |= 1u << (j % 32);
  return out;
}

}  // namespace bftest

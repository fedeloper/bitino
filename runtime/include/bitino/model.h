// Graph IR and .bitino loading.
//
// .bitino layout (all integers little-endian):
//   [0,8)   magic "BFMODEL1"
//   [8,12)  u32 format_version (currently 1)
//   [12,16) u32 header_size (64)
//   [16,24) u64 json_offset   [24,32) u64 json_size
//   [32,40) u64 data_offset   [40,48) u64 data_size   (data_offset is 64-aligned)
//   [48,64) reserved (zero)
// The JSON section describes the graph; constant tensors reference byte
// ranges of the data section, each 64-byte aligned. `data_sha256` in the JSON
// is the SHA-256 of the whole data section. See docs/model-format.md.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bitino/json.h"
#include "bitino/tensor.h"

namespace bitino {

struct TensorInfo {
  std::string name;
  DType dtype = DType::kF32;
  Shape shape;
  bool is_constant = false;
  std::string layout;  // informational for constants, e.g. "OHWI", "OHWI_bit"
  Tensor constant;     // valid iff is_constant (view into the model buffer)
};

struct Node {
  std::string name;
  std::string op;
  std::vector<int> inputs;   // tensor ids; -1 = optional input omitted
  std::vector<int> outputs;  // tensor ids
  Json attrs;
};

struct Graph {
  std::vector<TensorInfo> tensors;
  std::vector<Node> nodes;  // topologically ordered
  std::vector<int> inputs;
  std::vector<int> outputs;
  std::map<std::string, int> tensor_ids;

  int find(const std::string& name) const;  // -1 if absent
};

class Model {
 public:
  struct LoadOptions {
    bool verify_hash = true;
    // load(): map the file (MAP_PRIVATE, POSIX) instead of reading it into a heap copy. Constants are then views
    // into the mapping, and a Session can drop the pages of constants its kernels prepacked
    // (SessionOptions::release_constants); dropped pages are read again from the file if anything touches them.
    bool map_file = true;
  };
  static std::shared_ptr<Model> load(const std::string& path, const LoadOptions& opts);
  static std::shared_ptr<Model> load(const std::string& path) { return load(path, LoadOptions{}); }
  // Parses an in-memory image (takes ownership of a copy).
  static std::shared_ptr<Model> from_bytes(const uint8_t* data, size_t size, const LoadOptions& opts);

  const Graph& graph() const { return graph_; }
  const Json& manifest() const { return manifest_; }
  const std::string& data_sha256() const { return data_sha256_; }
  size_t file_bytes() const { return file_bytes_; }
  size_t constant_bytes() const { return constant_bytes_; }
  double load_ms() const { return load_ms_; }
  bool hash_verified() const { return hash_verified_; }
  // Constants are views into a private file mapping (load() with map_file).
  bool mapped() const { return mapped_; }

 private:
  static std::shared_ptr<Model> parse(std::shared_ptr<void> buffer, size_t size, const LoadOptions& opts, bool mapped);
  Graph graph_;
  Json manifest_;
  std::string data_sha256_;
  std::shared_ptr<void> buffer_;
  size_t file_bytes_ = 0;
  size_t constant_bytes_ = 0;
  double load_ms_ = 0;
  bool hash_verified_ = false;
  bool mapped_ = false;
};

}  // namespace bitino

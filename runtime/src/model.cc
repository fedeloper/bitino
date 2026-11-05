#include "bitino/model.h"

#include <cstring>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "bitino/error.h"
#include "bitino/util.h"

namespace bitino {
namespace {

constexpr char kMagic[8] = {'B', 'F', 'M', 'O', 'D', 'E', 'L', '1'};
constexpr uint32_t kFormatVersion = 1;

template <typename T>
T read_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);  // host is little-endian on all supported targets
  return v;
}

}  // namespace

int Graph::find(const std::string& name) const {
  auto it = tensor_ids.find(name);
  return it == tensor_ids.end() ? -1 : it->second;
}

std::shared_ptr<Model> Model::load(const std::string& path, const LoadOptions& opts) {
#if defined(__unix__) || defined(__APPLE__)
  if (opts.map_file) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail("cannot open file '", path, "'");
    struct stat st;
    void* p = MAP_FAILED;
    size_t size = 0;
    if (::fstat(fd, &st) == 0 && st.st_size >= 64) {
      size = static_cast<size_t>(st.st_size);
      // Writable private mapping: copy-on-write if anything ever wrote a constant, like the heap copy allowed.
      p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    }
    ::close(fd);
    if (p != MAP_FAILED) {
      std::shared_ptr<void> buf(p, [size](void* q) { ::munmap(q, size); });
      try {
        return parse(std::move(buf), size, opts, true);
      } catch (const Error& e) {
        fail("failed to load model '", path, "': ", e.what());
      }
    }
  }
#endif
  std::vector<uint8_t> bytes = read_file(path);
  try {
    return from_bytes(bytes.data(), bytes.size(), opts);
  } catch (const Error& e) {
    fail("failed to load model '", path, "': ", e.what());
  }
}

std::shared_ptr<Model> Model::from_bytes(const uint8_t* src, size_t size, const LoadOptions& opts) {
  // Copy into an aligned buffer so constant tensors are 64-byte aligned views.
  auto buf = aligned_alloc_shared(size);
  std::memcpy(buf.get(), src, size);
  return parse(std::move(buf), size, opts, false);
}

std::shared_ptr<Model> Model::parse(std::shared_ptr<void> buffer, size_t size, const LoadOptions& opts, bool mapped) {
  double t0 = now_ms();
  auto model = std::make_shared<Model>();
  const auto* src = static_cast<const uint8_t*>(buffer.get());
  BF_CHECK(size >= 64, "file too small (", size, " bytes)");
  if (std::memcmp(src, kMagic, 8) != 0) fail("bad magic: not a .bitino file");
  uint32_t version = read_le<uint32_t>(src + 8);
  if (version != kFormatVersion) fail("unsupported .bitino format version ", version, " (runtime supports ", kFormatVersion, ")");
  uint64_t json_off = read_le<uint64_t>(src + 16), json_size = read_le<uint64_t>(src + 24);
  uint64_t data_off = read_le<uint64_t>(src + 32), data_size = read_le<uint64_t>(src + 40);
  BF_CHECK(json_off + json_size <= size && data_off + data_size <= size, "section out of bounds (truncated file?)");
  BF_CHECK(data_off % kTensorAlignment == 0, "data section not 64-byte aligned");

  model->buffer_ = std::move(buffer);  // 64-byte aligned (heap copy) or page aligned (mapping)
  model->mapped_ = mapped;
  auto* base = static_cast<uint8_t*>(model->buffer_.get());
  model->file_bytes_ = size;

  Json doc = Json::parse(std::string(reinterpret_cast<const char*>(base + json_off), json_size));
  model->manifest_ = doc.contains("manifest") ? doc["manifest"] : Json(Json::Object{});
  model->data_sha256_ = doc.get_string("data_sha256", "");
  if (opts.verify_hash) {
    BF_CHECK(!model->data_sha256_.empty(), "model has no data_sha256; reload with verify_hash=false to skip");
    std::string actual = sha256_hex(base + data_off, data_size);
    if (actual != model->data_sha256_) fail("data section SHA-256 mismatch: expected ", model->data_sha256_, ", got ", actual);
    model->hash_verified_ = true;
  }

  Graph& g = model->graph_;
  auto add_tensor = [&](TensorInfo info) {
    if (g.tensor_ids.count(info.name)) fail("duplicate tensor name '", info.name, "'");
    g.tensor_ids[info.name] = static_cast<int>(g.tensors.size());
    g.tensors.push_back(std::move(info));
    return static_cast<int>(g.tensors.size()) - 1;
  };
  auto parse_shape = [](const Json& j) {
    Shape s;
    for (const auto& d : j.as_array()) s.push_back(d.as_int());
    return s;
  };

  for (const auto& t : doc["tensors"].as_array()) {
    TensorInfo info;
    info.name = t["name"].as_string();
    info.dtype = dtype_from_name(t["dtype"].as_string());
    info.shape = parse_shape(t["shape"]);
    info.layout = t.get_string("layout", "");
    info.is_constant = true;
    uint64_t off = static_cast<uint64_t>(t["offset"].as_int());
    uint64_t nbytes = static_cast<uint64_t>(t["nbytes"].as_int());
    BF_CHECK(nbytes == storage_bytes(info.dtype, info.shape), "tensor '", info.name, "' nbytes ", nbytes,
             " != expected ", storage_bytes(info.dtype, info.shape), " for ", dtype_name(info.dtype), shape_str(info.shape));
    BF_CHECK(off + nbytes <= data_size, "tensor '", info.name, "' out of data bounds");
    BF_CHECK(off % kTensorAlignment == 0, "tensor '", info.name, "' not 64-byte aligned");
    info.constant = Tensor(info.dtype, info.shape, base + data_off + off, model->buffer_);
    model->constant_bytes_ += nbytes;
    add_tensor(std::move(info));
  }
  for (const auto& t : doc["inputs"].as_array()) {
    TensorInfo info;
    info.name = t["name"].as_string();
    info.dtype = dtype_from_name(t["dtype"].as_string());
    info.shape = parse_shape(t["shape"]);
    g.inputs.push_back(add_tensor(std::move(info)));
  }
  for (const auto& n : doc["nodes"].as_array()) {
    Node node;
    node.op = n["op"].as_string();
    node.name = n.get_string("name", node.op + "_" + std::to_string(g.nodes.size()));
    node.attrs = n.contains("attrs") ? n["attrs"] : Json(Json::Object{});
    for (const auto& in : n["inputs"].as_array()) {
      if (in.is_null() || (in.is_string() && in.as_string().empty())) {
        node.inputs.push_back(-1);
        continue;
      }
      int id = g.find(in.as_string());
      if (id < 0) fail("node '", node.name, "' references unknown tensor '", in.as_string(), "' (graph must be topologically sorted)");
      node.inputs.push_back(id);
    }
    for (const auto& out : n["outputs"].as_array()) {
      TensorInfo info;
      info.name = out.as_string();
      node.outputs.push_back(add_tensor(std::move(info)));  // dtype/shape set by shape inference
    }
    g.nodes.push_back(std::move(node));
  }
  for (const auto& o : doc["outputs"].as_array()) {
    int id = g.find(o.as_string());
    if (id < 0) fail("unknown graph output '", o.as_string(), "'");
    g.outputs.push_back(id);
  }
  model->load_ms_ = now_ms() - t0;
  return model;
}

}  // namespace bitino

// Tensor storage and data types.
//
// Activations are NHWC. Binary tensors (DType::kBit) are bit-packed along the
// innermost (channel) dimension into uint32 words, little-endian bit order:
// element c lives in word c/32, bit c%32. Bit value 1 encodes -1, bit value 0
// encodes +1. Padding bits beyond the logical channel count are always 0.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bitino {

enum class DType : uint8_t { kF32, kI32, kI8, kU8, kBit };

const char* dtype_name(DType t);
DType dtype_from_name(const std::string& s);  // throws on unknown

using Shape = std::vector<int64_t>;

int64_t num_elements(const Shape& s);
std::string shape_str(const Shape& s);

constexpr int kBitsPerWord = 32;
inline int64_t bit_words(int64_t channels) { return (channels + kBitsPerWord - 1) / kBitsPerWord; }

// Bytes needed to store a tensor of `shape` and `dtype` (kBit packs the last dim).
size_t storage_bytes(DType dtype, const Shape& shape);

constexpr size_t kTensorAlignment = 64;

// Non-owning or owning view over tensor memory. Ownership is shared through
// `owner_`; a Tensor never frees memory it does not (co-)own.
class Tensor {
 public:
  Tensor() = default;
  Tensor(DType dtype, Shape shape, void* data, std::shared_ptr<void> owner = nullptr)
      : dtype_(dtype), shape_(std::move(shape)), data_(data), owner_(std::move(owner)) {}

  // Allocates zero-initialised, 64-byte aligned storage.
  static Tensor allocate(DType dtype, const Shape& shape);

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  int64_t dim(int i) const { return shape_[i < 0 ? shape_.size() + i : i]; }
  int rank() const { return static_cast<int>(shape_.size()); }
  int64_t size() const { return num_elements(shape_); }
  size_t bytes() const { return storage_bytes(dtype_, shape_); }
  bool empty() const { return data_ == nullptr; }

  void* raw() { return data_; }
  const void* raw() const { return data_; }
  template <typename T> T* data() { return static_cast<T*>(data_); }
  template <typename T> const T* data() const { return static_cast<const T*>(data_); }

  // Same storage, different logical shape (element count must match).
  Tensor reshaped(const Shape& s) const;

 private:
  DType dtype_ = DType::kF32;
  Shape shape_;
  void* data_ = nullptr;
  std::shared_ptr<void> owner_;
};

// 64-byte aligned buffer helper for kernel scratch and packed weights.
std::shared_ptr<void> aligned_alloc_shared(size_t bytes);

}  // namespace bitino

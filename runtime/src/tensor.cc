#include "bitino/tensor.h"

#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <malloc.h>
#endif

#include "bitino/error.h"

namespace bitino {

const char* dtype_name(DType t) {
  switch (t) {
    case DType::kF32: return "f32";
    case DType::kI32: return "i32";
    case DType::kI8: return "i8";
    case DType::kU8: return "u8";
    case DType::kBit: return "bit";
  }
  return "?";
}

DType dtype_from_name(const std::string& s) {
  if (s == "f32") return DType::kF32;
  if (s == "i32") return DType::kI32;
  if (s == "i8") return DType::kI8;
  if (s == "u8") return DType::kU8;
  if (s == "bit") return DType::kBit;
  fail("unknown dtype '", s, "'");
}

int64_t num_elements(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::string shape_str(const Shape& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

size_t storage_bytes(DType dtype, const Shape& shape) {
  switch (dtype) {
    case DType::kF32:
    case DType::kI32: return static_cast<size_t>(num_elements(shape)) * 4;
    case DType::kI8:
    case DType::kU8: return static_cast<size_t>(num_elements(shape));
    case DType::kBit: {
      if (shape.empty()) return 4;
      int64_t outer = num_elements(shape) / (shape.back() ? shape.back() : 1);
      if (shape.back() == 0) outer = 0;
      return static_cast<size_t>(outer * bit_words(shape.back()) * 4);
    }
  }
  return 0;
}

std::shared_ptr<void> aligned_alloc_shared(size_t bytes) {
  size_t rounded = (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;
  if (rounded == 0) rounded = kTensorAlignment;
#if defined(_WIN32)
  // The Windows C runtimes (MSVCRT / UCRT) have no aligned_alloc: _aligned_malloc pairs with _aligned_free.
  void* p = _aligned_malloc(rounded, kTensorAlignment);
  if (!p) fail("out of memory allocating ", bytes, " bytes");
  std::memset(p, 0, rounded);
  return std::shared_ptr<void>(p, [](void* q) { _aligned_free(q); });
#else
  // posix_memalign rather than std::aligned_alloc: bionic has aligned_alloc only from API 28, and the Android
  // library (android/bitino) targets API 24+. Both are released with free().
  void* p = nullptr;
  if (posix_memalign(&p, kTensorAlignment, rounded) != 0) p = nullptr;
  if (!p) fail("out of memory allocating ", bytes, " bytes");
  std::memset(p, 0, rounded);
  return std::shared_ptr<void>(p, [](void* q) { std::free(q); });
#endif
}

Tensor Tensor::allocate(DType dtype, const Shape& shape) {
  auto buf = aligned_alloc_shared(storage_bytes(dtype, shape));
  void* p = buf.get();
  return Tensor(dtype, shape, p, std::move(buf));
}

Tensor Tensor::reshaped(const Shape& s) const {
  BF_CHECK(num_elements(s) == size(), "reshape ", shape_str(shape_), " -> ", shape_str(s));
  BF_CHECK(dtype_ != DType::kBit || s.back() == shape_.back(), "bit tensors keep their last dim");
  return Tensor(dtype_, s, data_, owner_);
}

}  // namespace bitino

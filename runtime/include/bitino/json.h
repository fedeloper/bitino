// Minimal JSON DOM used for .bitino graph metadata and report output.
// Only what the runtime needs: parse, typed access with clear errors, dump.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bitino {

class Json {
 public:
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : kind_(Kind::kBool), b_(b) {}
  Json(double d) : kind_(Kind::kNumber), n_(d) {}
  Json(int v) : kind_(Kind::kNumber), n_(v) {}
  Json(int64_t v) : kind_(Kind::kNumber), n_(static_cast<double>(v)) {}
  Json(size_t v) : kind_(Kind::kNumber), n_(static_cast<double>(v)) {}
  Json(const char* s) : kind_(Kind::kString), s_(std::make_shared<std::string>(s)) {}
  Json(std::string s) : kind_(Kind::kString), s_(std::make_shared<std::string>(std::move(s))) {}
  Json(Array a) : kind_(Kind::kArray), a_(std::make_shared<Array>(std::move(a))) {}
  Json(Object o) : kind_(Kind::kObject), o_(std::make_shared<Object>(std::move(o))) {}

  static Json parse(const std::string& text);  // throws bitino::Error
  std::string dump(int indent = -1) const;

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::kNull; }
  bool is_object() const { return kind_ == Kind::kObject; }
  bool is_array() const { return kind_ == Kind::kArray; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_number() const { return kind_ == Kind::kNumber; }

  bool as_bool() const;
  double as_number() const;
  int64_t as_int() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;
  Array& mutable_array();
  Object& mutable_object();

  bool contains(const std::string& key) const;
  const Json& operator[](const std::string& key) const;  // throws if missing
  Json& operator[](const std::string& key);               // inserts (object only)
  const Json& operator[](size_t i) const;
  size_t size() const;

  // Typed lookups with defaults for optional attributes.
  int64_t get_int(const std::string& key, int64_t def) const;
  double get_number(const std::string& key, double def) const;
  std::string get_string(const std::string& key, const std::string& def) const;
  std::vector<int64_t> get_ints(const std::string& key, std::vector<int64_t> def) const;

  void push_back(Json v);

 private:
  void dump_to(std::string& out, int indent, int depth) const;
  Kind kind_ = Kind::kNull;
  bool b_ = false;
  double n_ = 0;
  std::shared_ptr<std::string> s_;
  std::shared_ptr<Array> a_;
  std::shared_ptr<Object> o_;
};

}  // namespace bitino

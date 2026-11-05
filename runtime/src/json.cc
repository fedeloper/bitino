#include "bitino/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "bitino/error.h"

namespace bitino {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& t) : t_(t) {}

  Json parse_document() {
    Json v = parse_value();
    skip_ws();
    if (i_ != t_.size()) error("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void error(const char* what) { fail("JSON parse error at offset ", i_, ": ", what); }

  void skip_ws() {
    while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\n' || t_[i_] == '\r' || t_[i_] == '\t')) ++i_;
  }

  bool consume(const char* lit) {
    size_t n = std::char_traits<char>::length(lit);
    if (t_.compare(i_, n, lit) == 0) {
      i_ += n;
      return true;
    }
    return false;
  }

  Json parse_value() {
    skip_ws();
    if (i_ >= t_.size()) error("unexpected end");
    char c = t_[i_];
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Json(parse_string());
    if (consume("true")) return Json(true);
    if (consume("false")) return Json(false);
    if (consume("null")) return Json();
    return parse_number();
  }

  Json parse_number() {
    const char* begin = t_.c_str() + i_;
    char* end = nullptr;
    double d = std::strtod(begin, &end);
    if (end == begin) error("invalid value");
    i_ += static_cast<size_t>(end - begin);
    return Json(d);
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  uint32_t parse_hex4() {
    if (i_ + 4 > t_.size()) error("bad \\u escape");
    uint32_t v = static_cast<uint32_t>(std::strtoul(t_.substr(i_, 4).c_str(), nullptr, 16));
    i_ += 4;
    return v;
  }

  std::string parse_string() {
    ++i_;  // opening quote
    std::string out;
    while (true) {
      if (i_ >= t_.size()) error("unterminated string");
      char c = t_[i_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= t_.size()) error("bad escape");
      char e = t_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && consume("\\u")) {
            uint32_t lo = parse_hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: error("bad escape");
      }
    }
    return out;
  }

  Json parse_array() {
    ++i_;
    Json::Array arr;
    skip_ws();
    if (i_ < t_.size() && t_[i_] == ']') {
      ++i_;
      return Json(std::move(arr));
    }
    while (true) {
      arr.push_back(parse_value());
      skip_ws();
      if (i_ < t_.size() && t_[i_] == ',') { ++i_; continue; }
      if (i_ < t_.size() && t_[i_] == ']') { ++i_; break; }
      error("expected , or ]");
    }
    return Json(std::move(arr));
  }

  Json parse_object() {
    ++i_;
    Json::Object obj;
    skip_ws();
    if (i_ < t_.size() && t_[i_] == '}') {
      ++i_;
      return Json(std::move(obj));
    }
    while (true) {
      skip_ws();
      if (i_ >= t_.size() || t_[i_] != '"') error("expected key");
      std::string key = parse_string();
      skip_ws();
      if (i_ >= t_.size() || t_[i_] != ':') error("expected :");
      ++i_;
      obj[key] = parse_value();
      skip_ws();
      if (i_ < t_.size() && t_[i_] == ',') { ++i_; continue; }
      if (i_ < t_.size() && t_[i_] == '}') { ++i_; break; }
      error("expected , or }");
    }
    return Json(std::move(obj));
  }

  const std::string& t_;
  size_t i_ = 0;
};

void escape_to(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

const char* kind_name(Json::Kind k) {
  switch (k) {
    case Json::Kind::kNull: return "null";
    case Json::Kind::kBool: return "bool";
    case Json::Kind::kNumber: return "number";
    case Json::Kind::kString: return "string";
    case Json::Kind::kArray: return "array";
    case Json::Kind::kObject: return "object";
  }
  return "?";
}

}  // namespace

Json Json::parse(const std::string& text) { return Parser(text).parse_document(); }

bool Json::as_bool() const {
  if (kind_ != Kind::kBool) fail("JSON: expected bool, got ", kind_name(kind_));
  return b_;
}
double Json::as_number() const {
  if (kind_ != Kind::kNumber) fail("JSON: expected number, got ", kind_name(kind_));
  return n_;
}
int64_t Json::as_int() const {
  double d = as_number();
  if (std::floor(d) != d) fail("JSON: expected integer, got ", d);
  return static_cast<int64_t>(d);
}
const std::string& Json::as_string() const {
  if (kind_ != Kind::kString) fail("JSON: expected string, got ", kind_name(kind_));
  return *s_;
}
const Json::Array& Json::as_array() const {
  if (kind_ != Kind::kArray) fail("JSON: expected array, got ", kind_name(kind_));
  return *a_;
}
const Json::Object& Json::as_object() const {
  if (kind_ != Kind::kObject) fail("JSON: expected object, got ", kind_name(kind_));
  return *o_;
}
// Copy-on-write: Json copies share storage until mutated.
Json::Array& Json::mutable_array() {
  if (kind_ != Kind::kArray) fail("JSON: expected array, got ", kind_name(kind_));
  if (a_.use_count() > 1) a_ = std::make_shared<Array>(*a_);
  return *a_;
}
Json::Object& Json::mutable_object() {
  if (kind_ == Kind::kNull) {
    kind_ = Kind::kObject;
    o_ = std::make_shared<Object>();
  }
  if (kind_ != Kind::kObject) fail("JSON: expected object, got ", kind_name(kind_));
  if (o_.use_count() > 1) o_ = std::make_shared<Object>(*o_);
  return *o_;
}

bool Json::contains(const std::string& key) const {
  return kind_ == Kind::kObject && o_->count(key) > 0;
}
const Json& Json::operator[](const std::string& key) const {
  const auto& o = as_object();
  auto it = o.find(key);
  if (it == o.end()) fail("JSON: missing key '", key, "'");
  return it->second;
}
Json& Json::operator[](const std::string& key) { return mutable_object()[key]; }
const Json& Json::operator[](size_t i) const {
  const auto& a = as_array();
  if (i >= a.size()) fail("JSON: index ", i, " out of range ", a.size());
  return a[i];
}
size_t Json::size() const {
  if (kind_ == Kind::kArray) return a_->size();
  if (kind_ == Kind::kObject) return o_->size();
  return 0;
}
void Json::push_back(Json v) {
  if (kind_ == Kind::kNull) {
    kind_ = Kind::kArray;
    a_ = std::make_shared<Array>();
  }
  mutable_array().push_back(std::move(v));
}

int64_t Json::get_int(const std::string& key, int64_t def) const {
  return contains(key) ? (*this)[key].as_int() : def;
}
double Json::get_number(const std::string& key, double def) const {
  return contains(key) ? (*this)[key].as_number() : def;
}
std::string Json::get_string(const std::string& key, const std::string& def) const {
  return contains(key) ? (*this)[key].as_string() : def;
}
std::vector<int64_t> Json::get_ints(const std::string& key, std::vector<int64_t> def) const {
  if (!contains(key)) return def;
  std::vector<int64_t> out;
  for (const auto& v : (*this)[key].as_array()) out.push_back(v.as_int());
  return out;
}

void Json::dump_to(std::string& out, int indent, int depth) const {
  auto newline = [&](int d) {
    if (indent < 0) return;
    out += '\n';
    out.append(static_cast<size_t>(indent * d), ' ');
  };
  switch (kind_) {
    case Kind::kNull: out += "null"; break;
    case Kind::kBool: out += b_ ? "true" : "false"; break;
    case Kind::kNumber: {
      if (!std::isfinite(n_)) {
        out += "null";
      } else if (std::floor(n_) == n_ && std::fabs(n_) < 9e15) {
        out += std::to_string(static_cast<int64_t>(n_));
      } else {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.9g", n_);
        out += buf;
      }
      break;
    }
    case Kind::kString: escape_to(out, *s_); break;
    case Kind::kArray: {
      out += '[';
      for (size_t i = 0; i < a_->size(); ++i) {
        if (i) out += ',';
        newline(depth + 1);
        (*a_)[i].dump_to(out, indent, depth + 1);
      }
      if (!a_->empty()) newline(depth);
      out += ']';
      break;
    }
    case Kind::kObject: {
      out += '{';
      bool first = true;
      for (const auto& kv : *o_) {
        if (!first) out += ',';
        first = false;
        newline(depth + 1);
        escape_to(out, kv.first);
        out += indent < 0 ? ":" : ": ";
        kv.second.dump_to(out, indent, depth + 1);
      }
      if (!o_->empty()) newline(depth);
      out += '}';
      break;
    }
  }
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_to(out, indent, 0);
  return out;
}

}  // namespace bitino

#include "config.hpp"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace cuscope {
namespace {

constexpr int kMinStackDepth = 1;
constexpr int kMaxStackDepth = 64;

// Parses the whole string as a base-10 integer; false on any trailing junk.
bool parse_int(const char* s, long long* out) {
  if (s == nullptr || *s == '\0') return false;
  errno = 0;
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (errno != 0 || *end != '\0') return false;
  *out = v;
  return true;
}

void warn(Config& c, const char* var, const char* value, const char* fallback) {
  c.warnings.push_back(std::string("ignoring ") + var + "=\"" + value + "\", using " + fallback);
}

}  // namespace

Config parse_config(const GetenvFn& getenv_fn) {
  Config c;

  if (const char* v = getenv_fn("CUSCOPE_OUT"); v != nullptr && *v != '\0') c.out = v;
  if (const char* v = getenv_fn("CUSCOPE_LOG"); v != nullptr) c.log_path = v;
  if (const char* v = getenv_fn("CUSCOPE_CENSUS"); v != nullptr) c.census_path = v;

  if (const char* v = getenv_fn("CUSCOPE_MODE"); v != nullptr) {
    if (std::strcmp(v, "alloc") == 0) {
      c.mode = Config::Mode::Alloc;
    } else if (std::strcmp(v, "full") == 0) {
      c.mode = Config::Mode::Full;
    } else {
      warn(c, "CUSCOPE_MODE", v, "full");
    }
  }

  if (const char* v = getenv_fn("CUSCOPE_STACK_DEPTH"); v != nullptr) {
    long long n = 0;
    if (!parse_int(v, &n)) {
      warn(c, "CUSCOPE_STACK_DEPTH", v, "16");
    } else if (n < kMinStackDepth || n > kMaxStackDepth) {
      c.stack_depth = n < kMinStackDepth ? kMinStackDepth : kMaxStackDepth;
      warn(c, "CUSCOPE_STACK_DEPTH", v, std::to_string(c.stack_depth).c_str());
    } else {
      c.stack_depth = static_cast<int>(n);
    }
  }

  if (const char* v = getenv_fn("CUSCOPE_SAMPLE_EVERY"); v != nullptr) {
    long long n = 0;
    if (!parse_int(v, &n) || n < 1 || n > UINT32_MAX) {
      warn(c, "CUSCOPE_SAMPLE_EVERY", v, "1");
    } else {
      c.sample_every = static_cast<uint32_t>(n);
    }
  }

  if (const char* v = getenv_fn("CUSCOPE_MEM_LIMIT"); v != nullptr) {
    if (!parse_size(v, &c.mem_limit_bytes)) {
      c.mem_limit_bytes = 0;
      warn(c, "CUSCOPE_MEM_LIMIT", v, "no cap");
    }
  }

  if (const char* v = getenv_fn("CUSCOPE_FLUSH_TIMEOUT_MS"); v != nullptr) {
    long long n = 0;
    if (!parse_int(v, &n) || n < 0 || n > UINT32_MAX) {
      warn(c, "CUSCOPE_FLUSH_TIMEOUT_MS", v, "2000");
    } else {
      c.flush_timeout_ms = static_cast<uint32_t>(n);
    }
  }

  return c;
}

bool parse_size(const char* s, uint64_t* out) {
  if (s == nullptr || !std::isdigit(static_cast<unsigned char>(*s))) return false;
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(s, &end);
  if (errno != 0 || end == s) return false;
  bool has_fraction = false;  // strtod also takes hex and exponents: allow digits and one '.'
  for (const char* p = s; p < end; ++p) {
    if (*p == '.' && !has_fraction) {
      has_fraction = true;
    } else if (!std::isdigit(static_cast<unsigned char>(*p))) {
      return false;
    }
  }

  int shift = 0;
  switch (std::toupper(static_cast<unsigned char>(*end))) {
    case 'K': shift = 10; break;
    case 'M': shift = 20; break;
    case 'G': shift = 30; break;
    case 'T': shift = 40; break;
    case '\0': break;
    default: return false;
  }
  if (shift != 0) {
    ++end;
    if (std::toupper(static_cast<unsigned char>(*end)) == 'I') ++end;
    if (std::toupper(static_cast<unsigned char>(*end)) == 'B') ++end;
  }
  if (*end != '\0' || (has_fraction && shift == 0)) return false;

  const double bytes = value * static_cast<double>(1ull << shift);
  if (!(bytes >= 1.0) || bytes >= 18446744073709551615.0) return false;
  *out = static_cast<uint64_t>(bytes);
  return true;
}

const Config& config() {
  static const Config c = parse_config([](const char* key) { return std::getenv(key); });
  return c;
}

std::string expand_out_path(const std::string& pattern, int pid, bool root) {
  const std::string pid_str = std::to_string(pid);
  std::string out;
  out.reserve(pattern.size() + pid_str.size());
  for (size_t i = 0; i < pattern.size(); ++i) {
    if (pattern[i] == '%' && i + 1 < pattern.size() && pattern[i + 1] == 'p') {
      out += pid_str;
      ++i;
    } else {
      out += pattern[i];
    }
  }
  if (root || pattern.find("%p") != std::string::npos) return out;
  const size_t slash = out.rfind('/');
  const size_t dot = out.rfind('.');
  const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash + 1);
  const size_t at = has_ext ? dot : out.size();
  return out.substr(0, at) + "." + pid_str + out.substr(at);
}

const char* mode_name(Config::Mode mode) {
  return mode == Config::Mode::Alloc ? "alloc" : "full";
}

}  // namespace cuscope

#include "config.hpp"

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

const Config& config() {
  static const Config c = parse_config([](const char* key) { return std::getenv(key); });
  return c;
}

std::string expand_out_path(const std::string& pattern, int pid) {
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
  return out;
}

const char* mode_name(Config::Mode mode) {
  return mode == Config::Mode::Alloc ? "alloc" : "full";
}

}  // namespace cuscope

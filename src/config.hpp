#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cuscope {

// Settings read once from CUSCOPE_* environment variables.
struct Config {
  enum class Mode { Alloc, Full };

  std::string out = "cuscope-%p.json";  // CUSCOPE_OUT; %p becomes the pid
  std::string log_path;                 // CUSCOPE_LOG; empty = stderr
  std::string census_path;              // CUSCOPE_CENSUS; empty = census off
  Mode mode = Mode::Full;               // CUSCOPE_MODE: alloc | full
  int stack_depth = 16;                 // CUSCOPE_STACK_DEPTH, clamped to 1..64
  uint32_t sample_every = 1;            // CUSCOPE_SAMPLE_EVERY, >= 1

  // One message per variable that was invalid and fell back to its default.
  std::vector<std::string> warnings;
};

using GetenvFn = std::function<const char*(const char*)>;

Config parse_config(const GetenvFn& getenv_fn);

// Parsed from the real environment on first call. Safe to call before
// cuscope's constructors have run.
const Config& config();

std::string expand_out_path(const std::string& pattern, int pid);

const char* mode_name(Config::Mode mode);

}  // namespace cuscope

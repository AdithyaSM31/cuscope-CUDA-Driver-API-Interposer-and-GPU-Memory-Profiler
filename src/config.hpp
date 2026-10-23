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
  uint32_t flush_timeout_ms = 2000;     // CUSCOPE_FLUSH_TIMEOUT_MS, >= 0
  uint64_t mem_limit_bytes = 0;         // CUSCOPE_MEM_LIMIT; 0 = no cap

  // One message per variable that was invalid and fell back to its default.
  std::vector<std::string> warnings;
};

using GetenvFn = std::function<const char*(const char*)>;

Config parse_config(const GetenvFn& getenv_fn);

// A positive size: plain bytes ("268435456"), or a number with a binary
// suffix K, M, G or T, optionally followed by B or iB, case-insensitive
// ("512M", "1.5GiB"). Decimals need a suffix. False for anything else,
// including zero.
bool parse_size(const char* s, uint64_t* out);

// True if PYTORCH_CUDA_ALLOC_CONF (or its newer name PYTORCH_ALLOC_CONF) turns
// on expandable_segments. PyTorch then allocates through the virtual memory
// APIs (cuMemCreate, cuMemMap), which the memory cap does not see.
bool expandable_segments_enabled(const GetenvFn& getenv_fn);

// Parsed from the real environment on first call. Safe to call before
// cuscope's constructors have run.
const Config& config();

// The trace path for process `pid`: %p becomes the pid. A pattern without %p
// is the exact path only for the root process (the one that started the
// run); any other process inserts ".<pid>" before the extension, so a child
// never overwrites the root's trace.
std::string expand_out_path(const std::string& pattern, int pid, bool root = true);

const char* mode_name(Config::Mode mode);

}  // namespace cuscope

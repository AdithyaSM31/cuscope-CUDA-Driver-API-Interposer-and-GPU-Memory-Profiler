#pragma once

#include <atomic>
#include <cstdint>

// Counters behind cuscope_get_stats(). Zero-initialised atomics are constant
// initialised, so they work before cuscope's constructors have run.
namespace cuscope::stats {

inline std::atomic<uint64_t> dlsym_wrapped{0};
inline std::atomic<uint64_t> gpa_wrapped{0};
inline std::atomic<uint64_t> gpa_passthrough{0};
inline std::atomic<uint64_t> direct_calls{0};
inline std::atomic<uint64_t> hooked_calls{0};
inline std::atomic<uint64_t> live_bytes{0};
inline std::atomic<uint64_t> live_allocs{0};
inline std::atomic<uint64_t> invalid_frees{0};

}  // namespace cuscope::stats

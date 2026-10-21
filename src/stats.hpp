#pragma once

#include <atomic>
#include <cstdint>
#include <initializer_list>

// Interception counters behind cuscope_get_stats(). Zero-initialised atomics
// are constant initialised, so they work before cuscope's constructors have
// run. Live-memory figures come from the tracker instead.
namespace cuscope::stats {

inline std::atomic<uint64_t> dlsym_wrapped{0};
inline std::atomic<uint64_t> gpa_wrapped{0};
inline std::atomic<uint64_t> gpa_passthrough{0};
inline std::atomic<uint64_t> direct_calls{0};
inline std::atomic<uint64_t> hooked_calls{0};

// True once the program has made a CUDA driver call through cuscope. A
// process that never did (a shell, a compiler, a worker that inherited
// LD_PRELOAD) gets no trace and no summary.
inline bool saw_cuda() { return hooked_calls.load() != 0 || direct_calls.load() != 0; }

// A forked child counts only its own calls.
inline void after_fork_child() {
  for (auto* counter : {&dlsym_wrapped, &gpa_wrapped, &gpa_passthrough, &direct_calls, &hooked_calls}) {
    counter->store(0);
  }
}

}  // namespace cuscope::stats

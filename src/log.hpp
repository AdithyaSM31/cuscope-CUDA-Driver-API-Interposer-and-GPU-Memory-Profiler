#pragma once

#include <atomic>
#include <cstddef>

namespace cuscope {

// Writes "[cuscope] <message>\n" to CUSCOPE_LOG (appending) or stderr.
// Never writes to stdout. Each message is a single write(2).
void log_msg(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Writes preformatted text (the exit summary) to the same place, unprefixed.
void log_write(const char* text, size_t length);

}  // namespace cuscope

// Logs only the first time this call site runs.
#define CS_LOG_ONCE(...)                                  \
  do {                                                    \
    static std::atomic<bool> cs_log_once_done_{false};    \
    if (!cs_log_once_done_.exchange(true)) {              \
      ::cuscope::log_msg(__VA_ARGS__);                    \
    }                                                     \
  } while (0)

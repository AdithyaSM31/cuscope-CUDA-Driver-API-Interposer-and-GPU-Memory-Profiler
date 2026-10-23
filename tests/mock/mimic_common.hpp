// Output shared by runtime_mimic and direct_mimic:
//   OK <scenario>
//   STATS {...}   only when cuscope is loaded
//   MOCK {...}    the mock driver's counters
#pragma once

#include <dlfcn.h>

#include <cstdio>

#include "cuscope/cuscope.h"
#include "mock_counters.h"

inline void print_stats() {
  using StatsFn = int (*)(cuscope_stats*);
  auto fn = reinterpret_cast<StatsFn>(dlsym(RTLD_DEFAULT, "cuscope_get_stats"));
  if (fn == nullptr) return;
  cuscope_stats s{};
  fn(&s);
  std::printf(
      "STATS {\"dlsym_wrapped\": %llu, \"gpa_wrapped\": %llu, \"gpa_passthrough\": %llu, "
      "\"direct_calls\": %llu, \"hooked_calls\": %llu, \"live_bytes\": %llu, "
      "\"live_allocs\": %llu, \"invalid_frees\": %llu}\n",
      (unsigned long long)s.dlsym_wrapped, (unsigned long long)s.gpa_wrapped,
      (unsigned long long)s.gpa_passthrough, (unsigned long long)s.direct_calls,
      (unsigned long long)s.hooked_calls, (unsigned long long)s.live_bytes,
      (unsigned long long)s.live_allocs, (unsigned long long)s.invalid_frees);
}

inline void print_mock(void (*get_counters)(mock_counters*)) {
  mock_counters c{};
  get_counters(&c);
  std::printf(
      "MOCK {\"alloc_v1\": %llu, \"alloc_v2\": %llu, \"free_\": %llu, \"memcpy_htod\": %llu, "
      "\"memcpy_dtoh\": %llu, \"memcpy_async\": %llu, \"memcpy_async_ptsz\": %llu, "
      "\"launch\": %llu, \"launch_ptsz\": %llu, \"event_record\": %llu, "
      "\"event_record_on_capturing\": %llu, \"ctx_destroy\": %llu, \"event_query\": %llu, \"event_sync\": %llu, \"live_hwm\": %llu}\n",
      (unsigned long long)c.alloc_v1, (unsigned long long)c.alloc_v2, (unsigned long long)c.free_,
      (unsigned long long)c.memcpy_htod, (unsigned long long)c.memcpy_dtoh,
      (unsigned long long)c.memcpy_async, (unsigned long long)c.memcpy_async_ptsz,
      (unsigned long long)c.launch, (unsigned long long)c.launch_ptsz,
      (unsigned long long)c.event_record, (unsigned long long)c.event_record_on_capturing,
      (unsigned long long)c.ctx_destroy, (unsigned long long)c.event_query,
      (unsigned long long)c.event_sync, (unsigned long long)c.live_hwm);
}

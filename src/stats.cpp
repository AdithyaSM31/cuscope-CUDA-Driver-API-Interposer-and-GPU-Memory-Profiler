#include "stats.hpp"

#include "cuscope/cuscope.h"

extern "C" CUSCOPE_API int cuscope_get_stats(cuscope_stats* out) {
  if (out == nullptr) return -1;
  using namespace cuscope::stats;
  constexpr auto r = std::memory_order_relaxed;
  out->dlsym_wrapped = dlsym_wrapped.load(r);
  out->gpa_wrapped = gpa_wrapped.load(r);
  out->gpa_passthrough = gpa_passthrough.load(r);
  out->direct_calls = direct_calls.load(r);
  out->hooked_calls = hooked_calls.load(r);
  out->live_bytes = live_bytes.load(r);
  out->live_allocs = live_allocs.load(r);
  out->invalid_frees = invalid_frees.load(r);
  return 0;
}

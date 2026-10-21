#include "stats.hpp"

#include "cuscope/cuscope.h"
#include "tracker.hpp"

extern "C" CUSCOPE_API int cuscope_get_stats(cuscope_stats* out) {
  if (out == nullptr) return -1;
  using namespace cuscope::stats;
  constexpr auto r = std::memory_order_relaxed;
  out->dlsym_wrapped = dlsym_wrapped.load(r);
  out->gpa_wrapped = gpa_wrapped.load(r);
  out->gpa_passthrough = gpa_passthrough.load(r);
  out->direct_calls = direct_calls.load(r);
  out->hooked_calls = hooked_calls.load(r);
  const cuscope::Tracker& t = cuscope::tracker();
  out->live_bytes = t.live_bytes();
  out->live_allocs = t.live_allocs();
  out->invalid_frees = t.invalid_free_count();
  return 0;
}

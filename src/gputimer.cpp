#include "gputimer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "guard.hpp"
#include "loader.hpp"
#include "log.hpp"
#include "recorder.hpp"

namespace cuscope {

struct Pending {
  CUevent start, end;
  GpuSpan meta;
};

// Everything cuscope keeps per CUDA context. Guarded by mu.
struct CtxState {
  CUcontext ctx = nullptr;
  int device = -1;
  CUevent reference = nullptr;
  uint64_t reference_cpu_ns = 0;

  std::mutex mu;
  std::vector<CUevent> pool;  // idle events
  size_t created = 0;         // events created for this context
  std::vector<Pending> pending;
  bool retired = false;       // flushed for teardown; must not be used again
};

namespace {

constexpr size_t kPoolBatch = 256;  // events created at a time
constexpr size_t kPoolMax = 4096;   // per context; beyond this, calls go untimed

// ---- Driver functions cuscope calls itself ----

struct DriverFns {
  decltype(&::cuCtxGetCurrent) ctx_get_current = nullptr;
  decltype(&::cuCtxSetCurrent) ctx_set_current = nullptr;
  decltype(&::cuCtxGetDevice) ctx_get_device = nullptr;
  decltype(&::cuEventCreate) event_create = nullptr;
  decltype(&::cuEventRecord) event_record[2] = {};  // per stream variant
  decltype(&::cuStreamIsCapturing) is_capturing[2] = {};
  decltype(&::cuEventQuery) event_query = nullptr;
  decltype(&::cuEventSynchronize) event_sync = nullptr;
  decltype(&::cuEventElapsedTime) elapsed = nullptr;
  bool ok = false;
};

const DriverFns& fns() {
  static const DriverFns f = [] {
    DriverFns d;
    d.ctx_get_current = CS_DRIVER_FN(cuCtxGetCurrent);
    d.ctx_set_current = CS_DRIVER_FN(cuCtxSetCurrent);
    d.ctx_get_device = CS_DRIVER_FN(cuCtxGetDevice);
    d.event_create = CS_DRIVER_FN(cuEventCreate);
    d.event_record[0] = CS_DRIVER_FN(cuEventRecord);
    d.event_record[1] = CS_DRIVER_FN_PTSZ(cuEventRecord);
    d.is_capturing[0] = CS_DRIVER_FN(cuStreamIsCapturing);
    d.is_capturing[1] = CS_DRIVER_FN_PTSZ(cuStreamIsCapturing);
    d.event_query = CS_DRIVER_FN(cuEventQuery);
    d.event_sync = CS_DRIVER_FN(cuEventSynchronize);
    d.elapsed = CS_DRIVER_FN(cuEventElapsedTime);
    d.ok = d.ctx_get_current && d.ctx_set_current && d.ctx_get_device && d.event_create && d.event_record[0] &&
           d.event_record[1] && d.is_capturing[0] && d.is_capturing[1] && d.event_query && d.event_sync &&
           d.elapsed;
    if (!d.ok) log_msg("CUDA event functions not found; GPU timing disabled");
    return d;
  }();
  return f;
}

// Makes `ctx` current on this thread for the scope, then restores whatever
// was current before, so the program's thread state is unchanged.
class ScopedContext {
 public:
  explicit ScopedContext(CUcontext ctx) {
    fns().ctx_get_current(&previous_);
    ok_ = previous_ == ctx || fns().ctx_set_current(ctx) == CUDA_SUCCESS;
  }
  ~ScopedContext() {
    CUcontext now = nullptr;
    fns().ctx_get_current(&now);
    if (now != previous_) fns().ctx_set_current(previous_);
  }
  bool ok() const { return ok_; }

 private:
  CUcontext previous_ = nullptr;
  bool ok_ = false;
};

// ---- Process-wide state (heap, never destroyed) ----

std::mutex g_states_mu;
std::unordered_map<uint64_t, CtxState*>& states() {
  static auto* m = new std::unordered_map<uint64_t, CtxState*>();
  return *m;
}
std::atomic<uint64_t> g_generation{1};  // bumped whenever a context is forgotten

std::mutex g_spans_mu;
std::vector<GpuSpan>& spans() {
  static auto* v = new std::vector<GpuSpan>();
  return *v;
}

std::atomic<uint64_t> g_unresolved{0};
std::atomic<uint64_t> g_untimed{0};

// Each thread remembers the last context it timed on, to skip the map lookup.
struct StateCache {
  uint64_t ctx;
  CtxState* state;
  uint64_t generation;
};
thread_local StateCache t_cache __attribute__((tls_model("initial-exec"))) = {0, nullptr, 0};

void poll_all();
void resolve_all_before_driver_exit();

// The driver registers its own exit-time cleanup when CUDA is initialised
// and, once that has run, every call fails with CUDA_ERROR_DEINITIALIZED. On
// WSL it runs before the CUDA runtime's cleanup (so before the context-
// teardown hooks) and before cuscope's own exit handler. Exit handlers run in
// reverse order of registration, so a handler registered here, after the
// driver is initialised, runs before the driver's and can still read events.
void register_exit_flush_once() {
  static std::atomic<bool> registered{false};
  if (!registered.exchange(true)) std::atexit(resolve_all_before_driver_exit);
}

// First timed call on a context: record a reference event on the legacy
// default stream, wait for it, and note the CPU time. Every later GPU time is
// measured from it. Runs once per context, so the wait is a one-off cost.
CtxState* create_state(CUcontext ctx) {
  const DriverFns& f = fns();
  auto* st = new CtxState();
  st->ctx = ctx;
  CUdevice dev = -1;
  if (f.ctx_get_device(&dev) == CUDA_SUCCESS) st->device = dev;
  if (f.event_create(&st->reference, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
      f.event_record[0](st->reference, nullptr) != CUDA_SUCCESS || f.event_sync(st->reference) != CUDA_SUCCESS) {
    CS_LOG_ONCE("cannot create a reference event; GPU timing disabled for this context");
    delete st;
    return nullptr;
  }
  st->reference_cpu_ns = now_ns();
  set_collector_poll(poll_all);
  register_exit_flush_once();
  return st;
}

CtxState* state_for_current() {
  CUcontext ctx = nullptr;
  if (fns().ctx_get_current(&ctx) != CUDA_SUCCESS || ctx == nullptr) return nullptr;
  const uint64_t key = reinterpret_cast<uint64_t>(ctx);
  const uint64_t generation = g_generation.load(std::memory_order_acquire);
  if (t_cache.ctx == key && t_cache.generation == generation) return t_cache.state;

  std::lock_guard<std::mutex> lock(g_states_mu);
  CtxState*& st = states()[key];
  if (st == nullptr) st = create_state(ctx);
  if (st == nullptr) {
    states().erase(key);
    return nullptr;
  }
  t_cache = {key, st, generation};
  return st;
}

bool take_events(CtxState* st, CUevent* a, CUevent* b) {
  std::lock_guard<std::mutex> lock(st->mu);
  if (st->retired) return false;
  while (st->pool.size() < 2 && st->created < kPoolMax) {
    const size_t batch = std::min(kPoolBatch, kPoolMax - st->created);
    for (size_t i = 0; i < batch; ++i) {
      CUevent ev = nullptr;
      if (fns().event_create(&ev, CU_EVENT_DEFAULT) != CUDA_SUCCESS) return false;
      st->pool.push_back(ev);
      st->created++;
    }
  }
  if (st->pool.size() < 2) return false;  // every event is in flight
  *a = st->pool.back();
  st->pool.pop_back();
  *b = st->pool.back();
  st->pool.pop_back();
  return true;
}

void give_back(CtxState* st, CUevent a, CUevent b) {
  std::lock_guard<std::mutex> lock(st->mu);
  if (st->retired) return;  // the context is going away; so are its events
  st->pool.push_back(a);
  st->pool.push_back(b);
}

// Converts a completed pair to a span. False if the driver cannot say.
bool resolve(const CtxState* st, const Pending& p, GpuSpan* out) {
  float start_ms = 0.f, end_ms = 0.f;
  CUresult r = fns().elapsed(&start_ms, st->reference, p.start);
  if (r == CUDA_SUCCESS) r = fns().elapsed(&end_ms, st->reference, p.end);
  if (r != CUDA_SUCCESS) {
    CS_LOG_ONCE("cannot read GPU event times (CUresult %d); those spans are counted as unresolved", r);
    return false;
  }
  *out = p.meta;
  out->start_ns = st->reference_cpu_ns + static_cast<uint64_t>(std::llround(std::max(0.f, start_ms) * 1e6));
  out->end_ns = st->reference_cpu_ns + static_cast<uint64_t>(std::llround(std::max(0.f, end_ms) * 1e6));
  return true;
}

// Resolves whatever has completed. With wait, first waits for each end event
// (teardown and exit, when nothing may be left behind).
void poll_state(CtxState* st, bool wait) {
  std::lock_guard<std::mutex> lock(st->mu);
  if (st->pending.empty()) return;
  ScopedContext scoped(st->ctx);
  std::vector<GpuSpan> done;
  std::vector<Pending> still;
  for (const Pending& p : st->pending) {
    CUresult r = scoped.ok() ? (wait ? fns().event_sync(p.end) : fns().event_query(p.end)) : CUDA_ERROR_INVALID_CONTEXT;
    GpuSpan span;
    if (r == CUDA_SUCCESS && resolve(st, p, &span)) {
      done.push_back(span);
      st->pool.push_back(p.start);
      st->pool.push_back(p.end);
    } else if (r == CUDA_ERROR_NOT_READY) {
      still.push_back(p);
    } else {
      if (r != CUDA_SUCCESS) CS_LOG_ONCE("cannot query a GPU event (CUresult %d); it is counted as unresolved", r);
      g_unresolved.fetch_add(1, std::memory_order_relaxed);  // the driver can no longer tell us
    }
  }
  st->pending.swap(still);
  if (!done.empty()) {
    std::lock_guard<std::mutex> spans_lock(g_spans_mu);
    spans().insert(spans().end(), done.begin(), done.end());
  }
}

std::vector<CtxState*> known_states() {
  std::lock_guard<std::mutex> lock(g_states_mu);
  std::vector<CtxState*> out;
  for (auto& [key, st] : states()) out.push_back(st);
  return out;
}

// The collector's hook, every 2 ms.
void poll_all() {
  for (CtxState* st : known_states()) poll_state(st, /*wait=*/false);
}

// Exit handler (see register_exit_flush_once): wait for and resolve every
// pending event while the driver still answers. Contexts stay known; the
// teardown hooks and gputimer_finish deal with whatever comes after.
void resolve_all_before_driver_exit() {
  InternalScope internal;
  for (CtxState* st : known_states()) poll_state(st, /*wait=*/true);
}

// Removes the context from the map (so no new call uses it), resolves all its
// pending events, and retires it. The CtxState itself is never freed: a
// thread may still hold a pointer from its cache.
void flush_and_forget(CtxState* st) {
  poll_state(st, /*wait=*/true);
  std::lock_guard<std::mutex> lock(st->mu);
  st->retired = true;
  g_unresolved.fetch_add(st->pending.size(), std::memory_order_relaxed);
  st->pending.clear();
}

}  // namespace

bool gpu_begin(int variant, CUstream stream, Timing* t) {
  if (config().mode == Config::Mode::Alloc || in_internal()) return false;
  const DriverFns& f = fns();
  if (!f.ok) return false;
  InternalScope internal;

  // An event recorded on a capturing stream would become a node in the
  // program's graph. If the driver cannot tell us, do not risk it.
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (f.is_capturing[variant](stream, &status) != CUDA_SUCCESS || status != CU_STREAM_CAPTURE_STATUS_NONE) {
    g_untimed.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  CtxState* st = state_for_current();
  CUevent a = nullptr, b = nullptr;
  if (st == nullptr || !take_events(st, &a, &b)) {
    g_untimed.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (f.event_record[variant](a, stream) != CUDA_SUCCESS) {
    give_back(st, a, b);
    g_untimed.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  *t = Timing{a, b, st};
  return true;
}

bool gpu_sample_launch() {
  static thread_local uint64_t t_launches __attribute__((tls_model("initial-exec"))) = 0;
  const uint32_t every = config().sample_every;
  if (every <= 1 || config().mode == Config::Mode::Alloc || in_internal()) return true;
  if (t_launches++ % every == 0) return true;
  g_untimed.fetch_add(1, std::memory_order_relaxed);
  return false;
}

void gpu_end(int variant, CUstream stream, const Timing& t, const GpuSpan& meta) {
  InternalScope internal;
  if (fns().event_record[variant](t.end, stream) != CUDA_SUCCESS) {
    give_back(t.state, t.start, t.end);
    g_untimed.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  std::lock_guard<std::mutex> lock(t.state->mu);
  if (t.state->retired) {
    g_unresolved.fetch_add(1, std::memory_order_relaxed);  // raced with teardown
    return;
  }
  t.state->pending.push_back({t.start, t.end, meta});
}

void gpu_cancel(const Timing& t) { give_back(t.state, t.start, t.end); }

void gpu_flush_ctx(uint64_t ctx) {
  if (!fns().ok) return;
  InternalScope internal;
  CtxState* st = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_states_mu);
    auto it = states().find(ctx);
    if (it == states().end()) return;
    st = it->second;
    states().erase(it);
    g_generation.fetch_add(1, std::memory_order_acq_rel);
  }
  flush_and_forget(st);
}

void gpu_flush_device(int device) {
  if (!fns().ok) return;
  InternalScope internal;
  std::vector<CtxState*> gone;
  {
    std::lock_guard<std::mutex> lock(g_states_mu);
    for (auto it = states().begin(); it != states().end();) {
      if (it->second->device == device) {
        gone.push_back(it->second);
        it = states().erase(it);
      } else {
        ++it;
      }
    }
    if (!gone.empty()) g_generation.fetch_add(1, std::memory_order_acq_rel);
  }
  for (CtxState* st : gone) flush_and_forget(st);
}

void gputimer_finish() {
  if (!fns().ok) return;
  InternalScope internal;
  std::vector<CtxState*> all;
  {
    std::lock_guard<std::mutex> lock(g_states_mu);
    for (auto& [key, st] : states()) all.push_back(st);
    states().clear();
    g_generation.fetch_add(1, std::memory_order_acq_rel);
  }
  for (CtxState* st : all) flush_and_forget(st);
}

void gputimer_fill(Snapshot& s) {
  {
    std::lock_guard<std::mutex> lock(g_spans_mu);
    s.spans = spans();
  }
  s.unresolved_events = g_unresolved.load(std::memory_order_relaxed);
  s.gpu_untimed = g_untimed.load(std::memory_order_relaxed);
}

}  // namespace cuscope

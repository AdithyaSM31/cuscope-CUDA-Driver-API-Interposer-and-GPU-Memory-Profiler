#include "sched_client.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>

#include "config.hpp"
#include "guard.hpp"
#include "loader.hpp"
#include "log.hpp"
#include "recorder.hpp"
#include "sched_core.hpp"

namespace cuscope {
namespace {

uint64_t mono_now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

void sleep_ns(uint64_t ns) {
  timespec ts{static_cast<time_t>(ns / 1'000'000'000), static_cast<long>(ns % 1'000'000'000)};
  nanosleep(&ts, nullptr);
}

// The daemon's segment, mapped read-write; nullptr if there is none, or if it
// is not private to this user (*unsafe is then set), or if its version/size mismatches (*version_mismatch is set).
SchedShm* map_segment(bool* unsafe, bool* version_mismatch) {
  *unsafe = false;
  *version_mismatch = false;
  const int fd = shm_open(sched_shm_name().c_str(), O_RDWR, 0);
  if (fd < 0) return nullptr;
  if (!sched_shm_private(fd)) {
    close(fd);
    *unsafe = true;
    return nullptr;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < sizeof(SchedShm)) {
    close(fd);
    *version_mismatch = true;
    return nullptr;
  }
  void* p = mmap(nullptr, sizeof(SchedShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return nullptr;
  auto* shm = static_cast<SchedShm*>(p);
  if (shm->magic != kSchedMagic || shm->version != kSchedVersion) {
    munmap(p, sizeof(SchedShm));
    *version_mismatch = true;
    return nullptr;
  }
  return shm;
}

// A wait, on the launching thread's timeline.
void record_wait(uint64_t start_ns, uint64_t dur_ns) {
  Record r{};
  r.type = RecType::Throttled;
  r.cpu_ns = start_ns;
  r.dur_ns = dur_ns > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(dur_ns);
  record(r);
}

// Launches on a stream being captured into a graph do not run now: they are
// neither charged nor delayed. If the driver cannot say, do not charge.
bool capturing(int variant, CUstream stream) {
  static const decltype(&::cuStreamIsCapturing) is_capturing[2] = {CS_DRIVER_FN(cuStreamIsCapturing),
                                                                   CS_DRIVER_FN_PTSZ(cuStreamIsCapturing)};
  const auto fn = is_capturing[variant];
  if (fn == nullptr) return true;
  InternalScope internal;
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  return fn(stream, &status) != CUDA_SUCCESS || status != CU_STREAM_CAPTURE_STATUS_NONE;
}

std::atomic<bool> g_stop_logged{false};

// Created at the first launch and never destroyed (an exit handler may still
// use it). No lock: a fork must never inherit one held by another thread.
std::atomic<SchedClient*> g_client{nullptr};

SchedClient* client() {
  if (SchedClient* c = g_client.load(std::memory_order_acquire)) return c;
  const uint32_t weight = config().share_weight;
  SchedClient* made = nullptr;
  if (config().mode == Config::Mode::Alloc) {
    made = new SchedClient(nullptr, static_cast<int32_t>(getpid()), weight, mono_now, sleep_ns);
    made->fail_open(SchedState::AllocMode);
  } else {
    bool unsafe = false, mismatch = false;
    SchedShm* shm = map_segment(&unsafe, &mismatch);
    made = new SchedClient(shm, static_cast<int32_t>(getpid()), weight, mono_now, sleep_ns);
    if (unsafe) made->fail_open(SchedState::Unsafe);
    if (mismatch) made->fail_open(SchedState::VersionMismatch);
  }
  made->set_on_wait(record_wait);
  SchedClient* expected = nullptr;
  if (!g_client.compare_exchange_strong(expected, made, std::memory_order_acq_rel)) {
    made->release();  // another thread registered first
    return expected;  // `made` is leaked on purpose: it is tiny and this race is one-off
  }
  switch (made->state()) {
    case SchedState::Registered: break;
    case SchedState::NoScheduler:
      log_msg("compute sharing off: no scheduler running (start cuscope sched)");
      break;
    default:
      log_msg("compute sharing off: %s", sched_reason(made->state()));
      break;
  }
  return made;
}

}  // namespace

uint64_t sched_before_launch(int variant, CUstream stream, uint64_t key) {
  if (config().share_weight == 0 || in_internal()) return 0;
  SchedClient* c = client();
  if (c->state() != SchedState::Registered || capturing(variant, stream)) return 0;
  const uint64_t charged = c->before_launch(key);
  if (c->state() == SchedState::Stopped && !g_stop_logged.exchange(true)) {
    log_msg("compute sharing off: scheduler stopped (running unthrottled from now on)");
  }
  return charged;
}

void sched_on_span(const GpuSpan& span) {
  if (span.charged_ns == 0) return;
  if (SchedClient* c = g_client.load(std::memory_order_acquire)) {
    c->on_span(span.func, span.charged_ns, span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0);
  }
}

void sched_refund(uint64_t charged_ns) {
  if (charged_ns == 0) return;
  if (SchedClient* c = g_client.load(std::memory_order_acquire)) {
    c->refund(charged_ns);
  }
}

void sched_fill(Snapshot& s) {
  s.sched = SchedStats{};
  if (config().share_weight == 0) return;
  if (SchedClient* c = g_client.load(std::memory_order_acquire)) {
    s.sched = c->stats();
  } else {
    s.sched.enabled = true;  // asked to share, but never launched a kernel
    s.sched.weight = config().share_weight;
    s.sched.reason = "no kernel launches";
  }
}

void sched_release_at_exit() {
  if (SchedClient* c = g_client.load(std::memory_order_acquire)) c->release();
}

void sched_after_fork_child() {
  g_client.store(nullptr, std::memory_order_release);
  g_stop_logged.store(false, std::memory_order_relaxed);
}

}  // namespace cuscope

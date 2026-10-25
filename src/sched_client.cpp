#include "sched_client.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>

#include "config.hpp"
#include "guard.hpp"
#include "log.hpp"
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

// The daemon's segment, mapped read-write; nullptr if there is none.
SchedShm* map_segment() {
  const int fd = shm_open(sched_shm_name().c_str(), O_RDWR, 0);
  if (fd < 0) return nullptr;
  struct stat st {};
  void* p = MAP_FAILED;
  if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= sizeof(SchedShm)) {
    p = mmap(nullptr, sizeof(SchedShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  }
  close(fd);
  return p == MAP_FAILED ? nullptr : static_cast<SchedShm*>(p);
}

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
    made = new SchedClient(map_segment(), static_cast<int32_t>(getpid()), weight, mono_now, sleep_ns);
  }
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
  (void)variant;
  (void)stream;
  if (config().share_weight == 0 || in_internal()) return 0;
  return client()->before_launch(key);
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

void sched_after_fork_child() { g_client.store(nullptr, std::memory_order_release); }

}  // namespace cuscope

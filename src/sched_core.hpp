#pragma once

// The scheduler's token math, with no CUDA and no system calls, so it is
// unit-tested directly.
//
// Every job has a balance of GPU time. Each tick the daemon hands the elapsed
// time to the active jobs (waiting, or launched within kActiveWindowNs), split
// by weight, and caps each balance at kCapNs; credit a job cannot take
// because it is at the cap goes to the other active jobs. A job that no other
// active job competes with is "solo": its balance is set to the cap and it
// never waits. So a share is a guaranteed minimum, never a ceiling.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "model.hpp"
#include "sched_shm.hpp"

namespace cuscope {

struct ClientView {
  bool used, active;
  uint32_t weight;
  int64_t balance = 0;
};

struct RefillStep {
  int64_t add_ns;
  bool solo;
};

// What each slot gets for `delta_ns` of elapsed time.
std::vector<RefillStep> plan_refill(const std::vector<ClientView>& clients, uint64_t delta_ns);

// One daemon tick: reads the slots, plans the refill and applies it.
void apply_refill(SchedShm* shm, uint64_t now_ns, uint64_t delta_ns);

// Frees dead slots safely without ABA clobbering: CAS pid -> kReapingPid, resets fields, then stores 0.
bool reap_dead_slot(SchedSlot& slot, const std::function<bool(int32_t)>& is_alive);
void reap_dead_slots(SchedShm* shm, const std::function<bool(int32_t)>& is_alive);

// The daemon tick interval: kTickNs (1 ms) when any slot is in use, kIdleTickNs (10 ms) when idle.
int64_t daemon_tick_interval(const SchedShm* shm);

// A launch's expected GPU time, per launch key (CUfunction or CUgraphExec): a
// running average of what it measured, kDefaultEstimateNs until it has been.
class Estimator {
 public:
  static constexpr size_t kMaxEntries = 4096;

  uint64_t estimate(uint64_t key) const;
  void observe(uint64_t key, uint64_t actual_ns);
  size_t size() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, uint64_t> average_ns_;
};

// Released: the slot was given back at exit; other threads stop waiting.
// Unsafe: the segment is owned by another user or open to others; not used.
// VersionMismatch: segment magic or version does not match kSchedMagic / kSchedVersion.
enum class SchedState { Registered, NoScheduler, NoFreeSlot, AllocMode, Stopped, Released, Unsafe, VersionMismatch };

// One job's side of the scheduler, on a mapped segment (or an in-memory one
// in tests). The clock and sleep are passed in so tests control time.
class SchedClient {
 public:
  using Clock = uint64_t (*)();
  using Sleep = void (*)(uint64_t ns);

  // Claims a free slot. A null, foreign or closed segment, or a full table,
  // leaves the job unregistered: it then never charges or waits.
  SchedClient(SchedShm* shm, int32_t pid, uint32_t weight, Clock now, Sleep sleep);
  SchedClient(const SchedClient&) = delete;
  SchedClient& operator=(const SchedClient&) = delete;

  SchedState state() const { return state_.load(std::memory_order_acquire); }
  void fail_open(SchedState why) { state_.store(why, std::memory_order_release); }

  // Charges a launch's estimate and waits while the job is over its share;
  // returns what was charged (0 when not registered).
  uint64_t before_launch(uint64_t key);

  // A launch's GPU time is now known: pay back (or pay more than) the
  // estimate charged for it, and learn from it. Spans charged nothing (not
  // shared, or captured) are ignored.
  void on_span(uint64_t key, uint64_t charged_ns, uint64_t actual_ns);

  // A launch failed: return the estimate to the balance without touching the estimator.
  void refund(uint64_t charged_ns);

  // Called after each wait with when it started and how long it took (the
  // glue records it on the timeline).
  using WaitHook = void (*)(uint64_t start_ns, uint64_t dur_ns);
  void set_on_wait(WaitHook hook) { on_wait_ = hook; }

  // Gives the slot back (at exit). Other threads may still be inside
  // before_launch: they see the Released state and return. The slot pointer
  // stays valid (the segment is never unmapped). Stats stay readable.
  void release();

  // Re-registers with a new scheduler segment after being Stopped.
  bool re_register(SchedShm* new_shm);

  SchedStats stats() const;

 private:
  std::atomic<SchedShm*> shm_{nullptr};
  std::atomic<SchedSlot*> slot_{nullptr};
  int32_t pid_;
  uint32_t weight_;
  Clock now_;
  Sleep sleep_;
  std::atomic<SchedState> state_{SchedState::NoScheduler};
  Estimator estimator_;
  WaitHook on_wait_ = nullptr;
  std::atomic<uint64_t> accum_charged_{0}, accum_waits_{0}, accum_wait_ns_{0};
  uint64_t released_charged_ = 0, released_waits_ = 0, released_wait_ns_ = 0;
};

// The reason a job is not registered, as the trace and summary give it.
const char* sched_reason(SchedState state);

}  // namespace cuscope

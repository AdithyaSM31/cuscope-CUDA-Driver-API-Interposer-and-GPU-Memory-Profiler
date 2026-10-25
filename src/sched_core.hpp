#pragma once

// The scheduler's token math, with no CUDA and no system calls, so it is
// unit-tested directly.
//
// Every job has a balance of GPU time. Each tick the daemon hands the elapsed
// time to the active jobs (waiting, or launched within kActiveWindowNs), split
// by weight, and caps each balance at kCapNs. A job that is the only active
// one is "solo": its balance is set to the cap and it never waits, so a share
// is a guaranteed minimum, never a ceiling.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "model.hpp"
#include "sched_shm.hpp"

namespace cuscope {

struct ClientView {
  bool used, active;
  uint32_t weight;
};

struct RefillStep {
  int64_t add_ns;
  bool solo;
};

// What each slot gets for `delta_ns` of elapsed time.
std::vector<RefillStep> plan_refill(const std::vector<ClientView>& clients, uint64_t delta_ns);

// One daemon tick: reads the slots, plans the refill and applies it.
void apply_refill(SchedShm* shm, uint64_t now_ns, uint64_t delta_ns);

// A launch's expected GPU time, per launch key (CUfunction or CUgraphExec): a
// running average of what it measured, kDefaultEstimateNs until it has been.
class Estimator {
 public:
  uint64_t estimate(uint64_t key) const;
  void observe(uint64_t key, uint64_t actual_ns);

 private:
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, uint64_t> average_ns_;
};

enum class SchedState { Registered, NoScheduler, NoFreeSlot, AllocMode, Stopped };

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

  // Gives the slot back (at exit). Stats stay readable afterwards.
  void release();

  SchedStats stats() const;

 private:
  SchedShm* shm_;
  SchedSlot* slot_ = nullptr;
  uint32_t weight_;
  Clock now_;
  Sleep sleep_;
  std::atomic<SchedState> state_{SchedState::NoScheduler};
  Estimator estimator_;
  uint64_t released_charged_ = 0, released_waits_ = 0, released_wait_ns_ = 0;
};

// The reason a job is not registered, as the trace and summary give it.
const char* sched_reason(SchedState state);

}  // namespace cuscope

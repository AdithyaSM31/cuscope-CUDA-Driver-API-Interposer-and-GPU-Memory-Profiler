#pragma once

// The scheduler's token math, with no CUDA and no system calls, so it is
// unit-tested directly.
//
// Every job has a balance of GPU time. Each tick the daemon hands the elapsed
// time to the active jobs (waiting, or launched within kActiveWindowNs), split
// by weight, and caps each balance at kCapNs. A job that is the only active
// one is "solo": its balance is set to the cap and it never waits, so a share
// is a guaranteed minimum, never a ceiling.

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

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

}  // namespace cuscope

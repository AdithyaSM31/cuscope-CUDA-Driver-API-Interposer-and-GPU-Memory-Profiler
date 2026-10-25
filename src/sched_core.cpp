#include "sched_core.hpp"

#include <unistd.h>

#include <cstdlib>

namespace cuscope {

std::string sched_shm_name() {
  if (const char* name = std::getenv("CUSCOPE_SCHED_NAME"); name != nullptr && *name != '\0') return name;
  return "/cuscope-sched-" + std::to_string(getuid());
}

void sched_shm_init(SchedShm* shm, int32_t daemon_pid, uint64_t now_ns) {
  shm->magic = kSchedMagic;
  shm->version = kSchedVersion;
  shm->daemon_pid = daemon_pid;
  shm->heartbeat_ns = now_ns;
  shm->closed = 0;
  for (SchedSlot& s : shm->slots) {
    s.pid = 0;
    s.weight = 0;
    s.balance_ns = 0;
    s.last_launch_ns = 0;
    s.waiting = 0;
    s.solo = 0;
    s.charged_ns = 0;
    s.waits = 0;
    s.wait_ns = 0;
  }
}

std::vector<RefillStep> plan_refill(const std::vector<ClientView>& clients, uint64_t delta_ns) {
  std::vector<RefillStep> steps(clients.size(), RefillStep{0, false});
  unsigned __int128 total_weight = 0;
  size_t active = 0, last_active = 0;
  for (size_t i = 0; i < clients.size(); ++i) {
    if (!clients[i].used || !clients[i].active) continue;
    total_weight += clients[i].weight;
    ++active;
    last_active = i;
  }
  if (active == 0 || total_weight == 0) return steps;
  if (active == 1) {
    steps[last_active] = RefillStep{static_cast<int64_t>(delta_ns), true};
    return steps;
  }
  for (size_t i = 0; i < clients.size(); ++i) {
    if (!clients[i].used || !clients[i].active) continue;
    // 128-bit: delta x weight cannot overflow, and the split loses only rounding.
    steps[i].add_ns =
        static_cast<int64_t>(static_cast<unsigned __int128>(delta_ns) * clients[i].weight / total_weight);
  }
  return steps;
}

void apply_refill(SchedShm* shm, uint64_t now_ns, uint64_t delta_ns) {
  std::vector<ClientView> views(kSchedSlots);
  for (int i = 0; i < kSchedSlots; ++i) {
    const SchedSlot& s = shm->slots[i];
    const uint64_t last = s.last_launch_ns.load(std::memory_order_relaxed);
    views[i].used = s.pid.load(std::memory_order_acquire) != 0;
    views[i].active =
        s.waiting.load(std::memory_order_relaxed) != 0 || (now_ns >= last && now_ns - last <= kActiveWindowNs);
    views[i].weight = s.weight.load(std::memory_order_relaxed);
  }
  const std::vector<RefillStep> steps = plan_refill(views, delta_ns);
  for (int i = 0; i < kSchedSlots; ++i) {
    SchedSlot& s = shm->slots[i];
    if (steps[i].solo) {
      // Alone: full credit, debt forgiven, and no waiting at all.
      s.balance_ns.store(kCapNs, std::memory_order_relaxed);
      s.solo.store(1, std::memory_order_release);
      continue;
    }
    s.solo.store(0, std::memory_order_release);
    if (steps[i].add_ns == 0) continue;
    // Jobs change their own balance concurrently: add, then clamp to the cap.
    const int64_t after = s.balance_ns.fetch_add(steps[i].add_ns, std::memory_order_relaxed) + steps[i].add_ns;
    int64_t current = after;
    while (current > kCapNs && !s.balance_ns.compare_exchange_weak(current, kCapNs, std::memory_order_relaxed)) {
    }
  }
}

uint64_t Estimator::estimate(uint64_t key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = average_ns_.find(key);
  return it == average_ns_.end() ? kDefaultEstimateNs : it->second;
}

void Estimator::observe(uint64_t key, uint64_t actual_ns) {
  std::lock_guard<std::mutex> lock(mu_);
  auto [it, inserted] = average_ns_.try_emplace(key, actual_ns);
  if (!inserted) {
    const int64_t diff = static_cast<int64_t>(actual_ns) - static_cast<int64_t>(it->second);
    it->second = static_cast<uint64_t>(static_cast<int64_t>(it->second) + diff / 8);  // alpha = 1/8
  }
}

}  // namespace cuscope

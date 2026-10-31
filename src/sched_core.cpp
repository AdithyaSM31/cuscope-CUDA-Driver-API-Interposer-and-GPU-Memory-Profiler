#include "sched_core.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>

#include <cstdlib>

namespace cuscope {

std::string sched_shm_name() {
  if (const char* name = std::getenv("CUSCOPE_SCHED_NAME"); name != nullptr && *name != '\0') return name;
  return "/cuscope-sched-" + std::to_string(getuid());
}

bool sched_shm_private(int fd) {
  struct stat st {};
  return fstat(fd, &st) == 0 && st.st_uid == geteuid() && (st.st_mode & 077) == 0;
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
  const size_t n = clients.size();
  std::vector<RefillStep> steps(n, RefillStep{0, false});
  auto competing = [&](size_t i) { return clients[i].used && clients[i].active && clients[i].weight > 0; };

  // Solo: used, and no other job is active. A lone job stays solo while idle
  // too, so its next burst does not wait either.
  size_t active = 0;
  for (size_t i = 0; i < n; ++i) active += competing(i) ? 1 : 0;
  for (size_t i = 0; i < n; ++i) {
    if (clients[i].used && active - (competing(i) ? 1 : 0) == 0) steps[i].solo = true;
  }
  if (active < 2) return steps;

  // Water-filling: split by weight; a job that reaches the cap takes only
  // what fits, and the rest is split again among the others. 128-bit:
  // delta x weight cannot overflow; only rounding is lost.
  std::vector<bool> open(n);
  for (size_t i = 0; i < n; ++i) open[i] = competing(i);
  int64_t remaining = static_cast<int64_t>(delta_ns);
  for (size_t round = 0; round < n && remaining > 0; ++round) {
    unsigned __int128 total_weight = 0;
    for (size_t i = 0; i < n; ++i) total_weight += open[i] ? clients[i].weight : 0;
    if (total_weight == 0) break;
    int64_t handed = 0;
    bool capped = false;
    for (size_t i = 0; i < n; ++i) {
      if (!open[i]) continue;
      const int64_t share = static_cast<int64_t>(static_cast<unsigned __int128>(remaining) * clients[i].weight /
                                                 total_weight);
      const int64_t room = std::max<int64_t>(0, kCapNs - clients[i].balance - steps[i].add_ns);
      const int64_t take = std::min(share, room);
      steps[i].add_ns += take;
      handed += take;
      if (take < share) {  // at the cap: out of the next round
        open[i] = false;
        capped = true;
      }
    }
    remaining -= handed;
    if (!capped) break;
  }
  return steps;
}

namespace {

// Jobs change their own balance concurrently: add, then clamp to the cap.
void add_capped(SchedSlot& s, int64_t add_ns) {
  if (add_ns == 0) return;
  int64_t current = s.balance_ns.fetch_add(add_ns, std::memory_order_relaxed) + add_ns;
  while (current > kCapNs && !s.balance_ns.compare_exchange_weak(current, kCapNs, std::memory_order_relaxed)) {
  }
}

// Every active job waiting for credit means nobody may launch and the GPU
// idles. That happens when jobs run side by side: each one's measured kernel
// time includes the other's time slices, so both are charged more than the
// GPU time they had. Grant credit by weight until the least indebted job (by
// weight) can launch: the split is unchanged and the GPU is never held idle.
void fast_forward_if_all_waiting(SchedShm* shm, const std::vector<ClientView>& views) {
  unsigned __int128 total_weight = 0;
  int active = 0;
  for (int i = 0; i < kSchedSlots; ++i) {
    if (!views[i].used || !views[i].active || views[i].weight == 0) continue;
    if (shm->slots[i].waiting.load(std::memory_order_acquire) == 0) return;  // someone can run
    total_weight += views[i].weight;
    ++active;
  }
  if (active < 2) return;  // a job alone is solo and never waits
  unsigned __int128 step = ~static_cast<unsigned __int128>(0);  // total credit to grant
  for (int i = 0; i < kSchedSlots; ++i) {
    if (!views[i].used || !views[i].active || views[i].weight == 0) continue;
    const int64_t balance = shm->slots[i].balance_ns.load(std::memory_order_relaxed);
    if (balance >= 0) return;  // it can run as soon as it looks
    const unsigned __int128 debt = static_cast<unsigned __int128>(-balance);
    const unsigned __int128 need = (debt * total_weight + views[i].weight - 1) / views[i].weight;
    step = std::min(step, need);
  }
  for (int i = 0; i < kSchedSlots; ++i) {
    if (!views[i].used || !views[i].active || views[i].weight == 0) continue;
    add_capped(shm->slots[i], static_cast<int64_t>(step * views[i].weight / total_weight));
  }
}

}  // namespace

void apply_refill(SchedShm* shm, uint64_t now_ns, uint64_t delta_ns) {
  std::vector<ClientView> views(kSchedSlots);
  for (int i = 0; i < kSchedSlots; ++i) {
    const SchedSlot& s = shm->slots[i];
    const uint64_t last = s.last_launch_ns.load(std::memory_order_relaxed);
    views[i].used = s.pid.load(std::memory_order_acquire) > 0;
    // A launch after this tick read the clock (last > now) is as recent as can be.
    views[i].balance = s.balance_ns.load(std::memory_order_relaxed);
    // Waiting counts only while in debt: a job frozen (Ctrl-Z, a debugger)
    // with its flag set stops competing once it has credit again.
    const bool waiting = s.waiting.load(std::memory_order_relaxed) != 0 && views[i].balance < 0;
    views[i].active = waiting || last > now_ns || now_ns - last <= kActiveWindowNs;
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
    add_capped(s, steps[i].add_ns);
  }
  fast_forward_if_all_waiting(shm, views);
}

bool reap_dead_slot(SchedSlot& slot, const std::function<bool(int32_t)>& is_alive) {
  int32_t pid = slot.pid.load(std::memory_order_acquire);
  if (pid <= 0 || is_alive(pid)) return false;
  if (!slot.pid.compare_exchange_strong(pid, kReapingPid, std::memory_order_acq_rel)) return false;
  slot.waiting.store(0, std::memory_order_relaxed);
  slot.solo.store(0, std::memory_order_relaxed);
  slot.balance_ns.store(0, std::memory_order_relaxed);
  slot.last_launch_ns.store(0, std::memory_order_relaxed);
  slot.pid.store(0, std::memory_order_release);
  return true;
}

void reap_dead_slots(SchedShm* shm, const std::function<bool(int32_t)>& is_alive) {
  if (shm == nullptr) return;
  for (SchedSlot& s : shm->slots) {
    reap_dead_slot(s, is_alive);
  }
}

int64_t daemon_tick_interval(const SchedShm* shm) {
  if (shm == nullptr) return kIdleTickNs;
  for (const SchedSlot& s : shm->slots) {
    if (s.pid.load(std::memory_order_relaxed) > 0) return kTickNs;
  }
  return kIdleTickNs;
}

SchedClient::SchedClient(SchedShm* shm, int32_t pid, uint32_t weight, Clock now, Sleep sleep)
    : pid_(pid), weight_(weight), now_(now), sleep_(sleep) {
  shm_.store(shm, std::memory_order_relaxed);
  if (shm != nullptr && (shm->magic != kSchedMagic || shm->version != kSchedVersion)) {
    state_ = SchedState::VersionMismatch;
    return;
  }
  const uint64_t t = now();
  const uint64_t beat = shm != nullptr ? shm->heartbeat_ns.load(std::memory_order_acquire) : 0;
  // A segment nobody serves (left by a daemon that died) is no scheduler:
  // a job registered there would wait for refills that never come.
  if (shm == nullptr || shm->closed.load(std::memory_order_acquire) != 0 || (t > beat && t - beat > kStaleNs)) {
    state_ = SchedState::NoScheduler;
    return;
  }
  for (SchedSlot& s : shm->slots) {
    int32_t expected = 0;
    if (!s.pid.compare_exchange_strong(expected, pid, std::memory_order_acq_rel)) continue;
    s.weight.store(weight, std::memory_order_relaxed);
    s.balance_ns.store(kCapNs, std::memory_order_relaxed);  // a new job starts with full credit
    s.last_launch_ns.store(now(), std::memory_order_relaxed);
    s.waiting.store(0, std::memory_order_relaxed);
    s.solo.store(0, std::memory_order_relaxed);
    s.charged_ns.store(0, std::memory_order_relaxed);
    s.waits.store(0, std::memory_order_relaxed);
    s.wait_ns.store(0, std::memory_order_relaxed);
    slot_.store(&s, std::memory_order_release);
    state_ = SchedState::Registered;
    return;
  }
  state_ = SchedState::NoFreeSlot;
}

uint64_t SchedClient::before_launch(uint64_t key) {
  if (state() != SchedState::Registered) return 0;
  SchedSlot* const slot = slot_.load(std::memory_order_acquire);
  if (slot == nullptr) return 0;
  SchedShm* const shm = shm_.load(std::memory_order_acquire);
  if (shm == nullptr) return 0;
  const uint64_t start = now_();
  // Every launch, not only a waiting one, notices a daemon that is gone: a
  // solo job never waits, and must still find a new daemon when one starts.
  const uint64_t beat = shm->heartbeat_ns.load(std::memory_order_acquire);
  if (shm->closed.load(std::memory_order_acquire) != 0 || (start > beat && start - beat > kStaleNs)) {
    state_.store(SchedState::Stopped, std::memory_order_release);
    return 0;
  }
  const uint64_t est = estimator_.estimate(key);
  slot->last_launch_ns.store(start, std::memory_order_relaxed);
  slot->charged_ns.fetch_add(est, std::memory_order_relaxed);
  int64_t balance = slot->balance_ns.fetch_sub(static_cast<int64_t>(est), std::memory_order_acq_rel) -
                    static_cast<int64_t>(est);

  // Over its share: wait for the daemon's refill. A solo job never waits,
  // and a daemon that has stopped (closed segment) or stopped beating
  // (stale heartbeat) ends throttling until a new daemon is found: never stuck.
  bool waited = false;
  while (balance < 0 && slot->solo.load(std::memory_order_acquire) == 0 && state() == SchedState::Registered) {
    const uint64_t now = now_();
    const uint64_t beat = shm->heartbeat_ns.load(std::memory_order_acquire);
    // Stop throttling for good when the daemon has stopped (closed segment),
    // stopped beating, or no longer serves this slot (a restarted daemon
    // re-initialised it): never stuck.
    // A wait longer than kMaxWaitNs means the daemon is not refilling at all
    // (a corrupt or foreign segment that still beats): give up as well.
    if (shm->closed.load(std::memory_order_acquire) != 0 || (now > beat && now - beat > kStaleNs) ||
        slot->pid.load(std::memory_order_acquire) != pid_ || now - start >= kMaxWaitNs) {
      if (now - start >= kMaxWaitNs) {
        gave_up_on_daemon_.store(shm->daemon_pid.load(std::memory_order_relaxed), std::memory_order_relaxed);
      }
      state_.store(SchedState::Stopped, std::memory_order_release);
      break;
    }
    slot->waiting.store(1, std::memory_order_release);
    waited = true;
    sleep_(kWaitStepNs);
    balance = slot->balance_ns.load(std::memory_order_acquire);
  }
  if (waited) {
    const uint64_t dur = now_() - start;
    if (state() != SchedState::Released) {  // a released slot may be another job's now
      slot->waiting.store(0, std::memory_order_release);
      slot->waits.fetch_add(1, std::memory_order_relaxed);
      slot->wait_ns.fetch_add(dur, std::memory_order_relaxed);
    }
    if (on_wait_ != nullptr) on_wait_(start, dur);
  }
  return est;
}

void SchedClient::on_span(uint64_t key, uint64_t charged_ns, uint64_t actual_ns) {
  if (charged_ns == 0 || state() == SchedState::Released) return;
  SchedSlot* const slot = slot_.load(std::memory_order_acquire);
  if (slot == nullptr) return;
  const int64_t diff = static_cast<int64_t>(charged_ns) - static_cast<int64_t>(actual_ns);
  slot->balance_ns.fetch_add(diff, std::memory_order_acq_rel);
  slot->charged_ns.fetch_sub(static_cast<uint64_t>(diff), std::memory_order_relaxed);  // now = actual
  estimator_.observe(key, actual_ns);
}

void SchedClient::refund(uint64_t charged_ns) {
  if (charged_ns == 0 || state() == SchedState::Released) return;
  SchedSlot* const slot = slot_.load(std::memory_order_acquire);
  if (slot == nullptr) return;
  slot->balance_ns.fetch_add(static_cast<int64_t>(charged_ns), std::memory_order_acq_rel);
  slot->charged_ns.fetch_sub(charged_ns, std::memory_order_relaxed);
}

void SchedClient::release() {
  SchedSlot* const slot = slot_.load(std::memory_order_acquire);
  if (slot == nullptr || state() == SchedState::Released) return;
  released_charged_ = slot->charged_ns.load(std::memory_order_relaxed);
  released_waits_ = slot->waits.load(std::memory_order_relaxed);
  released_wait_ns_ = slot->wait_ns.load(std::memory_order_relaxed);
  // First: waiting threads leave their loop and spans stop correcting.
  state_.store(SchedState::Released, std::memory_order_release);
  slot->waiting.store(0, std::memory_order_relaxed);
  slot->pid.store(0, std::memory_order_release);  // last: the slot is free
}

bool SchedClient::re_register(SchedShm* new_shm) {
  if (state() == SchedState::Released || state() == SchedState::AllocMode) return false;
  if (new_shm == nullptr || new_shm->magic != kSchedMagic || new_shm->version != kSchedVersion) {
    return false;
  }
  const uint64_t t = now_();
  const uint64_t beat = new_shm->heartbeat_ns.load(std::memory_order_acquire);
  if (new_shm->closed.load(std::memory_order_acquire) != 0 || (t > beat && t - beat > kStaleNs)) {
    return false;
  }
  const int32_t gave_up = gave_up_on_daemon_.load(std::memory_order_relaxed);
  if (gave_up != 0 && new_shm->daemon_pid.load(std::memory_order_relaxed) == gave_up) return false;
  // A slot holding this pid is this job's own from before the stop (the
  // daemon kept serving it through a heartbeat gap): take it back rather
  // than hold a second one. Otherwise any free slot.
  auto claim = [&](int32_t from) -> SchedSlot* {
    for (SchedSlot& s : new_shm->slots) {
      int32_t expected = from;
      if (s.pid.compare_exchange_strong(expected, pid_, std::memory_order_acq_rel)) return &s;
    }
    return nullptr;
  };
  SchedSlot* claimed = claim(pid_);
  if (claimed == nullptr) claimed = claim(0);
  if (claimed != nullptr) {
    SchedSlot& s = *claimed;
    SchedSlot* prev_slot = slot_.load(std::memory_order_relaxed);
    if (prev_slot != nullptr) {
      accum_charged_.fetch_add(prev_slot->charged_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
      accum_waits_.fetch_add(prev_slot->waits.load(std::memory_order_relaxed), std::memory_order_relaxed);
      accum_wait_ns_.fetch_add(prev_slot->wait_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    s.weight.store(weight_, std::memory_order_relaxed);
    s.balance_ns.store(kCapNs, std::memory_order_relaxed);
    s.last_launch_ns.store(now_(), std::memory_order_relaxed);
    s.waiting.store(0, std::memory_order_relaxed);
    s.solo.store(0, std::memory_order_relaxed);
    s.charged_ns.store(0, std::memory_order_relaxed);
    s.waits.store(0, std::memory_order_relaxed);
    s.wait_ns.store(0, std::memory_order_relaxed);
    shm_.store(new_shm, std::memory_order_release);
    slot_.store(&s, std::memory_order_release);
    state_.store(SchedState::Registered, std::memory_order_release);
    return true;
  }
  return false;
}

SchedStats SchedClient::stats() const {
  SchedStats s;
  s.enabled = true;
  s.weight = weight_;
  const SchedState st = state();
  s.registered = st == SchedState::Registered || st == SchedState::Stopped || st == SchedState::Released;
  s.reason = sched_reason(st);
  SchedSlot* const slot = slot_.load(std::memory_order_acquire);
  if (slot != nullptr && st != SchedState::Released) {
    s.charged_ns = accum_charged_.load(std::memory_order_relaxed) + slot->charged_ns.load(std::memory_order_relaxed);
    s.waits = accum_waits_.load(std::memory_order_relaxed) + slot->waits.load(std::memory_order_relaxed);
    s.wait_ns = accum_wait_ns_.load(std::memory_order_relaxed) + slot->wait_ns.load(std::memory_order_relaxed);
  } else {
    s.charged_ns = accum_charged_.load(std::memory_order_relaxed) + released_charged_;
    s.waits = accum_waits_.load(std::memory_order_relaxed) + released_waits_;
    s.wait_ns = accum_wait_ns_.load(std::memory_order_relaxed) + released_wait_ns_;
  }
  return s;
}

const char* sched_reason(SchedState state) {
  switch (state) {
    case SchedState::Registered: return "";
    case SchedState::NoScheduler: return "no scheduler running";
    case SchedState::NoFreeSlot: return "no free slot";
    case SchedState::AllocMode: return "alloc mode has no GPU timing";
    case SchedState::Stopped: return "scheduler stopped";
    case SchedState::Released: return "";
    case SchedState::Unsafe: return "scheduler segment is not private to this user";
    case SchedState::VersionMismatch: return "scheduler is a different version (restart cuscope sched)";
  }
  return "";
}

uint64_t Estimator::estimate(uint64_t key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = average_ns_.find(key);
  return it == average_ns_.end() ? kDefaultEstimateNs : it->second;
}

void Estimator::observe(uint64_t key, uint64_t actual_ns) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint64_t clamped = std::max<uint64_t>(1, actual_ns);
  auto it = average_ns_.find(key);
  if (it == average_ns_.end()) {
    if (average_ns_.size() >= kMaxEntries) {
      average_ns_.clear();
    }
    average_ns_.emplace(key, clamped);
  } else {
    const int64_t diff = static_cast<int64_t>(actual_ns) - static_cast<int64_t>(it->second);
    const int64_t updated = static_cast<int64_t>(it->second) + diff / 8;  // alpha = 1/8
    it->second = static_cast<uint64_t>(std::max<int64_t>(1, updated));
  }
}

size_t Estimator::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return average_ns_.size();
}

}  // namespace cuscope

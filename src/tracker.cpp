#include "tracker.hpp"

#include <algorithm>

namespace cuscope {
namespace {

void atomic_max(std::atomic<uint64_t>& peak, uint64_t value) {
  uint64_t seen = peak.load(std::memory_order_relaxed);
  while (value > seen && !peak.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
  }
}

}  // namespace

Tracker::Shard& Tracker::shard_for(uint64_t ptr) {
  // Allocations are aligned, so the low bits carry no information; mix first.
  uint64_t h = (ptr >> 4) * 0x9E3779B97F4A7C15ull;
  return shards_[h >> 60];
}

void Tracker::add_live(const AllocInfo& info) {
  const int k = static_cast<int>(info.kind);
  atomic_max(peak_by_kind_[k], live_by_kind_[k].fetch_add(info.bytes, std::memory_order_relaxed) + info.bytes);
  if (is_device_memory(info.kind)) {
    atomic_max(peak_device_total_,
               live_device_total_.fetch_add(info.bytes, std::memory_order_relaxed) + info.bytes);
  }
  live_bytes_.fetch_add(info.bytes, std::memory_order_relaxed);
  live_allocs_.fetch_add(1, std::memory_order_relaxed);
}

void Tracker::remove_live(const AllocInfo& info) {
  live_by_kind_[static_cast<int>(info.kind)].fetch_sub(info.bytes, std::memory_order_relaxed);
  if (is_device_memory(info.kind)) live_device_total_.fetch_sub(info.bytes, std::memory_order_relaxed);
  live_bytes_.fetch_sub(info.bytes, std::memory_order_relaxed);
  live_allocs_.fetch_sub(1, std::memory_order_relaxed);
}

void Tracker::on_alloc(const AllocInfo& info) {
  std::optional<AllocInfo> replaced;
  {
    Shard& shard = shard_for(info.ptr);
    std::lock_guard<std::mutex> lock(shard.mu);
    auto [it, inserted] = shard.live.try_emplace(info.ptr, info);
    if (!inserted) {
      // The driver reused an address cuscope still thought was live: its free
      // went through a path cuscope does not see. Drop the stale entry.
      replaced = it->second;
      it->second = info;
    }
  }
  if (replaced) remove_live(*replaced);
  add_live(info);
  alloc_count_.fetch_add(1, std::memory_order_relaxed);
}

std::optional<AllocInfo> Tracker::on_free(uint64_t ptr) {
  std::optional<AllocInfo> found;
  {
    Shard& shard = shard_for(ptr);
    std::lock_guard<std::mutex> lock(shard.mu);
    auto it = shard.live.find(ptr);
    if (it == shard.live.end()) return std::nullopt;
    found = it->second;
    shard.live.erase(it);
  }
  remove_live(*found);
  free_count_.fetch_add(1, std::memory_order_relaxed);
  return found;
}

void Tracker::restore(const AllocInfo& info) {
  {
    Shard& shard = shard_for(info.ptr);
    std::lock_guard<std::mutex> lock(shard.mu);
    shard.live.insert_or_assign(info.ptr, info);
  }
  add_live(info);
  free_count_.fetch_sub(1, std::memory_order_relaxed);
}

void Tracker::on_invalid_free(uint64_t ptr, uint32_t stack_id) {
  invalid_count_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(results_mu_);
  invalid_frees_.push_back({ptr, stack_id});
}

void Tracker::add_leaks(std::vector<AllocInfo>& gone, bool at_ctx_teardown) {
  std::sort(gone.begin(), gone.end(), [](const AllocInfo& a, const AllocInfo& b) { return a.ptr < b.ptr; });
  for (const AllocInfo& info : gone) remove_live(info);
  std::lock_guard<std::mutex> lock(results_mu_);
  for (const AllocInfo& info : gone) {
    leaks_.push_back({info.ptr, info.bytes, info.kind, info.stack_id, at_ctx_teardown});
  }
}

void Tracker::on_ctx_teardown(uint64_t ctx) {
  std::vector<AllocInfo> gone;
  for (Shard& shard : shards_) {
    std::lock_guard<std::mutex> lock(shard.mu);
    for (auto it = shard.live.begin(); it != shard.live.end();) {
      if (it->second.ctx == ctx) {
        gone.push_back(it->second);
        it = shard.live.erase(it);
      } else {
        ++it;
      }
    }
  }
  add_leaks(gone, /*at_ctx_teardown=*/true);
}

void Tracker::finalize() {
  std::vector<AllocInfo> gone;
  for (Shard& shard : shards_) {
    std::lock_guard<std::mutex> lock(shard.mu);
    for (auto& [ptr, info] : shard.live) gone.push_back(info);
    shard.live.clear();
  }
  add_leaks(gone, /*at_ctx_teardown=*/false);
}

void Tracker::fill(Snapshot& s) const {
  for (int k = 0; k < kAllocKinds; ++k) s.peak_bytes[k] = peak_by_kind_[k].load(std::memory_order_relaxed);
  s.peak_device_total = peak_device_total_.load(std::memory_order_relaxed);
  s.alloc_count = alloc_count_.load(std::memory_order_relaxed);
  s.free_count = free_count_.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(results_mu_);
  s.leaks = leaks_;
  s.invalid_frees = invalid_frees_;
}

namespace {

Tracker*& tracker_slot() {
  static Tracker* t = new Tracker();
  return t;
}

}  // namespace

Tracker& tracker() { return *tracker_slot(); }

void tracker_after_fork_child() { tracker_slot() = new Tracker(); }  // the old one is leaked on purpose

}  // namespace cuscope

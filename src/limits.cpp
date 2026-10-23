#include "limits.hpp"

#include <algorithm>

#include "config.hpp"

namespace cuscope {

MemLimit::MemLimit(uint64_t limit_bytes) : limit_(limit_bytes) {}

bool MemLimit::try_charge(AllocKind kind, uint64_t bytes) {
  if (!enabled() || !is_device_memory(kind)) return true;
  uint64_t current = charged_.load(std::memory_order_relaxed);
  do {
    // Compared as "room left" rather than "current + bytes", so a huge request
    // cannot wrap around to a small total.
    if (current > limit_ || bytes > limit_ - current) return false;
  } while (!charged_.compare_exchange_weak(current, current + bytes, std::memory_order_relaxed));
  raise_peak(current + bytes);
  return true;
}

void MemLimit::charge(AllocKind kind, uint64_t bytes) {
  if (!enabled() || !is_device_memory(kind)) return;
  raise_peak(charged_.fetch_add(bytes, std::memory_order_relaxed) + bytes);
}

void MemLimit::release(AllocKind kind, uint64_t bytes) {
  if (!enabled() || !is_device_memory(kind)) return;
  charged_.fetch_sub(bytes, std::memory_order_relaxed);
}

void MemLimit::raise_peak(uint64_t value) {
  uint64_t peak = peak_.load(std::memory_order_relaxed);
  while (value > peak && !peak_.compare_exchange_weak(peak, value, std::memory_order_relaxed)) {
  }
}

void MemLimit::on_denied(AllocKind kind, uint64_t bytes, uint32_t stack_id) {
  denied_count_.fetch_add(1, std::memory_order_relaxed);
  denied_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(largest_mu_);
  if (bytes > largest_bytes_) {
    largest_bytes_ = bytes;
    largest_kind_ = kind;
    largest_stack_ = stack_id;
  }
}

void MemLimit::cap_info(size_t* free_bytes, size_t* total_bytes) const {
  if (!enabled() || free_bytes == nullptr || total_bytes == nullptr) return;
  const uint64_t total = std::min<uint64_t>(limit_, *total_bytes);
  const uint64_t used = charged();
  const uint64_t room = used >= total ? 0 : total - used;
  *total_bytes = static_cast<size_t>(total);
  *free_bytes = static_cast<size_t>(std::min<uint64_t>(*free_bytes, room));
}

void MemLimit::fill(Snapshot& s) const {
  MemLimitStats& out = s.mem_limit;
  out = MemLimitStats{};
  if (!enabled()) return;
  out.enabled = true;
  out.limit_bytes = limit_;
  out.peak_charged_bytes = peak_.load(std::memory_order_relaxed);
  out.denied_count = denied_count_.load(std::memory_order_relaxed);
  out.denied_bytes = denied_bytes_.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(largest_mu_);
  out.largest_denied_bytes = largest_bytes_;
  out.largest_denied_kind = largest_kind_;
  out.largest_denied_stack = largest_stack_;
}

namespace {

MemLimit*& mem_limit_slot() {
  static MemLimit* m = new MemLimit(config().mem_limit_bytes);
  return m;
}

}  // namespace

MemLimit& mem_limit() { return *mem_limit_slot(); }

void mem_limit_after_fork_child() {
  // The old one is leaked: another thread may have been inside it at the fork.
  mem_limit_slot() = new MemLimit(mem_limit_slot()->limit());
}

}  // namespace cuscope

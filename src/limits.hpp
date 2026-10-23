#pragma once

// The per-process GPU memory budget (CUSCOPE_MEM_LIMIT). Pure bookkeeping, no
// CUDA, so it is unit-tested directly.
//
// Allocation hooks reserve bytes here *before* calling the driver: a request
// that would take the process past its cap is refused, and the hook returns
// CUDA_ERROR_OUT_OF_MEMORY without calling the driver. Because the
// reservation comes first, threads allocating at once can never all slip
// under the cap together. Only GPU-resident kinds count (is_device_memory);
// pinned host memory lives in system RAM.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "model.hpp"

namespace cuscope {

class MemLimit {
 public:
  explicit MemLimit(uint64_t limit_bytes);  // 0 = no cap

  bool enabled() const { return limit_ != 0; }
  uint64_t limit() const { return limit_; }
  uint64_t charged() const { return charged_.load(std::memory_order_relaxed); }

  // Reserves `bytes`. False if that would pass the cap (exactly at the cap is
  // allowed); nothing is reserved then. Uncounted kinds, and every request
  // when there is no cap, always succeed without charging.
  bool try_charge(AllocKind kind, uint64_t bytes);

  // Adds bytes without checking the cap: a rejected free whose allocation is
  // live again, or the pitch padding found out after cuMemAllocPitch.
  void charge(AllocKind kind, uint64_t bytes);

  void release(AllocKind kind, uint64_t bytes);

  void on_denied(AllocKind kind, uint64_t bytes, uint32_t stack_id);

  // Rewrites the driver's cuMemGetInfo figures for a GPU the size of the cap:
  // total = min(cap, total), free = min(free, total - charged), floored at 0.
  void cap_info(size_t* free_bytes, size_t* total_bytes) const;

  void fill(Snapshot& s) const;

 private:
  void raise_peak(uint64_t value);

  const uint64_t limit_;
  std::atomic<uint64_t> charged_{0}, peak_{0};
  std::atomic<uint64_t> denied_count_{0}, denied_bytes_{0};

  mutable std::mutex largest_mu_;  // only taken on denial and at exit
  uint64_t largest_bytes_ = 0;
  AllocKind largest_kind_ = AllocKind::Device;
  uint32_t largest_stack_ = 0;
};

// The process-wide budget, with config()'s cap. Created on first use and never
// destroyed, so it still works in exit handlers.
MemLimit& mem_limit();

// In a forked child: a fresh budget with the same cap. The parent's
// allocations are not the child's.
void mem_limit_after_fork_child();

}  // namespace cuscope

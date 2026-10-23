#pragma once

// The allocation tracker: live allocations, peak bytes per kind, leaks and
// invalid frees. Pure bookkeeping, no CUDA, so it is unit-tested directly.
//
// Live allocations are split over 16 shards by pointer hash, each with its own
// mutex, so threads allocating at once rarely contend. Byte and count totals
// are atomics; peaks are updated with a compare-and-swap maximum.

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "model.hpp"

namespace cuscope {

struct AllocInfo {
  uint64_t ptr, bytes;
  AllocKind kind;
  uint32_t stack_id;
  uint64_t ctx;  // CUcontext current when allocated; 0 if unknown
};

class Tracker {
 public:
  void on_alloc(const AllocInfo& info);

  // Removes a live allocation and returns it; nullopt if ptr is not live.
  // Called before the real free, so the driver can reuse the address at once.
  std::optional<AllocInfo> on_free(uint64_t ptr);

  // Undoes on_free after the driver rejected the free.
  void restore(const AllocInfo& info);

  void on_invalid_free(uint64_t ptr, uint32_t stack_id);

  // The context is gone, and the driver freed its memory with it: whatever the
  // program still held there becomes a leak marked at_ctx_teardown. Returns
  // the bytes of each kind that went with it (the memory cap gives them back).
  std::array<uint64_t, kAllocKinds> on_ctx_teardown(uint64_t ctx);

  // At exit: everything still live becomes a leak.
  void finalize();

  void fill(Snapshot& s) const;

  uint64_t live_bytes() const { return live_bytes_.load(std::memory_order_relaxed); }
  uint64_t live_allocs() const { return live_allocs_.load(std::memory_order_relaxed); }
  uint64_t invalid_free_count() const { return invalid_count_.load(std::memory_order_relaxed); }

 private:
  static constexpr int kShards = 16;

  struct Shard {
    std::mutex mu;
    std::unordered_map<uint64_t, AllocInfo> live;
  };

  Shard& shard_for(uint64_t ptr);
  void add_live(const AllocInfo& info);
  void remove_live(const AllocInfo& info);
  void add_leaks(std::vector<AllocInfo>& gone, bool at_ctx_teardown);

  std::array<Shard, kShards> shards_;

  std::array<std::atomic<uint64_t>, kAllocKinds> live_by_kind_{};
  std::array<std::atomic<uint64_t>, kAllocKinds> peak_by_kind_{};
  std::atomic<uint64_t> live_device_total_{0}, peak_device_total_{0};
  std::atomic<uint64_t> live_bytes_{0}, live_allocs_{0};
  std::atomic<uint64_t> alloc_count_{0}, free_count_{0}, invalid_count_{0};

  mutable std::mutex results_mu_;  // guards leaks_ and invalid_frees_
  std::vector<Leak> leaks_;
  std::vector<InvalidFree> invalid_frees_;
};

// The process-wide tracker. Created on first use and never destroyed, so it
// still works in exit handlers that run after static destructors.
Tracker& tracker();

// In a forked child: start a new, empty tracker. The parent's allocations are
// not the child's, and another thread may have held a shard lock at the fork.
void tracker_after_fork_child();

}  // namespace cuscope

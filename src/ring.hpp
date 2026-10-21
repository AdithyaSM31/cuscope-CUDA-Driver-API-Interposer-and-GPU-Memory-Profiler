#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace cuscope {

// A fixed-size single-producer, single-consumer ring buffer.
//
// One thread pushes (the program thread that owns the ring), one thread pops
// (the collector). Neither ever blocks: push on a full ring drops the item
// and counts it, so a slow collector costs completeness, never latency.
//
// head_ and tail_ only ever grow; the slot is the counter modulo N. Each side
// writes only its own counter, publishing with release and reading the other
// side's with acquire, so a popped slot is always fully written.
template <class T, size_t N>
class SpscRing {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "N must be a power of two");

 public:
  // Producer only. False (and one more drop) if the ring is full.
  bool push(const T& item) {
    const uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_.load(std::memory_order_acquire) == N) {
      drops_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    slots_[head & (N - 1)] = item;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  // Consumer only. False if the ring is empty.
  bool pop(T& out) {
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) return false;
    out = slots_[tail & (N - 1)];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  uint64_t drops() const { return drops_.load(std::memory_order_relaxed); }

 private:
  // Separate cache lines: the producer and consumer each write one counter.
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> drops_{0};
  alignas(64) std::array<T, N> slots_;
};

}  // namespace cuscope

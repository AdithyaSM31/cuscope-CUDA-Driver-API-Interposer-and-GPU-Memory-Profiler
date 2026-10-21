#include "tracker.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "model.hpp"

using namespace cuscope;

namespace {

AllocInfo info(uint64_t ptr, uint64_t bytes, AllocKind kind = AllocKind::Device, uint32_t stack = 0,
               uint64_t ctx = 1) {
  return AllocInfo{ptr, bytes, kind, stack, ctx};
}

Snapshot snapshot(const Tracker& t) {
  Snapshot s;
  t.fill(s);
  return s;
}

uint64_t peak(const Snapshot& s, AllocKind kind) { return s.peak_bytes[static_cast<int>(kind)]; }

}  // namespace

TEST(Tracker, AllocFreeUpdatesLiveAndPeak) {
  Tracker t;
  t.on_alloc(info(0x1000, 100));
  t.on_alloc(info(0x2000, 50));
  auto freed = t.on_free(0x1000);
  ASSERT_TRUE(freed.has_value());
  EXPECT_EQ(freed->bytes, 100u);
  EXPECT_EQ(t.live_bytes(), 50u);
  EXPECT_EQ(t.live_allocs(), 1u);

  Snapshot s = snapshot(t);
  EXPECT_EQ(peak(s, AllocKind::Device), 150u);
  EXPECT_EQ(s.peak_device_total, 150u);
  EXPECT_EQ(s.alloc_count, 2u);
  EXPECT_EQ(s.free_count, 1u);
}

TEST(Tracker, PeakPerKind) {
  Tracker t;
  t.on_alloc(info(0x10, 10, AllocKind::Device));
  t.on_alloc(info(0x20, 20, AllocKind::Managed));
  t.on_alloc(info(0x30, 30, AllocKind::HostPinned));
  t.on_alloc(info(0x40, 40, AllocKind::Async));
  Snapshot s = snapshot(t);
  EXPECT_EQ(peak(s, AllocKind::Device), 10u);
  EXPECT_EQ(peak(s, AllocKind::Managed), 20u);
  EXPECT_EQ(peak(s, AllocKind::HostPinned), 30u);
  EXPECT_EQ(peak(s, AllocKind::Async), 40u);
  EXPECT_EQ(s.peak_device_total, 70u);  // pinned host memory is not device memory
}

TEST(Tracker, DeviceTotalPeakIsSimultaneousNotSummedPeaks) {
  Tracker t;
  t.on_alloc(info(0x10, 100, AllocKind::Device));
  t.on_free(0x10);
  t.on_alloc(info(0x20, 60, AllocKind::Async));
  EXPECT_EQ(snapshot(t).peak_device_total, 100u);
}

TEST(Tracker, FreeUnknownReturnsNullopt) {
  Tracker t;
  EXPECT_FALSE(t.on_free(0xdead).has_value());
  EXPECT_EQ(snapshot(t).free_count, 0u);
}

TEST(Tracker, InvalidFreeRecorded) {
  Tracker t;
  t.on_invalid_free(0x5, 9);
  Snapshot s = snapshot(t);
  ASSERT_EQ(s.invalid_frees.size(), 1u);
  EXPECT_EQ(s.invalid_frees[0].ptr, 0x5u);
  EXPECT_EQ(s.invalid_frees[0].stack_id, 9u);
}

// Review Focus 3: the wrapper removes the entry before the real free, and puts
// it back if the driver rejects the free.
TEST(Tracker, RestoreAfterFailedFree) {
  Tracker t;
  t.on_alloc(info(0x1000, 64, AllocKind::Device, 7));
  auto freed = t.on_free(0x1000);
  ASSERT_TRUE(freed.has_value());
  t.restore(*freed);
  EXPECT_EQ(t.live_allocs(), 1u);
  EXPECT_EQ(t.live_bytes(), 64u);

  t.finalize();
  Snapshot s = snapshot(t);
  EXPECT_EQ(s.free_count, 0u);
  ASSERT_EQ(s.leaks.size(), 1u);
  EXPECT_EQ(s.leaks[0].stack_id, 7u);
}

// Review Focus 3: the driver may hand the same address to another allocation
// as soon as it is freed.
TEST(Tracker, AddressReuseAfterFree) {
  Tracker t;
  t.on_alloc(info(0x1000, 100));
  t.on_free(0x1000);
  t.on_alloc(info(0x1000, 200));
  EXPECT_EQ(t.live_allocs(), 1u);
  EXPECT_EQ(t.live_bytes(), 200u);
  t.finalize();
  Snapshot s = snapshot(t);
  ASSERT_EQ(s.leaks.size(), 1u);
  EXPECT_EQ(s.leaks[0].bytes, 200u);
}

TEST(Tracker, CtxTeardownMovesToLeaks) {
  Tracker t;
  t.on_alloc(info(0xA, 10, AllocKind::Device, 1, /*ctx=*/100));
  t.on_alloc(info(0xB, 20, AllocKind::Device, 2, /*ctx=*/200));
  t.on_ctx_teardown(100);
  EXPECT_EQ(t.live_allocs(), 1u);
  Snapshot s = snapshot(t);
  ASSERT_EQ(s.leaks.size(), 1u);
  EXPECT_EQ(s.leaks[0].ptr, 0xAu);
  EXPECT_TRUE(s.leaks[0].at_ctx_teardown);

  t.finalize();
  s = snapshot(t);
  ASSERT_EQ(s.leaks.size(), 2u);
  EXPECT_EQ(s.leaks[1].ptr, 0xBu);
  EXPECT_FALSE(s.leaks[1].at_ctx_teardown);
}

TEST(Tracker, ConcurrentAllocFree) {
  Tracker t;
  constexpr int kThreads = 8, kPerThread = 10000;
  std::vector<std::thread> threads;
  for (int th = 0; th < kThreads; ++th) {
    threads.emplace_back([&t, th] {
      for (int i = 0; i < kPerThread; ++i) {
        uint64_t ptr = (static_cast<uint64_t>(th + 1) << 32) | (static_cast<uint64_t>(i) * 16);
        t.on_alloc(info(ptr, 16));
        ASSERT_TRUE(t.on_free(ptr).has_value());
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(t.live_allocs(), 0u);
  EXPECT_EQ(t.live_bytes(), 0u);
  Snapshot s = snapshot(t);
  EXPECT_EQ(s.alloc_count, 80000u);
  EXPECT_EQ(s.free_count, 80000u);
}

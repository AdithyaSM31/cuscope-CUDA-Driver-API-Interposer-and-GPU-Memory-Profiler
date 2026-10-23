#include "limits.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using namespace cuscope;

namespace {
constexpr uint64_t MiB = 1ull << 20;
}  // namespace

TEST(MemLimit, ChargesUpToTheCapExactly) {
  MemLimit m(256 * MiB);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(m.try_charge(AllocKind::Device, 64 * MiB));
  EXPECT_FALSE(m.try_charge(AllocKind::Device, 1));
  EXPECT_EQ(m.charged(), 256 * MiB);
  m.release(AllocKind::Device, 64 * MiB);
  EXPECT_TRUE(m.try_charge(AllocKind::Async, 64 * MiB));
}

TEST(MemLimit, PinnedHostNeverCounts) {
  MemLimit m(1 * MiB);
  EXPECT_TRUE(m.try_charge(AllocKind::HostPinned, 64 * MiB));
  EXPECT_EQ(m.charged(), 0u);
  m.release(AllocKind::HostPinned, 64 * MiB);
  EXPECT_EQ(m.charged(), 0u);
}

TEST(MemLimit, HugeRequestDenied) {  // Review Focus 1
  MemLimit m(256 * MiB);
  EXPECT_TRUE(m.try_charge(AllocKind::Device, 64 * MiB));
  EXPECT_FALSE(m.try_charge(AllocKind::Device, UINT64_MAX));
  EXPECT_FALSE(m.try_charge(AllocKind::Managed, UINT64_MAX - 32 * MiB));
  EXPECT_EQ(m.charged(), 64 * MiB);
}

TEST(MemLimit, NoCapNeverDenies) {
  MemLimit m(0);
  EXPECT_FALSE(m.enabled());
  EXPECT_TRUE(m.try_charge(AllocKind::Device, UINT64_MAX));
}

TEST(MemLimit, CapInfo) {
  MemLimit m(2048 * MiB);
  m.charge(AllocKind::Device, 512 * MiB);
  size_t free_b = 5000 * MiB, total = 6144 * MiB;
  m.cap_info(&free_b, &total);
  EXPECT_EQ(total, 2048 * MiB);
  EXPECT_EQ(free_b, 1536 * MiB);
  free_b = 100 * MiB, total = 6144 * MiB;  // the GPU itself is nearly full
  m.cap_info(&free_b, &total);
  EXPECT_EQ(free_b, 100 * MiB);
}

TEST(MemLimit, CapInfoFloorsFree) {  // Review Focus 4
  MemLimit m(256 * MiB);
  m.charge(AllocKind::Device, 300 * MiB);  // pitch padding past the cap
  size_t free_b = 5000 * MiB, total = 6144 * MiB;
  m.cap_info(&free_b, &total);
  EXPECT_EQ(free_b, 0u);
}

TEST(MemLimit, StatsAndPeak) {
  MemLimit m(256 * MiB);
  m.try_charge(AllocKind::Device, 200 * MiB);
  m.release(AllocKind::Device, 200 * MiB);
  m.on_denied(AllocKind::Device, 512 * MiB, 7);
  m.on_denied(AllocKind::Managed, 300 * MiB, 9);
  Snapshot s;
  m.fill(s);
  EXPECT_TRUE(s.mem_limit.enabled);
  EXPECT_EQ(s.mem_limit.limit_bytes, 256 * MiB);
  EXPECT_EQ(s.mem_limit.peak_charged_bytes, 200 * MiB);
  EXPECT_EQ(s.mem_limit.denied_count, 2u);
  EXPECT_EQ(s.mem_limit.denied_bytes, 812 * MiB);
  EXPECT_EQ(s.mem_limit.largest_denied_bytes, 512 * MiB);
  EXPECT_EQ(s.mem_limit.largest_denied_stack, 7u);
}

TEST(MemLimit, ThreadsNeverOvershoot) {
  MemLimit m(4 * MiB);
  std::atomic<uint64_t> held{0}, worst{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 20000; ++i) {
        if (!m.try_charge(AllocKind::Device, 1 * MiB)) continue;
        uint64_t now = held.fetch_add(1 * MiB) + 1 * MiB;
        uint64_t w = worst.load();
        while (now > w && !worst.compare_exchange_weak(w, now)) {
        }
        held.fetch_sub(1 * MiB);
        m.release(AllocKind::Device, 1 * MiB);
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_LE(worst.load(), 4 * MiB);
  EXPECT_EQ(m.charged(), 0u);
}

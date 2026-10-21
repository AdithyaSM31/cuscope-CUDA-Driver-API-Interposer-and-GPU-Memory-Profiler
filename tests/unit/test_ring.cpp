#include "ring.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <thread>

using cuscope::SpscRing;

TEST(Ring, PushPopOrder) {
  SpscRing<int, 8> ring;
  for (int i = 1; i <= 5; ++i) ASSERT_TRUE(ring.push(i));
  int v = 0;
  for (int i = 1; i <= 5; ++i) {
    ASSERT_TRUE(ring.pop(v));
    EXPECT_EQ(v, i);
  }
  EXPECT_FALSE(ring.pop(v));
}

TEST(Ring, FullDropsAndCounts) {
  SpscRing<int, 8> ring;
  for (int i = 0; i < 8; ++i) ASSERT_TRUE(ring.push(i));
  EXPECT_FALSE(ring.push(99));  // full: dropped, never blocks
  EXPECT_EQ(ring.drops(), 1u);
  int v = 0;
  ASSERT_TRUE(ring.pop(v));
  EXPECT_EQ(v, 0);
  EXPECT_TRUE(ring.push(8));  // room again
  EXPECT_EQ(ring.drops(), 1u);
}

TEST(Ring, WrapsAroundManyTimes) {
  SpscRing<int, 4> ring;
  int v = 0;
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(ring.push(i));
    ASSERT_TRUE(ring.pop(v));
    ASSERT_EQ(v, i);
  }
}

TEST(Ring, SpscStress) {
  constexpr uint64_t kItems = 1'000'000;
  SpscRing<uint64_t, 1024> ring;
  uint64_t produced_sum = 0;
  std::thread producer([&] {
    for (uint64_t i = 1; i <= kItems; ++i) {
      while (!ring.push(i)) std::this_thread::yield();  // test only: retry instead of dropping
      produced_sum += i;
    }
  });
  uint64_t consumed = 0, consumed_sum = 0, v = 0, expected = 1;
  while (consumed < kItems) {
    if (ring.pop(v)) {
      ASSERT_EQ(v, expected++);  // FIFO, nothing lost or duplicated
      consumed_sum += v;
      ++consumed;
    }
  }
  producer.join();
  EXPECT_EQ(consumed_sum, produced_sum);
}

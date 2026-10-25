#include "sched_core.hpp"

#include <gtest/gtest.h>

using namespace cuscope;

TEST(SchedCore, SplitsByWeightAmongActive) {
  auto steps = plan_refill({{true, true, 80}, {true, true, 20}, {true, false, 50}, {false, false, 0}}, 1'000'000);
  EXPECT_EQ(steps[0].add_ns, 800'000);
  EXPECT_EQ(steps[1].add_ns, 200'000);
  EXPECT_EQ(steps[2].add_ns, 0);  // idle: nothing
  EXPECT_EQ(steps[3].add_ns, 0);
  EXPECT_FALSE(steps[0].solo);
}

TEST(SchedCore, AloneIsSolo) {
  auto steps = plan_refill({{true, true, 5}, {true, false, 95}}, 1'000'000);
  EXPECT_TRUE(steps[0].solo);
  EXPECT_FALSE(steps[1].solo);
}

TEST(SchedCore, ExtremeWeightsStillRefill) {  // Review Focus 1
  auto steps = plan_refill({{true, true, 1}, {true, true, 1000}}, 1'000'000);
  EXPECT_GT(steps[0].add_ns, 0);
  EXPECT_GE(steps[0].add_ns + steps[1].add_ns, 1'000'000 - 2);  // only rounding is lost
}

TEST(SchedCore, ApplyRefillCapsAndForgivesSolo) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].balance_ns = 19'900'000;
  shm.slots[0].last_launch_ns = 1;
  shm.slots[1].pid = 11;
  shm.slots[1].weight = 1;
  shm.slots[1].balance_ns = -5'000'000;
  shm.slots[1].last_launch_ns = 1;
  apply_refill(&shm, 2, 1'000'000);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);                // capped
  EXPECT_EQ(shm.slots[1].balance_ns.load(), -5'000'000 + 500'000);  // debt pays down
  shm.slots[1].pid = 0;                                             // the other job leaves
  apply_refill(&shm, 3, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 1u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);
}

TEST(SchedCore, SoloClearsWhenSecondJobActive) {  // Review Focus 4
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].last_launch_ns = 1;
  apply_refill(&shm, 2, 1'000'000);
  ASSERT_EQ(shm.slots[0].solo.load(), 1u);
  shm.slots[1].pid = 11;
  shm.slots[1].weight = 1;
  shm.slots[1].waiting = 1;
  apply_refill(&shm, 3, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 0u);
  EXPECT_EQ(shm.slots[1].solo.load(), 0u);
}

TEST(SchedCore, IdleSlotsAreNotActive) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].last_launch_ns = 0;
  apply_refill(&shm, kActiveWindowNs + 1, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 0u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), 0);
}

TEST(Estimator, DefaultThenRunningAverage) {
  Estimator e;
  EXPECT_EQ(e.estimate(7), kDefaultEstimateNs);
  e.observe(7, 800'000);
  EXPECT_EQ(e.estimate(7), 800'000u);
  e.observe(7, 1'600'000);
  EXPECT_EQ(e.estimate(7), 900'000u);  // 800 + (1600 - 800) / 8
}

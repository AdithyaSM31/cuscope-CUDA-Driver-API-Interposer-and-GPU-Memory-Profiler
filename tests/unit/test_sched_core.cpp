#include "sched_core.hpp"

#include <gtest/gtest.h>

using namespace cuscope;

namespace {

// A clock that only moves when a waiting client sleeps.
uint64_t g_fake_ns = 1'000'000'000;
uint64_t fake_now() { return g_fake_ns; }
void fake_sleep(uint64_t ns) { g_fake_ns += ns; }

uint64_t g_fake_wait_ns = 0;

// A sleep during which a live daemon beats and grants 100 us of credit.
SchedShm* g_refill_shm = nullptr;
void refill_on_sleep(uint64_t ns) {
  g_fake_ns += ns;
  g_refill_shm->heartbeat_ns = g_fake_ns;
  g_refill_shm->slots[0].balance_ns += 100'000;
}

}  // namespace

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

TEST(SchedClient, ClaimsAndReleasesASlot) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::Registered);
  EXPECT_EQ(shm.slots[0].pid.load(), 4242);
  EXPECT_EQ(shm.slots[0].weight.load(), 80u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);  // starts with full credit
  c.release();
  EXPECT_EQ(shm.slots[0].pid.load(), 0);
}

TEST(SchedClient, FullTableFailsOpen) {  // Review Focus 3
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  for (auto& s : shm.slots) s.pid = 99;
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::NoFreeSlot);
  EXPECT_EQ(c.before_launch(7), 0u);  // never charges or waits
  for (auto& s : shm.slots) EXPECT_EQ(s.pid.load(), 99);  // nothing overwritten
}

TEST(SchedClient, ClosedSegmentFailsOpen) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  shm.closed = 1;
  EXPECT_EQ(SchedClient(&shm, 4242, 80, fake_now, fake_sleep).state(), SchedState::NoScheduler);
  EXPECT_EQ(SchedClient(nullptr, 4242, 80, fake_now, fake_sleep).state(), SchedState::NoScheduler);
}

TEST(SchedClient, StatsCarryWeightAndReason) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient ok(&shm, 4242, 80, fake_now, fake_sleep);
  SchedStats s = ok.stats();
  EXPECT_TRUE(s.enabled && s.registered);
  EXPECT_EQ(s.weight, 80u);
  SchedStats off = SchedClient(nullptr, 4243, 20, fake_now, fake_sleep).stats();
  EXPECT_TRUE(off.enabled);
  EXPECT_FALSE(off.registered);
  EXPECT_EQ(off.reason, "no scheduler running");
}

TEST(SchedClient, ChargesTheEstimate) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  EXPECT_EQ(c.before_launch(7), kDefaultEstimateNs);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs - int64_t(kDefaultEstimateNs));
  EXPECT_EQ(shm.slots[0].charged_ns.load(), kDefaultEstimateNs);
  EXPECT_EQ(shm.slots[0].waits.load(), 0u);
}

TEST(SchedClient, WaitsUntilRefilled) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  g_refill_shm = &shm;
  SchedClient c(&shm, 4242, 50, fake_now, refill_on_sleep);
  shm.slots[0].balance_ns = -300'000;
  uint64_t waited_ns = 0;
  c.set_on_wait([](uint64_t, uint64_t dur) { g_fake_wait_ns = dur; });
  c.before_launch(7);
  waited_ns = g_fake_wait_ns;
  EXPECT_EQ(shm.slots[0].waits.load(), 1u);
  EXPECT_GE(shm.slots[0].balance_ns.load(), 0);
  EXPECT_EQ(shm.slots[0].wait_ns.load(), waited_ns);
  EXPECT_EQ(shm.slots[0].waiting.load(), 0u);  // cleared after the wait
  EXPECT_EQ(c.state(), SchedState::Registered);
}

TEST(SchedClient, SoloNeverWaits) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  shm.slots[0].balance_ns = -50'000'000;
  shm.slots[0].solo = 1;
  c.before_launch(7);
  EXPECT_EQ(shm.slots[0].waits.load(), 0u);
}

TEST(SchedClient, StaleHeartbeatStopsThrottling) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);  // fake_sleep advances time; nobody beats
  shm.slots[0].balance_ns = -50'000'000;
  c.before_launch(7);
  EXPECT_EQ(c.state(), SchedState::Stopped);
  EXPECT_EQ(c.before_launch(7), 0u);
  EXPECT_TRUE(c.stats().registered);
  EXPECT_EQ(c.stats().reason, "scheduler stopped");
}

TEST(SchedClient, ClosedSegmentWhileWaitingStops) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  shm.slots[0].balance_ns = -50'000'000;
  shm.closed = 1;  // the daemon was stopped
  c.before_launch(7);
  EXPECT_EQ(c.state(), SchedState::Stopped);
  EXPECT_EQ(shm.slots[0].waits.load(), 0u);
}

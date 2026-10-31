#include "sched_core.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace cuscope;

namespace {

// A clock that only moves when a waiting client sleeps.
uint64_t g_fake_ns = 1'000'000'000;
uint64_t fake_now() { return g_fake_ns; }
void fake_sleep(uint64_t ns) { g_fake_ns += ns; }

uint64_t g_fake_wait_ns = 0;

// A sleep during which a live daemon beats but grants nothing.
SchedShm* g_beat_shm = nullptr;
void beat_on_sleep(uint64_t ns);

// A sleep during which a live daemon beats and grants 100 us of credit.
SchedShm* g_refill_shm = nullptr;
void beat_on_sleep(uint64_t ns) {
  g_fake_ns += ns;
  g_refill_shm->heartbeat_ns = g_fake_ns;
}

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
  EXPECT_EQ(shm.slots[1].balance_ns.load(), -5'000'000 + 900'000);  // and gets what job 0 could not take
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
  shm.slots[1].balance_ns = -1'000'000;
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
  shm.slots[1].pid = 11;
  shm.slots[1].weight = 1;
  shm.slots[1].balance_ns = -1'000'000;
  shm.slots[1].last_launch_ns = kActiveWindowNs;
  apply_refill(&shm, kActiveWindowNs + 1, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 0u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), 0);  // idle: nothing
  EXPECT_EQ(shm.slots[1].solo.load(), 1u);       // the only active job
}

TEST(SchedCore, IdleLoneJobKeepsSolo) {
  // Nothing competes with it, so its next burst must not wait either.
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].last_launch_ns = 0;
  apply_refill(&shm, kActiveWindowNs + 1, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 1u);
}

TEST(Estimator, DefaultThenRunningAverage) {
  Estimator e;
  EXPECT_EQ(e.estimate(7), kDefaultEstimateNs);
  e.observe(7, 800'000);
  EXPECT_EQ(e.estimate(7), 800'000u);
  e.observe(7, 1'600'000);
  EXPECT_EQ(e.estimate(7), 900'000u);  // 800 + (1600 - 800) / 8
}

TEST(Estimator, ClampsZeroEstimate) {
  Estimator e;
  e.observe(7, 0);
  EXPECT_GE(e.estimate(7), 1u);
  e.observe(7, 800'000);
  EXPECT_GT(e.estimate(7), 1u);
}

TEST(Estimator, BoundedCapacity) {
  Estimator e;
  for (uint64_t i = 1; i <= Estimator::kMaxEntries; ++i) {
    e.observe(i, 100'000);
  }
  EXPECT_EQ(e.size(), Estimator::kMaxEntries);
  EXPECT_EQ(e.estimate(1), 100'000u);

  // Exceeding capacity clears the table and inserts the new entry
  e.observe(Estimator::kMaxEntries + 1, 200'000);
  EXPECT_EQ(e.size(), 1u);
  EXPECT_EQ(e.estimate(Estimator::kMaxEntries + 1), 200'000u);
  EXPECT_EQ(e.estimate(1), kDefaultEstimateNs);  // evicted
}

TEST(SchedClient, ZeroMeasurementDoesNotStick) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  uint64_t charge1 = c.before_launch(7);
  EXPECT_EQ(charge1, kDefaultEstimateNs);
  c.on_span(7, charge1, 0);  // first span measures 0 ns

  uint64_t charge2 = c.before_launch(7);
  EXPECT_GT(charge2, 0u);  // next charge is > 0

  c.on_span(7, charge2, 800'000);  // later real measurement
  uint64_t charge3 = c.before_launch(7);
  EXPECT_GT(charge3, charge2);  // real measurement is applied
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

TEST(SchedClient, FailedLaunchRefundsCharge) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  uint64_t charged = c.before_launch(7);
  EXPECT_EQ(charged, kDefaultEstimateNs);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), kDefaultEstimateNs);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs - static_cast<int64_t>(kDefaultEstimateNs));

  c.refund(charged);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), 0u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);

  // Estimator should be untouched (still default estimate, not 0)
  uint64_t next_charge = c.before_launch(7);
  EXPECT_EQ(next_charge, kDefaultEstimateNs);
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

TEST(SchedClient, VersionMismatchFailsOpen) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  shm.version = kSchedVersion + 1;
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::VersionMismatch);
  EXPECT_EQ(c.stats().reason, "scheduler is a different version (restart cuscope sched)");
  EXPECT_FALSE(c.stats().registered);
  EXPECT_EQ(c.before_launch(7), 0u);
}

TEST(SchedClient, WrongMagicFailsOpen) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  shm.magic = 0xdeadbeefULL;
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::VersionMismatch);
  EXPECT_EQ(c.stats().reason, "scheduler is a different version (restart cuscope sched)");
  EXPECT_FALSE(c.stats().registered);
}

TEST(SchedClient, ReRegistersAfterDaemonRestart) {
  SchedShm shm1{};
  sched_shm_init(&shm1, 1, fake_now());
  SchedClient c(&shm1, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::Registered);

  // Daemon 1 stops
  shm1.slots[0].balance_ns = -50'000'000;
  shm1.closed = 1;
  c.before_launch(7);
  EXPECT_EQ(c.state(), SchedState::Stopped);

  // While stopped, launches return 0 and do not charge
  EXPECT_EQ(c.before_launch(7), 0u);

  // Daemon 2 starts with a new segment
  SchedShm shm2{};
  sched_shm_init(&shm2, 2, fake_now());
  EXPECT_TRUE(c.re_register(&shm2));
  EXPECT_EQ(c.state(), SchedState::Registered);
  EXPECT_EQ(shm2.slots[0].pid.load(), 4242);

  // Now it charges and throttles again
  uint64_t charged = c.before_launch(7);
  EXPECT_GT(charged, 0u);
  EXPECT_EQ(shm2.slots[0].charged_ns.load(), charged);
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

TEST(SchedClient, CorrectionMakesTheChargeExact) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  const uint64_t est = c.before_launch(7);
  c.on_span(7, est, 1'000'000);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs - 1'000'000);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), 1'000'000u);
  EXPECT_EQ(c.before_launch(7), 1'000'000u);  // the estimate learned
}

TEST(SchedClient, UnchargedSpansAreIgnored) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  c.on_span(7, 0, 5'000'000);  // a capture launch, or one made before registering
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), 0u);
  EXPECT_EQ(c.before_launch(7), kDefaultEstimateNs);  // and taught the estimator nothing
}

TEST(SchedCore, LaunchAfterTheTickStartedCountsAsActive) {
  // The daemon reads the clock, then the slots: a job that launched in between
  // has a last-launch time after the tick's "now" and must still be active.
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].last_launch_ns = 5000001;
  apply_refill(&shm, 5000000, 1000000);
  EXPECT_EQ(shm.slots[0].solo.load(), 1u);
}

TEST(SchedCore, AllWaitingIsFastForwarded) {
  // Every active job waiting means nobody may launch: the GPU would idle. The
  // refill then grants credit by weight until the least indebted job can go.
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 3;
  shm.slots[0].balance_ns = -6000000;
  shm.slots[0].waiting = 1;
  shm.slots[1].pid = 11;
  shm.slots[1].weight = 1;
  shm.slots[1].balance_ns = -1000000;
  shm.slots[1].waiting = 1;
  apply_refill(&shm, 2, 1000000);
  // The tick adds 750 us and 250 us; job 1 then needs 750 us more, which at a
  // quarter of the weight is 3 ms of fast-forward: job 0 gets 2.25 ms of it.
  EXPECT_EQ(shm.slots[1].balance_ns.load(), 0);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), -3000000);
}

TEST(SchedCore, NoFastForwardWhileAJobCanRun) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].balance_ns = -6000000;
  shm.slots[0].waiting = 1;
  shm.slots[1].pid = 11;
  shm.slots[1].weight = 1;
  shm.slots[1].balance_ns = 1000000;  // can launch: the GPU is busy
  shm.slots[1].last_launch_ns = 1;
  apply_refill(&shm, 2, 1000000);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), -5500000);
  EXPECT_EQ(shm.slots[1].balance_ns.load(), 1500000);
}

TEST(SchedClient, SlotTakenAwayWhileWaitingFailsOpen) {  // Final review C1
  // A restarted daemon re-initialised the segment: this job's slot is no
  // longer its own, nobody refills it, and the heartbeat is fresh again.
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  g_refill_shm = &shm;
  SchedClient c(&shm, 4242, 50, fake_now, beat_on_sleep);
  shm.slots[0].balance_ns = -50'000'000;
  shm.slots[0].pid = 0;  // re-initialised by the new daemon
  c.before_launch(7);
  EXPECT_EQ(c.state(), SchedState::Stopped);
}

TEST(SchedClient, StaleSegmentAtRegistrationFailsOpen) {  // Final review C1
  // A segment left behind by a dead daemon: magic and version are fine, but
  // nobody has beaten for a long time.
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now() - kStaleNs - 1);
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::NoScheduler);
  EXPECT_EQ(shm.slots[0].pid.load(), 0);  // and it did not claim a slot
}

TEST(SchedCore, LightPeerDoesNotHoldTheBusyJobBack) {  // Final review I1
  // A job that launches a little now and then stays active at the cap: the
  // credit it cannot take goes to the busy job, not nowhere.
  auto steps = plan_refill({{true, true, 1, kCapNs}, {true, true, 1, -5'000'000}}, 1'000'000);
  EXPECT_EQ(steps[0].add_ns, 0);
  EXPECT_EQ(steps[1].add_ns, 1'000'000);
}

TEST(SchedCore, FrozenWaiterWithCreditIsNotActive) {  // Final review I1
  // Stopped (Ctrl-Z, a debugger) while its waiting flag was set: once it has
  // credit again it is not competing, and the other job is alone.
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10;
  shm.slots[0].weight = 1;
  shm.slots[0].balance_ns = 3'000'000;
  shm.slots[0].waiting = 1;
  shm.slots[1].pid = 11;
  shm.slots[1].weight = 1;
  shm.slots[1].balance_ns = -1'000'000;
  shm.slots[1].last_launch_ns = 99'000'000;
  apply_refill(&shm, 100'000'000, 1'000'000);
  EXPECT_EQ(shm.slots[1].solo.load(), 1u);
}

namespace {
uint64_t frozen_now() { return 5'000'000'000; }  // the heartbeat never goes stale
void short_sleep(uint64_t) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
}  // namespace

TEST(SchedClient, ReleaseWhileAnotherThreadWaits) {  // Final review I2
  // exit() runs the release handler while a worker thread is throttled: the
  // worker must stop waiting and return, not touch a released slot.
  SchedShm shm{};
  sched_shm_init(&shm, 1, frozen_now());
  SchedClient c(&shm, 4242, 50, frozen_now, short_sleep);
  shm.slots[0].balance_ns = -50'000'000;  // and nobody refills it
  std::atomic<bool> returned{false};
  std::thread worker([&] {
    c.before_launch(7);
    returned = true;
  });
  while (shm.slots[0].waiting.load() == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
  c.release();
  worker.join();
  EXPECT_TRUE(returned.load());
  EXPECT_EQ(shm.slots[0].pid.load(), 0);
  c.on_span(7, 50'000, 1'000'000);  // a span resolved after release changes nothing
  EXPECT_EQ(shm.slots[0].balance_ns.load(), -50'000'000 - int64_t(kDefaultEstimateNs));
  EXPECT_TRUE(c.stats().registered);
}

TEST(SchedClient, AWaitThatNeverEndsIsCutOff) {  // Final review I3
  // A segment that keeps beating but never refills (corrupt, or written by
  // someone else) must not hold a launch forever.
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  g_refill_shm = &shm;
  SchedClient c(&shm, 4242, 50, fake_now, beat_on_sleep);
  shm.slots[0].balance_ns = -50'000'000;
  const uint64_t start = fake_now();
  c.before_launch(7);
  EXPECT_EQ(c.state(), SchedState::Stopped);
  EXPECT_GE(fake_now() - start, kMaxWaitNs);
  EXPECT_LT(fake_now() - start, kMaxWaitNs + 1'000'000);
}

TEST(SchedCore, ReapsDeadSlot) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 100;
  shm.slots[0].balance_ns = 5'000'000;
  shm.slots[0].waiting = 1;
  shm.slots[0].solo = 1;
  shm.slots[0].last_launch_ns = 12345;

  EXPECT_TRUE(reap_dead_slot(shm.slots[0], [](int32_t) { return false; }));
  EXPECT_EQ(shm.slots[0].pid.load(), 0);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), 0);
  EXPECT_EQ(shm.slots[0].waiting.load(), 0u);
  EXPECT_EQ(shm.slots[0].solo.load(), 0u);
  EXPECT_EQ(shm.slots[0].last_launch_ns.load(), 0u);
}

TEST(SchedCore, ReaperRaceDoesNotClobberNewJob) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 100;
  shm.slots[0].balance_ns = 5'000'000;

  // Between checking alive and CAS, dead pid 100 is replaced by new job 200
  auto alive_check_with_race = [&](int32_t pid) {
    if (pid == 100) {
      shm.slots[0].pid = 200;
      shm.slots[0].balance_ns = 20'000'000;
      shm.slots[0].last_launch_ns = 999;
      return false;  // 100 was dead
    }
    return true;
  };

  EXPECT_FALSE(reap_dead_slot(shm.slots[0], alive_check_with_race));
  EXPECT_EQ(shm.slots[0].pid.load(), 200);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), 20'000'000);
  EXPECT_EQ(shm.slots[0].last_launch_ns.load(), 999u);
}

TEST(SchedCore, NegativePidIgnoredInRefill) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = kReapingPid;
  shm.slots[0].weight = 10;
  shm.slots[0].last_launch_ns = 1;

  apply_refill(&shm, 1'000'000, 1'000'000);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), 0);
}

TEST(SchedCore, DaemonTickIntervalIdleVsActive) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);

  // Empty table: idle interval (10 ms)
  EXPECT_EQ(daemon_tick_interval(&shm), kIdleTickNs);
  EXPECT_EQ(daemon_tick_interval(nullptr), kIdleTickNs);

  // Slot being reaped (-1): still idle
  shm.slots[0].pid = kReapingPid;
  EXPECT_EQ(daemon_tick_interval(&shm), kIdleTickNs);

  // One slot in use: active interval (1 ms)
  shm.slots[0].pid = 42;
  EXPECT_EQ(daemon_tick_interval(&shm), kTickNs);

  // Job exits: returns to idle
  shm.slots[0].pid = 0;
  EXPECT_EQ(daemon_tick_interval(&shm), kIdleTickNs);
}

TEST(SchedClient, GraphLaunchKeyChargingAndEstimator) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);

  const uint64_t graph_key = 0xabcdef1234ull;
  // First launch of graph key uses default estimate
  uint64_t charge1 = c.before_launch(graph_key);
  EXPECT_EQ(charge1, kDefaultEstimateNs);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), kDefaultEstimateNs);

  // Graph execution resolves on GPU taking 2 ms
  c.on_span(graph_key, charge1, 2'000'000);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), 2'000'000u);

  // Subsequent launch of the same graph key uses updated estimate
  uint64_t charge2 = c.before_launch(graph_key);
  EXPECT_GT(charge2, kDefaultEstimateNs);
}

TEST(SchedClient, SeveralThreadsWaitingAtOnce) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  shm.slots[1].pid = 999;  // Peer job makes this non-solo
  shm.slots[1].weight = 10;
  shm.slots[1].last_launch_ns = fake_now();

  SchedClient c(&shm, 4242, 10, fake_now, short_sleep);
  shm.slots[0].balance_ns = -1'000'000;  // Start in debt

  std::atomic<int> completed{0};
  constexpr int kThreads = 4;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      c.before_launch(100 + i);
      completed.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // Wait until threads enter waiting state
  while (shm.slots[0].waiting.load() == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  // Refill credit to release waiters
  shm.slots[0].balance_ns.store(10'000'000, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(completed.load(), kThreads);
  EXPECT_EQ(shm.slots[0].waiting.load(), 0u);
}



TEST(SchedClient, ReRegisterOnTheSameSegmentKeepsOneSlot) {
  // A heartbeat gap (a VM paused) stops a job while its daemon lives on; when
  // the daemon beats again the job rejoins the same segment, in one slot.
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  shm.slots[0].balance_ns = -1'000'000;  // nobody refills: the heartbeat goes stale
  c.before_launch(7);
  ASSERT_EQ(c.state(), SchedState::Stopped);
  shm.heartbeat_ns = fake_now();
  ASSERT_TRUE(c.re_register(&shm));
  int held = 0;
  for (const SchedSlot& s : shm.slots) held += s.pid.load() == 4242;
  EXPECT_EQ(held, 1);
}

TEST(SchedClient, AStopForWaitingTooLongIsFinalForThatDaemon) {
  // A daemon that beats but never refills held a launch for kMaxWaitNs:
  // rejoining it would hold the next launch just as long. A new daemon is fine.
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  g_refill_shm = &shm;
  SchedClient c(&shm, 4242, 50, fake_now, beat_on_sleep);
  shm.slots[0].balance_ns = -50'000'000;
  c.before_launch(7);
  ASSERT_EQ(c.state(), SchedState::Stopped);
  EXPECT_FALSE(c.re_register(&shm));
  EXPECT_EQ(c.state(), SchedState::Stopped);
  SchedShm fresh{};
  sched_shm_init(&fresh, 2, fake_now());
  EXPECT_TRUE(c.re_register(&fresh));
}


TEST(SchedClient, ASoloJobNoticesTheDaemonStopped) {
  // A solo job never waits, so it must not rely on the wait loop to notice
  // the daemon is gone: otherwise it never re-registers with a new one.
  SchedShm shm{};
  sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  shm.slots[0].solo = 1;
  c.before_launch(7);
  ASSERT_EQ(c.state(), SchedState::Registered);
  g_fake_ns += kStaleNs + 1;  // the daemon stopped beating
  EXPECT_EQ(c.before_launch(7), 0u);
  EXPECT_EQ(c.state(), SchedState::Stopped);
}

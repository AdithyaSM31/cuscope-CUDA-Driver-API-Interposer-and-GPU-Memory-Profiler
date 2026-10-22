#include "recorder.hpp"

#include <gtest/gtest.h>

#include <set>
#include <thread>
#include <vector>

#include "model.hpp"

using namespace cuscope;

// Uses the process-wide recorder, so it is the only recorder test in this binary.
TEST(Recorder, MultiThreadAllDrained) {
  constexpr int kThreads = 8, kPerThread = 1000;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([] {
      for (int i = 0; i < kPerThread; ++i) {
        Record r{};
        r.type = RecType::ApiCall;
        r.cpu_ns = now_ns();
        r.corr = next_corr();
        record(r);
      }
    });
  }
  for (auto& t : threads) t.join();

  Snapshot s;
  collector_stop_and_drain(s);
  ASSERT_EQ(s.records.size(), static_cast<size_t>(kThreads * kPerThread));
  EXPECT_EQ(s.dropped_records, 0u);

  std::set<uint64_t> corr, tids;
  for (const Record& r : s.records) {
    corr.insert(r.corr);
    tids.insert(r.tid);
    EXPECT_NE(r.corr, 0u);
  }
  EXPECT_EQ(corr.size(), s.records.size());  // correlation ids are unique
  EXPECT_EQ(tids.size(), static_cast<size_t>(kThreads));  // record() fills the thread id
}

TEST(Recorder, NoCollectorRestartAfterFinalDrain) {
  // Hooked calls made later in exit (the CUDA runtime's own cleanup, say) must
  // not start a collector thread that nothing will ever join.
  Snapshot first;
  collector_stop_and_drain(first);
  Record r{};
  r.type = RecType::ApiCall;
  r.corr = next_corr();
  record(r);
  EXPECT_FALSE(collector_running());
  Snapshot second;
  collector_stop_and_drain(second);
  ASSERT_EQ(second.records.size(), 1u);  // still drained by a later drain
  EXPECT_EQ(second.records[0].corr, r.corr);
}

TEST(Recorder, RingsOfExitedThreadsAreFreed) {
  // A program that runs many short-lived threads (a thread pool resized, a
  // worker per request) must not keep 1 MiB per thread it ever had, nor scan
  // them all on every drain. Nothing they recorded may be lost.
  constexpr int kThreads = 1000;
  for (int t = 0; t < kThreads; ++t) {
    std::thread([] {
      Record r{};
      r.type = RecType::ApiCall;
      r.corr = next_corr();
      record(r);
    }).join();
  }
  Snapshot s;
  collector_stop_and_drain(s);
  EXPECT_EQ(s.records.size(), static_cast<size_t>(kThreads));
  EXPECT_LE(live_rings(), 2u);
}

TEST(Recorder, ClockIsMonotonic) {
  uint64_t a = now_ns(), b = now_ns();
  EXPECT_GT(a, 0u);
  EXPECT_GE(b, a);
}

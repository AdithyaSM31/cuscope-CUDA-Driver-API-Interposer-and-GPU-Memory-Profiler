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

TEST(Recorder, ClockIsMonotonic) {
  uint64_t a = now_ns(), b = now_ns();
  EXPECT_GT(a, 0u);
  EXPECT_GE(b, a);
}

#include "summary.hpp"

#include <gtest/gtest.h>

#include "hook_names.hpp"
#include "model.hpp"

using namespace cuscope;

namespace {

Record launch(uint64_t corr, uint64_t func) {
  Record r{};
  r.type = RecType::Launch;
  r.api = H_cuLaunchKernel;
  r.corr = corr;
  r.func = func;
  return r;
}

GpuSpan span(uint64_t corr, uint64_t func, uint64_t duration_ns) {
  GpuSpan s{};
  s.corr = corr;
  s.func = func;
  s.start_ns = 1000;
  s.end_ns = 1000 + duration_ns;
  return s;
}

}  // namespace

TEST(Summary, KernelsSortedByTotal) {
  Snapshot s;
  s.kernel_names[1] = "short";
  s.kernel_names[2] = "long";
  s.records = {launch(1, 1), launch(2, 1), launch(3, 2)};
  s.spans = {span(1, 1, 100), span(2, 1, 300), span(3, 2, 1000)};

  Summary sum = summarize(s);
  ASSERT_EQ(sum.kernels.size(), 2u);
  EXPECT_EQ(sum.kernels[0].name, "long");
  EXPECT_EQ(sum.kernels[0].count, 1u);
  EXPECT_EQ(sum.kernels[0].gpu_ns_total, 1000u);
  EXPECT_EQ(sum.kernels[1].name, "short");
  EXPECT_EQ(sum.kernels[1].count, 2u);
  EXPECT_EQ(sum.kernels[1].gpu_ns_total, 400u);
  EXPECT_EQ(sum.kernels[1].gpu_ns_max, 300u);
}

TEST(Summary, TimedCountsOnlyLaunchesWithSpans) {
  // With sampling, some launches have no GPU span: the stat says how many did.
  Snapshot s;
  s.records = {launch(1, 9), launch(2, 9), launch(3, 9), launch(4, 9)};
  s.spans = {span(1, 9, 100), span(3, 9, 300)};
  Summary sum = summarize(s);
  ASSERT_EQ(sum.kernels.size(), 1u);
  EXPECT_EQ(sum.kernels[0].count, 4u);
  EXPECT_EQ(sum.kernels[0].timed, 2u);
  EXPECT_EQ(sum.kernels[0].gpu_ns_total, 400u);
}

TEST(Summary, LaunchesCountWithoutGpuTiming) {
  // alloc mode records launches but no GPU spans.
  Snapshot s;
  s.records = {launch(1, 0xbeef), launch(2, 0xbeef)};
  Summary sum = summarize(s);
  ASSERT_EQ(sum.kernels.size(), 1u);
  EXPECT_EQ(sum.kernels[0].name, "0xbeef");
  EXPECT_EQ(sum.kernels[0].count, 2u);
  EXPECT_EQ(sum.kernels[0].gpu_ns_total, 0u);
}

TEST(Summary, CopyBandwidth) {
  Snapshot s;
  Record copy{};
  copy.type = RecType::Copy;
  copy.api = H_cuMemcpyHtoD;
  copy.aux = copy_aux(CopyDir::HtoD, true);
  copy.bytes = 1'000'000'000;
  copy.corr = 7;
  s.records = {copy};
  s.spans = {span(7, 0, 100'000'000)};

  Summary sum = summarize(s);
  ASSERT_EQ(sum.copies.size(), 1u);
  EXPECT_EQ(sum.copies[0].dir, CopyDir::HtoD);
  EXPECT_TRUE(sum.copies[0].pinned);
  EXPECT_EQ(sum.copies[0].count, 1u);
  EXPECT_EQ(sum.copies[0].bytes, 1'000'000'000u);
  EXPECT_DOUBLE_EQ(sum.copies[0].gbps, 10.0);
}

TEST(Summary, CopiesGroupedByDirectionAndPinning) {
  Snapshot s;
  for (bool pinned : {false, true, false}) {
    Record c{};
    c.type = RecType::Copy;
    c.aux = copy_aux(CopyDir::DtoH, pinned);
    c.bytes = 10;
    s.records.push_back(c);
  }
  Summary sum = summarize(s);
  ASSERT_EQ(sum.copies.size(), 2u);
  EXPECT_FALSE(sum.copies[0].pinned);
  EXPECT_EQ(sum.copies[0].count, 2u);
  EXPECT_TRUE(sum.copies[1].pinned);
  EXPECT_DOUBLE_EQ(sum.copies[1].gbps, 0.0);  // no GPU time measured
}

TEST(Summary, GraphLaunchesCounted) {
  Snapshot s;
  Record g{};
  g.type = RecType::GraphLaunch;
  s.records = {g, g};
  EXPECT_EQ(summarize(s).graph_launches, 2u);
}

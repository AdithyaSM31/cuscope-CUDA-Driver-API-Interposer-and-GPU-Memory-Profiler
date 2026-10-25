#include "report.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "hook_names.hpp"
#include "model.hpp"
#include "summary.hpp"

using namespace cuscope;

namespace {

std::string render(const Snapshot& s) {
  char* buf = nullptr;
  size_t size = 0;
  FILE* out = open_memstream(&buf, &size);
  print_summary(s, summarize(s), out);
  std::fclose(out);
  std::string text(buf, size);
  std::free(buf);
  return text;
}

Snapshot sample() {
  Snapshot s;
  s.pid = 42;
  s.mode = "full";
  s.peak_bytes[static_cast<int>(AllocKind::Device)] = 3 << 20;
  s.peak_device_total = 3 << 20;
  s.alloc_count = 3;
  s.free_count = 1;
  // The static CUDA runtime lives inside the program, so its frames come first.
  s.stacks[7] = {{"/bin/app", 0x100, "libcudart_static_5382377d5c77"},
                 {"/bin/app", 0x200, "cudaMalloc"},
                 {"/bin/app", 0x300, "main"}};
  s.leaks.push_back({0x1000, 1 << 20, AllocKind::Device, 7, false});
  s.leaks.push_back({0x2000, 2 << 20, AllocKind::Device, 7, true});
  s.invalid_frees.push_back({0xdead, 7});

  Record launch{};
  launch.type = RecType::Launch;
  launch.api = H_cuLaunchKernel;
  launch.func = 0x55;
  launch.corr = 1;
  s.records.push_back(launch);
  GpuSpan span{};
  span.corr = 1;
  span.start_ns = 0;
  span.end_ns = 2'500'000;
  s.spans.push_back(span);
  s.kernel_names[0x55] = "spin_kernel(long long)";

  Record copy{};
  copy.type = RecType::Copy;
  copy.aux = copy_aux(CopyDir::HtoD, true);
  copy.bytes = 64 << 20;
  s.records.push_back(copy);
  s.dropped_records = 2;
  return s;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

TEST(Report, SectionsInOrder) {
  const std::string text = render(sample());
  size_t last = 0;
  for (const char* header : {"== cuscope summary (pid 42, mode full) ==", "Memory", "Leaks", "Invalid frees",
                             "Top kernels", "Copies", "Completeness: dropped=2 unresolved=0 untimed=0"}) {
    size_t at = text.find(header, last);
    ASSERT_NE(at, std::string::npos) << header << "\n" << text;
    last = at;
  }
}

TEST(Report, MemoryFigures) {
  const std::string text = render(sample());
  EXPECT_TRUE(contains(text, "3.0 MiB")) << text;
  EXPECT_TRUE(contains(text, "allocations 3, frees 1")) << text;
}

TEST(Report, LeaksPointPastTheCudaRuntime) {
  const std::string text = render(sample());
  EXPECT_TRUE(contains(text, "Leaks (2, 3.0 MiB)")) << text;
  EXPECT_TRUE(contains(text, "1.0 MiB device at main")) << text;
  EXPECT_TRUE(contains(text, "at context teardown")) << text;
  EXPECT_FALSE(contains(text, "at libcudart_static")) << text;
}

TEST(Report, ProgramFunctionNamedCudaIsTheSite) {
  // Runtime API functions are cudaX with X uppercase; a program's own
  // cuda_check helper is where the program allocated, not CUDA internals.
  Snapshot s = sample();
  s.stacks[7] = {{"/bin/app", 0x100, "libcudart_static_5382377d5c77"},
                 {"/bin/app", 0x200, "cudaMalloc"},
                 {"/bin/app", 0x300, "cuda_check_alloc(unsigned long)"},
                 {"/bin/app", 0x400, "main"}};
  const std::string text = render(s);
  EXPECT_TRUE(contains(text, "1.0 MiB device at cuda_check_alloc(unsigned long)")) << text;
}

TEST(Report, KernelsAndCopies) {
  const std::string text = render(sample());
  EXPECT_TRUE(contains(text, "spin_kernel(long long)")) << text;
  EXPECT_TRUE(contains(text, "2.500 ms")) << text;
  EXPECT_TRUE(contains(text, "HtoD pinned")) << text;
}

TEST(Report, SamplingIsStated) {
  Snapshot s = sample();
  s.sample_every = 4;
  const std::string text = render(s);
  EXPECT_TRUE(contains(text, "(pid 42, mode full, timing 1 in 4 launches)")) << text;
  EXPECT_TRUE(contains(text, "timed")) << text;
}

TEST(Report, EmptySnapshotSaysSo) {
  Snapshot s;
  s.mode = "alloc";
  const std::string text = render(s);
  EXPECT_TRUE(contains(text, "Leaks: none")) << text;
  EXPECT_TRUE(contains(text, "Invalid frees: none")) << text;
}

TEST(Report, MemoryCapSection) {
  Snapshot s = sample();
  s.mem_limit = {true, 2ull << 30, 2140000000, 3, 1610612736, 512ull << 20, AllocKind::Device, 7};
  const std::string text = render(s);
  EXPECT_TRUE(contains(text, "Memory cap: 2.00 GiB, peak charged 1.99 GiB (99.7%)")) << text;
  EXPECT_TRUE(contains(text, "3 allocations denied (1.50 GiB); largest 512.0 MiB device at main")) << text;
  EXPECT_LT(text.find("Memory cap"), text.find("Leaks"));
}

TEST(Report, NoCapNoSection) { EXPECT_FALSE(contains(render(sample()), "Memory cap")); }

TEST(Report, SharingSection) {
  Snapshot s = sample();
  s.sched = {true, true, 80, 4'210'000'000, 1534, 812'000'000, ""};
  const std::string text = render(s);
  EXPECT_TRUE(contains(text, "Sharing: weight 80, 4.210 s GPU time; 1534 launches waited, 812.000 ms in total")) << text;
  EXPECT_LT(text.find("Memory"), text.find("Sharing"));
  EXPECT_LT(text.find("Sharing"), text.find("Leaks"));
  s.sched = {true, false, 80, 0, 0, 0, "no scheduler running"};
  EXPECT_TRUE(contains(render(s), "Sharing: weight 80, not registered (no scheduler running)")) << render(s);
  EXPECT_FALSE(contains(render(sample()), "Sharing"));
}

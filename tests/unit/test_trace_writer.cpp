#include "trace_writer.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

#include "hook_names.hpp"
#include "model.hpp"

using namespace cuscope;
using nlohmann::json;

namespace {

json write(const Snapshot& s) {
  std::ostringstream out;
  write_trace(s, out);
  return json::parse(out.str());
}

Record make_record(RecType type, uint16_t api, uint64_t cpu_ns, uint32_t dur_ns = 0) {
  Record r{};
  r.type = type;
  r.api = api;
  r.cpu_ns = cpu_ns;
  r.dur_ns = dur_ns;
  r.tid = 1;
  return r;
}

std::vector<json> events_with(const json& trace, const std::string& ph) {
  std::vector<json> out;
  for (const json& e : trace["traceEvents"]) {
    if (e["ph"] == ph) out.push_back(e);
  }
  return out;
}

}  // namespace

TEST(TraceWriter, ApiCallBecomesCompleteEvent) {
  Snapshot s;
  s.pid = 42;
  Record r = make_record(RecType::ApiCall, H_cuMemAlloc, 1000, 500);
  r.tid = 7;
  s.records.push_back(r);

  auto complete = events_with(write(s), "X");
  ASSERT_EQ(complete.size(), 1u);
  EXPECT_EQ(complete[0]["name"], "cuMemAlloc");
  EXPECT_DOUBLE_EQ(complete[0]["ts"].get<double>(), 1.0);
  EXPECT_DOUBLE_EQ(complete[0]["dur"].get<double>(), 0.5);
  EXPECT_EQ(complete[0]["pid"], 42);
  EXPECT_EQ(complete[0]["tid"], 7);
}

TEST(TraceWriter, ApiEventCarriesCorrelationId) {
  // Lets a reader pair a launch with its GPU span (flow arrows do it visually).
  Snapshot s;
  Record r = make_record(RecType::Launch, H_cuLaunchKernel, 1000, 10);
  r.corr = 5;
  s.records.push_back(r);
  EXPECT_EQ(events_with(write(s), "X")[0]["args"]["corr"], 5);
}

TEST(TraceWriter, MemoryCounterRisesAndFalls) {
  Snapshot s;
  Record alloc = make_record(RecType::Alloc, H_cuMemAlloc, 1000);
  alloc.aux = static_cast<uint8_t>(AllocKind::Device);
  alloc.bytes = 1 << 20;
  Record free_ = make_record(RecType::Free, H_cuMemFree, 2000);
  free_.aux = static_cast<uint8_t>(AllocKind::Device);
  free_.bytes = 1 << 20;
  s.records = {free_, alloc};  // out of order: the writer sorts by time

  std::vector<uint64_t> values;
  for (const json& e : events_with(write(s), "C")) {
    EXPECT_EQ(e["name"], "device memory");
    values.push_back(e["args"]["bytes"].get<uint64_t>());
  }
  EXPECT_EQ(values, (std::vector<uint64_t>{1 << 20, 0}));
}

TEST(TraceWriter, PinnedHostMemoryIsNotDeviceMemory) {
  Snapshot s;
  Record alloc = make_record(RecType::Alloc, H_cuMemHostAlloc, 1000);
  alloc.aux = static_cast<uint8_t>(AllocKind::HostPinned);
  alloc.bytes = 4096;
  s.records = {alloc};
  for (const json& e : events_with(write(s), "C")) EXPECT_EQ(e["args"]["bytes"], 0);
}

TEST(TraceWriter, GpuSpanOnStreamTrack) {
  Snapshot s;
  GpuSpan span{};
  span.stream = 0xabc;
  span.start_ns = 5000;
  span.end_ns = 9000;
  span.api = H_cuLaunchKernel;
  span.func = 0x1234;
  s.spans.push_back(span);
  json trace = write(s);

  auto complete = events_with(trace, "X");
  ASSERT_EQ(complete.size(), 1u);
  EXPECT_GE(complete[0]["tid"].get<int64_t>(), 1000000);
  EXPECT_EQ(complete[0]["name"], "0x1234");  // no kernel name known
  EXPECT_DOUBLE_EQ(complete[0]["dur"].get<double>(), 4.0);

  bool named = false;
  for (const json& e : events_with(trace, "M")) {
    if (e["name"] == "thread_name" && e["tid"] == complete[0]["tid"]) {
      named = e["args"]["name"] == "GPU stream 0xabc";
    }
  }
  EXPECT_TRUE(named);
}

TEST(TraceWriter, KernelNameFromSnapshot) {
  Snapshot s;
  GpuSpan span{};
  span.api = H_cuLaunchKernel;
  span.func = 0x1234;
  span.end_ns = 1;
  s.spans.push_back(span);
  s.kernel_names[0x1234] = "spin_kernel(long long)";
  EXPECT_EQ(events_with(write(s), "X")[0]["name"], "spin_kernel(long long)");
}

TEST(TraceWriter, EscapesNames) {
  Snapshot s;
  GpuSpan span{};
  span.api = H_cuLaunchKernel;
  span.func = 1;
  span.end_ns = 1;
  s.spans.push_back(span);
  const std::string tricky = "k<\"q\", \\path\\>\n\ttab\x01";
  s.kernel_names[1] = tricky;
  EXPECT_EQ(events_with(write(s), "X")[0]["name"], tricky);  // parses and round-trips
}

TEST(TraceWriter, ReportSectionMatchesSchema) {
  Snapshot s;
  s.pid = 9;
  s.mode = "full";
  s.leaks.push_back({0x10, 64, AllocKind::Managed, 3, true});
  s.invalid_frees.push_back({0x20, 4});
  s.stacks[3] = {{"/bin/app", 0x40, "main"}};
  json r = write(s)["cuscope"];

  for (const char* key : {"version", "pid", "mode", "peak_bytes", "alloc_count", "free_count", "leaks",
                          "invalid_frees", "stacks", "kernels", "copies", "graph_launches", "dropped_records",
                          "unresolved_events", "gpu_untimed"}) {
    EXPECT_TRUE(r.contains(key)) << key;
  }
  for (const char* key : {"device", "managed", "host_pinned", "async", "device_total"}) {
    EXPECT_TRUE(r["peak_bytes"].contains(key)) << key;
  }
  EXPECT_EQ(r["pid"], 9);
  EXPECT_EQ(r["leaks"][0], json({{"ptr", "0x10"}, {"bytes", 64}, {"kind", "managed"}, {"stack", 3},
                                 {"at_ctx_teardown", true}}));
  EXPECT_EQ(r["invalid_frees"][0], json({{"ptr", "0x20"}, {"stack", 4}}));
  EXPECT_EQ(r["stacks"]["3"][0], json({{"module", "/bin/app"}, {"offset", 64}, {"symbol", "main"}}));
}

TEST(TraceWriter, UnwritablePathReturnsFalse) {
  EXPECT_FALSE(write_trace_file(Snapshot{}, "/nonexistent/dir/x.json"));
}

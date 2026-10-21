#include "trace_writer.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#include "hook_names.hpp"
#include "log.hpp"
#include "summary.hpp"

#ifndef CUSCOPE_VERSION
#define CUSCOPE_VERSION "dev"
#endif

namespace cuscope {
namespace {

constexpr int64_t kGpuTidBase = 1000000;

// Chrome trace timestamps are microseconds; keep nanosecond precision.
std::string micros(uint64_t ns) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(ns) / 1000.0);
  return buf;
}

std::string hex(uint64_t v) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
  return buf;
}

std::string quoted(const std::string& s) { return "\"" + json_escape(s) + "\""; }

bool is_launch(uint16_t api) { return api == H_cuLaunchKernel || api == H_cuLaunchKernelEx; }

std::string span_name(const Snapshot& s, const GpuSpan& span) {
  if (is_launch(span.api)) return kernel_name(s, span.func);
  if (span.api == H_cuGraphLaunch) return "graph " + hex(span.func);
  return std::string(copy_dir_name(copy_dir(span.aux))) + " " + std::to_string(span.bytes) + " B";
}

// Writes the events of the "traceEvents" array, comma-separated.
class EventList {
 public:
  explicit EventList(std::ostream& out) : out_(out) {}
  void add(const std::string& json_object) {
    out_ << (first_ ? "\n" : ",\n") << json_object;
    first_ = false;
  }

 private:
  std::ostream& out_;
  bool first_ = true;
};

void write_cpu_events(const Snapshot& s, EventList& events) {
  std::vector<const Record*> sorted;
  sorted.reserve(s.records.size());
  for (const Record& r : s.records) sorted.push_back(&r);
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const Record* a, const Record* b) { return a->cpu_ns < b->cpu_ns; });

  const std::string pid = std::to_string(s.pid);
  uint64_t device_bytes = 0;
  for (const Record* r : sorted) {
    std::string args;
    auto add_arg = [&args](const std::string& key_value) { args += (args.empty() ? "" : ",") + key_value; };
    if (r->bytes != 0) add_arg("\"bytes\":" + std::to_string(r->bytes));
    if (r->ptr != 0) add_arg("\"ptr\":" + quoted(hex(r->ptr)));
    if (r->corr != 0) add_arg("\"corr\":" + std::to_string(r->corr));
    events.add("{\"name\":" + quoted(hook_name(r->api)) + ",\"cat\":\"api\",\"ph\":\"X\",\"ts\":" +
               micros(r->cpu_ns) + ",\"dur\":" + micros(r->dur_ns) + ",\"pid\":" + pid +
               ",\"tid\":" + std::to_string(r->tid) + ",\"args\":{" + args + "}}");

    const bool alloc = r->type == RecType::Alloc, freed = r->type == RecType::Free;
    if (alloc || freed) {
      if (is_device_memory(static_cast<AllocKind>(r->aux))) {
        device_bytes = alloc ? device_bytes + r->bytes : device_bytes - std::min(device_bytes, r->bytes);
      }
      events.add("{\"name\":\"device memory\",\"ph\":\"C\",\"ts\":" + micros(r->cpu_ns) + ",\"pid\":" + pid +
                 ",\"args\":{\"bytes\":" + std::to_string(device_bytes) + "}}");
    }
  }
}

// Where each correlated call happened on the CPU, for flow arrows.
struct CallSite {
  uint64_t cpu_ns;
  uint32_t tid;
};

void write_gpu_events(const Snapshot& s, EventList& events) {
  const std::string pid = std::to_string(s.pid);
  std::map<uint64_t, CallSite> calls;  // corr -> launching call
  for (const Record& r : s.records) {
    if (r.corr != 0) calls[r.corr] = {r.cpu_ns, r.tid};
  }
  std::map<uint64_t, int64_t> stream_tid;  // stream handle -> track, in order of first use
  for (const GpuSpan& span : s.spans) {
    auto [it, added] = stream_tid.try_emplace(span.stream, kGpuTidBase + static_cast<int64_t>(stream_tid.size()));
    const std::string tid = std::to_string(it->second);
    if (added) {
      events.add("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" + pid + ",\"tid\":" + tid +
                 ",\"args\":{\"name\":" + quoted("GPU stream " + hex(span.stream)) + "}}");
    }
    const uint64_t dur = span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0;
    events.add("{\"name\":" + quoted(span_name(s, span)) + ",\"cat\":\"gpu\",\"ph\":\"X\",\"ts\":" +
               micros(span.start_ns) + ",\"dur\":" + micros(dur) + ",\"pid\":" + pid + ",\"tid\":" + tid +
               ",\"args\":{\"corr\":" + std::to_string(span.corr) + "}}");

    // An arrow from the call (flow start inside the API slice) to the GPU
    // work (flow end bound to the enclosing GPU slice).
    auto call = calls.find(span.corr);
    if (span.corr == 0 || call == calls.end()) continue;
    const std::string id = std::to_string(span.corr);
    events.add("{\"name\":\"launch\",\"cat\":\"flow\",\"ph\":\"s\",\"id\":" + id + ",\"ts\":" +
               micros(call->second.cpu_ns) + ",\"pid\":" + pid + ",\"tid\":" + std::to_string(call->second.tid) +
               "}");
    events.add("{\"name\":\"launch\",\"cat\":\"flow\",\"ph\":\"f\",\"bp\":\"e\",\"id\":" + id + ",\"ts\":" +
               micros(span.start_ns) + ",\"pid\":" + pid + ",\"tid\":" + tid + "}");
  }
}

void write_report(const Snapshot& s, std::ostream& out) {
  const Summary sum = summarize(s);
  auto kind_peak = [&](AllocKind k) { return std::to_string(s.peak_bytes[static_cast<int>(k)]); };

  out << "{\"version\":" << quoted(CUSCOPE_VERSION) << ",\"pid\":" << s.pid << ",\"mode\":" << quoted(s.mode)
      << ",\"sample_every\":" << s.sample_every
      << ",\n\"peak_bytes\":{\"device\":" << kind_peak(AllocKind::Device)
      << ",\"managed\":" << kind_peak(AllocKind::Managed) << ",\"host_pinned\":" << kind_peak(AllocKind::HostPinned)
      << ",\"async\":" << kind_peak(AllocKind::Async) << ",\"device_total\":" << s.peak_device_total << "}"
      << ",\n\"alloc_count\":" << s.alloc_count << ",\"free_count\":" << s.free_count;

  out << ",\n\"leaks\":[";
  for (size_t i = 0; i < s.leaks.size(); ++i) {
    const Leak& l = s.leaks[i];
    out << (i ? "," : "") << "{\"ptr\":" << quoted(hex(l.ptr)) << ",\"bytes\":" << l.bytes
        << ",\"kind\":" << quoted(alloc_kind_name(l.kind)) << ",\"stack\":" << l.stack_id
        << ",\"at_ctx_teardown\":" << (l.at_ctx_teardown ? "true" : "false") << "}";
  }
  out << "],\n\"invalid_frees\":[";
  for (size_t i = 0; i < s.invalid_frees.size(); ++i) {
    out << (i ? "," : "") << "{\"ptr\":" << quoted(hex(s.invalid_frees[i].ptr))
        << ",\"stack\":" << s.invalid_frees[i].stack_id << "}";
  }
  out << "],\n\"stacks\":{";
  bool first_stack = true;
  for (const auto& [id, frames] : s.stacks) {
    out << (first_stack ? "" : ",") << quoted(std::to_string(id)) << ":[";
    first_stack = false;
    for (size_t i = 0; i < frames.size(); ++i) {
      out << (i ? "," : "") << "{\"module\":" << quoted(frames[i].module) << ",\"offset\":" << frames[i].offset
          << ",\"symbol\":" << quoted(frames[i].symbol) << "}";
    }
    out << "]";
  }
  out << "},\n\"kernels\":[";
  for (size_t i = 0; i < sum.kernels.size(); ++i) {
    const KernelStat& k = sum.kernels[i];
    out << (i ? "," : "") << "{\"name\":" << quoted(k.name) << ",\"count\":" << k.count << ",\"timed\":" << k.timed
        << ",\"gpu_ns_total\":" << k.gpu_ns_total << ",\"gpu_ns_max\":" << k.gpu_ns_max << "}";
  }
  out << "],\n\"copies\":[";
  for (size_t i = 0; i < sum.copies.size(); ++i) {
    const CopyStat& c = sum.copies[i];
    char gbps[32];
    std::snprintf(gbps, sizeof(gbps), "%.3f", c.gbps);
    out << (i ? "," : "") << "{\"dir\":" << quoted(copy_dir_name(c.dir))
        << ",\"pinned\":" << (c.pinned ? "true" : "false") << ",\"count\":" << c.count << ",\"bytes\":" << c.bytes
        << ",\"gpu_ns_total\":" << c.gpu_ns_total << ",\"gbps\":" << gbps << "}";
  }
  out << "],\n\"graph_launches\":" << sum.graph_launches << ",\"dropped_records\":" << s.dropped_records
      << ",\"unresolved_events\":" << s.unresolved_events << ",\"gpu_untimed\":" << s.gpu_untimed << "}";
}

}  // namespace

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

void write_trace(const Snapshot& s, std::ostream& out) {
  out << "{\"displayTimeUnit\":\"ns\",\n\"traceEvents\":[";
  EventList events(out);
  events.add("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" + std::to_string(s.pid) +
             ",\"args\":{\"name\":\"cuscope\"}}");
  write_cpu_events(s, events);
  write_gpu_events(s, events);
  out << "\n],\n\"cuscope\":";
  write_report(s, out);
  out << "}\n";
}

bool write_trace_file(const Snapshot& s, const std::string& path) {
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file) {
    log_msg("cannot write trace to %s: %s", path.c_str(), std::strerror(errno));
    return false;
  }
  write_trace(s, file);
  file.flush();
  if (!file) {
    log_msg("error while writing trace to %s", path.c_str());
    return false;
  }
  return true;
}

}  // namespace cuscope

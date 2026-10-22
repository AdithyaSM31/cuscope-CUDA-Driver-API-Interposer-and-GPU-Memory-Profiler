#include "report.hpp"

#include <cctype>
#include <cinttypes>
#include <cstring>
#include <string>

namespace cuscope {
namespace {

constexpr size_t kMaxListed = 20;  // leaks and invalid frees listed before "... and N more"
constexpr size_t kTopKernels = 10;

std::string bytes_text(uint64_t bytes) {
  char buf[32];
  if (bytes < 1024) {
    std::snprintf(buf, sizeof(buf), "%" PRIu64 " B", bytes);
  } else if (bytes < (1ull << 20)) {
    std::snprintf(buf, sizeof(buf), "%.1f KiB", bytes / 1024.0);
  } else if (bytes < (1ull << 30)) {
    std::snprintf(buf, sizeof(buf), "%.1f MiB", bytes / double(1 << 20));
  } else {
    std::snprintf(buf, sizeof(buf), "%.2f GiB", bytes / double(1ull << 30));
  }
  return buf;
}

std::string time_text(uint64_t ns) {
  char buf[32];
  if (ns >= 1'000'000'000) {
    std::snprintf(buf, sizeof(buf), "%.3f s", ns / 1e9);
  } else if (ns >= 1'000'000) {
    std::snprintf(buf, sizeof(buf), "%.3f ms", ns / 1e6);
  } else if (ns >= 1'000) {
    std::snprintf(buf, sizeof(buf), "%.1f us", ns / 1e3);
  } else {
    std::snprintf(buf, sizeof(buf), "%" PRIu64 " ns", ns);
  }
  return buf;
}

bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

std::string basename_of(const std::string& path) {
  size_t slash = path.rfind('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Frames inside the CUDA runtime or driver say nothing about the program. The
// static runtime is linked into the program itself, so module alone is not
// enough: its functions are recognised by name.
// Runtime API functions are cudaX with X uppercase (cudaMalloc), so a
// program helper such as cuda_check stays a program frame.
bool is_cuda_frame(const StackFrame& f) {
  const std::string module = basename_of(f.module);
  const bool runtime_api =
      starts_with(f.symbol, "cuda") && f.symbol.size() > 4 && std::isupper(static_cast<unsigned char>(f.symbol[4]));
  return runtime_api || starts_with(f.symbol, "libcudart_static") || starts_with(f.symbol, "__cuda") ||
         starts_with(f.symbol, "cudart::") || starts_with(module, "libcudart") || starts_with(module, "libcuda.");
}

// "main (app+0x8a7f)": the first frame that belongs to the program.
std::string call_site(const Snapshot& s, uint32_t stack_id) {
  auto it = s.stacks.find(stack_id);
  if (it == s.stacks.end() || it->second.empty()) return "unknown site";
  const std::vector<StackFrame>& frames = it->second;
  const StackFrame* site = &frames.front();
  for (const StackFrame& f : frames) {
    if (!is_cuda_frame(f)) {
      site = &f;
      break;
    }
  }
  char where[64];
  std::snprintf(where, sizeof(where), "+0x%" PRIx64, site->offset);
  return (site->symbol.empty() ? std::string("?") : site->symbol) + " (" + basename_of(site->module) + where + ")";
}

void print_memory(const Snapshot& s, FILE* out) {
  auto peak = [&](AllocKind k) { return bytes_text(s.peak_bytes[static_cast<int>(k)]); };
  std::fprintf(out, "Memory\n");
  std::fprintf(out, "  peak device memory  %s  (device %s, managed %s, async %s; pinned host %s)\n",
               bytes_text(s.peak_device_total).c_str(), peak(AllocKind::Device).c_str(),
               peak(AllocKind::Managed).c_str(), peak(AllocKind::Async).c_str(), peak(AllocKind::HostPinned).c_str());
  std::fprintf(out, "  allocations %" PRIu64 ", frees %" PRIu64 "\n", s.alloc_count, s.free_count);
}

void print_leaks(const Snapshot& s, FILE* out) {
  if (s.leaks.empty()) {
    std::fprintf(out, "Leaks: none\n");
    return;
  }
  uint64_t total = 0;
  for (const Leak& l : s.leaks) total += l.bytes;
  std::fprintf(out, "Leaks (%zu, %s)\n", s.leaks.size(), bytes_text(total).c_str());
  for (size_t i = 0; i < s.leaks.size() && i < kMaxListed; ++i) {
    const Leak& l = s.leaks[i];
    std::fprintf(out, "  %s %s at %s%s\n", bytes_text(l.bytes).c_str(), alloc_kind_name(l.kind),
                 call_site(s, l.stack_id).c_str(), l.at_ctx_teardown ? ", freed at context teardown" : "");
  }
  if (s.leaks.size() > kMaxListed) std::fprintf(out, "  ... and %zu more\n", s.leaks.size() - kMaxListed);
}

void print_invalid_frees(const Snapshot& s, FILE* out) {
  if (s.invalid_frees.empty()) {
    std::fprintf(out, "Invalid frees: none\n");
    return;
  }
  std::fprintf(out, "Invalid frees (%zu)\n", s.invalid_frees.size());
  for (size_t i = 0; i < s.invalid_frees.size() && i < kMaxListed; ++i) {
    const InvalidFree& f = s.invalid_frees[i];
    std::fprintf(out, "  0x%" PRIx64 " at %s\n", f.ptr, call_site(s, f.stack_id).c_str());
  }
  if (s.invalid_frees.size() > kMaxListed) {
    std::fprintf(out, "  ... and %zu more\n", s.invalid_frees.size() - kMaxListed);
  }
}

// With sampling, GPU time covers only the timed launches, so a "timed"
// column says how many that was.
void print_kernels(const Summary& sum, bool sampled, FILE* out) {
  if (sum.kernels.empty()) {
    std::fprintf(out, "Top kernels: none\n");
  } else if (sampled) {
    std::fprintf(out, "Top kernels\n  %12s  %6s  %6s  %s\n", "GPU time", "calls", "timed", "name");
    for (size_t i = 0; i < sum.kernels.size() && i < kTopKernels; ++i) {
      const KernelStat& k = sum.kernels[i];
      std::fprintf(out, "  %12s  %6" PRIu64 "  %6" PRIu64 "  %s\n",
                   k.gpu_ns_total ? time_text(k.gpu_ns_total).c_str() : "untimed", k.count, k.timed, k.name.c_str());
    }
  } else {
    std::fprintf(out, "Top kernels\n  %12s  %6s  %s\n", "GPU time", "calls", "name");
    for (size_t i = 0; i < sum.kernels.size() && i < kTopKernels; ++i) {
      const KernelStat& k = sum.kernels[i];
      std::fprintf(out, "  %12s  %6" PRIu64 "  %s\n", k.gpu_ns_total ? time_text(k.gpu_ns_total).c_str() : "untimed",
                   k.count, k.name.c_str());
    }
  }
  if (sum.graph_launches > 0) std::fprintf(out, "  graph launches: %" PRIu64 "\n", sum.graph_launches);
}

void print_copies(const Summary& sum, FILE* out) {
  if (sum.copies.empty()) {
    std::fprintf(out, "Copies: none\n");
    return;
  }
  std::fprintf(out, "Copies\n");
  for (const CopyStat& c : sum.copies) {
    char rate[32];
    if (c.gbps > 0) {
      std::snprintf(rate, sizeof(rate), "%.2f GB/s", c.gbps);
    } else {
      std::snprintf(rate, sizeof(rate), "untimed");
    }
    std::fprintf(out, "  %-4s %-8s  %6" PRIu64 " x %-10s %s\n", copy_dir_name(c.dir), c.pinned ? "pinned" : "pageable",
                 c.count, bytes_text(c.count ? c.bytes / c.count : 0).c_str(), rate);
  }
}

}  // namespace

void print_summary(const Snapshot& s, const Summary& sum, FILE* out) {
  const bool sampled = s.sample_every > 1;
  if (sampled) {
    std::fprintf(out, "== cuscope summary (pid %d, mode %s, timing 1 in %u launches) ==\n", s.pid, s.mode.c_str(),
                 s.sample_every);
  } else {
    std::fprintf(out, "== cuscope summary (pid %d, mode %s) ==\n", s.pid, s.mode.c_str());
  }
  print_memory(s, out);
  print_leaks(s, out);
  print_invalid_frees(s, out);
  print_kernels(sum, sampled, out);
  print_copies(sum, out);
  std::fprintf(out, "Completeness: dropped=%" PRIu64 " unresolved=%" PRIu64 " untimed=%" PRIu64 "\n",
               s.dropped_records, s.unresolved_events, s.gpu_untimed);
}

}  // namespace cuscope

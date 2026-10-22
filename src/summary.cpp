#include "summary.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <tuple>
#include <unordered_map>

namespace cuscope {

std::string kernel_name(const Snapshot& s, uint64_t func) {
  auto it = s.kernel_names.find(func);
  if (it != s.kernel_names.end()) return it->second;
  char hex[24];
  std::snprintf(hex, sizeof(hex), "0x%llx", static_cast<unsigned long long>(func));
  return hex;
}

Summary summarize(const Snapshot& s) {
  std::unordered_map<uint64_t, const GpuSpan*> span_by_corr;
  for (const GpuSpan& span : s.spans) {
    if (span.corr != 0) span_by_corr[span.corr] = &span;
  }
  auto span_of = [&](uint64_t corr) -> const GpuSpan* {
    auto it = span_by_corr.find(corr);
    return corr == 0 || it == span_by_corr.end() ? nullptr : it->second;
  };
  auto gpu_ns = [&](uint64_t corr) -> uint64_t {
    const GpuSpan* span = span_of(corr);
    return span != nullptr && span->end_ns > span->start_ns ? span->end_ns - span->start_ns : 0;
  };

  Summary out;
  std::map<std::string, KernelStat> kernels;
  std::map<std::pair<CopyDir, bool>, CopyStat> copies;

  for (const Record& r : s.records) {
    switch (r.type) {
      case RecType::Launch: {
        std::string name = kernel_name(s, r.func);
        KernelStat& k = kernels[name];
        k.name = name;
        uint64_t ns = gpu_ns(r.corr);
        k.count++;
        if (span_of(r.corr) != nullptr) k.timed++;
        k.gpu_ns_total += ns;
        k.gpu_ns_max = std::max(k.gpu_ns_max, ns);
        break;
      }
      case RecType::Copy: {
        auto key = std::make_pair(copy_dir(r.aux), copy_pinned(r.aux));
        CopyStat& c = copies.try_emplace(key, CopyStat{key.first, key.second}).first->second;
        c.count++;
        c.bytes += r.bytes;
        if (span_of(r.corr) != nullptr) {
          c.timed_bytes += r.bytes;
          c.gpu_ns_total += gpu_ns(r.corr);
        }
        break;
      }
      case RecType::GraphLaunch:
        out.graph_launches++;
        break;
      default:
        break;
    }
  }

  for (auto& [name, k] : kernels) out.kernels.push_back(k);
  std::sort(out.kernels.begin(), out.kernels.end(), [](const KernelStat& a, const KernelStat& b) {
    return std::tie(b.gpu_ns_total, a.name) < std::tie(a.gpu_ns_total, b.name);
  });

  for (auto& [key, c] : copies) {
    c.gbps = c.gpu_ns_total > 0 ? static_cast<double>(c.timed_bytes) / static_cast<double>(c.gpu_ns_total) : 0.0;
    out.copies.push_back(c);
  }
  return out;
}

}  // namespace cuscope

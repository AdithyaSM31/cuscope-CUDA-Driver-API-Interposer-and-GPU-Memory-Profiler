#pragma once

// Aggregates a Snapshot into the per-kernel and per-copy tables the report and
// trace show. Counts come from the CPU-side records, so they exist in alloc
// mode too; GPU time comes from the spans matched by correlation id.

#include <cstdint>
#include <string>
#include <vector>

#include "model.hpp"

namespace cuscope {

struct KernelStat {
  std::string name;
  uint64_t count = 0, gpu_ns_total = 0, gpu_ns_max = 0;
};

struct CopyStat {
  CopyDir dir;
  bool pinned;
  uint64_t count = 0, bytes = 0, gpu_ns_total = 0;
  double gbps = 0.0;  // bytes per GPU nanosecond, which equals GB/s; 0 without GPU time
};

struct Summary {
  std::vector<KernelStat> kernels;  // by gpu_ns_total descending, then name
  std::vector<CopyStat> copies;     // by direction, then pageable before pinned
  uint64_t graph_launches = 0;
};

Summary summarize(const Snapshot& s);

// The kernel's name from the snapshot, or its handle as "0x<hex>".
std::string kernel_name(const Snapshot& s, uint64_t func);

}  // namespace cuscope

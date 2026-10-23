#pragma once

// What cuscope records, and the snapshot handed to the trace writer and
// reporter at exit. Pure data: no CUDA, no locking.

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cuscope {

enum class AllocKind : uint8_t { Device = 0, Managed = 1, HostPinned = 2, Async = 3 };
constexpr int kAllocKinds = 4;

// Device-resident kinds count toward "device memory"; pinned host memory does not.
constexpr bool is_device_memory(AllocKind kind) { return kind != AllocKind::HostPinned; }

constexpr const char* alloc_kind_name(AllocKind kind) {
  switch (kind) {
    case AllocKind::Device: return "device";
    case AllocKind::Managed: return "managed";
    case AllocKind::HostPinned: return "host_pinned";
    case AllocKind::Async: return "async";
  }
  return "?";
}

enum class CopyDir : uint8_t { HtoD = 0, DtoH = 1, DtoD = 2, HtoH = 3, Unknown = 4 };

constexpr const char* copy_dir_name(CopyDir dir) {
  switch (dir) {
    case CopyDir::HtoD: return "HtoD";
    case CopyDir::DtoH: return "DtoH";
    case CopyDir::DtoD: return "DtoD";
    case CopyDir::HtoH: return "HtoH";
    case CopyDir::Unknown: return "Unknown";
  }
  return "?";
}

// A copy's Record::aux and GpuSpan::aux: direction in the low nibble, pinned host memory in bit 4.
constexpr uint8_t copy_aux(CopyDir dir, bool pinned) {
  return static_cast<uint8_t>(static_cast<uint8_t>(dir) | (pinned ? 0x10 : 0));
}
constexpr CopyDir copy_dir(uint8_t aux) { return static_cast<CopyDir>(aux & 0x0F); }
constexpr bool copy_pinned(uint8_t aux) { return (aux & 0x10) != 0; }

enum class RecType : uint8_t { ApiCall, Alloc, AllocFailed, Free, InvalidFree, Launch, Copy, GraphLaunch };

// One hooked call, written by the calling thread into its ring buffer.
struct alignas(64) Record {
  RecType type;
  uint8_t aux;         // AllocKind for Alloc/AllocFailed/Free, copy_aux() for Copy
  uint16_t api;        // HookId
  uint32_t tid;
  uint64_t cpu_ns;     // CLOCK_MONOTONIC when the call started
  uint32_t dur_ns;     // CPU duration of the real call, saturating
  uint32_t stack_id;   // 0 = none
  uint64_t ptr;        // allocation, freed pointer, or copy destination
  uint64_t bytes;
  uint64_t stream;
  uint64_t func;       // CUfunction, CUgraphExec, or copy source
  uint64_t corr;       // ties a call to its GpuSpan; 0 = none
};
static_assert(sizeof(Record) == 64, "a Record is one cache line");

// GPU execution of one launch or copy, on the CPU clock.
struct GpuSpan {
  uint64_t corr, stream, start_ns, end_ns, func, bytes;
  uint16_t api;
  uint8_t aux;
};

struct StackFrame {
  std::string module;  // absolute path of the object
  uint64_t offset;     // address minus the object's load base
  std::string symbol;  // demangled, or empty
};

struct Leak {
  uint64_t ptr, bytes;
  AllocKind kind;
  uint32_t stack_id;
  bool at_ctx_teardown;  // still live when its context was destroyed
};

struct InvalidFree {
  uint64_t ptr;
  uint32_t stack_id;
};

// The memory cap's figures at exit (CUSCOPE_MEM_LIMIT). enabled is false when
// no cap was set, and then nothing else is meaningful.
struct MemLimitStats {
  bool enabled = false;
  uint64_t limit_bytes = 0, peak_charged_bytes = 0, denied_count = 0, denied_bytes = 0;
  uint64_t largest_denied_bytes = 0;
  AllocKind largest_denied_kind = AllocKind::Device;
  uint32_t largest_denied_stack = 0;
};

// Everything known at exit.
struct Snapshot {
  int pid = 0;
  std::string mode;
  uint32_t sample_every = 1;  // CUSCOPE_SAMPLE_EVERY: 1 in N launches is timed
  std::vector<Record> records;
  std::vector<GpuSpan> spans;
  std::map<uint64_t, std::string> kernel_names;  // function handle -> demangled name
  std::array<uint64_t, kAllocKinds> peak_bytes{};
  uint64_t peak_device_total = 0;  // peak of Device + Managed + Async together
  uint64_t alloc_count = 0, free_count = 0;
  std::vector<Leak> leaks;
  std::vector<InvalidFree> invalid_frees;
  std::map<uint32_t, std::vector<StackFrame>> stacks;
  uint64_t dropped_records = 0, unresolved_events = 0, gpu_untimed = 0;
  MemLimitStats mem_limit;
};

}  // namespace cuscope

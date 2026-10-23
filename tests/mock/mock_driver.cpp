// Mock libcuda for GPU-free tests: a few dozen driver functions on host memory.
//
// Functions are defined by their cuda.h names, so the header's macros give
// them the same exported symbol names as the real driver (cuMemAlloc ->
// cuMemAlloc_v2). Per-thread variants (_ptsz) and the old 4-argument
// cuGetProcAddress are defined explicitly.
#include <cuda.h>

#include "mock_counters.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#define MOCK_API extern "C" __attribute__((visibility("default")))

namespace {

struct Counters {
  std::atomic<uint64_t> alloc_v1{0}, alloc_v2{0}, free_{0}, memcpy_htod{0}, memcpy_dtoh{0},
      memcpy_async{0}, memcpy_async_ptsz{0}, launch{0}, launch_ptsz{0}, event_record{0},
      event_record_on_capturing{0}, ctx_destroy{0}, event_query{0}, event_sync{0};
};
Counters g_count;

const CUcontext kPrimaryCtx = reinterpret_cast<CUcontext>(0x1000);
const CUstream kCapturingStream = reinterpret_cast<CUstream>(MOCK_CAPTURING_STREAM_HANDLE);

std::mutex g_mu;
int g_ctx_refs = 0;
bool g_ctx_destroyed = false;
uint64_t g_ctx_generation = 0;  // bumped on destroy; events from older generations are invalid
uint32_t g_next_v1_ptr = 0x10000;
uint64_t g_completed_until_ns = 0;  // work recorded up to here has finished (in-order streams)
thread_local CUcontext t_current = nullptr;

// Heap-allocated and never freed, so the mock still works from exit handlers
// that run after static destructors.
std::unordered_map<CUdeviceptr, size_t>& device_allocs() {  // pointer -> bytes
  static auto* m = new std::unordered_map<CUdeviceptr, size_t>();
  return *m;
}
std::unordered_set<void*>& host_allocs() {
  static auto* s = new std::unordered_set<void*>();
  return *s;
}
uint64_t g_live_bytes = 0, g_live_hwm = 0;  // device memory, guarded by g_mu

void add_device_alloc_locked(CUdeviceptr p, size_t bytes) {
  device_allocs()[p] = bytes;
  g_live_bytes += bytes;
  if (g_live_bytes > g_live_hwm) g_live_hwm = g_live_bytes;
}

struct MockEvent {
  uint64_t generation;
  uint64_t t_ns = 0;
  int queries = 0;
  bool synced = false;
};

uint64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// Queries that return CUDA_ERROR_NOT_READY before an event counts as complete.
int event_delay() {
  static const int delay = [] {
    const char* v = std::getenv("MOCK_EVENT_DELAY_QUERIES");
    return v ? std::atoi(v) : 0;
  }();
  return delay;
}

bool event_alive(const MockEvent* e) { return e->generation == g_ctx_generation && !g_ctx_destroyed; }

// Like a real stream, work completes in order: once an event is known to be
// complete, so is everything recorded before it.
bool event_ready(const MockEvent* e) {
  return e->synced || e->queries >= event_delay() || (e->t_ns != 0 && e->t_ns <= g_completed_until_ns);
}

void destroy_context_locked() {
  g_ctx_destroyed = true;
  g_ctx_refs = 0;
  ++g_ctx_generation;
  g_count.ctx_destroy++;
}

CUresult record_event(CUevent ev, CUstream stream) {
  auto* e = reinterpret_cast<MockEvent*>(ev);
  g_count.event_record++;
  if (stream == kCapturingStream) g_count.event_record_on_capturing++;
  std::lock_guard<std::mutex> lock(g_mu);
  e->t_ns = now_ns();
  e->queries = 0;
  e->synced = false;
  return CUDA_SUCCESS;
}

CUresult mem_alloc_v1(unsigned int* dptr, unsigned int bytes) {
  if (dptr == nullptr || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mu);
  *dptr = g_next_v1_ptr;
  g_next_v1_ptr += 0x1000;
  g_count.alloc_v1++;
  return CUDA_SUCCESS;
}

}  // namespace

MOCK_API void mock_get_counters(mock_counters* out) {
  out->alloc_v1 = g_count.alloc_v1;
  out->alloc_v2 = g_count.alloc_v2;
  out->free_ = g_count.free_;
  out->memcpy_htod = g_count.memcpy_htod;
  out->memcpy_dtoh = g_count.memcpy_dtoh;
  out->memcpy_async = g_count.memcpy_async;
  out->memcpy_async_ptsz = g_count.memcpy_async_ptsz;
  out->launch = g_count.launch;
  out->launch_ptsz = g_count.launch_ptsz;
  out->event_record = g_count.event_record;
  out->event_record_on_capturing = g_count.event_record_on_capturing;
  out->ctx_destroy = g_count.ctx_destroy;
  out->event_query = g_count.event_query;
  out->event_sync = g_count.event_sync;
  std::lock_guard<std::mutex> lock(g_mu);
  out->live_hwm = g_live_hwm;
}

// ---- Initialisation and contexts ----

MOCK_API CUresult cuInit(unsigned int) { return CUDA_SUCCESS; }

MOCK_API CUresult cuDriverGetVersion(int* version) {
  if (version == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *version = CUDA_VERSION;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuCtxGetCurrent(CUcontext* ctx) {
  if (ctx == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *ctx = t_current;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuCtxSetCurrent(CUcontext ctx) {
  t_current = ctx;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuCtxGetDevice(CUdevice* dev) {
  if (dev == nullptr) return CUDA_ERROR_INVALID_VALUE;
  if (t_current == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
  *dev = 0;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuCtxSynchronize() { return CUDA_SUCCESS; }

MOCK_API CUresult cuCtxDestroy(CUcontext ctx) {
  if (ctx != kPrimaryCtx) return CUDA_ERROR_INVALID_CONTEXT;
  std::lock_guard<std::mutex> lock(g_mu);
  destroy_context_locked();
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, CUdevice dev) {
  if (ctx == nullptr || dev != 0) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_ctx_refs == 0) g_ctx_destroyed = false;
  ++g_ctx_refs;
  *ctx = kPrimaryCtx;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_ctx_refs > 0 && --g_ctx_refs == 0) destroy_context_locked();
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
  if (dev != 0 || flags == nullptr || active == nullptr) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mu);
  *flags = 0;
  *active = (g_ctx_refs > 0 && !g_ctx_destroyed) ? 1 : 0;
  return CUDA_SUCCESS;
}

// ---- Memory ----

MOCK_API CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytes) {
  if (dptr == nullptr || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  void* p = std::malloc(bytes);
  if (p == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
  std::lock_guard<std::mutex> lock(g_mu);
  *dptr = reinterpret_cast<CUdeviceptr>(p);
  add_device_alloc_locked(*dptr, bytes);
  g_count.alloc_v2++;
  return CUDA_SUCCESS;
}

// Rows are padded to 512 bytes, like a real device's pitch alignment.
MOCK_API CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height, unsigned int) {
  if (dptr == nullptr || pitch == nullptr || width == 0 || height == 0) return CUDA_ERROR_INVALID_VALUE;
  const size_t padded = (width + 511) / 512 * 512;
  void* p = std::malloc(padded * height);
  if (p == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
  std::lock_guard<std::mutex> lock(g_mu);
  *dptr = reinterpret_cast<CUdeviceptr>(p);
  *pitch = padded;
  add_device_alloc_locked(*dptr, padded * height);
  return CUDA_SUCCESS;
}

// A 6 GiB GPU; free is what the program has not allocated.
MOCK_API CUresult cuMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
  constexpr size_t kTotal = size_t{6} << 30;
  std::lock_guard<std::mutex> lock(g_mu);
  *total_bytes = kTotal;
  *free_bytes = kTotal - g_live_bytes;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuMemAllocHost(void** pp, size_t bytes) {
  if (pp == nullptr || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  void* p = std::malloc(bytes);
  if (p == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
  std::lock_guard<std::mutex> lock(g_mu);
  host_allocs().insert(p);
  *pp = p;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuMemFreeHost(void* p) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (host_allocs().erase(p) == 0) return CUDA_ERROR_INVALID_VALUE;
  std::free(p);
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuMemFree(CUdeviceptr dptr) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = device_allocs().find(dptr);
  if (it == device_allocs().end()) return CUDA_ERROR_INVALID_VALUE;
  g_live_bytes -= it->second;
  device_allocs().erase(it);
  std::free(reinterpret_cast<void*>(dptr));
  g_count.free_++;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
  if (data == nullptr || attribute != CU_POINTER_ATTRIBUTE_MEMORY_TYPE) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mu);
  if (device_allocs().count(ptr) == 0) return CUDA_ERROR_INVALID_VALUE;  // pageable host memory
  *static_cast<CUmemorytype*>(data) = CU_MEMORYTYPE_DEVICE;
  return CUDA_SUCCESS;
}

// ---- Copies ----

MOCK_API CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytes) {
  std::memcpy(reinterpret_cast<void*>(dst), src, bytes);
  g_count.memcpy_htod++;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t bytes) {
  std::memcpy(dst, reinterpret_cast<const void*>(src), bytes);
  g_count.memcpy_dtoh++;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream) {
  std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
  g_count.memcpy_async++;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream) {
  std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
  g_count.memcpy_async_ptsz++;
  return CUDA_SUCCESS;
}

// ---- Launches and streams ----

MOCK_API CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
  g_count.launch++;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuLaunchKernel_ptsz(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
  g_count.launch_ptsz++;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuFuncGetName(const char** name, CUfunction) {
  if (name == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *name = "_Z11mock_kernelv";
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuStreamIsCapturing(CUstream stream, CUstreamCaptureStatus* status) {
  if (status == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *status = stream == kCapturingStream ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuStreamIsCapturing_ptsz(CUstream stream, CUstreamCaptureStatus* status) {
  return cuStreamIsCapturing(stream, status);
}

MOCK_API CUresult cuStreamSynchronize(CUstream) { return CUDA_SUCCESS; }

// ---- Events ----

MOCK_API CUresult cuEventCreate(CUevent* ev, unsigned int) {
  if (ev == nullptr) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mu);
  *ev = reinterpret_cast<CUevent>(new MockEvent{g_ctx_generation});
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuEventDestroy(CUevent ev) {
  delete reinterpret_cast<MockEvent*>(ev);
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuEventRecord(CUevent ev, CUstream stream) { return record_event(ev, stream); }
MOCK_API CUresult cuEventRecord_ptsz(CUevent ev, CUstream stream) { return record_event(ev, stream); }

MOCK_API CUresult cuEventQuery(CUevent ev) {
  g_count.event_query++;
  auto* e = reinterpret_cast<MockEvent*>(ev);
  std::lock_guard<std::mutex> lock(g_mu);
  if (!event_alive(e)) return CUDA_ERROR_INVALID_CONTEXT;
  if (event_ready(e)) {
    // Complete, so everything recorded before it is too.
    if (e->t_ns > g_completed_until_ns) g_completed_until_ns = e->t_ns;
    return CUDA_SUCCESS;
  }
  e->queries++;
  return CUDA_ERROR_NOT_READY;
}

MOCK_API CUresult cuEventSynchronize(CUevent ev) {
  g_count.event_sync++;
  auto* e = reinterpret_cast<MockEvent*>(ev);
  std::lock_guard<std::mutex> lock(g_mu);
  if (!event_alive(e)) return CUDA_ERROR_INVALID_CONTEXT;
  e->synced = true;
  if (e->t_ns > g_completed_until_ns) g_completed_until_ns = e->t_ns;
  return CUDA_SUCCESS;
}

MOCK_API CUresult cuEventElapsedTime(float* ms, CUevent start, CUevent end) {
  auto* a = reinterpret_cast<MockEvent*>(start);
  auto* b = reinterpret_cast<MockEvent*>(end);
  if (ms == nullptr) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!event_alive(a) || !event_alive(b)) return CUDA_ERROR_INVALID_CONTEXT;
  if (!event_ready(a) || !event_ready(b)) return CUDA_ERROR_NOT_READY;
  *ms = static_cast<float>(static_cast<double>(b->t_ns - a->t_ns) / 1e6);
  return CUDA_SUCCESS;
}

// ---- Entry points ----

extern "C" __attribute__((visibility("default"))) CUresult mock_get_proc_address_v1(
    const char* symbol, void** pfn, int version, cuuint64_t flags) __asm__("cuGetProcAddress");

namespace {

struct ProcEntry {
  const char* name;
  void* legacy;
  void* per_thread;  // null when the function has no per-thread variant
};

#define ENTRY(fn) {#fn, reinterpret_cast<void*>(&fn), nullptr}
#define ENTRY_PTSZ(fn) {#fn, reinterpret_cast<void*>(&fn), reinterpret_cast<void*>(&fn##_ptsz)}

const ProcEntry kProcTable[] = {
    ENTRY(cuInit), ENTRY(cuDriverGetVersion), ENTRY(cuCtxGetCurrent), ENTRY(cuCtxSetCurrent),
    ENTRY(cuCtxGetDevice), ENTRY(cuCtxSynchronize), ENTRY(cuCtxDestroy),
    ENTRY(cuDevicePrimaryCtxRetain), ENTRY(cuDevicePrimaryCtxRelease), ENTRY(cuDevicePrimaryCtxGetState),
    ENTRY(cuMemFree), ENTRY(cuMemAllocPitch), ENTRY(cuMemAllocHost), ENTRY(cuMemFreeHost), ENTRY(cuMemGetInfo),
    ENTRY(cuPointerGetAttribute), ENTRY(cuMemcpyHtoD), ENTRY(cuMemcpyDtoH),
    ENTRY_PTSZ(cuMemcpyAsync), ENTRY_PTSZ(cuLaunchKernel), ENTRY(cuFuncGetName),
    ENTRY_PTSZ(cuStreamIsCapturing), ENTRY(cuStreamSynchronize), ENTRY(cuEventCreate),
    ENTRY(cuEventDestroy), ENTRY_PTSZ(cuEventRecord), ENTRY(cuEventQuery), ENTRY(cuEventSynchronize),
    ENTRY(cuEventElapsedTime),
};

#undef ENTRY
#undef ENTRY_PTSZ

// Resolves a base name the way the driver does: by version for functions
// whose signature changed, and by flags for per-thread-stream variants.
void* lookup(const char* symbol, int version, cuuint64_t flags) {
  if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
    return version >= 12000 ? reinterpret_cast<void*>(&cuGetProcAddress)
                            : reinterpret_cast<void*>(&mock_get_proc_address_v1);
  }
  if (std::strcmp(symbol, "cuMemAlloc") == 0) {
    return version >= 3020 ? reinterpret_cast<void*>(&cuMemAlloc) : reinterpret_cast<void*>(&mem_alloc_v1);
  }
  for (const ProcEntry& e : kProcTable) {
    if (std::strcmp(symbol, e.name) != 0) continue;
    bool per_thread = (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0;
    return per_thread && e.per_thread != nullptr ? e.per_thread : e.legacy;
  }
  return nullptr;
}

}  // namespace

MOCK_API CUresult cuGetProcAddress(const char* symbol, void** pfn, int version, cuuint64_t flags,
                                   CUdriverProcAddressQueryResult* status) {
  if (symbol == nullptr || pfn == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *pfn = lookup(symbol, version, flags);
  if (status != nullptr) *status = *pfn ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

CUresult mock_get_proc_address_v1(const char* symbol, void** pfn, int version, cuuint64_t flags) {
  return cuGetProcAddress(symbol, pfn, version, flags, nullptr);
}

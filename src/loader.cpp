#include "loader.hpp"

#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>

#include <atomic>

#include "guard.hpp"
#include "hook_table.hpp"
#include "log.hpp"

namespace cuscope {
namespace {

using GpaV2Fn = CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

pthread_once_t g_dlsym_once = PTHREAD_ONCE_INIT;
DlsymFn g_real_dlsym = nullptr;

pthread_once_t g_libcuda_once = PTHREAD_ONCE_INIT;
void* g_libcuda = nullptr;

pthread_once_t g_loader_once = PTHREAD_ONCE_INIT;
std::atomic<bool> g_loader_ready{false};

std::atomic<void*> g_real_gpa[2] = {};

void find_real_dlsym() {
  // RTLD_NEXT searches the libraries loaded after libcuscope, which includes
  // libc. glibc 2.34 moved dlsym into libc under a new version; older
  // binaries and libraries use the GLIBC_2.2.5 one.
  void* fn = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
  if (fn == nullptr) fn = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  g_real_dlsym = reinterpret_cast<DlsymFn>(fn);
}

void open_libcuda() {
  // Reuse the program's copy if it already loaded one; otherwise load it.
  g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (g_libcuda == nullptr) g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
  if (g_libcuda == nullptr) log_msg("cannot load libcuda.so.1: %s", dlerror());
}

void* dlsym_libcuda(const char* name) {
  void* lib = libcuda_handle();
  DlsymFn real = real_dlsym();
  if (lib == nullptr || real == nullptr) return nullptr;
  void* fn = real(lib, name);
  return fn != nullptr && !inside_cuscope(fn) ? fn : nullptr;
}

void load_real_functions() {
  InternalScope internal;
  hook_table_init();

  auto gpa = reinterpret_cast<GpaV2Fn>(resolve_real_gpa(GpaVer::V2));
  int missing = 0;
  for (int id = 0; id < H_COUNT; ++id) {
    const HookSpec& spec = hook_spec(static_cast<HookId>(id));
    for (int v = kLegacy; v <= kPerThread; ++v) {
      const cuuint64_t flags = v == kLegacy ? CU_GET_PROC_ADDRESS_LEGACY_STREAM
                                            : CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
      void* fn = nullptr;
      CUdriverProcAddressQueryResult status;
      if (gpa != nullptr && gpa(spec.base, &fn, spec.version, flags, &status) != CUDA_SUCCESS) fn = nullptr;
      if (fn == nullptr && spec.export_name[v] != nullptr) fn = dlsym_libcuda(spec.export_name[v]);
      if (fn != nullptr && inside_cuscope(fn)) fn = nullptr;
      set_real(static_cast<HookId>(id), v, fn);
      if (fn == nullptr && (v == kLegacy || spec.export_name[v] != nullptr)) ++missing;
    }
  }
  if (missing > 0) log_msg("%d hooked driver functions not found; calls to them pass through", missing);
  g_loader_ready.store(true, std::memory_order_release);
}

}  // namespace

DlsymFn real_dlsym() {
  pthread_once(&g_dlsym_once, find_real_dlsym);
  return g_real_dlsym;
}

void* libcuda_handle() {
  pthread_once(&g_libcuda_once, open_libcuda);
  return g_libcuda;
}

void* libcuda_if_loaded() {
  void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (lib == nullptr) return nullptr;
  dlclose(lib);  // RTLD_NOLOAD still took a reference
  return libcuda_handle();
}

bool inside_cuscope(const void* addr) {
  Dl_info self{}, other{};
  if (dladdr(reinterpret_cast<const void*>(&inside_cuscope), &self) == 0) return false;
  return dladdr(addr, &other) != 0 && other.dli_fbase == self.dli_fbase;
}

void set_real_gpa(GpaVer ver, void* fn) {
  g_real_gpa[static_cast<int>(ver)].store(fn, std::memory_order_release);
}

void* real_gpa(GpaVer ver) {
  return g_real_gpa[static_cast<int>(ver)].load(std::memory_order_acquire);
}

void* resolve_real_gpa(GpaVer ver) {
  if (void* fn = real_gpa(ver)) return fn;
  void* fn = dlsym_libcuda(ver == GpaVer::V1 ? "cuGetProcAddress" : "cuGetProcAddress_v2");
  if (fn != nullptr) set_real_gpa(ver, fn);
  return fn;
}

void ensure_loader() { pthread_once(&g_loader_once, load_real_functions); }

bool loader_ready() { return g_loader_ready.load(std::memory_order_acquire); }

void* resolve_by_export(int hook_id, int variant) {
  const char* name = hook_spec(static_cast<HookId>(hook_id)).export_name[variant];
  return name != nullptr ? dlsym_libcuda(name) : nullptr;
}

void* driver_proc(const char* export_name) {
  InternalScope internal;
  return dlsym_libcuda(export_name);
}

}  // namespace cuscope

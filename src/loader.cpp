#include "loader.hpp"

#include "log.hpp"

#include <dlfcn.h>
#include <pthread.h>

#include <atomic>

namespace cuscope {
namespace {

pthread_once_t g_dlsym_once = PTHREAD_ONCE_INIT;
DlsymFn g_real_dlsym = nullptr;

pthread_once_t g_libcuda_once = PTHREAD_ONCE_INIT;
void* g_libcuda = nullptr;

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

}  // namespace

DlsymFn real_dlsym() {
  pthread_once(&g_dlsym_once, find_real_dlsym);
  return g_real_dlsym;
}

void* libcuda_handle() {
  pthread_once(&g_libcuda_once, open_libcuda);
  return g_libcuda;
}

void set_real_gpa(GpaVer ver, void* fn) {
  g_real_gpa[static_cast<int>(ver)].store(fn, std::memory_order_release);
}

void* real_gpa(GpaVer ver) {
  return g_real_gpa[static_cast<int>(ver)].load(std::memory_order_acquire);
}

}  // namespace cuscope

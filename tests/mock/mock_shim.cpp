// Mock of the WSL libcuda.so.1 loader. Its constructor dlopen()s the mock
// driver and fills one slot per export with dlsym(); the generated
// trampolines (shim_stubs.s) jump through those slots.
//
// The dlsym() calls here are what make WSL tricky for cuscope: they reach
// cuscope's dlsym override before cuscope's own constructor may have run.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

extern "C" {
__attribute__((visibility("hidden"))) extern void* mock_shim_slots[];
__attribute__((visibility("hidden"))) extern const char* const mock_shim_names[];
__attribute__((visibility("hidden"))) extern const unsigned mock_shim_count;
}

namespace {

__attribute__((constructor)) void shim_init() {
  void* driver = dlopen(MOCK_DRIVER_PATH, RTLD_NOW | RTLD_LOCAL);
  if (driver == nullptr) {
    std::fprintf(stderr, "mock shim: dlopen failed: %s\n", dlerror());
    std::abort();
  }
  for (unsigned i = 0; i < mock_shim_count; ++i) {
    mock_shim_slots[i] = dlsym(driver, mock_shim_names[i]);
    if (mock_shim_slots[i] == nullptr) {
      std::fprintf(stderr, "mock shim: missing %s\n", mock_shim_names[i]);
      std::abort();
    }
  }
}

}  // namespace

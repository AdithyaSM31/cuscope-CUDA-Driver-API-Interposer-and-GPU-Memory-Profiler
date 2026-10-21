// Stands in for the shared CUDA runtime, libcudart.so.13: it loads libcuda and
// fetches cuGetProcAddress_v2 with dlsym from inside a library whose name
// starts with "libcuda", which a careless caller check mistakes for the driver.
#include <dlfcn.h>

extern "C" __attribute__((visibility("default"))) void* fake_cudart_get_gpa() {
  void* driver = dlopen("libcuda.so.1", RTLD_NOW);
  return driver ? dlsym(driver, "cuGetProcAddress_v2") : nullptr;
}

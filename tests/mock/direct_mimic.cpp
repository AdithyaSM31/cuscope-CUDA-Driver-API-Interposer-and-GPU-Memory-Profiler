// A driver-API program linked with -lcuda: calls reach libcuda by symbol
// name, which is cuscope's interception path 1 (exported symbols).
#include <cuda.h>

#include <cstdio>
#include <cstdlib>

#include "mimic_common.hpp"

#define CHECK(call)                                                       \
  do {                                                                    \
    CUresult r_ = (call);                                                 \
    if (r_ != CUDA_SUCCESS) {                                             \
      std::fprintf(stderr, "FAIL %s -> %d\n", #call, static_cast<int>(r_)); \
      return 1;                                                           \
    }                                                                     \
  } while (0)

int main() {
  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
  CHECK(cuCtxSetCurrent(ctx));
  CUdeviceptr p = 0;
  CHECK(cuMemAlloc(&p, 4096));
  CHECK(cuMemFree(p));

  std::printf("OK direct\n");
  print_stats();
  print_mock(&mock_get_counters);
  return 0;
}

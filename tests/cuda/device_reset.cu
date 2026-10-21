// Destroys the primary context while the driver is alive (cudaDeviceReset ->
// cuDevicePrimaryCtxReset), with one allocation still live: cuscope must
// report it as a leak at context teardown. A second allocation, made in the
// new context and never freed, is an ordinary leak at exit.
#include "check.cuh"

int main() {
  void* before = nullptr;
  CHECK(cudaMalloc(&before, 1 << 20));  // LEAK_AT_RESET
  CHECK(cudaDeviceReset());

  void* after = nullptr;
  CHECK(cudaMalloc(&after, 2 << 20));  // LEAK_AT_EXIT
  std::printf("device_reset: done\n");
  return 0;
}

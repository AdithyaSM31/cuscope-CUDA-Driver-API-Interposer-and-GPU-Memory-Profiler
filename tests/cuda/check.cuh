// Shared helpers for the GPU test programs. Each program prints a fixed line
// to stdout so runs with and without cuscope can be diffed.
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorName(err_));                                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Busy-waits on the GPU for `cycles` SM clock cycles.
__global__ void spin_kernel(long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

// SM clock cycles in `ms` milliseconds on device 0.
inline long long cycles_for_ms(int ms) {
  int clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  return static_cast<long long>(clock_khz) * ms;
}

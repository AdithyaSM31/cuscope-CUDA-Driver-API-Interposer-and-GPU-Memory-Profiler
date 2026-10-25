// A GPU load for the compute-sharing benchmark: launches batches of kernels on
// the default stream and waits for each batch, for a fixed time.
//
//   share_load <seconds> <kernel_ms> <batch> [iters]
//
// Each kernel fills the GPU (6 blocks of 256 threads per SM, one wave) with a
// dependent multiply-add loop, calibrated at start to take kernel_ms alone. It
// does real work, so two jobs on one GPU slow each other down, as training
// steps do. (A clock64() spin would finish after a fixed wall time even while
// the driver had switched it out, and look full speed under contention.)
//
// Given iters (as the benchmark does, from one calibration run alone), it
// skips calibrating, so every job runs the same kernel.
//
// Prints one JSON line: kernels launched, the median and 99th percentile time
// per batch (launch to completion) in milliseconds, and the iterations used.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

#include "check.cuh"

__global__ void work_kernel(float* out, int iters) {
  float a = static_cast<float>(threadIdx.x), b = 1.000001f;
  for (int i = 0; i < iters; ++i) a = a * b + 0.5f;
  out[blockIdx.x * blockDim.x + threadIdx.x] = a;  // keeps the loop
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: %s <seconds> <kernel_ms> <batch> [iters]\n", argv[0]);
    return 2;
  }
  const double seconds = std::atof(argv[1]);
  const double kernel_ms = std::atof(argv[2]);
  const int batch = std::atoi(argv[3]);

  int sms = 0;
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int blocks = sms * 6, threads = 256;
  float* out = nullptr;
  CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));

  // Calibrate: how many iterations take kernel_ms on this GPU, alone.
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  const int probe = 20000;
  work_kernel<<<blocks, threads>>>(out, probe);  // warm-up
  CHECK(cudaEventRecord(start));
  work_kernel<<<blocks, threads>>>(out, probe);
  CHECK(cudaEventRecord(end));
  CHECK(cudaEventSynchronize(end));
  float probe_ms = 0.f;
  CHECK(cudaEventElapsedTime(&probe_ms, start, end));
  const int iters = argc == 5 ? std::atoi(argv[4])
                              : std::max(1, static_cast<int>(probe * kernel_ms / std::max(probe_ms, 0.001f)));

  using clock = std::chrono::steady_clock;
  const auto stop = clock::now() + std::chrono::duration<double>(seconds);
  std::vector<double> batch_ms;
  long kernels = 0;
  while (clock::now() < stop) {
    const auto t0 = clock::now();
    for (int i = 0; i < batch; ++i) work_kernel<<<blocks, threads>>>(out, iters);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    batch_ms.push_back(std::chrono::duration<double, std::milli>(clock::now() - t0).count());
    kernels += batch;
  }
  std::sort(batch_ms.begin(), batch_ms.end());
  auto pct = [&](double p) { return batch_ms.empty() ? 0.0 : batch_ms[static_cast<size_t>(p * (batch_ms.size() - 1))]; };
  std::printf("{\"kernels\": %ld, \"p50_ms\": %.3f, \"p99_ms\": %.3f, \"iters\": %d}\n", kernels, pct(0.50),
              pct(0.99), iters);
  CHECK(cudaFree(out));
  return 0;
}

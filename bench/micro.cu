// Microbenchmarks for cuscope's per-call overhead.
//
// usage: micro <launch|alloc|copy4k> [--iters N]
// Prints ns_per_call=<float>: wall time per call, after a warmup that creates
// the context (and cuscope's per-context state) outside the measurement.
//   launch  an empty kernel, <<<1, 1>>>, N times, then one synchronise
//   alloc   cudaMalloc(4096) + cudaFree, N times
//   copy4k  a synchronous 4 KiB host-to-device copy from pageable memory, N times
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../tests/cuda/check.cuh"

__global__ void empty_kernel() {}

namespace {

constexpr int kWarmup = 200;

template <class Body>
double ns_per_call(int iters, Body&& body) {
  for (int i = 0; i < kWarmup; ++i) body();
  CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) body();
  CHECK(cudaDeviceSynchronize());
  auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double, std::nano>(elapsed).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <launch|alloc|copy4k> [--iters N]\n", argv[0]);
    return 2;
  }
  const char* which = argv[1];
  int iters = 20000;
  if (argc == 4 && std::strcmp(argv[2], "--iters") == 0) iters = std::atoi(argv[3]);

  double ns = 0;
  if (std::strcmp(which, "launch") == 0) {
    ns = ns_per_call(iters, [] { empty_kernel<<<1, 1>>>(); });
  } else if (std::strcmp(which, "alloc") == 0) {
    ns = ns_per_call(iters, [] {
      void* p = nullptr;
      CHECK(cudaMalloc(&p, 4096));
      CHECK(cudaFree(p));
    });
  } else if (std::strcmp(which, "copy4k") == 0) {
    std::vector<char> host(4096, 1);
    void* device = nullptr;
    CHECK(cudaMalloc(&device, host.size()));
    ns = ns_per_call(iters, [&] { CHECK(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice)); });
    CHECK(cudaFree(device));
  } else {
    std::fprintf(stderr, "unknown benchmark %s\n", which);
    return 2;
  }
  std::printf("ns_per_call=%.1f\n", ns);
  return 0;
}

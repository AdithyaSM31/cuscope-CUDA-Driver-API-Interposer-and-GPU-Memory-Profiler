// Eight threads allocate and free concurrently, then each leaks one block.
// Expected: 8 * 1001 allocations, 8000 frees, 8 leaks of 1..8 KiB, no invalid frees.
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(err_));  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main() {
  constexpr int kThreads = 8;
  constexpr int kIterations = 1000;
  CHECK(cudaFree(nullptr));  // create the context before the threads start

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([t] {
      for (int i = 0; i < kIterations; ++i) {
        void* p = nullptr;
        CHECK(cudaMalloc(&p, static_cast<size_t>(t + 1) * 4096));
        CHECK(cudaFree(p));
      }
      void* leaked = nullptr;
      CHECK(cudaMalloc(&leaked, static_cast<size_t>(t + 1) * 1024));  // LEAK_PER_THREAD
    });
  }
  for (auto& th : threads) th.join();

  std::printf("multithread: done\n");
  return 0;
}

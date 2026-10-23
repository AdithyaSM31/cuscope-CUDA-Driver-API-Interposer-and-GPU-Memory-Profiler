// Allocates 64 MiB at a time until the memory cap says no, then checks what
// cudaMemGetInfo reports and that freeing makes room again. Run with
// CUSCOPE_MEM_LIMIT=256M; without a cap it stops after 16 allocations.
#include "check.cuh"

int main() {
  constexpr size_t kChunk = size_t{64} << 20;
  constexpr int kMax = 16;
  CHECK(cudaFree(nullptr));  // start the runtime and its context

  void* chunks[kMax] = {};
  int got = 0;
  for (; got < kMax; ++got) {
    cudaError_t err = cudaMalloc(&chunks[got], kChunk);
    if (err != cudaSuccess) {
      std::printf("%s\n", cudaGetErrorName(err));
      break;
    }
    std::printf("ok\n");
  }

  size_t free_bytes = 0, total_bytes = 0;
  CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("MEMINFO %zu %zu\n", free_bytes, total_bytes);

  if (got > 0) {
    CHECK(cudaFree(chunks[0]));
    void* again = nullptr;
    cudaError_t err = cudaMalloc(&again, kChunk);
    if (err == cudaSuccess) CHECK(cudaFree(again));
    for (int i = 1; i < got; ++i) CHECK(cudaFree(chunks[i]));
    std::printf("after free: %s\n", cudaGetErrorName(err));
  }
  return 0;
}

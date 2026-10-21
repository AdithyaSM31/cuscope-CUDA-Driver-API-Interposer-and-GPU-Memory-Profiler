// Frees one pointer twice and frees nullptr: cuscope must report exactly one
// invalid free (the second), and the program's own output must not change.
#include "check.cuh"

int main() {
  void* p = nullptr;
  CHECK(cudaMalloc(&p, 4096));
  CHECK(cudaFree(p));

  cudaError_t second = cudaFree(p);  // SECOND_FREE
  cudaGetLastError();                // clear it so later calls are unaffected
  std::printf("double_free: second free -> %s\n", cudaGetErrorName(second));

  cudaError_t null_free = cudaFree(nullptr);  // not an invalid free
  std::printf("double_free: null free -> %s\n", cudaGetErrorName(null_free));
  return 0;
}

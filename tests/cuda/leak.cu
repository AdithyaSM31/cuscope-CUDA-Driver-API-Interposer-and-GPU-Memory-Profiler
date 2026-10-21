// Three device allocations, one freed: cuscope must report exactly two leaks
// (1 MiB and 3 MiB) with call sites on the marked lines.
#include "check.cuh"

int main() {
  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  CHECK(cudaMalloc(&a, 1 << 20));  // LEAK_A
  CHECK(cudaMalloc(&b, 2 << 20));  // FREED
  CHECK(cudaMalloc(&c, 3 << 20));  // LEAK_C
  CHECK(cudaFree(b));
  std::printf("leak: done\n");
  return 0;
}

// Stream-ordered allocations on two streams. In call order the live bytes are
// 8, 12, 4, 20, 16, 0 MiB, with a peak of 20 MiB.
#include "check.cuh"

int main() {
  constexpr size_t MiB = 1 << 20;
  cudaStream_t s1, s2;
  CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

  void *a = nullptr, *b = nullptr, *c = nullptr;
  CHECK(cudaMallocAsync(&a, 8 * MiB, s1));   //  8
  CHECK(cudaMallocAsync(&b, 4 * MiB, s2));   // 12
  CHECK(cudaFreeAsync(a, s1));               //  4
  CHECK(cudaMallocAsync(&c, 16 * MiB, s2));  // 20
  CHECK(cudaFreeAsync(b, s2));               // 16
  CHECK(cudaFreeAsync(c, s2));               //  0

  CHECK(cudaStreamSynchronize(s1));
  CHECK(cudaStreamSynchronize(s2));
  CHECK(cudaStreamDestroy(s1));
  CHECK(cudaStreamDestroy(s2));
  std::printf("async_pool: done\n");
  return 0;
}

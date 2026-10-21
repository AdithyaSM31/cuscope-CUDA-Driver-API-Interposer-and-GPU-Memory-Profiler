// Same-size copies from pageable and pinned host memory, five each way.
// cuscope must report both, with pinned faster. Timings are not printed, so
// stdout stays identical across runs.
#include <vector>

#include "check.cuh"

int main() {
  constexpr size_t kBytes = 64u << 20;
  constexpr int kRepeats = 5;

  std::vector<char> pageable(kBytes, 1);
  char* pinned = nullptr;
  CHECK(cudaMallocHost(&pinned, kBytes));
  void* device = nullptr;
  CHECK(cudaMalloc(&device, kBytes));

  for (int i = 0; i < kRepeats; ++i) CHECK(cudaMemcpy(device, pageable.data(), kBytes, cudaMemcpyHostToDevice));
  for (int i = 0; i < kRepeats; ++i) CHECK(cudaMemcpy(pageable.data(), device, kBytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kRepeats; ++i) CHECK(cudaMemcpy(device, pinned, kBytes, cudaMemcpyHostToDevice));
  for (int i = 0; i < kRepeats; ++i) CHECK(cudaMemcpy(pinned, device, kBytes, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(device));
  CHECK(cudaFreeHost(pinned));
  std::printf("bandwidth: done\n");
  return 0;
}

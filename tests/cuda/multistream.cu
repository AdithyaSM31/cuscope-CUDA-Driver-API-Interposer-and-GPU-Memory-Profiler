// One ~5 ms spin_kernel on each of four streams: four GPU tracks with
// overlapping spans.
#include "check.cuh"

int main() {
  constexpr int kStreams = 4;
  const long long cycles = cycles_for_ms(5);

  cudaStream_t streams[kStreams];
  for (auto& s : streams) CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (auto& s : streams) spin_kernel<<<1, 1, 0, s>>>(cycles);
  CHECK(cudaGetLastError());
  for (auto& s : streams) CHECK(cudaStreamSynchronize(s));
  for (auto& s : streams) CHECK(cudaStreamDestroy(s));

  std::printf("multistream: done\n");
  return 0;
}

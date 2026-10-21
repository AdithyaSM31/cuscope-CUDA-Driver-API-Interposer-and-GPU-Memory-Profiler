// Captures a CUDA graph in global mode (PyTorch's default) while timed work
// is still pending on another stream. In global mode, event queries from any
// other thread are prohibited during the capture and invalidate it, so
// cuscope's collector must not make them in that mode.
#include <chrono>
#include <thread>

#include "check.cuh"

int main() {
  cudaStream_t busy, captured;
  CHECK(cudaStreamCreateWithFlags(&busy, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&captured, cudaStreamNonBlocking));

  const long long cycles = cycles_for_ms(50);
  for (int i = 0; i < 4; ++i) spin_kernel<<<1, 1, 0, busy>>>(cycles);  // ~200 ms still pending
  CHECK(cudaGetLastError());

  cudaGraph_t graph;
  CHECK(cudaStreamBeginCapture(captured, cudaStreamCaptureModeGlobal));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // host work: cuscope's collector runs meanwhile
  spin_kernel<<<1, 1, 0, captured>>>(1000);
  cudaError_t launch = cudaGetLastError();
  cudaError_t end = cudaStreamEndCapture(captured, &graph);
  std::printf("capture_race: launch %s, end capture %s\n", cudaGetErrorName(launch), cudaGetErrorName(end));
  if (end == cudaSuccess) CHECK(cudaGraphDestroy(graph));

  CHECK(cudaStreamSynchronize(busy));
  CHECK(cudaStreamDestroy(busy));
  CHECK(cudaStreamDestroy(captured));
  return launch == cudaSuccess && end == cudaSuccess ? 0 : 1;
}

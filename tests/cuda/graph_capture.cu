// Captures three launches into a CUDA graph and launches the graph twice.
// cuscope must not add event nodes to the capture, and records each graph
// launch as one event.
#include "check.cuh"

int main() {
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  cudaGraph_t graph;
  CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 3; ++i) spin_kernel<<<1, 1, 0, s>>>(1000);  // CAPTURED_LAUNCH
  CHECK(cudaStreamEndCapture(s, &graph));

  size_t nodes = 0;
  CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));

  cudaGraphExec_t exec;
  CHECK(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < 2; ++i) CHECK(cudaGraphLaunch(exec, s));
  CHECK(cudaStreamSynchronize(s));

  CHECK(cudaGraphExecDestroy(exec));
  CHECK(cudaGraphDestroy(graph));
  CHECK(cudaStreamDestroy(s));
  std::printf("graph_capture: %zu nodes, done\n", nodes);  // 3 unless something added nodes
  return 0;
}

// Returns while a 1.5 s kernel is still running, without synchronising (a job
// exiting after a timeout, say). The process must exit as promptly as it
// does without cuscope: cuscope's exit flush waits only a bounded time.
// 1.5 s stays under Windows' 2 s GPU timeout detection.
#include "check.cuh"

int main() {
  spin_kernel<<<1, 1>>>(cycles_for_ms(1500));
  CHECK(cudaGetLastError());
  std::printf("exit_busy: returning with a kernel still running\n");
  return 3;
}

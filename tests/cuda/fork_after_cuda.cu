// Uses CUDA, then forks a child that does no CUDA work and exits normally (a
// subprocess launched from a training script, say). The child still runs the
// exit handlers it inherited, the CUDA runtime's cleanup among them. Only
// the parent did CUDA work, so only the parent writes a trace.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>

#include "check.cuh"

int main() {
  float* buf = nullptr;
  CHECK(cudaMalloc(&buf, 1 << 20));
  spin_kernel<<<1, 1>>>(cycles_for_ms(5));
  CHECK(cudaDeviceSynchronize());
  std::fflush(stdout);

  pid_t child = fork();
  if (child < 0) return 1;
  if (child == 0) std::exit(0);
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 1;

  CHECK(cudaFree(buf));
  std::printf("fork_after_cuda: child exited cleanly\n");
  return 0;
}

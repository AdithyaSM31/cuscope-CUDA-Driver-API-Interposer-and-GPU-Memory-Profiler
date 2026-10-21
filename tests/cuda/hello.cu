// Smallest program that allocates, copies, launches and frees.
// Built twice: hello_static (nvcc default runtime) and hello_shared (-cudart shared).
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

#define CHECK(call)                                                       \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(err_)); \
      return 1;                                                           \
    }                                                                     \
  } while (0)

int main() {
  constexpr int n = 1 << 20;
  constexpr size_t bytes = n * sizeof(float);
  std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

  vec_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dc));

  double sum = 0.0;
  for (float v : c) sum += v;
  std::printf("hello: sum=%.0f\n", sum);
  return 0;
}

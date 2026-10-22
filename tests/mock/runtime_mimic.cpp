// Behaves like the CUDA runtime: never linked against libcuda, it dlopen()s
// libcuda.so.1, fetches cuGetProcAddress_v2 with dlsym(), and looks up every
// other function by base name, version and flags.
//
// usage: runtime_mimic <scenario>   (see conftest.SCENARIOS)
// Exit code 0 means every driver call returned CUDA_SUCCESS.
#include <cuda.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mimic_common.hpp"

namespace {

using GpaFn = decltype(&cuGetProcAddress);  // the v2 signature
using MemAllocV1Fn = CUresult (*)(unsigned int*, unsigned int);

GpaFn g_gpa = nullptr;
void* g_driver = nullptr;

#define CHECK(call)                                                       \
  do {                                                                    \
    CUresult r_ = (call);                                                 \
    if (r_ != CUDA_SUCCESS) {                                             \
      std::fprintf(stderr, "FAIL %s -> %d\n", #call, static_cast<int>(r_)); \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

[[noreturn]] void die(const char* what, const char* detail) {
  std::fprintf(stderr, "FAIL %s: %s\n", what, detail ? detail : "");
  std::exit(1);
}

template <class Fn>
Fn get(const char* symbol, int version = CUDA_VERSION,
       cuuint64_t flags = CU_GET_PROC_ADDRESS_LEGACY_STREAM) {
  void* p = nullptr;
  CUdriverProcAddressQueryResult status;
  CHECK(g_gpa(symbol, &p, version, flags, &status));
  if (p == nullptr) die("lookup", symbol);
  return reinterpret_cast<Fn>(p);
}

// via_default: first ask dlsym(RTLD_DEFAULT, ...) before libcuda is loaded,
// the order that exercises cuscope loading libcuda itself.
// via_cudart: let the fake libcudart.so.13 make the dlsym call, as -cudart shared does.
void open_driver(bool via_default, bool via_cudart) {
  if (via_default) g_gpa = reinterpret_cast<GpaFn>(dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2"));
  if (via_cudart) {
    void* cudart = dlopen(FAKE_CUDART_PATH, RTLD_NOW);
    if (cudart == nullptr) die("dlopen fake libcudart", dlerror());
    auto get_gpa = reinterpret_cast<void* (*)()>(dlsym(cudart, "fake_cudart_get_gpa"));
    if (get_gpa == nullptr) die("dlsym fake_cudart_get_gpa", dlerror());
    g_gpa = reinterpret_cast<GpaFn>(get_gpa());
  }
  g_driver = dlopen("libcuda.so.1", RTLD_NOW);
  if (g_driver == nullptr) die("dlopen libcuda.so.1", dlerror());
  if (g_gpa == nullptr) g_gpa = reinterpret_cast<GpaFn>(dlsym(g_driver, "cuGetProcAddress_v2"));
  if (g_gpa == nullptr) die("dlsym cuGetProcAddress_v2", dlerror());
}

CUcontext init_context() {
  CHECK(get<decltype(&cuInit)>("cuInit")(0));
  CUcontext ctx = nullptr;
  CHECK(get<decltype(&cuDevicePrimaryCtxRetain)>("cuDevicePrimaryCtxRetain")(&ctx, 0));
  CHECK(get<decltype(&cuCtxSetCurrent)>("cuCtxSetCurrent")(ctx));
  return ctx;
}

CUdeviceptr alloc(size_t bytes) {
  CUdeviceptr p = 0;
  CHECK(get<decltype(&cuMemAlloc)>("cuMemAlloc")(&p, bytes));
  return p;
}

void release(CUdeviceptr p) { CHECK(get<decltype(&cuMemFree)>("cuMemFree")(p)); }

void launch(CUstream stream, cuuint64_t flags = CU_GET_PROC_ADDRESS_LEGACY_STREAM) {
  auto fn = get<decltype(&cuLaunchKernel)>("cuLaunchKernel", CUDA_VERSION, flags);
  CHECK(fn(reinterpret_cast<CUfunction>(0x1234), 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
}

void run(const char* scenario) {
  const bool via_default = std::strcmp(scenario, "dlsym_default") == 0;
  const bool via_cudart = std::strcmp(scenario, "via_cudart") == 0;
  open_driver(via_default, via_cudart);
  CUcontext ctx = init_context();

  if (std::strcmp(scenario, "dlsym_gpa") == 0 || via_default || via_cudart) {
    release(alloc(4096));
  } else if (std::strcmp(scenario, "ptsz") == 0) {
    CUdeviceptr p = alloc(64);
    auto copy = get<decltype(&cuMemcpyAsync)>("cuMemcpyAsync", CUDA_VERSION,
                                              CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
    CHECK(copy(p, p, 16, nullptr));
    launch(nullptr, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
    release(p);
  } else if (std::strcmp(scenario, "old_version") == 0) {
    auto alloc_v1 = get<MemAllocV1Fn>("cuMemAlloc", 3000, CU_GET_PROC_ADDRESS_DEFAULT);
    unsigned int p = 0;
    CHECK(alloc_v1(&p, 4096));
  } else if (std::strcmp(scenario, "gpa_self") == 0) {
    g_gpa = get<GpaFn>("cuGetProcAddress");
    release(alloc(4096));
  } else if (std::strcmp(scenario, "capture") == 0) {
    launch(reinterpret_cast<CUstream>(MOCK_CAPTURING_STREAM_HANDLE));
  } else if (std::strcmp(scenario, "launch") == 0) {
    launch(nullptr);
    CHECK(get<decltype(&cuStreamSynchronize)>("cuStreamSynchronize")(nullptr));
  } else if (std::strcmp(scenario, "launch200") == 0) {
    // A deep queue: 200 launches on one stream, then 50 ms of host work while
    // cuscope's collector polls.
    for (int i = 0; i < 200; ++i) launch(nullptr);
    usleep(50 * 1000);
  } else if (std::strcmp(scenario, "launch8") == 0) {
    for (int i = 0; i < 8; ++i) launch(nullptr);
    CHECK(get<decltype(&cuStreamSynchronize)>("cuStreamSynchronize")(nullptr));
  } else if (std::strcmp(scenario, "copy") == 0) {
    CUdeviceptr p = alloc(4096);
    std::vector<char> host(4096, 1);
    CHECK(get<decltype(&cuMemcpyHtoD)>("cuMemcpyHtoD")(p, host.data(), host.size()));
    release(p);
  } else if (std::strcmp(scenario, "teardown") == 0) {
    alloc(1 << 20);  // deliberately not freed
    launch(nullptr);
    CHECK(get<decltype(&cuCtxDestroy)>("cuCtxDestroy")(ctx));
  } else if (std::strcmp(scenario, "release_destroys") == 0) {
    // The only reference: releasing it destroys the primary context.
    alloc(1 << 20);  // deliberately not freed
    CHECK(get<decltype(&cuDevicePrimaryCtxRelease)>("cuDevicePrimaryCtxRelease")(0));
  } else if (std::strcmp(scenario, "release_keeps") == 0) {
    // A second reference keeps the context alive across one release.
    CUcontext again = nullptr;
    CHECK(get<decltype(&cuDevicePrimaryCtxRetain)>("cuDevicePrimaryCtxRetain")(&again, 0));
    alloc(1 << 20);  // deliberately not freed
    CHECK(get<decltype(&cuDevicePrimaryCtxRelease)>("cuDevicePrimaryCtxRelease")(0));
  } else if (std::strcmp(scenario, "release_keeps_launch") == 0) {
    // Launches on both sides of a Release that leaves the context alive.
    launch(nullptr);
    CUcontext again = nullptr;
    CHECK(get<decltype(&cuDevicePrimaryCtxRetain)>("cuDevicePrimaryCtxRetain")(&again, 0));
    CHECK(get<decltype(&cuDevicePrimaryCtxRelease)>("cuDevicePrimaryCtxRelease")(0));
    launch(nullptr);
  } else if (std::strcmp(scenario, "fork") == 0) {
    alloc(1 << 20);  // deliberately not freed; the child must not report it
    launch(nullptr);  // timed; its span must not show up in the child's trace
    usleep(20 * 1000);
    std::fflush(stdout);
    pid_t child = fork();
    if (child < 0) die("fork", std::strerror(errno));
    if (child == 0) {
      // Real CUDA forbids this in a forked child; the mock allows it, which
      // shows the child's report is its own: one allocation, freed.
      release(alloc(4096));
      std::exit(0);  // exit handlers run
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      die("child", "did not exit cleanly");
    }
  } else if (std::strcmp(scenario, "fork_quiet") == 0) {
    // A child that does no CUDA work of its own but runs the CUDA runtime's
    // inherited exit cleanup, which releases the primary context.
    alloc(1 << 20);
    launch(nullptr);
    usleep(20 * 1000);
    std::fflush(stdout);
    pid_t child = fork();
    if (child < 0) die("fork", std::strerror(errno));
    if (child == 0) {
      get<decltype(&cuDevicePrimaryCtxRelease)>("cuDevicePrimaryCtxRelease")(0);
      std::exit(0);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      die("child", "did not exit cleanly");
    }
  } else {
    die("unknown scenario", scenario);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <scenario>\n", argv[0]);
    return 2;
  }
  run(argv[1]);
  std::printf("OK %s\n", argv[1]);
  print_stats();
  auto counters = reinterpret_cast<void (*)(mock_counters*)>(dlsym(g_driver, "mock_get_counters"));
  if (counters == nullptr) die("dlsym mock_get_counters", dlerror());
  print_mock(counters);
  return 0;
}

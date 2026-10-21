// Launch hooks. Each launch is recorded with its function, stream and a
// correlation id, and timed on the GPU with an event before and after it on
// its stream.
#include "wrappers.hpp"

#include "gputimer.hpp"
#include "names.hpp"

namespace cuscope {
namespace {

// After the real call. A launch the driver rejected never ran, so it stays a
// plain API call and its events go back unused; a successful one gets a
// correlation id shared by its record and its GPU span.
template <int V>
void finish_launch(ApiRecord& rec, CUresult result, RecType type, uint64_t func, CUstream stream, bool timed,
                   const Timing& timing) {
  rec.done();
  rec.rec.func = func;
  rec.rec.stream = handle(stream);
  if (result != CUDA_SUCCESS) {
    if (timed) gpu_cancel(timing);
    return;
  }
  rec.rec.type = type;
  rec.rec.corr = next_corr();
  if (type == RecType::Launch && !in_internal()) note_kernel(func);
  if (timed) {
    GpuSpan meta{};
    meta.corr = rec.rec.corr;
    meta.stream = rec.rec.stream;
    meta.func = func;
    meta.api = rec.rec.api;
    gpu_end(V, stream, timing, meta);
  }
}

}  // namespace

template <int V>
CUresult cs_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                           unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                           unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
  note_call();
  Timing timing;
  const bool timed = gpu_begin(V, hStream, &timing);
  ApiRecord rec(H_cuLaunchKernel);
  CUresult r = forward<V, decltype(&::cuLaunchKernel)>(H_cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                                       blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams,
                                                       extra);
  finish_launch<V>(rec, r, RecType::Launch, handle(f), hStream, timed, timing);
  return r;
}

template <int V>
CUresult cs_cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
  note_call();
  CUstream stream = config != nullptr ? config->hStream : nullptr;
  Timing timing;
  const bool timed = gpu_begin(V, stream, &timing);
  ApiRecord rec(H_cuLaunchKernelEx);
  CUresult r = forward<V, decltype(&::cuLaunchKernelEx)>(H_cuLaunchKernelEx, config, f, kernelParams, extra);
  finish_launch<V>(rec, r, RecType::Launch, handle(f), stream, timed, timing);
  return r;
}

// One span for the whole graph; per-node timing is a non-goal.
template <int V>
CUresult cs_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  note_call();
  Timing timing;
  const bool timed = gpu_begin(V, hStream, &timing);
  ApiRecord rec(H_cuGraphLaunch);
  CUresult r = forward<V, decltype(&::cuGraphLaunch)>(H_cuGraphLaunch, hGraphExec, hStream);
  finish_launch<V>(rec, r, RecType::GraphLaunch, handle(hGraphExec), hStream, timed, timing);
  return r;
}

void register_launch_hooks() {
  CS_REGISTER(cuLaunchKernel);
  CS_REGISTER(cuLaunchKernelEx);
  CS_REGISTER(cuGraphLaunch);
}

}  // namespace cuscope

#define CS_LAUNCH_PARAMS                                                                                        \
  (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, \
   unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,             \
   void** kernelParams, void** extra)
#define CS_LAUNCH_ARGS \
  (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
CS_EXPORT(cuLaunchKernel, cuLaunchKernel, 0, CS_LAUNCH_PARAMS, CS_LAUNCH_ARGS)
CS_EXPORT(cuLaunchKernel_ptsz, cuLaunchKernel, 1, CS_LAUNCH_PARAMS, CS_LAUNCH_ARGS)
#undef CS_LAUNCH_PARAMS
#undef CS_LAUNCH_ARGS

CS_EXPORT(cuLaunchKernelEx, cuLaunchKernelEx, 0,
          (const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra),
          (config, f, kernelParams, extra))
CS_EXPORT(cuLaunchKernelEx_ptsz, cuLaunchKernelEx, 1,
          (const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra),
          (config, f, kernelParams, extra))
CS_EXPORT(cuGraphLaunch, cuGraphLaunch, 0, (CUgraphExec hGraphExec, CUstream hStream), (hGraphExec, hStream))
CS_EXPORT(cuGraphLaunch_ptsz, cuGraphLaunch, 1, (CUgraphExec hGraphExec, CUstream hStream), (hGraphExec, hStream))

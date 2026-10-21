// Launch hooks. Pass-through for now; GPU timing (Task 10) records each
// launch's span on its stream.
#include "wrappers.hpp"

namespace cuscope {

template <int V>
CUresult cs_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                           unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                           unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
  note_call();
  return forward<V, decltype(&::cuLaunchKernel)>(H_cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                                 blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams,
                                                 extra);
}

template <int V>
CUresult cs_cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
  note_call();
  return forward<V, decltype(&::cuLaunchKernelEx)>(H_cuLaunchKernelEx, config, f, kernelParams, extra);
}

template <int V>
CUresult cs_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuGraphLaunch)>(H_cuGraphLaunch, hGraphExec, hStream);
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

// Synchronisation and context-teardown hooks, recorded as API calls. Task 11
// flushes pending GPU events here before a context goes away.
#include "wrappers.hpp"

namespace cuscope {

template <int V>
CUresult cs_cuStreamSynchronize(CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuStreamSynchronize);
  rec.rec.stream = handle(hStream);
  return forward<V, decltype(&::cuStreamSynchronize)>(H_cuStreamSynchronize, hStream);
}

// The (void) version. CUDA 13's cuCtxSynchronize_v2(CUcontext) is not wrapped;
// hooks.def resolves this hook at version 2000 so the two never meet.
template <int V>
CUresult cs_cuCtxSynchronize() {
  note_call();
  ApiRecord rec(H_cuCtxSynchronize);
  return forward<V, decltype(&::cuCtxSynchronize)>(H_cuCtxSynchronize);
}

template <int V>
CUresult cs_cuCtxDestroy(CUcontext ctx) {
  note_call();
  ApiRecord rec(H_cuCtxDestroy);
  rec.rec.ptr = handle(ctx);
  return forward<V, decltype(&::cuCtxDestroy)>(H_cuCtxDestroy, ctx);
}

template <int V>
CUresult cs_cuDevicePrimaryCtxRelease(CUdevice dev) {
  note_call();
  ApiRecord rec(H_cuDevicePrimaryCtxRelease);
  return forward<V, decltype(&::cuDevicePrimaryCtxRelease)>(H_cuDevicePrimaryCtxRelease, dev);
}

template <int V>
CUresult cs_cuDevicePrimaryCtxReset(CUdevice dev) {
  note_call();
  ApiRecord rec(H_cuDevicePrimaryCtxReset);
  return forward<V, decltype(&::cuDevicePrimaryCtxReset)>(H_cuDevicePrimaryCtxReset, dev);
}

void register_teardown_hooks() {
  CS_REGISTER(cuStreamSynchronize);
  CS_REGISTER(cuCtxSynchronize);
  CS_REGISTER(cuCtxDestroy);
  CS_REGISTER(cuDevicePrimaryCtxRelease);
  CS_REGISTER(cuDevicePrimaryCtxReset);
}

}  // namespace cuscope

CS_EXPORT(cuStreamSynchronize, cuStreamSynchronize, 0, (CUstream hStream), (hStream))
CS_EXPORT(cuStreamSynchronize_ptsz, cuStreamSynchronize, 1, (CUstream hStream), (hStream))
CS_EXPORT(cuCtxSynchronize, cuCtxSynchronize, 0, (void), ())
CS_EXPORT(cuCtxDestroy_v2, cuCtxDestroy, 0, (CUcontext ctx), (ctx))
// The v1 symbols, matching hooks.def. Programs that call the _v2 symbols by
// name reach the driver directly (a documented limitation).
CS_EXPORT_AS("cuDevicePrimaryCtxRelease", cuscope_primary_ctx_release_v1, cuDevicePrimaryCtxRelease, 0,
             (CUdevice dev), (dev))
CS_EXPORT_AS("cuDevicePrimaryCtxReset", cuscope_primary_ctx_reset_v1, cuDevicePrimaryCtxReset, 0,
             (CUdevice dev), (dev))

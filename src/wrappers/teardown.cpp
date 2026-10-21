// Synchronisation and context-teardown hooks, recorded as API calls.
//
// Before a context can go away, its pending GPU events are resolved: the CUDA
// runtime destroys its context in its own exit cleanup, which runs before
// cuscope's exit handler, and an event cannot be read once its context is gone.
//
// After a context really is destroyed, the driver has freed all its memory, so
// whatever the program still held there becomes a leak marked at_ctx_teardown.
#include "wrappers.hpp"

#include "gputimer.hpp"
#include "tracker.hpp"

namespace cuscope {
namespace {

bool primary_active(CUdevice dev) {
  static const auto get_state = CS_DRIVER_FN(cuDevicePrimaryCtxGetState);
  unsigned int flags = 0;
  int active = 0;
  return get_state != nullptr && get_state(dev, &flags, &active) == CUDA_SUCCESS && active != 0;
}

// The device's primary context, if it is active; 0 otherwise. Release and
// Reset name only the device, so cuscope reads the handle with a Retain and an
// immediate Release: the reference count ends where it started. Only done when
// the context is active, because Retain on an inactive device would create one.
uint64_t active_primary_ctx(CUdevice dev) {
  static const auto retain = CS_DRIVER_FN(cuDevicePrimaryCtxRetain);
  static const auto release = CS_DRIVER_FN(cuDevicePrimaryCtxRelease);
  if (retain == nullptr || release == nullptr || !primary_active(dev)) return 0;
  CUcontext ctx = nullptr;
  if (retain(&ctx, dev) != CUDA_SUCCESS) return 0;
  release(dev);
  return handle(ctx);
}

// Shared by Release and Reset. Both may destroy the primary context (Release
// only when it drops the last reference); afterwards the driver says which.
template <class RealCall>
CUresult primary_teardown(CUdevice dev, RealCall&& real_call) {
  if (in_internal()) return real_call();
  uint64_t primary = 0;
  {
    InternalScope internal;
    primary = active_primary_ctx(dev);
  }
  gpu_flush_device(dev);
  CUresult result = real_call();
  if (result == CUDA_SUCCESS && primary != 0) {
    InternalScope internal;
    if (!primary_active(dev)) tracker().on_ctx_teardown(primary);
  }
  return result;
}

}  // namespace

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
  if (!in_internal()) gpu_flush_ctx(handle(ctx));
  ApiRecord rec(H_cuCtxDestroy);
  rec.rec.ptr = handle(ctx);
  CUresult r = forward<V, decltype(&::cuCtxDestroy)>(H_cuCtxDestroy, ctx);
  rec.done();
  if (r == CUDA_SUCCESS && !in_internal()) tracker().on_ctx_teardown(handle(ctx));
  return r;
}

// Release may or may not destroy the primary context (it is reference
// counted). Flushing either way is safe: a surviving context simply gets
// fresh timing state on its next launch.
template <int V>
CUresult cs_cuDevicePrimaryCtxRelease(CUdevice dev) {
  note_call();
  ApiRecord rec(H_cuDevicePrimaryCtxRelease);
  return primary_teardown(dev, [&] {
    CUresult r = forward<V, decltype(&::cuDevicePrimaryCtxRelease)>(H_cuDevicePrimaryCtxRelease, dev);
    rec.done();
    return r;
  });
}

template <int V>
CUresult cs_cuDevicePrimaryCtxReset(CUdevice dev) {
  note_call();
  ApiRecord rec(H_cuDevicePrimaryCtxReset);
  return primary_teardown(dev, [&] {
    CUresult r = forward<V, decltype(&::cuDevicePrimaryCtxReset)>(H_cuDevicePrimaryCtxReset, dev);
    rec.done();
    return r;
  });
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

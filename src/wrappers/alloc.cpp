// Allocation hooks: forward the call, then tell the tracker and the timeline
// what happened.
//
// Allocations are recorded only when the driver succeeds. A free removes the
// allocation from the tracker *before* the real call, because the driver may
// hand the same address to another thread the moment it is freed; if the
// driver rejects the free, the allocation is put back.
//
// With a memory cap (CUSCOPE_MEM_LIMIT), an allocation of a GPU-resident kind
// first reserves its bytes; one that would pass the cap is denied with
// CUDA_ERROR_OUT_OF_MEMORY and never reaches the driver. The reservation is
// given back if the driver fails the call, and when the memory is freed.
#include "wrappers.hpp"

#include "limits.hpp"
#include "stacks.hpp"
#include "tracker.hpp"

namespace cuscope {
namespace {

// The context the calling thread has current, as the tracker's key for
// context teardown. 0 if there is none or the driver lacks the function.
uint64_t current_ctx() {
  static const auto get_current = CS_DRIVER_FN(cuCtxGetCurrent);
  if (get_current == nullptr) return 0;
  InternalScope internal;
  CUcontext ctx = nullptr;
  return get_current(&ctx) == CUDA_SUCCESS ? reinterpret_cast<uint64_t>(ctx) : 0;
}

// Reserves an allocation's bytes against the memory cap before the driver
// sees the call. *charged is what was reserved (0 when there is no cap, the
// kind does not count, or the call is cuscope's own). False when the cap
// would be passed: the denial is recorded with the caller's stack, and the
// wrapper returns CUDA_ERROR_OUT_OF_MEMORY without calling the driver, as a
// full GPU would.
bool reserve(ApiRecord& rec, AllocKind kind, uint64_t bytes, uint64_t* charged) {
  *charged = 0;
  MemLimit& limit = mem_limit();
  if (!limit.enabled() || !is_device_memory(kind) || in_internal()) return true;
  if (limit.try_charge(kind, bytes)) {
    *charged = bytes;
    return true;
  }
  rec.done();
  const uint32_t stack = capture_caller_stack();
  limit.on_denied(kind, bytes, stack);
  rec.rec.type = RecType::AllocDenied;
  rec.rec.aux = static_cast<uint8_t>(kind);
  rec.rec.bytes = bytes;
  rec.rec.stack_id = stack;
  return false;
}

// Fills in an allocation's record and, if it succeeded, tracks it. If it
// failed, the bytes reserved for it go back to the cap.
CUresult after_alloc(ApiRecord& rec, CUresult result, uint64_t ptr, uint64_t bytes, AllocKind kind,
                     uint64_t charged) {
  rec.done();
  rec.rec.bytes = bytes;
  rec.rec.aux = static_cast<uint8_t>(kind);
  if (result != CUDA_SUCCESS) {
    mem_limit().release(kind, charged);
    rec.rec.type = RecType::AllocFailed;
    return result;
  }
  if (in_internal()) return result;
  const uint32_t stack = capture_caller_stack();
  if (std::optional<AllocInfo> stale = tracker().on_alloc(AllocInfo{ptr, bytes, kind, stack, current_ctx()})) {
    mem_limit().release(stale->kind, stale->bytes);  // freed where cuscope could not see it
  }
  rec.rec.type = RecType::Alloc;
  rec.rec.ptr = ptr;
  rec.rec.stack_id = stack;
  return result;
}

// Runs real_free (the forwarded call) with the bookkeeping around it.
//
// An invalid free is a pointer cuscope never saw allocated that the driver
// also rejects (a double free, a stray pointer). A free the driver accepts
// but cuscope never saw allocated means memory came from somewhere cuscope
// does not hook; that is a gap in cuscope's view, not a bug in the program,
// so it is not reported.
template <class RealFree>
CUresult tracked_free(ApiRecord& rec, uint64_t ptr, RealFree&& real_free) {
  rec.rec.ptr = ptr;
  if (ptr == 0 || in_internal()) {  // freeing null is a no-op
    CUresult result = real_free();
    rec.done();
    return result;
  }
  std::optional<AllocInfo> freed = tracker().on_free(ptr);
  CUresult result = real_free();
  rec.done();
  if (result == CUDA_SUCCESS) {
    if (freed) {
      // Only a free the driver accepted gives bytes back: a rejected one
      // leaves the allocation live, and a stray or double free never had
      // a charge to give back.
      mem_limit().release(freed->kind, freed->bytes);
      rec.rec.type = RecType::Free;
      rec.rec.bytes = freed->bytes;
      rec.rec.aux = static_cast<uint8_t>(freed->kind);
    }
  } else if (freed) {
    tracker().restore(*freed);
  } else {
    const uint32_t stack = capture_caller_stack();
    tracker().on_invalid_free(ptr, stack);
    rec.rec.type = RecType::InvalidFree;
    rec.rec.stack_id = stack;
  }
  return result;
}

}  // namespace

template <int V>
CUresult cs_cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  note_call();
  ApiRecord rec(H_cuMemAlloc);
  uint64_t charged = 0;
  if (!reserve(rec, AllocKind::Device, bytesize, &charged)) return CUDA_ERROR_OUT_OF_MEMORY;
  CUresult r = forward<V, decltype(&::cuMemAlloc)>(H_cuMemAlloc, dptr, bytesize);
  return after_alloc(rec, r, r == CUDA_SUCCESS ? *dptr : 0, bytesize, AllocKind::Device, charged);
}

template <int V>
CUresult cs_cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  note_call();
  ApiRecord rec(H_cuMemAllocPitch);
  // The pitch is not known until the driver picks it: reserve the unpadded
  // size, then settle the difference. The settling never denies, so the
  // cap can be passed by one allocation's row padding.
  uint64_t charged = 0;
  if (!reserve(rec, AllocKind::Device, static_cast<uint64_t>(WidthInBytes) * Height, &charged)) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  CUresult r = forward<V, decltype(&::cuMemAllocPitch)>(H_cuMemAllocPitch, dptr, pPitch, WidthInBytes, Height,
                                                        ElementSizeBytes);
  const bool ok = r == CUDA_SUCCESS;
  const uint64_t bytes = ok ? static_cast<uint64_t>(*pPitch) * Height : 0;
  if (ok && charged != 0) {
    if (bytes > charged) mem_limit().charge(AllocKind::Device, bytes - charged);
    if (bytes < charged) mem_limit().release(AllocKind::Device, charged - bytes);
  }
  return after_alloc(rec, r, ok ? *dptr : 0, bytes, AllocKind::Device, charged);
}

template <int V>
CUresult cs_cuMemFree(CUdeviceptr dptr) {
  note_call();
  ApiRecord rec(H_cuMemFree);
  return tracked_free(rec, dptr, [&] { return forward<V, decltype(&::cuMemFree)>(H_cuMemFree, dptr); });
}

template <int V>
CUresult cs_cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemAllocAsync);
  rec.rec.stream = handle(hStream);
  uint64_t charged = 0;
  if (!reserve(rec, AllocKind::Async, bytesize, &charged)) return CUDA_ERROR_OUT_OF_MEMORY;
  CUresult r = forward<V, decltype(&::cuMemAllocAsync)>(H_cuMemAllocAsync, dptr, bytesize, hStream);
  return after_alloc(rec, r, r == CUDA_SUCCESS ? *dptr : 0, bytesize, AllocKind::Async, charged);
}

template <int V>
CUresult cs_cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemAllocFromPoolAsync);
  rec.rec.stream = handle(hStream);
  uint64_t charged = 0;
  if (!reserve(rec, AllocKind::Async, bytesize, &charged)) return CUDA_ERROR_OUT_OF_MEMORY;
  CUresult r = forward<V, decltype(&::cuMemAllocFromPoolAsync)>(H_cuMemAllocFromPoolAsync, dptr, bytesize, pool,
                                                                hStream);
  return after_alloc(rec, r, r == CUDA_SUCCESS ? *dptr : 0, bytesize, AllocKind::Async, charged);
}

// Stream-ordered: the memory is released when the stream reaches the free,
// but cuscope counts it free at the call (a documented approximation).
template <int V>
CUresult cs_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemFreeAsync);
  rec.rec.stream = handle(hStream);
  return tracked_free(rec, dptr,
                      [&] { return forward<V, decltype(&::cuMemFreeAsync)>(H_cuMemFreeAsync, dptr, hStream); });
}

template <int V>
CUresult cs_cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
  note_call();
  ApiRecord rec(H_cuMemAllocManaged);
  uint64_t charged = 0;
  if (!reserve(rec, AllocKind::Managed, bytesize, &charged)) return CUDA_ERROR_OUT_OF_MEMORY;
  CUresult r = forward<V, decltype(&::cuMemAllocManaged)>(H_cuMemAllocManaged, dptr, bytesize, flags);
  return after_alloc(rec, r, r == CUDA_SUCCESS ? *dptr : 0, bytesize, AllocKind::Managed, charged);
}

template <int V>
CUresult cs_cuMemAllocHost(void** pp, size_t bytesize) {
  note_call();
  ApiRecord rec(H_cuMemAllocHost);
  CUresult r = forward<V, decltype(&::cuMemAllocHost)>(H_cuMemAllocHost, pp, bytesize);
  return after_alloc(rec, r, r == CUDA_SUCCESS ? handle(*pp) : 0, bytesize, AllocKind::HostPinned, 0);
}

template <int V>
CUresult cs_cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
  note_call();
  ApiRecord rec(H_cuMemHostAlloc);
  CUresult r = forward<V, decltype(&::cuMemHostAlloc)>(H_cuMemHostAlloc, pp, bytesize, Flags);
  return after_alloc(rec, r, r == CUDA_SUCCESS ? handle(*pp) : 0, bytesize, AllocKind::HostPinned, 0);
}

template <int V>
CUresult cs_cuMemFreeHost(void* p) {
  note_call();
  ApiRecord rec(H_cuMemFreeHost);
  return tracked_free(rec, handle(p), [&] { return forward<V, decltype(&::cuMemFreeHost)>(H_cuMemFreeHost, p); });
}

void register_alloc_hooks() {
  CS_REGISTER(cuMemAlloc);
  CS_REGISTER(cuMemAllocPitch);
  CS_REGISTER(cuMemFree);
  CS_REGISTER(cuMemAllocAsync);
  CS_REGISTER(cuMemAllocFromPoolAsync);
  CS_REGISTER(cuMemFreeAsync);
  CS_REGISTER(cuMemAllocManaged);
  CS_REGISTER(cuMemAllocHost);
  CS_REGISTER(cuMemHostAlloc);
  CS_REGISTER(cuMemFreeHost);
}

}  // namespace cuscope

CS_EXPORT(cuMemAlloc_v2, cuMemAlloc, 0, (CUdeviceptr* dptr, size_t bytesize), (dptr, bytesize))
CS_EXPORT(cuMemAllocPitch_v2, cuMemAllocPitch, 0,
          (CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes),
          (dptr, pPitch, WidthInBytes, Height, ElementSizeBytes))
CS_EXPORT(cuMemFree_v2, cuMemFree, 0, (CUdeviceptr dptr), (dptr))
CS_EXPORT(cuMemAllocAsync, cuMemAllocAsync, 0, (CUdeviceptr* dptr, size_t bytesize, CUstream hStream),
          (dptr, bytesize, hStream))
CS_EXPORT(cuMemAllocAsync_ptsz, cuMemAllocAsync, 1, (CUdeviceptr* dptr, size_t bytesize, CUstream hStream),
          (dptr, bytesize, hStream))
CS_EXPORT(cuMemAllocFromPoolAsync, cuMemAllocFromPoolAsync, 0,
          (CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream), (dptr, bytesize, pool, hStream))
CS_EXPORT(cuMemAllocFromPoolAsync_ptsz, cuMemAllocFromPoolAsync, 1,
          (CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream), (dptr, bytesize, pool, hStream))
CS_EXPORT(cuMemFreeAsync, cuMemFreeAsync, 0, (CUdeviceptr dptr, CUstream hStream), (dptr, hStream))
CS_EXPORT(cuMemFreeAsync_ptsz, cuMemFreeAsync, 1, (CUdeviceptr dptr, CUstream hStream), (dptr, hStream))
CS_EXPORT(cuMemAllocManaged, cuMemAllocManaged, 0, (CUdeviceptr* dptr, size_t bytesize, unsigned int flags),
          (dptr, bytesize, flags))
CS_EXPORT(cuMemAllocHost_v2, cuMemAllocHost, 0, (void** pp, size_t bytesize), (pp, bytesize))
CS_EXPORT(cuMemHostAlloc, cuMemHostAlloc, 0, (void** pp, size_t bytesize, unsigned int Flags), (pp, bytesize, Flags))
CS_EXPORT(cuMemFreeHost, cuMemFreeHost, 0, (void* p), (p))

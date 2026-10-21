// Allocation hooks. Pass-through for now; the allocation tracker (Task 8)
// records what each one does.
#include "wrappers.hpp"

namespace cuscope {

template <int V>
CUresult cs_cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  note_call();
  return forward<V, decltype(&::cuMemAlloc)>(H_cuMemAlloc, dptr, bytesize);
}

template <int V>
CUresult cs_cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  note_call();
  return forward<V, decltype(&::cuMemAllocPitch)>(H_cuMemAllocPitch, dptr, pPitch, WidthInBytes, Height,
                                                  ElementSizeBytes);
}

template <int V>
CUresult cs_cuMemFree(CUdeviceptr dptr) {
  note_call();
  return forward<V, decltype(&::cuMemFree)>(H_cuMemFree, dptr);
}

template <int V>
CUresult cs_cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemAllocAsync)>(H_cuMemAllocAsync, dptr, bytesize, hStream);
}

template <int V>
CUresult cs_cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemAllocFromPoolAsync)>(H_cuMemAllocFromPoolAsync, dptr, bytesize, pool,
                                                          hStream);
}

template <int V>
CUresult cs_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemFreeAsync)>(H_cuMemFreeAsync, dptr, hStream);
}

template <int V>
CUresult cs_cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
  note_call();
  return forward<V, decltype(&::cuMemAllocManaged)>(H_cuMemAllocManaged, dptr, bytesize, flags);
}

template <int V>
CUresult cs_cuMemAllocHost(void** pp, size_t bytesize) {
  note_call();
  return forward<V, decltype(&::cuMemAllocHost)>(H_cuMemAllocHost, pp, bytesize);
}

template <int V>
CUresult cs_cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
  note_call();
  return forward<V, decltype(&::cuMemHostAlloc)>(H_cuMemHostAlloc, pp, bytesize, Flags);
}

template <int V>
CUresult cs_cuMemFreeHost(void* p) {
  note_call();
  return forward<V, decltype(&::cuMemFreeHost)>(H_cuMemFreeHost, p);
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

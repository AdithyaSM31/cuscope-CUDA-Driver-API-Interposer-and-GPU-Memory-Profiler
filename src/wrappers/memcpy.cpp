// Copy hooks. Pass-through for now; copy tracking (Task 13) adds direction,
// bytes and GPU time.
#include "wrappers.hpp"

namespace cuscope {

template <int V>
CUresult cs_cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  note_call();
  return forward<V, decltype(&::cuMemcpy)>(H_cuMemcpy, dst, src, ByteCount);
}

template <int V>
CUresult cs_cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemcpyAsync)>(H_cuMemcpyAsync, dst, src, ByteCount, hStream);
}

template <int V>
CUresult cs_cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  note_call();
  return forward<V, decltype(&::cuMemcpyHtoD)>(H_cuMemcpyHtoD, dstDevice, srcHost, ByteCount);
}

template <int V>
CUresult cs_cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  note_call();
  return forward<V, decltype(&::cuMemcpyDtoH)>(H_cuMemcpyDtoH, dstHost, srcDevice, ByteCount);
}

template <int V>
CUresult cs_cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  note_call();
  return forward<V, decltype(&::cuMemcpyDtoD)>(H_cuMemcpyDtoD, dstDevice, srcDevice, ByteCount);
}

template <int V>
CUresult cs_cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemcpyHtoDAsync)>(H_cuMemcpyHtoDAsync, dstDevice, srcHost, ByteCount, hStream);
}

template <int V>
CUresult cs_cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemcpyDtoHAsync)>(H_cuMemcpyDtoHAsync, dstHost, srcDevice, ByteCount, hStream);
}

template <int V>
CUresult cs_cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  note_call();
  return forward<V, decltype(&::cuMemcpyDtoDAsync)>(H_cuMemcpyDtoDAsync, dstDevice, srcDevice, ByteCount,
                                                    hStream);
}

void register_memcpy_hooks() {
  CS_REGISTER(cuMemcpy);
  CS_REGISTER(cuMemcpyAsync);
  CS_REGISTER(cuMemcpyHtoD);
  CS_REGISTER(cuMemcpyDtoH);
  CS_REGISTER(cuMemcpyDtoD);
  CS_REGISTER(cuMemcpyHtoDAsync);
  CS_REGISTER(cuMemcpyDtoHAsync);
  CS_REGISTER(cuMemcpyDtoDAsync);
}

}  // namespace cuscope

#define CS_COPY_PARAMS (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
CS_EXPORT(cuMemcpy, cuMemcpy, 0, CS_COPY_PARAMS, (dst, src, ByteCount))
CS_EXPORT(cuMemcpy_ptds, cuMemcpy, 1, CS_COPY_PARAMS, (dst, src, ByteCount))
#undef CS_COPY_PARAMS

CS_EXPORT(cuMemcpyAsync, cuMemcpyAsync, 0, (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream),
          (dst, src, ByteCount, hStream))
CS_EXPORT(cuMemcpyAsync_ptsz, cuMemcpyAsync, 1,
          (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream), (dst, src, ByteCount, hStream))

CS_EXPORT(cuMemcpyHtoD_v2, cuMemcpyHtoD, 0, (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount),
          (dstDevice, srcHost, ByteCount))
CS_EXPORT(cuMemcpyHtoD_v2_ptds, cuMemcpyHtoD, 1, (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount),
          (dstDevice, srcHost, ByteCount))
CS_EXPORT(cuMemcpyDtoH_v2, cuMemcpyDtoH, 0, (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount),
          (dstHost, srcDevice, ByteCount))
CS_EXPORT(cuMemcpyDtoH_v2_ptds, cuMemcpyDtoH, 1, (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount),
          (dstHost, srcDevice, ByteCount))
CS_EXPORT(cuMemcpyDtoD_v2, cuMemcpyDtoD, 0, (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount),
          (dstDevice, srcDevice, ByteCount))
CS_EXPORT(cuMemcpyDtoD_v2_ptds, cuMemcpyDtoD, 1, (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount),
          (dstDevice, srcDevice, ByteCount))

CS_EXPORT(cuMemcpyHtoDAsync_v2, cuMemcpyHtoDAsync, 0,
          (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream),
          (dstDevice, srcHost, ByteCount, hStream))
CS_EXPORT(cuMemcpyHtoDAsync_v2_ptsz, cuMemcpyHtoDAsync, 1,
          (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream),
          (dstDevice, srcHost, ByteCount, hStream))
CS_EXPORT(cuMemcpyDtoHAsync_v2, cuMemcpyDtoHAsync, 0,
          (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
          (dstHost, srcDevice, ByteCount, hStream))
CS_EXPORT(cuMemcpyDtoHAsync_v2_ptsz, cuMemcpyDtoHAsync, 1,
          (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
          (dstHost, srcDevice, ByteCount, hStream))
CS_EXPORT(cuMemcpyDtoDAsync_v2, cuMemcpyDtoDAsync, 0,
          (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
          (dstDevice, srcDevice, ByteCount, hStream))
CS_EXPORT(cuMemcpyDtoDAsync_v2_ptsz, cuMemcpyDtoDAsync, 1,
          (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
          (dstDevice, srcDevice, ByteCount, hStream))

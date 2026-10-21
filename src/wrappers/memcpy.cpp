// Copy hooks. Each call is recorded with its destination, source, size and
// stream; copy tracking (Task 13) adds direction and GPU time.
#include "wrappers.hpp"

namespace cuscope {
namespace {

void describe_copy(ApiRecord& rec, uint64_t dst, uint64_t src, size_t bytes, CUstream stream) {
  rec.rec.ptr = dst;
  rec.rec.func = src;
  rec.rec.bytes = bytes;
  rec.rec.stream = handle(stream);
}

}  // namespace

template <int V>
CUresult cs_cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  note_call();
  ApiRecord rec(H_cuMemcpy);
  describe_copy(rec, dst, src, ByteCount, nullptr);
  CUresult r = forward<V, decltype(&::cuMemcpy)>(H_cuMemcpy, dst, src, ByteCount);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemcpyAsync);
  describe_copy(rec, dst, src, ByteCount, hStream);
  CUresult r = forward<V, decltype(&::cuMemcpyAsync)>(H_cuMemcpyAsync, dst, src, ByteCount, hStream);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  note_call();
  ApiRecord rec(H_cuMemcpyHtoD);
  describe_copy(rec, dstDevice, handle(srcHost), ByteCount, nullptr);
  CUresult r = forward<V, decltype(&::cuMemcpyHtoD)>(H_cuMemcpyHtoD, dstDevice, srcHost, ByteCount);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  note_call();
  ApiRecord rec(H_cuMemcpyDtoH);
  describe_copy(rec, handle(dstHost), srcDevice, ByteCount, nullptr);
  CUresult r = forward<V, decltype(&::cuMemcpyDtoH)>(H_cuMemcpyDtoH, dstHost, srcDevice, ByteCount);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  note_call();
  ApiRecord rec(H_cuMemcpyDtoD);
  describe_copy(rec, dstDevice, srcDevice, ByteCount, nullptr);
  CUresult r = forward<V, decltype(&::cuMemcpyDtoD)>(H_cuMemcpyDtoD, dstDevice, srcDevice, ByteCount);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemcpyHtoDAsync);
  describe_copy(rec, dstDevice, handle(srcHost), ByteCount, hStream);
  CUresult r = forward<V, decltype(&::cuMemcpyHtoDAsync)>(H_cuMemcpyHtoDAsync, dstDevice, srcHost, ByteCount, hStream);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemcpyDtoHAsync);
  describe_copy(rec, handle(dstHost), srcDevice, ByteCount, hStream);
  CUresult r = forward<V, decltype(&::cuMemcpyDtoHAsync)>(H_cuMemcpyDtoHAsync, dstHost, srcDevice, ByteCount, hStream);
  rec.done();
  return r;
}

template <int V>
CUresult cs_cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  note_call();
  ApiRecord rec(H_cuMemcpyDtoDAsync);
  describe_copy(rec, dstDevice, srcDevice, ByteCount, hStream);
  CUresult r = forward<V, decltype(&::cuMemcpyDtoDAsync)>(H_cuMemcpyDtoDAsync, dstDevice, srcDevice, ByteCount,
                                                          hStream);
  rec.done();
  return r;
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

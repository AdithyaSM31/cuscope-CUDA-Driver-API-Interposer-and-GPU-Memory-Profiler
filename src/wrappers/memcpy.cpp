// Copy hooks: direction, size, pinned or pageable host memory, and the
// copy's time on the GPU.
//
// Explicit copies (HtoD, DtoH, DtoD) say their direction; for those cuscope
// only asks whether the host side is pinned. The generic cuMemcpy and
// cuMemcpyAsync do not, so cuscope asks the driver what each pointer is.
#include "wrappers.hpp"

#include "gputimer.hpp"

namespace cuscope {
namespace {

enum class Memory { Device, PinnedHost, PageableHost };

// Device (including managed) memory, page-locked host memory, or ordinary
// host memory, which the driver does not know and rejects with an error.
Memory classify(uint64_t ptr) {
  static const auto get_attribute = CS_DRIVER_FN(cuPointerGetAttribute);
  if (get_attribute == nullptr || ptr == 0) return Memory::PageableHost;
  InternalScope internal;
  unsigned int type = 0;  // the attribute's value is a CUmemorytype
  if (get_attribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr) != CUDA_SUCCESS) return Memory::PageableHost;
  return type == CU_MEMORYTYPE_HOST ? Memory::PinnedHost : Memory::Device;
}

struct CopyKind {
  CopyDir dir;
  bool pinned;
};

CopyKind explicit_copy(CopyDir dir, uint64_t host_ptr) {
  return {dir, host_ptr != 0 && !in_internal() && classify(host_ptr) == Memory::PinnedHost};
}

CopyKind generic_copy(uint64_t dst, uint64_t src) {
  if (in_internal()) return {CopyDir::Unknown, false};
  const Memory s = classify(src), d = classify(dst);
  const bool src_device = s == Memory::Device, dst_device = d == Memory::Device;
  const CopyDir dir = src_device ? (dst_device ? CopyDir::DtoD : CopyDir::DtoH)
                                 : (dst_device ? CopyDir::HtoD : CopyDir::HtoH);
  return {dir, s == Memory::PinnedHost || d == Memory::PinnedHost};
}

// Times and records one copy around real_copy (the forwarded call). The
// classification above runs before the clock and the start event, so it is
// never part of the copy's measured time.
template <int V, class RealCopy>
CUresult tracked_copy(HookId api, uint64_t dst, uint64_t src, size_t bytes, CUstream stream, CopyKind kind,
                      RealCopy&& real_copy) {
  Timing timing;
  const bool timed = gpu_begin(V, stream, &timing);
  ApiRecord rec(api);
  rec.rec.ptr = dst;
  rec.rec.func = src;
  rec.rec.bytes = bytes;
  rec.rec.stream = handle(stream);
  CUresult r = real_copy();
  rec.done();
  if (r != CUDA_SUCCESS) {  // a rejected copy moved nothing: it stays a plain API call
    if (timed) gpu_cancel(timing);
    return r;
  }
  rec.rec.type = RecType::Copy;
  rec.rec.aux = copy_aux(kind.dir, kind.pinned);
  rec.rec.corr = next_corr();
  if (timed) {
    GpuSpan meta{};
    meta.corr = rec.rec.corr;
    meta.stream = rec.rec.stream;
    meta.bytes = bytes;
    meta.api = api;
    meta.aux = rec.rec.aux;
    gpu_end(V, stream, timing, meta);
  }
  return r;
}

}  // namespace

template <int V>
CUresult cs_cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  note_call();
  return tracked_copy<V>(H_cuMemcpy, dst, src, ByteCount, nullptr, generic_copy(dst, src),
                         [&] { return forward<V, decltype(&::cuMemcpy)>(H_cuMemcpy, dst, src, ByteCount); });
}

template <int V>
CUresult cs_cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyAsync, dst, src, ByteCount, hStream, generic_copy(dst, src), [&] {
    return forward<V, decltype(&::cuMemcpyAsync)>(H_cuMemcpyAsync, dst, src, ByteCount, hStream);
  });
}

template <int V>
CUresult cs_cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyHtoD, dstDevice, handle(srcHost), ByteCount, nullptr,
                         explicit_copy(CopyDir::HtoD, handle(srcHost)), [&] {
                           return forward<V, decltype(&::cuMemcpyHtoD)>(H_cuMemcpyHtoD, dstDevice, srcHost, ByteCount);
                         });
}

template <int V>
CUresult cs_cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyDtoH, handle(dstHost), srcDevice, ByteCount, nullptr,
                         explicit_copy(CopyDir::DtoH, handle(dstHost)), [&] {
                           return forward<V, decltype(&::cuMemcpyDtoH)>(H_cuMemcpyDtoH, dstHost, srcDevice, ByteCount);
                         });
}

template <int V>
CUresult cs_cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyDtoD, dstDevice, srcDevice, ByteCount, nullptr, CopyKind{CopyDir::DtoD, false},
                         [&] {
                           return forward<V, decltype(&::cuMemcpyDtoD)>(H_cuMemcpyDtoD, dstDevice, srcDevice,
                                                                        ByteCount);
                         });
}

template <int V>
CUresult cs_cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyHtoDAsync, dstDevice, handle(srcHost), ByteCount, hStream,
                         explicit_copy(CopyDir::HtoD, handle(srcHost)), [&] {
                           return forward<V, decltype(&::cuMemcpyHtoDAsync)>(H_cuMemcpyHtoDAsync, dstDevice,
                                                                             srcHost, ByteCount, hStream);
                         });
}

template <int V>
CUresult cs_cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyDtoHAsync, handle(dstHost), srcDevice, ByteCount, hStream,
                         explicit_copy(CopyDir::DtoH, handle(dstHost)), [&] {
                           return forward<V, decltype(&::cuMemcpyDtoHAsync)>(H_cuMemcpyDtoHAsync, dstHost,
                                                                             srcDevice, ByteCount, hStream);
                         });
}

template <int V>
CUresult cs_cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  note_call();
  return tracked_copy<V>(H_cuMemcpyDtoDAsync, dstDevice, srcDevice, ByteCount, hStream,
                         CopyKind{CopyDir::DtoD, false}, [&] {
                           return forward<V, decltype(&::cuMemcpyDtoDAsync)>(H_cuMemcpyDtoDAsync, dstDevice,
                                                                             srcDevice, ByteCount, hStream);
                         });
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

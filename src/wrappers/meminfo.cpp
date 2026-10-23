// cuMemGetInfo: with a memory cap, the program sees a GPU the size of its cap.
//
// Tools and frameworks that size themselves to the free memory then plan for
// the cap rather than the whole GPU. Without a cap the driver's answer is
// returned untouched. No API event is recorded: it is a query, and PyTorch
// calls it on every out-of-memory retry.
#include "wrappers.hpp"

#include "limits.hpp"

namespace cuscope {

template <int V>
CUresult cs_cuMemGetInfo(size_t* free, size_t* total) {
  note_call();
  CUresult r = forward<V, decltype(&::cuMemGetInfo)>(H_cuMemGetInfo, free, total);
  if (r == CUDA_SUCCESS && !in_internal()) mem_limit().cap_info(free, total);
  return r;
}

void register_meminfo_hooks() { CS_REGISTER(cuMemGetInfo); }

}  // namespace cuscope

CS_EXPORT(cuMemGetInfo_v2, cuMemGetInfo, 0, (size_t* free, size_t* total), (free, total))

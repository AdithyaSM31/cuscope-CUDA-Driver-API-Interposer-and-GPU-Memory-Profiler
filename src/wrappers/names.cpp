// Kernel names: lookup at first launch, plus hooks on the calls that hand a
// program its function handles, as a fallback name map.
#include "names.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

#include "stacks.hpp"
#include "wrappers.hpp"

namespace cuscope {
namespace {

std::mutex g_names_mu;
std::unordered_map<uint64_t, std::string>& known_names() {  // handle -> mangled name
  static auto* m = new std::unordered_map<uint64_t, std::string>();
  return *m;
}

// A few recently launched functions per thread, so a launch loop over the
// same kernels skips the map entirely.
constexpr int kSeenSlots = 16;
thread_local uint64_t t_seen[kSeenSlots] __attribute__((tls_model("initial-exec"))) = {};

std::string driver_name(uint64_t func) {
  static const auto func_get_name = CS_DRIVER_FN(cuFuncGetName);
  static const auto kernel_get_name = CS_DRIVER_FN(cuKernelGetName);
  InternalScope internal;
  const char* name = nullptr;
  if (func_get_name != nullptr && func_get_name(&name, reinterpret_cast<CUfunction>(func)) == CUDA_SUCCESS &&
      name != nullptr) {
    return name;
  }
  name = nullptr;
  if (kernel_get_name != nullptr && kernel_get_name(&name, reinterpret_cast<CUkernel>(func)) == CUDA_SUCCESS &&
      name != nullptr) {
    return name;
  }
  return {};
}

}  // namespace

void note_kernel(uint64_t func) {
  uint64_t& slot = t_seen[(func >> 4) % kSeenSlots];
  if (slot == func) return;
  slot = func;
  {
    std::lock_guard<std::mutex> lock(g_names_mu);
    if (known_names().count(func) != 0) return;
  }
  std::string name = driver_name(func);  // outside the lock: a driver call
  if (name.empty()) return;
  std::lock_guard<std::mutex> lock(g_names_mu);
  known_names().emplace(func, std::move(name));
}

void remember_kernel_name(uint64_t handle, const char* name) {
  if (handle == 0 || name == nullptr) return;
  std::lock_guard<std::mutex> lock(g_names_mu);
  known_names()[handle] = name;
}

void alias_kernel_name(uint64_t function, uint64_t kernel) {
  std::lock_guard<std::mutex> lock(g_names_mu);
  auto it = known_names().find(kernel);
  if (it != known_names().end()) known_names()[function] = it->second;
}

void fill_kernel_names(Snapshot& s) {
  std::lock_guard<std::mutex> lock(g_names_mu);
  for (const Record& r : s.records) {
    if (r.type != RecType::Launch || s.kernel_names.count(r.func) != 0) continue;
    auto it = known_names().find(r.func);
    if (it != known_names().end()) s.kernel_names[r.func] = demangle(it->second.c_str());
  }
}

// ---- Hooks: the name map ----

template <int V>
CUresult cs_cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  note_call();
  CUresult r = forward<V, decltype(&::cuModuleGetFunction)>(H_cuModuleGetFunction, hfunc, hmod, name);
  if (r == CUDA_SUCCESS && hfunc != nullptr) remember_kernel_name(handle(*hfunc), name);
  return r;
}

template <int V>
CUresult cs_cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
  note_call();
  CUresult r = forward<V, decltype(&::cuLibraryGetKernel)>(H_cuLibraryGetKernel, pKernel, library, name);
  if (r == CUDA_SUCCESS && pKernel != nullptr) remember_kernel_name(handle(*pKernel), name);
  return r;
}

template <int V>
CUresult cs_cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
  note_call();
  CUresult r = forward<V, decltype(&::cuKernelGetFunction)>(H_cuKernelGetFunction, pFunc, kernel);
  if (r == CUDA_SUCCESS && pFunc != nullptr) alias_kernel_name(handle(*pFunc), handle(kernel));
  return r;
}

void register_name_hooks() {
  CS_REGISTER(cuModuleGetFunction);
  CS_REGISTER(cuLibraryGetKernel);
  CS_REGISTER(cuKernelGetFunction);
}

}  // namespace cuscope

CS_EXPORT(cuModuleGetFunction, cuModuleGetFunction, 0, (CUfunction * hfunc, CUmodule hmod, const char* name),
          (hfunc, hmod, name))
CS_EXPORT(cuLibraryGetKernel, cuLibraryGetKernel, 0, (CUkernel * pKernel, CUlibrary library, const char* name),
          (pKernel, library, name))
CS_EXPORT(cuKernelGetFunction, cuKernelGetFunction, 0, (CUfunction * pFunc, CUkernel kernel), (pFunc, kernel))

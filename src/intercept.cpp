// Interception paths 2 and 3: cuscope's dlsym and cuGetProcAddress.
//
// The CUDA runtime loads libcuda with dlopen and fetches cuGetProcAddress with
// dlsym(handle, ...). A lookup on a specific handle never consults
// LD_PRELOAD, so cuscope overrides dlsym itself and hands back its own
// cuGetProcAddress. Every later driver lookup then passes through cuscope,
// which swaps in a wrapper when the returned pointer is the exact variant the
// wrapper was written for (spec §4.3).
#include <cuda.h>
#include <dlfcn.h>

#include <cstring>

#include "census.hpp"
#include "cuscope/cuscope.h"
#include "guard.hpp"
#include "hook_table.hpp"
#include "loader.hpp"
#include "stats.hpp"

using cuscope::GpaVer;

using GpaV1Fn = CUresult (*)(const char*, void**, int, cuuint64_t);
using GpaV2Fn = CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

// cuda.h maps the name cuGetProcAddress to cuGetProcAddress_v2, so the
// 4-argument version is defined under another C++ name with an asm label.
extern "C" CUSCOPE_API CUresult cuscope_gpa_v1(const char* symbol, void** pfn, int version,
                                               cuuint64_t flags) __asm__("cuGetProcAddress");

namespace {

constexpr int kFirstV2GpaVersion = 12000;  // cuGetProcAddress_v2 arrived in CUDA 12.0
constexpr int kNoStatus = -1;               // census value when the caller gave no status pointer

void* gpa_wrapper(GpaVer ver) {
  return ver == GpaVer::V1 ? reinterpret_cast<void*>(&cuscope_gpa_v1)
                           : reinterpret_cast<void*>(&cuGetProcAddress);  // _v2
}

bool is_gpa_name(const char* symbol, GpaVer* ver) {
  if (std::strcmp(symbol, "cuGetProcAddress_v2") == 0) {
    *ver = GpaVer::V2;
    return true;
  }
  if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
    *ver = GpaVer::V1;
    return true;
  }
  return false;
}

// True when the code that called dlsym lives in libcuda: the WSL loader shim
// (libcuda.so.1 / libcuda_loader.so) or the real driver (libcuda.so.1.1).
// Decided by the caller's module, never by the handle it passed.
//
// "libcuda" must be followed by '.' or '_': libcudart.so.13, the shared CUDA
// runtime, also starts with "libcuda", and its lookups must be intercepted.
bool caller_is_libcuda(const void* return_address, const char** module) {
  Dl_info info{};
  *module = nullptr;
  if (dladdr(return_address, &info) == 0 || info.dli_fname == nullptr) return false;
  *module = info.dli_fname;
  const char* slash = std::strrchr(info.dli_fname, '/');
  const char* base = slash ? slash + 1 : info.dli_fname;
  return std::strncmp(base, "libcuda", 7) == 0 && (base[7] == '.' || base[7] == '_');
}

// Swaps *pfn for a wrapper when it is exactly the function that wrapper was
// written for. Any other pointer (an older version with a different
// signature, an unhooked function) is returned untouched. Returns the census
// action: "wrap", "mismatch" (hooked base name, different variant) or "pass".
const char* wrap_if_matching(const char* symbol, void** pfn, int version) {
  if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
    // Lookups made through the returned pointer must keep passing through
    // cuscope, so the lookup function itself is always wrapped.
    GpaVer ver = version >= kFirstV2GpaVersion ? GpaVer::V2 : GpaVer::V1;
    if (*pfn == gpa_wrapper(ver)) return "pass";
    if (cuscope::real_gpa(ver) == nullptr) cuscope::set_real_gpa(ver, *pfn);
    *pfn = gpa_wrapper(ver);
    cuscope::stats::gpa_wrapped++;
    return "wrap";
  }

  int id = cuscope::find_hook_by_base(symbol);
  if (id < 0 || !cuscope::loader_ready()) return "pass";
  auto hook = static_cast<cuscope::HookId>(id);
  for (int v = cuscope::kLegacy; v <= cuscope::kPerThread; ++v) {
    void* wrapper = cuscope::hook_wrapper(hook, v);
    if (wrapper != nullptr && *pfn == cuscope::hook_real(hook, v)) {
      *pfn = wrapper;
      cuscope::stats::gpa_wrapped++;
      return "wrap";
    }
  }
  cuscope::stats::gpa_passthrough++;
  return "mismatch";
}

// Shared tail of both cuGetProcAddress wrappers.
CUresult finish_lookup(const char* symbol, void** pfn, int version, cuuint64_t flags, CUresult result,
                       int status) {
  const char* action = "pass";
  if (result == CUDA_SUCCESS && symbol != nullptr && pfn != nullptr && *pfn != nullptr) {
    action = wrap_if_matching(symbol, pfn, version);
  }
  cuscope::census_gpa(symbol, version, flags, static_cast<int>(result), status, action);
  return result;
}

// Loads the real driver functions on the first lookup a program makes.
void prepare_lookup() {
  if (!cuscope::in_internal()) cuscope::ensure_loader();
}

}  // namespace

// Path 2. The exported dlsym is the assembly trampoline in
// dlsym_trampoline.S; this function decides what it does. Every lookup that
// is not cuscope's business is answered by *forwarding*: the trampoline jumps
// to the real dlsym with the original return address still on the stack, so
// glibc resolves RTLD_NEXT and RTLD_DEFAULT for the real caller.
struct DlsymDecision {
  void* value;       // the answer, when forward_to is null
  void* forward_to;  // the real dlsym, to jump to with the original arguments
};

extern "C" __attribute__((visibility("hidden"))) DlsymDecision cuscope_dlsym_decide(void* handle, const char* symbol,
                                                                                     const void* caller) noexcept {
  cuscope::DlsymFn real = cuscope::real_dlsym();
  if (real == nullptr) return {nullptr, nullptr};
  const DlsymDecision forward{nullptr, reinterpret_cast<void*>(real)};
  if (symbol == nullptr) return forward;

  // libcuda's own lookups (the WSL shim filling its table from the real
  // driver) must get real functions. A wrapper here would forward to the shim,
  // which would forward back to the wrapper, forever. This branch must not
  // touch libcuda_handle(): it can run inside that function's dlopen.
  const char* module = nullptr;
  if (caller_is_libcuda(caller, &module)) {
    cuscope::census_dlsym(symbol, module, "caller-libcuda");
    return forward;
  }

  GpaVer ver;
  if (is_gpa_name(symbol, &ver)) {
    // Pseudo-handles would find cuscope's own export first, so resolve those
    // through libcuda. Without libcuda the program would have got nullptr too.
    void* source = (handle == RTLD_DEFAULT || handle == RTLD_NEXT) ? cuscope::libcuda_handle() : handle;
    void* found = source ? real(source, symbol) : nullptr;
    if (found == nullptr || cuscope::inside_cuscope(found)) {
      cuscope::census_dlsym(symbol, module, "pass");
      return {found, nullptr};
    }
    cuscope::set_real_gpa(ver, found);
    cuscope::stats::dlsym_wrapped++;
    cuscope::census_dlsym(symbol, module, "wrap");
    return {gpa_wrapper(ver), nullptr};
  }

  // A hooked driver export fetched by name from a specific handle gets its
  // wrapper, as it would through path 1. Pseudo-handles depend on the caller,
  // so those go to glibc; it finds cuscope's export first, which is the same
  // wrapper. Specific handles do not depend on the caller, so asking here is
  // exact.
  int variant = 0;
  int id = cuscope::find_hook_by_export(symbol, &variant);
  if (id < 0 || handle == RTLD_DEFAULT || handle == RTLD_NEXT) return forward;
  void* found = real(handle, symbol);
  void* wrapper = found ? cuscope::hook_wrapper(static_cast<cuscope::HookId>(id), variant) : nullptr;
  if (wrapper == nullptr) return {found, nullptr};
  cuscope::stats::dlsym_wrapped++;
  cuscope::census_dlsym(symbol, module, "wrap");
  return {wrapper, nullptr};
}

// Path 3. cuda.h turns this name into cuGetProcAddress_v2.
extern "C" CUSCOPE_API CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pfn, int version,
                                                         cuuint64_t flags,
                                                         CUdriverProcAddressQueryResult* status) {
  auto real = reinterpret_cast<GpaV2Fn>(cuscope::resolve_real_gpa(GpaVer::V2));
  if (real == nullptr) return CUDA_ERROR_NOT_INITIALIZED;
  prepare_lookup();
  CUresult result = real(symbol, pfn, version, flags, status);
  return finish_lookup(symbol, pfn, version, flags, result,
                       status ? static_cast<int>(*status) : kNoStatus);
}

CUresult cuscope_gpa_v1(const char* symbol, void** pfn, int version, cuuint64_t flags) {
  auto real = reinterpret_cast<GpaV1Fn>(cuscope::resolve_real_gpa(GpaVer::V1));
  if (real == nullptr) return CUDA_ERROR_NOT_INITIALIZED;
  prepare_lookup();
  CUresult result = real(symbol, pfn, version, flags);
  return finish_lookup(symbol, pfn, version, flags, result, kNoStatus);
}

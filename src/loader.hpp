#pragma once

// Finding the real driver and the real dynamic-linker functions.
//
// Everything here is reachable from cuscope's dlsym override, which can run
// before cuscope's own constructors (the WSL libcuda shim calls dlsym from its
// constructor). So all state is constant-initialised and set up lazily with
// pthread_once.
namespace cuscope {

using DlsymFn = void* (*)(void*, const char*);

// glibc's dlsym, found with dlvsym so the lookup never re-enters cuscope's
// override. Null only if glibc has neither symbol version.
DlsymFn real_dlsym();

// A handle to libcuda.so.1, loading it if no one has yet. Null on failure.
// Never call this from a dlsym that libcuda itself made: that dlsym can run
// inside this function's own dlopen, and pthread_once would deadlock.
void* libcuda_handle();

// True if addr is inside libcuscope.so. A "real" driver function never is;
// RTLD_DEFAULT lookups would otherwise find cuscope's own exports.
bool inside_cuscope(const void* addr);

// The real cuGetProcAddress entry points, by signature version:
// V1 = (symbol, pfn, cudaVersion, flags), V2 adds symbolStatus.
enum class GpaVer { V1 = 0, V2 = 1 };

void set_real_gpa(GpaVer ver, void* fn);
void* real_gpa(GpaVer ver);

// real_gpa(ver), or else resolved from libcuda by export name (a program
// linked with -lcuda calls cuscope's export without any dlsym). Null if
// libcuda is unavailable.
void* resolve_real_gpa(GpaVer ver);

// Stage 2, once per process: register the wrappers, then resolve each hook's
// real function for both stream variants through the real cuGetProcAddress at
// the hook's version (hooks.def). Missing functions are logged once and their
// hooks pass through.
void ensure_loader();
bool loader_ready();

// Fallback when stage 2 has not resolved a hook: dlsym of the hook's export
// name on libcuda. Never blocks on the loader. Null if not found.
void* resolve_by_export(int hook_id, int variant);

// A driver function cuscope calls itself (cuCtxGetCurrent, cuEventCreate,
// ...), by exact exported symbol name, from libcuda. Null if the driver lacks
// it. Not cached here: callers keep the pointer. Use CS_DRIVER_FN below rather
// than naming symbols by hand.
void* driver_proc(const char* export_name);

}  // namespace cuscope

// The driver function `fn`, typed with its cuda.h prototype and resolved by
// the symbol cuda.h maps `fn` to (cuEventElapsedTime -> cuEventElapsedTime_v2,
// cuCtxGetDevice -> cuCtxGetDevice). The type and the symbol name come from
// the same token, so they always agree.
//
// Never resolve these with cuGetProcAddress at CUDA_VERSION: in CUDA 13 that
// returns cuCtxGetDevice_v2(CUdevice*, CUcontext) for "cuCtxGetDevice", and a
// call through the one-argument prototype passes the driver a garbage context.
#define CS_STR_(x) #x
#define CS_STR(x) CS_STR_(x)
#define CS_DRIVER_FN(fn) reinterpret_cast<decltype(&::fn)>(::cuscope::driver_proc(CS_STR(fn)))
// The per-thread-default-stream variant of a stream function.
#define CS_DRIVER_FN_PTSZ(fn) reinterpret_cast<decltype(&::fn)>(::cuscope::driver_proc(CS_STR(fn) "_ptsz"))

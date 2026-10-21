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

// The real cuGetProcAddress entry points, by signature version:
// V1 = (symbol, pfn, cudaVersion, flags), V2 adds symbolStatus.
enum class GpaVer { V1 = 0, V2 = 1 };

void set_real_gpa(GpaVer ver, void* fn);
void* real_gpa(GpaVer ver);

}  // namespace cuscope

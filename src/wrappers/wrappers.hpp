#pragma once

// Shared pieces of the hook wrappers.
//
// Each hooked function has one wrapper template, cs_<base><V>, where V is the
// stream variant (kLegacy or kPerThread) it forwards to. Both instantiations
// are registered in the hook table, and every exported driver symbol for that
// function (path 1) calls the matching one.
#include <cuda.h>

#include <atomic>
#include <type_traits>

#include "cuscope/cuscope.h"
#include "guard.hpp"
#include "hook_table.hpp"
#include "stats.hpp"

namespace cuscope {

// Counts a wrapper invocation, unless cuscope itself made the call.
inline void note_call() {
  if (!in_internal()) stats::hooked_calls.fetch_add(1, std::memory_order_relaxed);
}

// Counts a call that arrived through an exported driver symbol (path 1).
inline void note_direct_call() { stats::direct_calls.fetch_add(1, std::memory_order_relaxed); }

// Calls the real driver function for (id, V) with the same arguments and
// returns its result unchanged. A hook whose real function was never found
// cannot be forwarded, so the call fails as the driver would without it.
template <int V, class Fn, class... Args>
CUresult forward(HookId id, Args... args) {
  Fn fn = real_fn<Fn>(id, V);
  return fn != nullptr ? fn(args...) : CUDA_ERROR_NOT_INITIALIZED;
}

void register_alloc_hooks();
void register_memcpy_hooks();
void register_launch_hooks();
void register_teardown_hooks();

}  // namespace cuscope

// Registers both variants of cs_<b>. The static_assert proves at compile time
// that the wrapper has exactly the cuda.h prototype of b (after cuda.h's
// macros, so cuMemAlloc is checked against cuMemAlloc_v2).
#define CS_REGISTER(b)                                                                 \
  do {                                                                                 \
    static_assert(std::is_same_v<decltype(&cs_##b<0>), decltype(&::b)>,                \
                  #b ": wrapper signature differs from cuda.h");                       \
    ::cuscope::set_wrappers(::cuscope::H_##b, reinterpret_cast<void*>(&cs_##b<0>),     \
                            reinterpret_cast<void*>(&cs_##b<1>));                      \
  } while (0)

// Defines exported driver symbol `name` (path 1), forwarding to cs_<b><V>.
#define CS_EXPORT(name, b, V, params, args)                         \
  extern "C" CUSCOPE_API CUresult name params {                     \
    ::cuscope::note_direct_call();                                  \
    return ::cuscope::cs_##b<V> args;                               \
  }

// Like CS_EXPORT, for a symbol whose name cuda.h redefines as a macro (for
// example cuDevicePrimaryCtxRelease -> _v2): the C++ name is `cname`, and an
// asm label gives it the exported symbol name `symbol`.
#define CS_EXPORT_AS(symbol, cname, b, V, params, args)                       \
  extern "C" CUSCOPE_API CUresult cname params __asm__(symbol);               \
  extern "C" CUresult cname params {                                          \
    ::cuscope::note_direct_call();                                            \
    return ::cuscope::cs_##b<V> args;                                         \
  }

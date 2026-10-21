#pragma once

// Kernel names for launches.
//
// A launch carries only a function handle. Its name comes from the driver
// (cuFuncGetName, or cuKernelGetName for a library kernel handle) or, failing
// that, from the name the program passed to cuModuleGetFunction or
// cuLibraryGetKernel, which cuscope hooks. Names are looked up when a function
// is first launched: by exit the driver may already have shut down.

#include <cstdint>

#include "model.hpp"

namespace cuscope {

// Called after each successful launch. Cheap when func was seen recently.
void note_kernel(uint64_t func);

// Name map from the hooks: a handle the program got back for `name`.
void remember_kernel_name(uint64_t handle, const char* name);
// cuKernelGetFunction: the function handle inherits the kernel's name.
void alias_kernel_name(uint64_t function, uint64_t kernel);

// At exit: the demangled name of every launched function that has one.
// Functions without one keep showing their handle.
void fill_kernel_names(Snapshot& s);

}  // namespace cuscope

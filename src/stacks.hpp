#pragma once

// Call-site stacks. Capturing is cheap (backtrace into a stack buffer, hash,
// deduplicate); turning addresses into names happens once, at exit.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "model.hpp"

namespace cuscope {

// The current stack below the caller of capture_stack, skipping `skip` more
// frames, at most `depth` frames (-1: CUSCOPE_STACK_DEPTH). Identical stacks
// get the same id. Returns 0 if nothing was captured.
uint32_t capture_stack(int skip, int depth = -1);

// The stack starting at the first frame outside libcuscope, so a wrapper's
// report points at the program (or the CUDA runtime it uses), never at cuscope.
uint32_t capture_caller_stack();

// Every captured stack, with each frame's module, offset and function name.
// Names come from the module's ELF symbol table when it has one (that covers
// functions the program does not export, like main) and from dladdr otherwise.
std::map<uint32_t, std::vector<StackFrame>> symbolize_all();

// __cxa_demangle, or the input unchanged when it is not a mangled name.
std::string demangle(const char* mangled);

// backtrace() may allocate on its first call; call this once during
// initialisation so it never does on a wrapper's path.
void warm_up_backtrace();

// pthread_atfork handlers: hold the stack-table lock across fork. The child
// keeps the parent's stacks (they are only addresses and stay valid).
void stacks_before_fork();
void stacks_after_fork();

}  // namespace cuscope

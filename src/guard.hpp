#pragma once

// Recursion guard. cuscope's own driver calls (resolving functions, creating
// and recording events, capturing stacks) run inside an InternalScope, and
// wrappers reached from inside one skip all instrumentation.
namespace cuscope {

// initial-exec: libcuscope is always loaded at startup (LD_PRELOAD), so its
// TLS is static and needs no __tls_get_addr call on the hot path.
inline thread_local int t_internal_depth __attribute__((tls_model("initial-exec"))) = 0;

inline bool in_internal() { return t_internal_depth != 0; }

struct InternalScope {
  InternalScope() { ++t_internal_depth; }
  ~InternalScope() { --t_internal_depth; }
  InternalScope(const InternalScope&) = delete;
  InternalScope& operator=(const InternalScope&) = delete;
};

}  // namespace cuscope

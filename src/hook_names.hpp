#pragma once

#include <cstdint>

// Hook ids and names from hooks.def, with no dependency on cuda.h or the
// loader, so the trace writer and reporter can name API calls in pure code.
namespace cuscope {

enum HookId : uint16_t {
#define HOOK(base, legacy_export, per_thread_export, version) H_##base,
#include "hooks.def"
#undef HOOK
  H_COUNT
};

// Base name of a hook, or "?" when out of range.
const char* hook_name(uint16_t id);

}  // namespace cuscope

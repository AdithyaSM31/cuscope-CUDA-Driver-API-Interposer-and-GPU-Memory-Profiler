#pragma once

#include <cstdint>

#include "guard.hpp"
#include "loader.hpp"

// The hook table built from hooks.def.
//
// Fixed data (names, exports, versions) is constexpr. Wrapper and real
// pointers live in zero-initialised arrays filled at run time, so nothing here
// needs a constructor: the table works before cuscope's constructors run.
namespace cuscope {

enum HookId : uint16_t {
#define HOOK(base, legacy_export, per_thread_export, version) H_##base,
#include "hooks.def"
#undef HOOK
  H_COUNT
};

// Variant 0 is the legacy-stream function, variant 1 the per-thread one.
constexpr int kLegacy = 0;
constexpr int kPerThread = 1;

struct HookSpec {
  const char* base;
  const char* export_name[2];  // [kPerThread] may be null
  int version;
};

const HookSpec& hook_spec(HookId id);
const char* hook_name(uint16_t id);  // base name, or "?" when out of range

// -1 when not hooked.
int find_hook_by_base(const char* base);
// -1 when no hook exports `symbol`; otherwise sets *variant.
int find_hook_by_export(const char* symbol, int* variant);

// Registers every wrapper once (pthread_once). Safe to call from dlsym.
void hook_table_init();
void set_wrappers(HookId id, void* legacy, void* per_thread);
void* hook_wrapper(HookId id, int variant);  // null if the hook has no wrapper

void set_real(HookId id, int variant, void* fn);
void* hook_real(HookId id, int variant);  // null until the loader resolves it

// The real driver function behind a wrapper. Loads the driver on first use,
// except inside cuscope's own calls, where waiting on the loader could
// deadlock; those resolve by export name instead. Null if the driver lacks it.
template <class Fn>
Fn real_fn(HookId id, int variant) {
  if (!loader_ready() && !in_internal()) ensure_loader();
  void* fn = hook_real(id, variant);
  if (fn == nullptr) fn = resolve_by_export(id, variant);
  return reinterpret_cast<Fn>(fn);
}

}  // namespace cuscope

#include "hook_table.hpp"

#include <cuda.h>
#include <pthread.h>

#include <atomic>
#include <cstring>

#include "log.hpp"
#include "wrappers/wrappers.hpp"

namespace cuscope {
namespace {

constexpr HookSpec kSpecs[H_COUNT] = {
#define HOOK(base, legacy_export, per_thread_export, version) \
  {#base, {legacy_export, per_thread_export}, version},
#include "hooks.def"
#undef HOOK
};

// Static storage without initialisers: zero before any code runs.
void* g_wrappers[H_COUNT][2];
std::atomic<void*> g_real[H_COUNT][2];

pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

void register_all() {
  register_alloc_hooks();
  register_memcpy_hooks();
  register_launch_hooks();
  register_teardown_hooks();
  for (int id = 0; id < H_COUNT; ++id) {
    if (g_wrappers[id][kLegacy] == nullptr) {
      log_msg("hook %s has no wrapper; its calls pass through unrecorded", kSpecs[id].base);
    }
  }
}

}  // namespace

const HookSpec& hook_spec(HookId id) { return kSpecs[id]; }

const char* hook_name(uint16_t id) { return id < H_COUNT ? kSpecs[id].base : "?"; }

int find_hook_by_base(const char* base) {
  for (int id = 0; id < H_COUNT; ++id) {
    if (std::strcmp(kSpecs[id].base, base) == 0) return id;
  }
  return -1;
}

int find_hook_by_export(const char* symbol, int* variant) {
  for (int id = 0; id < H_COUNT; ++id) {
    for (int v = kLegacy; v <= kPerThread; ++v) {
      const char* name = kSpecs[id].export_name[v];
      if (name != nullptr && std::strcmp(name, symbol) == 0) {
        *variant = v;
        return id;
      }
    }
  }
  return -1;
}

void hook_table_init() { pthread_once(&g_init_once, register_all); }

void set_wrappers(HookId id, void* legacy, void* per_thread) {
  g_wrappers[id][kLegacy] = legacy;
  g_wrappers[id][kPerThread] = per_thread;
}

void* hook_wrapper(HookId id, int variant) {
  hook_table_init();
  return g_wrappers[id][variant];
}

void set_real(HookId id, int variant, void* fn) { g_real[id][variant].store(fn, std::memory_order_release); }

void* hook_real(HookId id, int variant) { return g_real[id][variant].load(std::memory_order_acquire); }

}  // namespace cuscope

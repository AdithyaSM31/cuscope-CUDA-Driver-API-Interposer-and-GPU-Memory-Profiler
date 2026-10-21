#include "hook_names.hpp"

namespace cuscope {
namespace {

constexpr const char* kNames[H_COUNT] = {
#define HOOK(base, legacy_export, per_thread_export, version) #base,
#include "hooks.def"
#undef HOOK
};

}  // namespace

const char* hook_name(uint16_t id) { return id < H_COUNT ? kNames[id] : "?"; }

}  // namespace cuscope

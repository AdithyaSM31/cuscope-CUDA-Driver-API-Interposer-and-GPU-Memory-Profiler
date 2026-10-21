#include "config.hpp"
#include "log.hpp"

#include <unistd.h>

namespace cuscope {
namespace {

// Runs when the dynamic loader initialises libcuscope.so, before main().
// That is not necessarily before other libraries call into cuscope: libraries
// that libcuscope does not depend on (such as the WSL libcuda shim) may run
// their own constructors first and call dlsym. Code reachable from the hooks
// therefore must not rely on anything this constructor sets up.
__attribute__((constructor)) void cuscope_ctor() {
  const Config& cfg = config();
  log_msg("cuscope %s loaded (pid %d, mode %s)", CUSCOPE_VERSION,
          static_cast<int>(getpid()), mode_name(cfg.mode));
  for (const std::string& warning : cfg.warnings) log_msg("%s", warning.c_str());
}

}  // namespace
}  // namespace cuscope

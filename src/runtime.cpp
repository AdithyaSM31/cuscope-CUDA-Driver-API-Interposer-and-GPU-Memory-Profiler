#include "config.hpp"
#include "guard.hpp"
#include "log.hpp"
#include "model.hpp"
#include "stacks.hpp"
#include "trace_writer.hpp"
#include "tracker.hpp"

#include <unistd.h>

#include <cstdlib>

namespace cuscope {
namespace {

// Builds the snapshot and writes the trace. Runs once, from atexit.
void write_report_at_exit() {
  InternalScope internal;
  Snapshot s;
  s.pid = static_cast<int>(getpid());
  s.mode = mode_name(config().mode);
  tracker().finalize();
  tracker().fill(s);
  s.stacks = symbolize_all();
  write_trace_file(s, expand_out_path(config().out, s.pid));
}

// Runs when the dynamic loader initialises libcuscope.so, before main().
// That is not necessarily before other libraries call into cuscope: libraries
// that libcuscope does not depend on (such as the WSL libcuda shim) may run
// their own constructors first and call dlsym. Code reachable from the hooks
// therefore must not rely on anything this constructor sets up.
__attribute__((constructor)) void cuscope_ctor() {
  const Config& cfg = config();
  log_msg("cuscope %s loaded (pid %d, mode %s)", CUSCOPE_VERSION, static_cast<int>(getpid()),
          mode_name(cfg.mode));
  for (const std::string& warning : cfg.warnings) log_msg("%s", warning.c_str());

  warm_up_backtrace();
  tracker();  // create it now rather than inside the first allocation

  // Registered after config() and tracker() are built, so this handler runs
  // before their destructors (exit runs handlers in reverse order). The CUDA
  // runtime registers its own cleanup later, so that cleanup runs first.
  std::atexit(write_report_at_exit);
}

}  // namespace
}  // namespace cuscope

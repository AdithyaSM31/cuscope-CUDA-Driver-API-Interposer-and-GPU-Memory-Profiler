#include "config.hpp"
#include "gputimer.hpp"
#include "guard.hpp"
#include "log.hpp"
#include "model.hpp"
#include "names.hpp"
#include "recorder.hpp"
#include "report.hpp"
#include "stacks.hpp"
#include "stats.hpp"
#include "summary.hpp"
#include "trace_writer.hpp"
#include "tracker.hpp"

#include <pthread.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace cuscope {
namespace {

// Builds the snapshot and writes the trace. Runs once, from atexit.
void write_report_at_exit() {
  if (!stats::saw_cuda()) return;  // nothing to report
  InternalScope internal;
  Snapshot s;
  s.pid = static_cast<int>(getpid());
  s.mode = mode_name(config().mode);
  s.sample_every = config().sample_every;
  collector_stop_and_drain(s);  // first, so no record arrives after the drain
  gputimer_finish();            // then resolve events of contexts still alive
  gputimer_fill(s);
  fill_kernel_names(s);
  tracker().finalize();
  tracker().fill(s);
  s.stacks = symbolize_all();

  const std::string path = expand_out_path(config().out, s.pid);
  if (write_trace_file(s, path)) log_msg("trace written to %s", path.c_str());

  // One write, so the summary is not interleaved with the program's stderr.
  char* text = nullptr;
  size_t length = 0;
  if (FILE* out = open_memstream(&text, &length)) {
    print_summary(s, summarize(s), out);
    std::fclose(out);
    log_write(text, length);
    std::free(text);
  }
}

// fork() copies only the calling thread. Every cuscope lock is taken before
// the fork, so none can be held by a thread that does not exist in the child;
// the child then starts with empty state of its own.
void before_fork() {
  stacks_before_fork();
  recorder_before_fork();
}

void after_fork_parent() {
  recorder_after_fork_parent();
  stacks_after_fork();
}

void after_fork_child() {
  recorder_after_fork_child();
  stacks_after_fork();
  tracker_after_fork_child();
  stats::after_fork_child();
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
  pthread_atfork(before_fork, after_fork_parent, after_fork_child);

  // Registered after config() and tracker() are built, so this handler runs
  // before their destructors (exit runs handlers in reverse order). The CUDA
  // runtime registers its own cleanup later, so that cleanup runs first.
  std::atexit(write_report_at_exit);
}

}  // namespace
}  // namespace cuscope

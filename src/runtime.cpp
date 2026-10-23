#include "config.hpp"
#include "gputimer.hpp"
#include "guard.hpp"
#include "limits.hpp"
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

// Which process of a run owns a CUSCOPE_OUT path without %p: the first one
// to do CUDA work (not a shell that merely started it). A forked child of a
// process never owns it. A process that claims it puts its pid in
// CUSCOPE_ROOT_PID, which programs it later runs inherit.
bool g_forked_child = false;

int thread_count() {
  FILE* status = std::fopen("/proc/self/status", "r");
  if (status == nullptr) return -1;
  char line[128];
  int threads = -1;
  while (std::fgets(line, sizeof(line), status) != nullptr) {
    if (std::sscanf(line, "Threads: %d", &threads) == 1) break;
  }
  std::fclose(status);
  return threads;
}

void claim_root() {
  if (g_forked_child || std::getenv("CUSCOPE_ROOT_PID") != nullptr) return;
  // setenv is not safe while other threads may read the environment; a
  // program that is already multithreaded keeps its exact path but does not
  // pass the claim on to programs it runs.
  if (thread_count() != 1) return;
  char pid[16];
  std::snprintf(pid, sizeof(pid), "%d", static_cast<int>(getpid()));
  setenv("CUSCOPE_ROOT_PID", pid, /*overwrite=*/0);
}

bool is_root_process() {
  if (g_forked_child) return false;
  const char* root = std::getenv("CUSCOPE_ROOT_PID");
  return root == nullptr || std::atoi(root) == static_cast<int>(getpid());
}

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
  mem_limit().fill(s);
  s.stacks = symbolize_all();

  const std::string path = expand_out_path(config().out, s.pid, is_root_process());
  if (path != expand_out_path(config().out, s.pid)) {
    log_msg("another process of this run writes %s; this one writes %s (use %%p in CUSCOPE_OUT for one file "
            "per process)",
            config().out.c_str(), path.c_str());
  }
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
  gputimer_before_fork();
  names_before_fork();
}

void after_fork_parent() {
  names_after_fork();
  gputimer_after_fork_parent();
  recorder_after_fork_parent();
  stacks_after_fork();
}

void after_fork_child() {
  names_after_fork();
  gputimer_after_fork_child();
  recorder_after_fork_child();
  stacks_after_fork();
  tracker_after_fork_child();
  mem_limit_after_fork_child();
  stats::after_fork_child();
  g_forked_child = true;
}

// At the first CUDA work in this process rather than at load, so the many
// processes that inherit LD_PRELOAD and never use CUDA stay silent.
void on_first_work() {
  if (mem_limit().enabled()) {
    log_msg("cuscope %s loaded (pid %d, mode %s, mem limit %s)", CUSCOPE_VERSION, static_cast<int>(getpid()),
            mode_name(config().mode), bytes_text(mem_limit().limit()).c_str());
  } else {
    log_msg("cuscope %s loaded (pid %d, mode %s)", CUSCOPE_VERSION, static_cast<int>(getpid()),
            mode_name(config().mode));
  }
  claim_root();
  if (mem_limit().enabled() && expandable_segments_enabled([](const char* key) { return std::getenv(key); })) {
    log_msg("PYTORCH_CUDA_ALLOC_CONF turns on expandable_segments: PyTorch then allocates through "
            "cuMemCreate, which is not capped. Set expandable_segments:False for CUSCOPE_MEM_LIMIT to apply.");
  }
}

// Runs when the dynamic loader initialises libcuscope.so, before main().
// That is not necessarily before other libraries call into cuscope: libraries
// that libcuscope does not depend on (such as the WSL libcuda shim) may run
// their own constructors first and call dlsym. Code reachable from the hooks
// therefore must not rely on anything this constructor sets up.
__attribute__((constructor)) void cuscope_ctor() {
  const Config& cfg = config();
  for (const std::string& warning : cfg.warnings) log_msg("%s", warning.c_str());
  set_on_first_work(on_first_work);

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

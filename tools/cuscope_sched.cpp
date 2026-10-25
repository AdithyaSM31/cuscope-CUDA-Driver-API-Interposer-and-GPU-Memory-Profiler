// cuscope-sched: the compute-sharing daemon.
//
//   cuscope-sched [--verbose]   run in the foreground
//   cuscope-sched --status      print the registered jobs of the running daemon
//
// It owns the shared-memory segment (sched_shm.hpp) that jobs started with
// CUSCOPE_SHARE register in. Every millisecond it hands the elapsed time out as
// GPU-time credit to the active jobs, split by weight (sched_core.cpp); every
// 100 ms it frees the slots of jobs that have died. It loads no CUDA: jobs
// measure their own GPU time and charge it to their balance.
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>

#include "sched_core.hpp"

using namespace cuscope;

namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

bool alive(int32_t pid) { return pid > 0 && (kill(pid, 0) == 0 || errno == EPERM); }

// Maps an existing segment, or returns nullptr.
SchedShm* map_existing(const std::string& name, bool writable) {
  const int fd = shm_open(name.c_str(), writable ? O_RDWR : O_RDONLY, 0);
  if (fd < 0) return nullptr;
  struct stat st {};
  void* p = MAP_FAILED;
  if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= sizeof(SchedShm)) {
    p = mmap(nullptr, sizeof(SchedShm), writable ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
  }
  close(fd);
  return p == MAP_FAILED ? nullptr : static_cast<SchedShm*>(p);
}

// A segment a live daemon is serving.
bool serving(const SchedShm* shm) {
  return shm != nullptr && shm->magic == kSchedMagic && shm->version == kSchedVersion && shm->closed.load() == 0 &&
         alive(shm->daemon_pid.load());
}

int print_status(const std::string& name) {
  SchedShm* shm = map_existing(name, /*writable=*/false);
  if (!serving(shm)) {
    std::fprintf(stderr, "cuscope-sched: no scheduler running\n");
    return 1;
  }
  for (int i = 0; i < kSchedSlots; ++i) {
    const SchedSlot& s = shm->slots[i];
    const int32_t pid = s.pid.load();
    if (pid == 0) continue;
    std::printf("%d %d %u %lld %u %u\n", i, pid, s.weight.load(), static_cast<long long>(s.balance_ns.load()),
                s.solo.load(), s.waiting.load());
  }
  return 0;
}

// Creates the segment, or takes over one whose daemon has died. nullptr (and a
// message) if another daemon is serving it or it cannot be created.
SchedShm* create(const std::string& name) {
  if (SchedShm* existing = map_existing(name, /*writable=*/true)) {
    if (serving(existing)) {
      std::fprintf(stderr, "cuscope-sched: already running (pid %d)\n", existing->daemon_pid.load());
      return nullptr;
    }
    // Left behind by a daemon that crashed: start it afresh.
    sched_shm_init(existing, static_cast<int32_t>(getpid()), now_ns());
    return existing;
  }
  const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) {
    std::fprintf(stderr, "cuscope-sched: cannot create %s: %s\n", name.c_str(), std::strerror(errno));
    return nullptr;
  }
  void* p = MAP_FAILED;
  if (ftruncate(fd, sizeof(SchedShm)) == 0) {
    p = mmap(nullptr, sizeof(SchedShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  }
  close(fd);
  if (p == MAP_FAILED) {
    std::fprintf(stderr, "cuscope-sched: cannot map %s: %s\n", name.c_str(), std::strerror(errno));
    shm_unlink(name.c_str());
    return nullptr;
  }
  auto* shm = static_cast<SchedShm*>(p);
  sched_shm_init(shm, static_cast<int32_t>(getpid()), now_ns());
  return shm;
}

void free_dead_slots(SchedShm* shm) {
  for (SchedSlot& s : shm->slots) {
    const int32_t pid = s.pid.load(std::memory_order_acquire);
    if (pid == 0 || alive(pid)) continue;
    s.waiting = 0;
    s.solo = 0;
    s.balance_ns = 0;
    s.last_launch_ns = 0;
    int32_t expected = pid;
    s.pid.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);  // last: the slot is free
  }
}

// --verbose: once a second, each job's share of the GPU time charged in that
// second and the time it spent waiting.
class Reporter {
 public:
  void report(const SchedShm* shm) {
    uint64_t charged[kSchedSlots] = {}, waited[kSchedSlots] = {}, total = 0;
    for (int i = 0; i < kSchedSlots; ++i) {
      const SchedSlot& s = shm->slots[i];
      const int32_t pid = s.pid.load();
      const uint64_t c = s.charged_ns.load(), w = s.wait_ns.load();
      if (pid != 0 && pid == pid_[i]) {
        charged[i] = c - charged_[i];
        waited[i] = w - waited_[i];
        total += charged[i];
      }
      pid_[i] = pid;
      charged_[i] = c;
      waited_[i] = w;
    }
    std::printf("%8s %6s %7s %9s\n", "pid", "weight", "share", "waited");
    for (int i = 0; i < kSchedSlots; ++i) {
      if (pid_[i] == 0) continue;
      const double share = total ? 100.0 * static_cast<double>(charged[i]) / static_cast<double>(total) : 0.0;
      std::printf("%8d %6u %6.1f%% %7.1f ms\n", pid_[i], shm->slots[i].weight.load(), share, waited[i] / 1e6);
    }
    std::fflush(stdout);
  }

 private:
  int32_t pid_[kSchedSlots] = {};
  uint64_t charged_[kSchedSlots] = {}, waited_[kSchedSlots] = {};
};

int run(const std::string& name, bool verbose) {
  SchedShm* shm = create(name);
  if (shm == nullptr) return 1;

  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  Reporter reporter;
  uint64_t last = now_ns(), last_reap = last, last_report = last;
  timespec deadline;
  clock_gettime(CLOCK_MONOTONIC, &deadline);
  while (!g_stop) {
    // Absolute deadlines: a late wake-up does not push later ticks back, and
    // whatever time passed is handed out in full.
    deadline.tv_nsec += kTickNs;
    if (deadline.tv_nsec >= 1'000'000'000) {
      deadline.tv_nsec -= 1'000'000'000;
      ++deadline.tv_sec;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr);
    const uint64_t now = now_ns();
    shm->heartbeat_ns.store(now, std::memory_order_release);
    apply_refill(shm, now, now - last);
    last = now;
    if (now - last_reap >= kReapEveryNs) {
      free_dead_slots(shm);
      last_reap = now;
    }
    if (verbose && now - last_report >= 1'000'000'000) {
      reporter.report(shm);
      last_report = now;
    }
  }
  shm->closed.store(1, std::memory_order_release);  // jobs that still map it run unthrottled
  shm_unlink(name.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false, status = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (std::strcmp(argv[i], "--status") == 0) {
      status = true;
    } else {
      std::fprintf(stderr, "usage: cuscope-sched [--verbose] [--status]\n");
      return 2;
    }
  }
  const std::string name = sched_shm_name();
  return status ? print_status(name) : run(name, verbose);
}

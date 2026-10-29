#pragma once

// The compute-sharing scheduler's shared memory: one POSIX segment per user,
// created by the daemon (cuscope-sched) and mapped by every job that runs with
// CUSCOPE_SHARE. Every field the other side changes is an atomic; the segment
// is only ever touched through this layout, so the daemon and the jobs must
// agree on kSchedVersion.

#include <atomic>
#include <cstdint>
#include <string>

namespace cuscope {

constexpr uint64_t kSchedMagic = 0x6375736368656421ULL;  // "cusched!"
constexpr uint32_t kSchedVersion = 1;
constexpr int kSchedSlots = 32;

constexpr int64_t kTickNs = 1'000'000;             // the daemon refills every 1 ms
constexpr int64_t kIdleTickNs = 10'000'000;         // tick every 10 ms when idle
constexpr uint64_t kActiveWindowNs = 50'000'000;   // launched within 50 ms = active
constexpr int64_t kCapNs = 20'000'000;             // at most 20 ms of credit
constexpr uint64_t kDefaultEstimateNs = 50'000;    // a launch never measured
constexpr uint64_t kWaitStepNs = 200'000;          // a waiting launch sleeps this long
constexpr uint64_t kStaleNs = 100'000'000;         // a heartbeat older than this: daemon gone
constexpr uint64_t kReapEveryNs = 100'000'000;     // the daemon frees dead jobs' slots
constexpr uint64_t kMaxWaitNs = 30'000'000'000;    // a single wait longer than this: give up throttling
constexpr int32_t kReapingPid = -1;                 // slot is being reaped; unclaimable

// One job. pid 0 means the slot is free.
struct SchedSlot {
  std::atomic<int32_t> pid;
  std::atomic<uint32_t> weight;
  std::atomic<int64_t> balance_ns;       // GPU time the job may still use; negative = in debt
  std::atomic<uint64_t> last_launch_ns;  // CLOCK_MONOTONIC, shared by all processes
  std::atomic<uint32_t> waiting;         // a launch is waiting for credit
  std::atomic<uint32_t> solo;            // the only active job: never waits
  std::atomic<uint64_t> charged_ns, waits, wait_ns;
};

struct SchedShm {
  uint64_t magic;
  uint32_t version;
  std::atomic<int32_t> daemon_pid;
  std::atomic<uint64_t> heartbeat_ns;
  std::atomic<uint32_t> closed;  // the daemon has stopped: jobs run unthrottled
  SchedSlot slots[kSchedSlots];
};

// "/cuscope-sched-<uid>", or CUSCOPE_SCHED_NAME when set (tests run their own).
std::string sched_shm_name();

// The segment must belong to this user and be closed to everyone else: any
// other user who could write it could hold this user's jobs.
bool sched_shm_private(int fd);

// A fresh segment: magic, version, the daemon's pid, empty slots.
void sched_shm_init(SchedShm* shm, int32_t daemon_pid, uint64_t now_ns);

}  // namespace cuscope

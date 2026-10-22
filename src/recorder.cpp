#include "recorder.hpp"

#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "guard.hpp"
#include "ring.hpp"
#include "stats.hpp"

namespace cuscope {
namespace {

constexpr size_t kRingRecords = 16384;  // 1 MiB of records per thread
constexpr long kCollectIntervalNs = 2'000'000;

struct ThreadRing {
  SpscRing<Record, kRingRecords> ring;
  uint32_t tid = 0;
};

// Rings are never freed: a thread that exits leaves its ring for the
// collector to drain. The lists are heap-allocated on first use (another
// library's constructor may make a CUDA call before ours runs) and never
// destroyed, so they survive into exit handlers. The fork child swaps in new ones.
std::mutex g_registry_mu;
std::vector<ThreadRing*>*& rings() {
  static std::vector<ThreadRing*>* list = new std::vector<ThreadRing*>();
  return list;
}

std::mutex g_collected_mu;  // the collector and the final drain both append
std::vector<Record>*& collected() {
  static std::vector<Record>* list = new std::vector<Record>();
  return list;
}

std::mutex g_collector_mu;  // starting and stopping the collector
std::atomic<bool> g_collector_running{false};
std::atomic<bool> g_collector_stop{false};
std::atomic<bool> g_collector_final{false};  // stopped at exit for good
pthread_t g_collector;

std::atomic<uint64_t> g_next_corr{1};
std::atomic<void (*)()> g_collector_poll{nullptr};
std::atomic<void (*)()> g_on_first_work{nullptr};

thread_local ThreadRing* t_ring __attribute__((tls_model("initial-exec"))) = nullptr;

// Moves everything currently in the rings into g_collected.
void drain_all() {
  std::vector<ThreadRing*> snapshot;
  {
    std::lock_guard<std::mutex> lock(g_registry_mu);
    snapshot = *rings();
  }
  std::lock_guard<std::mutex> lock(g_collected_mu);
  Record r;
  for (ThreadRing* tr : snapshot) {
    while (tr->ring.pop(r)) collected()->push_back(r);
  }
}

void* collector_main(void*) {
  InternalScope internal;  // nothing cuscope does here is the program's
  const timespec interval{0, kCollectIntervalNs};
  while (!g_collector_stop.load(std::memory_order_acquire)) {
    drain_all();
    if (auto poll = g_collector_poll.load(std::memory_order_acquire)) poll();
    nanosleep(&interval, nullptr);
  }
  return nullptr;
}

void start_collector() {
  std::lock_guard<std::mutex> lock(g_collector_mu);
  if (g_collector_running.load(std::memory_order_relaxed) || g_collector_final.load(std::memory_order_relaxed)) return;
  g_collector_stop.store(false, std::memory_order_relaxed);

  // The collector never handles the program's signals: those belong on the
  // program's own threads. A new thread inherits the creator's mask.
  sigset_t all, previous;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &previous);
  const bool started = pthread_create(&g_collector, nullptr, collector_main, nullptr) == 0;
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);

  g_collector_running.store(started, std::memory_order_release);
}

ThreadRing* this_thread_ring() {
  if (t_ring != nullptr) return t_ring;
  auto* tr = new ThreadRing();
  tr->tid = static_cast<uint32_t>(syscall(SYS_gettid));
  {
    std::lock_guard<std::mutex> lock(g_registry_mu);
    rings()->push_back(tr);
  }
  t_ring = tr;
  return tr;
}

}  // namespace

uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t next_corr() { return g_next_corr.fetch_add(1, std::memory_order_relaxed); }

void set_on_first_work(void (*hook)()) { g_on_first_work.store(hook, std::memory_order_release); }

bool collector_running() { return g_collector_running.load(std::memory_order_acquire); }

void set_collector_poll(void (*poll)()) { g_collector_poll.store(poll, std::memory_order_release); }

void record(Record r) {
  if (r.type != RecType::ApiCall && stats::work_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
    if (auto hook = g_on_first_work.load(std::memory_order_acquire)) hook();
  }
  ThreadRing* tr = this_thread_ring();
  if (!g_collector_running.load(std::memory_order_acquire)) start_collector();
  r.tid = tr->tid;
  tr->ring.push(r);
}

void collector_stop_and_drain(Snapshot& s) {
  {
    std::lock_guard<std::mutex> lock(g_collector_mu);
    if (g_collector_running.load(std::memory_order_acquire)) {
      g_collector_stop.store(true, std::memory_order_release);
      pthread_join(g_collector, nullptr);
      g_collector_running.store(false, std::memory_order_release);
    }
    g_collector_final.store(true, std::memory_order_relaxed);
  }
  drain_all();

  uint64_t drops = 0;
  {
    std::lock_guard<std::mutex> lock(g_registry_mu);
    for (ThreadRing* tr : *rings()) drops += tr->ring.drops();
  }
  std::lock_guard<std::mutex> lock(g_collected_mu);
  s.records = std::move(*collected());
  collected()->clear();
  s.dropped_records = drops;
}

// Take every lock before fork, so no other thread holds one at the moment
// the child is created; release them on both sides.
void recorder_before_fork() {
  g_collector_mu.lock();
  g_registry_mu.lock();
  g_collected_mu.lock();
}

void recorder_after_fork_parent() {
  g_collected_mu.unlock();
  g_registry_mu.unlock();
  g_collector_mu.unlock();
}

void recorder_after_fork_child() {
  // Only the forking thread exists in the child. The parent's rings, records
  // and collector thread are not the child's: start empty (the old objects
  // are leaked, since other threads' state may be mid-update).
  rings() = new std::vector<ThreadRing*>();
  collected() = new std::vector<Record>();
  t_ring = nullptr;
  g_collector_running.store(false, std::memory_order_relaxed);
  g_collector_stop.store(false, std::memory_order_relaxed);
  g_collector_final.store(false, std::memory_order_relaxed);
  g_collected_mu.unlock();
  g_registry_mu.unlock();
  g_collector_mu.unlock();
}

}  // namespace cuscope

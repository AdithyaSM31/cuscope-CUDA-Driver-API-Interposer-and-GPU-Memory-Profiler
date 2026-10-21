#pragma once

// Moves records off the program's threads.
//
// Each program thread gets its own SpscRing on its first record(); a
// background collector thread drains every ring every 2 ms into one list.
// The program thread's cost is a clock read and a 64-byte copy.

#include <cstdint>

#include "model.hpp"

namespace cuscope {

// CLOCK_MONOTONIC in nanoseconds (vDSO, no system call).
uint64_t now_ns();

// A process-wide id tying a call to its GPU span. Starts at 1; 0 means none.
uint64_t next_corr();

// Fills r.tid and pushes r to this thread's ring, creating the ring and
// starting the collector on first use. A full ring drops r and counts it.
void record(Record r);

// At exit: stops and joins the collector, drains every ring into s.records,
// and sets s.dropped_records.
void collector_stop_and_drain(Snapshot& s);

// pthread_atfork handlers (installed by runtime.cpp). The child keeps none of
// the parent's records, rings or collector.
void recorder_before_fork();
void recorder_after_fork_parent();
void recorder_after_fork_child();

}  // namespace cuscope

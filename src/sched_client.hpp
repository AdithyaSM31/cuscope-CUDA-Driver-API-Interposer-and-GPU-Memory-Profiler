#pragma once

// The job's side of compute sharing, wired into libcuscope.so: with
// CUSCOPE_SHARE set, the first kernel or graph launch maps the scheduler's
// segment and registers; every launch then goes through sched_before_launch.
// Without CUSCOPE_SHARE none of this does anything.

#include <cuda.h>

#include <cstdint>

#include "model.hpp"

namespace cuscope {

// Before a kernel or graph launch (key: its CUfunction or CUgraphExec).
// Returns the GPU time charged for it, 0 when not sharing.
uint64_t sched_before_launch(int variant, CUstream stream, uint64_t key);

// At exit: the job's sharing figures for the report.
void sched_fill(Snapshot& s);

// Exit handler: gives the slot back. Registered before the report's handler,
// so it runs after the report has read the slot's counters.
void sched_release_at_exit();

// fork(): the child is a new job; it must not release the parent's slot.
void sched_after_fork_child();

}  // namespace cuscope

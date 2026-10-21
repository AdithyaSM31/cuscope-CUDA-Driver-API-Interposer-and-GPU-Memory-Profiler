#pragma once

// GPU start and end times for launches (and, from Task 13, copies).
//
// Around each timed call cuscope records a CUDA event before and after it on
// the same stream. The collector thread later asks the driver whether the end
// event has completed and, once it has, converts both to CPU-clock times:
//
//   gpu_time = reference_cpu_ns + cuEventElapsedTime(reference_event, event)
//
// where the reference event was recorded and waited for when cuscope first
// saw the context. Events come from a per-context pool and go back to it once
// resolved, so steady state allocates nothing.

#include <cuda.h>

#include <cstdint>

#include "model.hpp"

namespace cuscope {

struct CtxState;

// The two events around one timed call.
struct Timing {
  CUevent start = nullptr;
  CUevent end = nullptr;
  CtxState* state = nullptr;
};

// Records the start event before the real call. False (and the call is
// counted as untimed) in alloc mode, inside cuscope, on a stream that is being
// captured into a graph, or when no event is available. `variant` is the
// wrapper's stream variant, so the event goes on the stream the call uses.
bool gpu_begin(int variant, CUstream stream, Timing* t);

// After a successful real call: records the end event and queues the pair.
void gpu_end(int variant, CUstream stream, const Timing& t, const GpuSpan& meta);

// After a failed real call: returns the events unused.
void gpu_cancel(const Timing& t);

// Before a context goes away: waits for and resolves its pending events, then
// forgets the context (its events die with it). Device form: every context
// cuscope knows on that device, for cuDevicePrimaryCtxRelease/Reset.
void gpu_flush_ctx(uint64_t ctx);
void gpu_flush_device(int device);

// At exit, after the collector has stopped: flush every context still known.
void gputimer_finish();

// Resolved spans, unresolved_events and gpu_untimed.
void gputimer_fill(Snapshot& s);

}  // namespace cuscope

#pragma once

// Writes a Snapshot as Chrome Trace Event JSON, which Perfetto opens directly.
//
//   API call        ph:X on the calling thread's track
//   kernel / copy   ph:X on a per-stream GPU track (tid 1000000 + stream index)
//   device memory   ph:C counter of Device + Managed + Async bytes in use
//
// The top-level object also carries a "cuscope" section with the report
// (peaks, leaks, invalid frees, stacks, kernels, copies, completeness counts),
// which the CLI and tests read. Pure C++: no CUDA.

#include <ostream>
#include <string>

#include "model.hpp"

namespace cuscope {

void write_trace(const Snapshot& s, std::ostream& out);

// False, with one log line, if the file cannot be written.
bool write_trace_file(const Snapshot& s, const std::string& path);

// s with JSON string escapes applied, without surrounding quotes.
std::string json_escape(const std::string& s);

}  // namespace cuscope

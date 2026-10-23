#pragma once

// The text summary printed at exit (to CUSCOPE_LOG or stderr, never stdout).
// cli/cuscope.py prints the same sections from a trace file, adding file:line.
//
//   == cuscope summary (pid N, mode M) ==
//   Memory / Leaks / Invalid frees / Top kernels / Copies
//   Completeness: dropped=X unresolved=Y untimed=Z

#include <cstdio>

#include "model.hpp"
#include "summary.hpp"

namespace cuscope {

void print_summary(const Snapshot& s, const Summary& sum, FILE* out);

// "512 B", "1.5 KiB", "64.0 MiB", "2.00 GiB": sizes as the summary prints them.
std::string bytes_text(uint64_t bytes);

}  // namespace cuscope

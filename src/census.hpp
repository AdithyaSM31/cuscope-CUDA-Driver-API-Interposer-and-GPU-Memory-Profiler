#pragma once

#include <cstdint>

// The symbol census: one TSV line per lookup that reaches cuscope, written to
// CUSCOPE_CENSUS when it is set. Safe to call from inside dlsym, before
// cuscope's constructors have run.
//
//   dlsym <TAB> symbol <TAB> caller module basename <TAB> action
//   gpa   <TAB> symbol <TAB> cudaVersion <TAB> flags <TAB> CUresult <TAB> symbolStatus <TAB> action
namespace cuscope {

bool census_enabled();

// caller_module may be a full path; only its basename is written. Null is "?".
void census_dlsym(const char* symbol, const char* caller_module, const char* action);

void census_gpa(const char* symbol, int cuda_version, uint64_t flags, int result, int status,
                const char* action);

}  // namespace cuscope

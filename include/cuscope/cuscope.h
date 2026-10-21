/* cuscope public C API. Tests use it to see what the interposer did. */
#ifndef CUSCOPE_CUSCOPE_H
#define CUSCOPE_CUSCOPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUSCOPE_API __attribute__((visibility("default")))

typedef struct cuscope_stats {
  uint64_t dlsym_wrapped;    /* wrappers handed out by the dlsym override */
  uint64_t gpa_wrapped;      /* wrappers handed out by the cuGetProcAddress hook */
  uint64_t gpa_passthrough;  /* hooked base names returned unwrapped (variant mismatch) */
  uint64_t direct_calls;     /* calls entering through exported driver symbols (path 1) */
  uint64_t hooked_calls;     /* wrapper invocations not marked internal */
  uint64_t live_bytes;
  uint64_t live_allocs;
  uint64_t invalid_frees;
} cuscope_stats;

/* Copies the current counters into *out. Returns 0, or -1 if out is NULL. */
CUSCOPE_API int cuscope_get_stats(cuscope_stats* out);

#ifdef __cplusplus
}
#endif

#endif /* CUSCOPE_CUSCOPE_H */

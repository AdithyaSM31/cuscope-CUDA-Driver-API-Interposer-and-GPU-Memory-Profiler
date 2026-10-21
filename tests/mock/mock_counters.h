/* Call counters kept by the mock libcuda, read back by tests. */
#ifndef CUSCOPE_MOCK_COUNTERS_H
#define CUSCOPE_MOCK_COUNTERS_H

#include <stdint.h>

/* Streams with this handle report CU_STREAM_CAPTURE_STATUS_ACTIVE. */
#define MOCK_CAPTURING_STREAM_HANDLE 0xCA97

typedef struct mock_counters {
  uint64_t alloc_v1;
  uint64_t alloc_v2;
  uint64_t free_;
  uint64_t memcpy_htod;
  uint64_t memcpy_dtoh;
  uint64_t memcpy_async;
  uint64_t memcpy_async_ptsz;
  uint64_t launch;
  uint64_t launch_ptsz;
  uint64_t event_record;
  uint64_t event_record_on_capturing;
  uint64_t ctx_destroy;
  uint64_t event_query;
} mock_counters;

#ifdef __cplusplus
extern "C"
#endif
void mock_get_counters(mock_counters* out);

#endif /* CUSCOPE_MOCK_COUNTERS_H */

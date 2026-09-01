#ifndef __PERF_H
#define __PERF_H

#include <time.h>
#include <stdint.h>

typedef struct _transformer_perf_t {
	float prefill_ms;
	float decode_ms;
	uint32_t prefill_tokens;
	uint32_t decode_tokens;
} transformer_perf_t;

static inline uint64_t perf_now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

#endif /* __PERF_H */

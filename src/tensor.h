#ifndef __TENSOR_H
#define __TENSOR_H

#include <stdint.h>
#include "gguf.h"

typedef struct _tensor_t {
	const float *data;
	uint64_t dimensions[MAX_TENSOR_N_DIMS];
	uint32_t n_dimensions;
} tensor_t;

#endif /* __TENSOR_H */

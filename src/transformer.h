#ifndef __TRANSFORMER_H
#define __TRANSFORMER_H

#include <stdint.h>
#include "model.h"

typedef struct _transformer_ctx_t {
	float *qkv_buff;
} transformer_ctx_t;

void layer_norm(const float *restrict in, const float *restrict weight,
		const float *restrict bias, float *restrict out,
		uint64_t seq_len, uint64_t hidden_dim);

void proj(const float *restrict in, const tensor_t *weight,
	  const tensor_t *bias, float *restrict out, uint64_t seq_len);

#endif /* __TRANSFORMER_H */

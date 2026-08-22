#ifndef __TRANSFORMER_H
#define __TRANSFORMER_H

#include <stdint.h>
#include "model.h"

typedef struct _transformer_ctx_t {
	float *hidden;

	float *norm;
	float *qkv;
	float *scores;
	float *ffn;
	float *proj;

	float *logits;

	float *k_cache;
	float *v_cache;
	uint64_t cache_len;

	uint64_t seq_len;
} transformer_ctx_t;

transformer_ctx_t *transformer_create(model_t *model, uint64_t seq_len);
void transformer_destroy(transformer_ctx_t *ctx);
uint32_t transformer_forward(transformer_ctx_t *ctx, model_t *model);

#endif /* __TRANSFORMER_H */

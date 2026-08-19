#ifndef __TRANSFORMER_H
#define __TRANSFORMER_H

#include <stdint.h>
#include "tensor.h"

typedef struct _transformer_ctx_t {
	float *hidden;

	float *norm;
	float *qkv;
	float *scores;
	float *ffn;
	float *proj;

	float *k_cache;
	float *v_cache;
	uint64_t cache_len;

	uint64_t curr_token;
	uint64_t seq_len;
} transformer_ctx_t;

typedef struct _transformer_blk_t {
	tensor_t attn_qkv_b;
	tensor_t attn_qkv_w;
	tensor_t attn_output_b;
	tensor_t attn_output_w;
	tensor_t attn_norm_b;
	tensor_t attn_norm_w;
	tensor_t ffn_up_b;
	tensor_t ffn_up_w;
	tensor_t ffn_down_b;
	tensor_t ffn_down_w;
	tensor_t ffn_norm_b;
	tensor_t ffn_norm_w;
} transformer_blk_t;

void layer_norm(const float *restrict in, const float *restrict weight,
		const float *restrict bias, float *restrict out,
		uint64_t seq_len, uint64_t hidden_dim);

void proj(const float *restrict in, const tensor_t *weight,
	  const tensor_t *bias, float *restrict out, uint64_t seq_len);

#endif /* __TRANSFORMER_H */

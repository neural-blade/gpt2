#ifndef __MODEL_H
#define __MODEL_H

#include <stdint.h>
#include "gguf.h"
#include "perf.h"

typedef struct _tensor_t {
	const float *data;
	uint64_t dimensions[MAX_TENSOR_N_DIMS];
	uint32_t n_dimensions;
} tensor_t;

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

typedef struct _model_t {
	transformer_blk_t *blocks;
	uint32_t block_count;

	tensor_t output_norm_b;
	tensor_t output_norm_w;
	tensor_t token_embd_w;
	tensor_t pos_embd_w;

	char **vocab;
	uint32_t vocab_count;

	uint32_t n_ctx;
	uint32_t n_embd;
	uint32_t n_head;
	uint32_t head_len;
	uint32_t n_ff;
	float epsilon;
} model_t;

int model_init(model_t *model, const gguf_file_t *file);
void model_free(model_t *model);
void model_run(model_t *model, uint32_t *token_ids, uint32_t token_count,
	       uint32_t max_tokens, transformer_perf_t *stats);

#endif /* __MODEL_H */

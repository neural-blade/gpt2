#ifndef __MODEL_H
#define __MODEL_H

#include <stdint.h>
#include "gguf.h"

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
	uint32_t blocks_count;

	tensor_t output_norm_b;
	tensor_t output_norm_w;
	tensor_t token_embd_w;
	tensor_t pos_embd_w;
} model_t;

int model_init(model_t *model, const gguf_file_t *file);
void model_free(model_t *model);

#endif /* __MODEL_H */

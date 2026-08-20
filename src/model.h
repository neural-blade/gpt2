#ifndef __MODEL_H
#define __MODEL_H

#include <stdint.h>
#include "gguf.h"
#include "tensor.h"
#include "transformer.h"

typedef struct _model_t {
	transformer_blk_t *blocks;
	uint32_t block_count;

	tensor_t output_norm_b;
	tensor_t output_norm_w;
	tensor_t token_embd_w;
	tensor_t pos_embd_w;

	uint32_t n_ctx;
	uint32_t n_embd;
	uint32_t n_head;
	uint32_t n_ff;
} model_t;

int model_init(model_t *model, const gguf_file_t *file);
void model_free(model_t *model);

#endif /* __MODEL_H */

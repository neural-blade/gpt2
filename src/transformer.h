#ifndef __TRANSFORMER_H
#define __TRANSFORMER_H

#include <stdint.h>
#include "model.h"

typedef struct _transformer_ctx_t transformer_ctx_t;

transformer_ctx_t *transformer_create(model_t *model, uint32_t *token_ids,
				      uint64_t seq_len);
void transformer_destroy(transformer_ctx_t *ctx);
uint32_t transformer_get_token(transformer_ctx_t *ctx, uint32_t idx);
void transformer_forward(transformer_ctx_t *ctx, model_t *model);

#endif /* __TRANSFORMER_H */

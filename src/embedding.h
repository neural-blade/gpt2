#ifndef __EMBEDDING_H
#define __EMBEDDING_H

#include "tensor.h"

void get_embd(const uint32_t *token_ids, const tensor_t *token_embd_w,
	      const tensor_t *pos_embd_w, float *embd, uint32_t token_count,
	      uint32_t initial_token);

#endif /* __EMBEDDING_H */

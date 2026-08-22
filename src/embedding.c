#include <stdint.h>
#include <string.h>
#include "tensor.h"
#include "linalg.h"

static void token_embd(const uint32_t *token_ids, const tensor_t *token_embd_w,
		       float *embd, uint32_t token_count)
{
	uint32_t hidden_dim = token_embd_w->dimensions[0];
	for (uint32_t i = 0; i < token_count; ++i)
		memcpy(&embd[i * hidden_dim],
		       &token_embd_w->data[token_ids[i] * hidden_dim],
		       hidden_dim * sizeof(*embd));
}

static void pos_embd(float *embd, const tensor_t *pos_embd_w,
		     uint32_t token_count, uint32_t initial_token)
{
	uint32_t hidden_dim = pos_embd_w->dimensions[0];
	for (uint32_t i = 0; i < token_count; ++i)
		add_f32v_inplace(
		    &embd[i * hidden_dim],
		    &pos_embd_w->data[(initial_token + i) * hidden_dim], 1.0f,
		    hidden_dim);
}

void get_embd(const uint32_t *token_ids, const tensor_t *token_embd_w,
	      const tensor_t *pos_embd_w, float *embd, uint32_t token_count,
	      uint32_t initial_token)
{
	token_embd(token_ids, token_embd_w, embd, token_count);
	pos_embd(embd, pos_embd_w, token_count, initial_token);
}

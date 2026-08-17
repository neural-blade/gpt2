#include <math.h>
#include "model.h"
#include "transformer.h"
#include "linalg.h"

void layer_norm(const float *restrict in, const float *restrict weight,
		const float *restrict bias, float *restrict out,
		uint64_t seq_len, uint64_t hidden_dim)
{
	for (uint64_t i = 0; i < seq_len; ++i) {
		const float *x = &in[i * hidden_dim];
		float *a       = &out[i * hidden_dim];

		float sum      = 0.0f;
		for (uint64_t i = 0; i < hidden_dim; ++i) sum += x[i];
		float mean   = sum / hidden_dim;

		float sum_sq = 0.0f;
		for (uint64_t i = 0; i < hidden_dim; ++i)
			sum_sq += (x[i] - mean) * (x[i] - mean);
		float var     = sum_sq / hidden_dim;

		float inv_std = 1.0f / sqrtf(var + 1e-5f);

		for (uint64_t i = 0; i < hidden_dim; ++i)
			a[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
	}
}

void proj(const float *restrict in, const tensor_t *weight,
	  const tensor_t *bias, float *restrict out, uint64_t seq_len)
{
	uint64_t hidden_dim = weight->dimensions[0];
	uint64_t out_dim    = weight->dimensions[1];

	gemm_f32(in, weight->data, out, seq_len, out_dim, hidden_dim);
	for (uint64_t i = 0; i < seq_len; ++i)
		add_f32v_inplace(&out[i * out_dim], bias->data, out_dim);
}

void transformer_block_forward(transformer_ctx_t *ctx, transformer_blk_t *block)
{
}

void transformer_forward(transformer_ctx_t *ctx, transformer_blk_t *blocks,
			 uint32_t blocks_count)
{
	for (uint32_t i = 0; i < blocks_count; ++i)
		transformer_block_forward(ctx, &blocks[i]);
}

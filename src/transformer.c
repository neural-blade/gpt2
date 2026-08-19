#include <math.h>
#include <string.h>
#include "transformer.h"
#include "linalg.h"
#include "model.h"

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

	gemm_f32(in, weight->data, out, 1.0f, seq_len, out_dim, hidden_dim);
	for (uint64_t i = 0; i < seq_len; ++i)
		add_f32v_inplace(&out[i * out_dim], bias->data, 1.0f, out_dim);
}

void attn_scores(const float *restrict q, const float *restrict k,
		 float *restrict scores, uint64_t q_len, uint64_t k_len,
		 uint64_t heads_count, uint64_t head_len)
{
	float inv_sqrt = 1.0f / sqrtf((float)head_len);

	for (uint64_t i = 0; i < q_len; ++i) {
		uint64_t token_offset = ((i * (i + 1)) / 2) * heads_count;
		for (uint64_t j = 0; j < heads_count; ++j) {
			uint64_t head_offset = token_offset + k_len * j;
			for (uint64_t t = 0; t < k_len; ++t) {
				scores[head_offset + t] =
				    dot_f32v(&q[i * heads_count * 3 * head_len +
						j * head_len],
					     &k[t * heads_count * head_len +
						j * head_len],
					     head_len) *
				    inv_sqrt;
			}
		}
		++k_len;
	}
}

void kv_cache_append(const float *qkv, float *k, float *v, uint64_t cache_len,
		     uint64_t hidden_len, uint64_t n_tokens)
{
	for (uint64_t i = 0; i < n_tokens; ++i) {
		uint64_t last_offset = cache_len * hidden_len;
		uint32_t t_offset    = i * hidden_len;
		uint64_t k_offset    = t_offset * 3 + hidden_len;

		memcpy(k + last_offset + t_offset, &qkv[k_offset], hidden_len);
		memcpy(v + last_offset + t_offset, &qkv[k_offset + hidden_len],
		       hidden_len);
	}
}

float max(const float *restrict v, uint64_t len)
{
	float max = v[0];
	for (uint64_t i = 0; i < len; ++i)
		if (v[i] > max) max = v[i];
	return max;
}

void softmax(float *restrict scores, uint64_t n_head, uint64_t initial_token,
	     uint64_t n_token)
{
	uint64_t initial_skip = (initial_token * (initial_token + 1)) / 2;

	for (uint64_t i = 0; i < n_token; ++i) {
		uint64_t curr_token = initial_token + i;
		uint64_t t_offset   = (((curr_token * (curr_token + 1)) / 2) -
				       initial_skip) *
				    n_head;
		uint64_t curr_head_len = curr_token + 1;

		for (uint64_t j = 0; j < n_head; ++j) {
			uint64_t h_offset = j * curr_head_len;
			float *row	  = &scores[t_offset + h_offset];

			float max_val	  = max(row, curr_head_len);
			float sum	  = 0.0f;
			for (uint64_t k = 0; k < curr_head_len; ++k) {
				row[k] = expf(row[k] - max_val);
				sum += max_val;
			}

			float inv_sum = 1.0f / sum;
			for (uint64_t k = 0; k < curr_head_len; ++k)
				row[k] *= inv_sum;
		}
	}
}

void attn_v_weighted_sum(const float *restrict p, const float *restrict v,
			 float *restrict out, uint64_t p_len, uint64_t v_len,
			 uint64_t n_head, uint64_t head_len)
{
	uint64_t t_len = n_head * head_len;

	for (uint64_t i = 0; i < p_len * t_len; ++i) out[i] = 0.0f;

	for (uint64_t i = 0; i < p_len; ++i) {
		uint64_t t_offset = (i * (i + 1) / 2) * n_head;
		for (uint64_t j = 0; j < n_head; ++j) {
			uint64_t h_offset = j * head_len;

			for (uint64_t k = 0; k < v_len; ++k) {
				add_f32v_inplace(&out[i * t_len + h_offset],
						 &v[k * t_len + h_offset],
						 p[t_offset + j * v_len + k],
						 head_len);
			}
		}

		++v_len;
	}
}

void transformer_forward(transformer_ctx_t *ctx, model_t *model,
			 float *embeddings, uint64_t seq_len)
{
	memcpy(ctx->hidden, embeddings, model->n_embd);
	for (uint32_t i = 0; i < model->block_count; ++i) {
		transformer_blk_t block = model->blocks[i];
		uint64_t head_len	= model->n_embd / model->n_head;

		// Attention
		layer_norm(ctx->hidden, block.attn_norm_w.data,
			   block.attn_norm_b.data, ctx->norm, seq_len,
			   model->n_embd);
		proj(ctx->norm, &block.attn_qkv_w, &block.attn_qkv_b, ctx->qkv,
		     seq_len);

		uint64_t c_offset = i * model->n_ctx * model->n_embd;
		kv_cache_append(ctx->qkv, &ctx->k_cache[c_offset],
				&ctx->v_cache[c_offset], ctx->cache_len,
				model->n_embd, seq_len);

		attn_scores(ctx->qkv, &ctx->k_cache[c_offset], ctx->scores,
			    seq_len, ctx->cache_len, model->n_head, head_len);

		softmax(ctx->scores, model->n_head, ctx->curr_token,
			ctx->cache_len);

		attn_v_weighted_sum(ctx->scores, &ctx->v_cache[c_offset],
				    ctx->norm, seq_len, ctx->cache_len,
				    model->n_head, head_len);

		proj(ctx->norm, &block.attn_output_w, &block.attn_output_b,
		     ctx->qkv, seq_len);

		add_f32v(ctx->hidden, ctx->qkv, ctx->norm,
			 seq_len * model->n_embd);

		// FFN
	}

	ctx->cache_len += seq_len;
}

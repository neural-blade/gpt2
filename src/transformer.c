#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "transformer.h"
#include "linalg.h"
#include "model.h"

transformer_ctx_t *transformer_create(model_t *model, uint64_t seq_len)
{
	transformer_ctx_t *ctx = malloc(sizeof(*ctx));

	ctx->hidden    = malloc(model->n_embd * seq_len * sizeof(float));
	ctx->norm      = malloc(model->n_embd * seq_len * sizeof(float));
	ctx->qkv       = malloc(3 * model->n_embd * seq_len * sizeof(float));
	ctx->scores    = malloc(model->n_ctx * model->n_head * seq_len *
				sizeof(float));
	ctx->ffn       = malloc(model->n_embd * seq_len * sizeof(float));
	ctx->proj      = malloc(model->n_ff * seq_len * sizeof(float));
	ctx->logits    = malloc(model->vocab_count * sizeof(float));
	ctx->k_cache   = malloc(model->block_count * model->n_ctx *
				model->n_embd * sizeof(float));
	ctx->v_cache   = malloc(model->block_count * model->n_ctx *
				model->n_embd * sizeof(float));

	ctx->cache_len = 0;
	ctx->seq_len   = seq_len;

	return ctx;
}

void transformer_destroy(transformer_ctx_t *ctx)
{
	free(ctx->hidden);
	free(ctx->norm);
	free(ctx->qkv);
	free(ctx->scores);
	free(ctx->ffn);
	free(ctx->proj);
	free(ctx->logits);
	free(ctx->k_cache);
	free(ctx->v_cache);
	free(ctx);
}

static void layer_norm(const float *restrict in, const float *restrict weight,
		       const float *restrict bias, float *restrict out,
		       uint64_t seq_len, uint64_t hidden_dim, float eps)
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

		float inv_std = 1.0f / sqrtf(var + eps);

		for (uint64_t i = 0; i < hidden_dim; ++i)
			a[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
	}
}

static void proj(const float *restrict in, const tensor_t *weight,
		 const tensor_t *bias, float *restrict out, uint64_t seq_len)
{
	uint64_t hidden_dim = weight->dimensions[0];
	uint64_t out_dim    = weight->dimensions[1];

	gemm_f32(in, weight->data, out, 1.0f, seq_len, out_dim, hidden_dim);
	for (uint64_t i = 0; i < seq_len; ++i)
		add_f32v_inplace(&out[i * out_dim], bias->data, 1.0f, out_dim);
}

static void attn_scores(const float *restrict q, const float *restrict k,
			float *restrict scores, uint64_t q_len, uint64_t k_len,
			uint64_t heads_count, uint64_t head_len)
{
	float inv_sqrt	= 1.0f / sqrtf((float)head_len);
	uint64_t anchor = k_len;

	for (uint64_t i = 0; i < q_len; ++i) {
		uint64_t g	      = anchor + i;
		uint64_t nk	      = g + 1;
		uint64_t token_offset = (((g * (g + 1)) / 2) -
					 ((anchor * (anchor + 1)) / 2)) *
					heads_count;
		for (uint64_t j = 0; j < heads_count; ++j) {
			uint64_t head_offset = token_offset + nk * j;
			for (uint64_t t = 0; t < nk; ++t) {
				scores[head_offset + t] =
				    dot_f32v(&q[i * heads_count * 3 * head_len +
						j * head_len],
					     &k[t * heads_count * head_len +
						j * head_len],
					     head_len) *
				    inv_sqrt;
			}
		}
	}
}

static void kv_cache_append(const float *qkv, float *k, float *v,
			    uint64_t cache_len, uint64_t hidden_len,
			    uint64_t n_tokens)
{
	for (uint64_t i = 0; i < n_tokens; ++i) {
		uint64_t last_offset = cache_len * hidden_len;
		uint32_t t_offset    = i * hidden_len;
		uint64_t k_offset    = t_offset * 3 + hidden_len;

		memcpy(k + last_offset + t_offset, &qkv[k_offset],
		       hidden_len * sizeof(*k));
		memcpy(v + last_offset + t_offset, &qkv[k_offset + hidden_len],
		       hidden_len * sizeof(*v));
	}
}

static uint32_t max_id(const float *restrict v, uint64_t len)
{
	float max   = v[0];
	uint32_t id = 0;
	for (uint32_t i = 1; i < len; ++i)
		if (v[i] > max) {
			max = v[i];
			id  = i;
		}
	return id;
}

static void softmax(float *restrict scores, uint64_t n_head,
		    uint64_t initial_token, uint64_t n_token)
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

			float max_val	  = row[max_id(row, curr_head_len)];
			float sum	  = 0.0f;
			for (uint64_t k = 0; k < curr_head_len; ++k) {
				row[k] = expf(row[k] - max_val);
				sum += row[k];
			}

			float inv_sum = 1.0f / sum;
			for (uint64_t k = 0; k < curr_head_len; ++k)
				row[k] *= inv_sum;
		}
	}
}

static void attn_v_weighted_sum(const float *restrict p,
				const float *restrict v, float *restrict out,
				uint64_t p_len, uint64_t v_len, uint64_t n_head,
				uint64_t head_len)
{
	uint64_t t_len	= n_head * head_len;
	uint64_t anchor = v_len;

	for (uint64_t i = 0; i < p_len * t_len; ++i) out[i] = 0.0f;

	for (uint64_t i = 0; i < p_len; ++i) {
		uint64_t g	      = anchor + i;
		uint64_t nk	      = g + 1;
		uint64_t token_offset = (((g * (g + 1)) / 2) -
					 ((anchor * (anchor + 1)) / 2)) *
					n_head;
		for (uint64_t j = 0; j < n_head; ++j) {
			uint64_t h_offset = j * head_len;

			for (uint64_t k = 0; k < nk; ++k) {
				add_f32v_inplace(&out[i * t_len + h_offset],
						 &v[k * t_len + h_offset],
						 p[token_offset + j * nk + k],
						 head_len);
			}
		}
	}
}

static void gelu_actv(float *restrict x, uint64_t len)
{
	for (uint64_t i = 0; i < len; ++i)
		x[i] = 0.5 * x[i] *
		       (1 + tanhf(0.7978845608f * x[i] +
				  0.0356774081f * x[i] * x[i] * x[i]));
}

void transformer_forward(transformer_ctx_t *ctx, model_t *model,
			 uint32_t *gen_token)
{
	for (uint32_t i = 0; i < model->block_count; ++i) {
		transformer_blk_t block = model->blocks[i];
		uint64_t head_len	= model->n_embd / model->n_head;

		// Attention
		layer_norm(ctx->hidden, block.attn_norm_w.data,
			   block.attn_norm_b.data, ctx->norm, ctx->seq_len,
			   model->n_embd, model->epsilon);

		proj(ctx->norm, &block.attn_qkv_w, &block.attn_qkv_b, ctx->qkv,
		     ctx->seq_len);

		uint64_t c_offset = i * model->n_ctx * model->n_embd;
		kv_cache_append(ctx->qkv, &ctx->k_cache[c_offset],
				&ctx->v_cache[c_offset], ctx->cache_len,
				model->n_embd, ctx->seq_len);

		attn_scores(ctx->qkv, &ctx->k_cache[c_offset], ctx->scores,
			    ctx->seq_len, ctx->cache_len, model->n_head,
			    head_len);

		softmax(ctx->scores, model->n_head, ctx->cache_len,
			ctx->seq_len);

		attn_v_weighted_sum(ctx->scores, &ctx->v_cache[c_offset],
				    ctx->norm, ctx->seq_len, ctx->cache_len,
				    model->n_head, head_len);

		proj(ctx->norm, &block.attn_output_w, &block.attn_output_b,
		     ctx->proj, ctx->seq_len);

		add_f32v(ctx->hidden, ctx->proj, ctx->qkv,
			 ctx->seq_len * model->n_embd);

		// FFN
		layer_norm(ctx->qkv, block.ffn_norm_w.data,
			   block.ffn_norm_b.data, ctx->norm, ctx->seq_len,
			   model->n_embd, model->epsilon);

		proj(ctx->norm, &block.ffn_up_w, &block.ffn_up_b, ctx->proj,
		     ctx->seq_len);

		gelu_actv(ctx->proj, ctx->seq_len * model->n_ff);

		proj(ctx->proj, &block.ffn_down_w, &block.ffn_down_b, ctx->ffn,
		     ctx->seq_len);

		add_f32v(ctx->qkv, ctx->ffn, ctx->hidden,
			 ctx->seq_len * model->n_embd);
	}

	layer_norm(ctx->hidden, model->output_norm_w.data,
		   model->output_norm_b.data, ctx->norm, ctx->seq_len,
		   model->n_embd, model->epsilon);

	gemm_f32(&ctx->norm[(ctx->seq_len - 1) * model->n_embd],
		 model->token_embd_w.data, ctx->logits, 1.0f, 1,
		 model->vocab_count, model->n_embd);

	ctx->cache_len += ctx->seq_len;
	ctx->seq_len = 1;

	*gen_token   = max_id(ctx->logits, model->vocab_count);
}

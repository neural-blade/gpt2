#include <string.h>
#include <stdlib.h>
#include "backend.h"
#include "model.h"

typedef struct _transformer_ctx_t {
	uint32_t *token_ids;
	float *hidden;

	float *norm;
	float *qkv;
	float *scores;
	float *ffn;
	float *proj;

	float *logits;

	float *k_cache;
	float *v_cache;
	uint64_t cache_len;

	uint64_t seq_len;
} transformer_ctx_t;

transformer_ctx_t *transformer_create(model_t *model, uint32_t *token_ids,
				      uint64_t seq_len)
{
	transformer_ctx_t *ctx = malloc(sizeof(*ctx));

	ctx->token_ids	       = malloc(model->n_ctx * sizeof(uint32_t));
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
	for (uint32_t i = 0; i < seq_len; ++i) ctx->token_ids[i] = token_ids[i];

	return ctx;
}

void transformer_destroy(transformer_ctx_t *ctx)
{
	free(ctx->token_ids);
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

uint32_t transformer_get_token(transformer_ctx_t *ctx, uint32_t idx)
{
	return ctx->token_ids[idx];
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

void transformer_forward(transformer_ctx_t *ctx, model_t *model)
{
	// Embedding
	token_embd(&ctx->token_ids[ctx->cache_len], model->token_embd_w.data,
		   ctx->hidden, ctx->seq_len, model->n_embd);
	pos_embd(ctx->hidden, model->pos_embd_w.data, ctx->seq_len,
		 ctx->cache_len, model->n_embd);

	// Transformer
	for (uint32_t i = 0; i < model->block_count; ++i) {
		transformer_blk_t *block = &model->blocks[i];

		// Attention
		layer_norm(ctx->hidden, block->attn_norm_w.data,
			   block->attn_norm_b.data, ctx->norm, ctx->seq_len,
			   model->n_embd, model->epsilon);

		proj(ctx->norm, block->attn_qkv_w.data, block->attn_qkv_b.data,
		     ctx->qkv, ctx->seq_len, block->attn_qkv_w.dimensions[0],
		     block->attn_qkv_w.dimensions[1]);

		uint64_t c_offset = i * model->n_ctx * model->n_embd;
		kv_cache_append(ctx->qkv, &ctx->k_cache[c_offset],
				&ctx->v_cache[c_offset], ctx->cache_len,
				model->n_embd, ctx->seq_len);

		attn_scores(ctx->qkv, &ctx->k_cache[c_offset], ctx->scores,
			    ctx->seq_len, ctx->cache_len, model->n_head,
			    model->head_len);

		softmax(ctx->scores, model->n_head, ctx->cache_len,
			ctx->seq_len);

		attn_v_weighted_sum(ctx->scores, &ctx->v_cache[c_offset],
				    ctx->norm, ctx->seq_len, ctx->cache_len,
				    model->n_head, model->head_len);

		proj(ctx->norm, block->attn_output_w.data,
		     block->attn_output_b.data, ctx->proj, ctx->seq_len,
		     block->attn_output_w.dimensions[0],
		     block->attn_output_w.dimensions[1]);

		add_f32v(ctx->hidden, ctx->proj, ctx->qkv,
			 ctx->seq_len * model->n_embd);

		// FFN
		layer_norm(ctx->qkv, block->ffn_norm_w.data,
			   block->ffn_norm_b.data, ctx->norm, ctx->seq_len,
			   model->n_embd, model->epsilon);

		proj(ctx->norm, block->ffn_up_w.data, block->ffn_up_b.data,
		     ctx->proj, ctx->seq_len, block->ffn_up_w.dimensions[0],
		     block->ffn_up_w.dimensions[1]);

		gelu_actv(ctx->proj, ctx->seq_len * model->n_ff);

		proj(ctx->proj, block->ffn_down_w.data, block->ffn_down_b.data,
		     ctx->ffn, ctx->seq_len, block->ffn_down_w.dimensions[0],
		     block->ffn_down_w.dimensions[1]);

		add_f32v(ctx->qkv, ctx->ffn, ctx->hidden,
			 ctx->seq_len * model->n_embd);
	}

	// Final layer norm
	layer_norm(ctx->hidden, model->output_norm_w.data,
		   model->output_norm_b.data, ctx->norm, ctx->seq_len,
		   model->n_embd, model->epsilon);

	// Output projection
	gemm_f32(&ctx->norm[(ctx->seq_len - 1) * model->n_embd],
		 model->token_embd_w.data, ctx->logits, 1.0f, 1,
		 model->vocab_count, model->n_embd);

	ctx->cache_len += ctx->seq_len;
	ctx->seq_len		       = 1;

	// Greedy decoding
	ctx->token_ids[ctx->cache_len] = argmax_f32v(ctx->logits,
						     model->vocab_count);
}

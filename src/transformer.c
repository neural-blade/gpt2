#include "transformer.h"
#include <string.h>
#include "backend.h"
#include "model.h"
#include "perf.h"

struct _transformer_ctx_t {
	uint32_t *h_token_ids;
	uint32_t *token_ids;
	float *hidden;

	float *buf_narr;
	float *buf_wide;

	float *logits;

	float *k_cache;
	float *v_cache;
	uint64_t cache_len;

	uint64_t seq_len;

	float *steps;
	uint32_t step_count;
};

transformer_ctx_t *transformer_create(model_t *model, uint32_t *token_ids,
				      uint64_t seq_len)
{
	transformer_ctx_t *ctx;
	backend_init();
	backend_malloc_host((void **)&ctx, sizeof(*ctx));
	memset(ctx, 0, sizeof(*ctx));

	backend_malloc_host((void **)&ctx->h_token_ids,
			    (model->n_ctx + 1) * sizeof(uint32_t));
	backend_malloc_device((void **)&ctx->token_ids,
			      (model->n_ctx + 1) * sizeof(uint32_t));
	backend_malloc_device((void **)&ctx->hidden,
			      model->n_embd * seq_len * sizeof(float));
	backend_malloc_device((void **)&ctx->buf_narr,
			      model->n_ff * seq_len * sizeof(float));
	backend_malloc_device((void **)&ctx->buf_wide,
			      model->n_head * model->n_ctx * seq_len *
				  sizeof(float));
	backend_malloc_device((void **)&ctx->logits,
			      model->vocab_count * sizeof(float));
	backend_malloc_device((void **)&ctx->k_cache,
			      model->block_count * model->n_ctx *
				  model->n_embd * sizeof(float));
	backend_malloc_device((void **)&ctx->v_cache,
			      model->block_count * model->n_ctx *
				  model->n_embd * sizeof(float));
	backend_malloc_host((void **)&ctx->steps,
			    model->n_ctx * sizeof(*ctx->steps));

	ctx->step_count = 0;
	ctx->cache_len	= 0;
	ctx->seq_len	= seq_len;
	for (uint32_t i = 0; i < seq_len; ++i)
		ctx->h_token_ids[i] = token_ids[i];

	if (ctx->token_ids == NULL) ctx->token_ids = ctx->h_token_ids;

	backend_h2d(ctx->token_ids, ctx->h_token_ids,
		    seq_len * sizeof(uint32_t));

	return ctx;
}

void transformer_destroy(transformer_ctx_t *ctx)
{
	backend_free_device(ctx->token_ids);
	backend_free_device(ctx->hidden);
	backend_free_device(ctx->buf_narr);
	backend_free_device(ctx->buf_wide);
	backend_free_device(ctx->logits);
	backend_free_device(ctx->k_cache);
	backend_free_device(ctx->v_cache);
	backend_free_host(ctx->h_token_ids);
	backend_free_host(ctx->steps);
	backend_free_host(ctx);
	backend_destroy();
}

uint32_t transformer_get_token(transformer_ctx_t *ctx, uint32_t idx)
{
	return ctx->h_token_ids[idx];
}

static void kv_cache_append(const float *qkv, float *k, float *v,
			    uint64_t cache_len, uint64_t hidden_len,
			    uint64_t n_tokens)
{
	for (uint64_t i = 0; i < n_tokens; ++i) {
		uint64_t last_offset = cache_len * hidden_len;
		uint32_t t_offset    = i * hidden_len;
		uint64_t k_offset    = t_offset * 3 + hidden_len;

		backend_d2d(k + last_offset + t_offset, &qkv[k_offset],
			    hidden_len * sizeof(*k));
		backend_d2d(v + last_offset + t_offset,
			    &qkv[k_offset + hidden_len],
			    hidden_len * sizeof(*v));
	}
}

void transformer_get_perf(transformer_ctx_t *ctx, transformer_perf_t *stats)
{
	stats->prefill_tokens = ctx->cache_len - ctx->step_count + 1;
	stats->prefill_ms     = ctx->steps[0];
	stats->decode_tokens  = ctx->step_count - 1;

	uint64_t decode_ns    = 0;
	for (uint32_t i = 1; i < ctx->step_count; ++i)
		decode_ns += ctx->steps[i];
	stats->decode_ms = decode_ns;
}

void transformer_forward(transformer_ctx_t *ctx, model_t *model)
{
	backend_time_start();

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
			   block->attn_norm_b.data, ctx->buf_wide, ctx->seq_len,
			   model->n_embd, model->epsilon);

		proj(ctx->buf_wide, block->attn_qkv_w.data,
		     block->attn_qkv_b.data, ctx->buf_narr, ctx->seq_len,
		     block->attn_qkv_w.dimensions[0],
		     block->attn_qkv_w.dimensions[1]);

		uint64_t c_offset = i * model->n_ctx * model->n_embd;
		kv_cache_append(ctx->buf_narr, &ctx->k_cache[c_offset],
				&ctx->v_cache[c_offset], ctx->cache_len,
				model->n_embd, ctx->seq_len);

		attn_scores(ctx->buf_narr, &ctx->k_cache[c_offset],
			    ctx->buf_wide, ctx->seq_len, ctx->cache_len,
			    model->n_head, model->head_len);

		softmax(ctx->buf_wide, model->n_head, ctx->cache_len,
			ctx->seq_len);

		attn_v_weighted_sum(ctx->buf_wide, &ctx->v_cache[c_offset],
				    ctx->buf_narr, ctx->seq_len, ctx->cache_len,
				    model->n_head, model->head_len);

		proj(ctx->buf_narr, block->attn_output_w.data,
		     block->attn_output_b.data, ctx->buf_wide, ctx->seq_len,
		     block->attn_output_w.dimensions[0],
		     block->attn_output_w.dimensions[1]);

		resid(ctx->hidden, ctx->buf_wide, ctx->seq_len * model->n_embd);

		// FFN
		layer_norm(ctx->hidden, block->ffn_norm_w.data,
			   block->ffn_norm_b.data, ctx->buf_wide, ctx->seq_len,
			   model->n_embd, model->epsilon);

		proj(ctx->buf_wide, block->ffn_up_w.data, block->ffn_up_b.data,
		     ctx->buf_narr, ctx->seq_len, block->ffn_up_w.dimensions[0],
		     block->ffn_up_w.dimensions[1]);

		gelu_actv(ctx->buf_narr, ctx->seq_len * model->n_ff);

		proj(ctx->buf_narr, block->ffn_down_w.data,
		     block->ffn_down_b.data, ctx->buf_wide, ctx->seq_len,
		     block->ffn_down_w.dimensions[0],
		     block->ffn_down_w.dimensions[1]);

		resid(ctx->hidden, ctx->buf_wide, ctx->seq_len * model->n_embd);
	}

	// Final layer norm
	layer_norm(ctx->hidden, model->output_norm_w.data,
		   model->output_norm_b.data, ctx->buf_narr, ctx->seq_len,
		   model->n_embd, model->epsilon);

	// Output projection
	out_proj(&ctx->buf_narr[(ctx->seq_len - 1) * model->n_embd],
		 model->token_embd_w.data, ctx->logits, model->vocab_count,
		 model->n_embd);

	ctx->cache_len += ctx->seq_len;
	ctx->seq_len = 1;

	// Greedy decoding
	greedy_decode(ctx->logits, model->vocab_count,
		      &ctx->token_ids[ctx->cache_len]);

	backend_time_stop();
	backend_time_elaps(&ctx->steps[ctx->step_count++]);

	backend_d2h(&ctx->h_token_ids[ctx->cache_len],
		    &ctx->token_ids[ctx->cache_len], sizeof(uint32_t));
}

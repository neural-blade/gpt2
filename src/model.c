#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "gguf.h"
#include "transformer.h"
#include "backend.h"

static int tensor_init(tensor_t *tensor, const gguf_file_t *file,
		       const char *t_name)
{
	const gguf_tensor_info_t *t_info = gguf_find_tensor_info(file, t_name);
	tensor->n_dimensions		 = t_info->n_dimensions;
	memcpy(tensor->dimensions, t_info->dimensions,
	       sizeof(tensor->dimensions));
	const float *gguf_tensor = gguf_get_f32tensor(file->tensor_data,
						      t_info);

	size_t tensor_size	 = sizeof(float);
	for (size_t i = 0; i < tensor->n_dimensions; ++i)
		tensor_size *= tensor->dimensions[i];

	backend_move_h2d((void **)&tensor->data, gguf_tensor, tensor_size);

	return 0;
}

static int transformer_block_init(transformer_blk_t *block,
				  const gguf_file_t *file, uint32_t block_id)
{
	char t_name[64];

	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".attn_qkv.bias",
		 block_id);
	tensor_init(&block->attn_qkv_b, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".attn_qkv.weight",
		 block_id);
	tensor_init(&block->attn_qkv_w, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".attn_output.bias",
		 block_id);
	tensor_init(&block->attn_output_b, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".attn_output.weight",
		 block_id);
	tensor_init(&block->attn_output_w, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".attn_norm.bias",
		 block_id);
	tensor_init(&block->attn_norm_b, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".attn_norm.weight",
		 block_id);
	tensor_init(&block->attn_norm_w, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".ffn_up.bias",
		 block_id);
	tensor_init(&block->ffn_up_b, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".ffn_up.weight",
		 block_id);
	tensor_init(&block->ffn_up_w, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".ffn_down.bias",
		 block_id);
	tensor_init(&block->ffn_down_b, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".ffn_down.weight",
		 block_id);
	tensor_init(&block->ffn_down_w, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".ffn_norm.bias",
		 block_id);
	tensor_init(&block->ffn_norm_b, file, t_name);
	snprintf(t_name, sizeof(t_name), "blk.%" PRIu32 ".ffn_norm.weight",
		 block_id);
	tensor_init(&block->ffn_norm_w, file, t_name);

	return 0;
}

static uint8_t unicode_to_byte[512];

static void init_unicode_to_byte(void)
{
	uint32_t n = 0;

	for (uint32_t b = 0; b < 256; ++b) {
		uint32_t idx = b;
		if (!(b >= 0x21 && b <= 0x7E) && !(b >= 0xA1 && b <= 0xAC) &&
		    !(b >= 0xAE && b <= 0xFF))
			idx = 256 + n++;

		unicode_to_byte[idx] = b;
	}
}

int model_init(model_t *model, const gguf_file_t *file)
{
	gguf_metadata_value_t value;

	char key[64];
	gguf_get_value(file, "general.architecture", &value);
	snprintf(key, sizeof(key), "%s.block_count", value.string.string);
	gguf_get_value(file, key, &value);

	model->block_count = value.uint32;
	model->blocks = malloc(model->block_count * sizeof(*model->blocks));

	for (uint32_t i = 0; i < model->block_count; ++i)
		transformer_block_init(&model->blocks[i], file, i);

	tensor_init(&model->output_norm_b, file, "output_norm.bias");
	tensor_init(&model->output_norm_w, file, "output_norm.weight");
	tensor_init(&model->pos_embd_w, file, "position_embd.weight");
	tensor_init(&model->token_embd_w, file, "token_embd.weight");

	gguf_get_value(file, "gpt2.context_length", &value);
	model->n_ctx = value.uint32;
	gguf_get_value(file, "gpt2.embedding_length", &value);
	model->n_embd = value.uint32;
	gguf_get_value(file, "gpt2.attention.head_count", &value);
	model->n_head	= value.uint32;
	model->head_len = model->n_embd / model->n_head;
	gguf_get_value(file, "gpt2.feed_forward_length", &value);
	model->n_ff = value.uint32;
	gguf_get_value(file, "gpt2.attention.layer_norm_epsilon", &value);
	model->epsilon = value.float32;
	gguf_get_value(file, "tokenizer.ggml.tokens", &value);
	model->vocab_count = value.array.len;
	model->vocab	   = malloc(model->vocab_count * sizeof(*model->vocab));
	for (uint32_t i = 0; i < model->vocab_count; ++i) {
		model->vocab[i] = malloc(value.array.array[i].string.len + 1);
		strcpy(model->vocab[i], value.array.array[i].string.string);
	}

	init_unicode_to_byte();

	return 0;
}

void model_free(model_t *model)
{
	free(model->blocks);
	for (uint32_t i = 0; i < model->vocab_count; ++i) free(model->vocab[i]);
	free(model->vocab);
}

static void print_token(const char *token)
{
	static uint32_t unicode_id	= 0;
	static uint32_t bytes_remaining = 0;

	uint8_t *bytes			= (uint8_t *)token;
	uint32_t len			= strlen(token);

	for (uint32_t i = 0; i < len; ++i) {
		uint8_t b = bytes[i];
		if (bytes_remaining == 0) {
			if ((b & 0x80) == 0) { // 1000 0000
				unicode_id	= b & 0x7F;
				bytes_remaining = 0;
			} else if ((b & 0xE0) == 0xC0) { // 1110 0000
				unicode_id	= b & 0x1F;
				bytes_remaining = 1;
			} else if ((b & 0xF0) == 0xE0) { // 1111 0000
				unicode_id	= b & 0x0F;
				bytes_remaining = 2;
			} else if ((b & 0xF8) == 0xF0) { // 1111 1000
				unicode_id	= b & 0x07;
				bytes_remaining = 3;
			}
		} else {
			unicode_id = (unicode_id << 6) | (b & 0x3F);
			--bytes_remaining;
		}

		if (bytes_remaining == 0) putchar(unicode_to_byte[unicode_id]);
	}
}

void model_run(model_t *model, uint32_t *token_ids, uint32_t token_count,
	       uint32_t max_tokens, transformer_perf_t *stats)
{
	transformer_ctx_t *ctx = transformer_create(model, token_ids,
						    token_count);

	for (uint32_t i = 0; i < max_tokens; ++i) {
		uint32_t total_token = token_count + i;
		uint32_t next_token_id;
		if (total_token > model->n_ctx) break;
		transformer_forward(ctx, model);
		next_token_id = transformer_get_token(ctx, total_token);
		print_token(model->vocab[next_token_id]);
		fflush(stdout);
	}
	printf("\n");

	if (stats) transformer_get_perf(ctx, stats);

	transformer_destroy(ctx);
}

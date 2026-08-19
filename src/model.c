#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "model.h"
#include "gguf.h"

static int tensor_init(tensor_t *tensor, const gguf_file_t *file,
		       const char *t_name)
{
	const gguf_tensor_info_t *t_info = gguf_find_tensor_info(file, t_name);
	tensor->n_dimensions		 = t_info->n_dimensions;
	memcpy(tensor->dimensions, t_info->dimensions,
	       sizeof(tensor->dimensions));
	tensor->data = gguf_get_f32tensor(file->tensor_data, t_info);

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

int model_init(model_t *model, const gguf_file_t *file)
{
	gguf_metadata_value_t value;

	char key[64];
	gguf_get_value((file), "general.architecture", &value);
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

	return 0;
}

void model_free(model_t *model)
{
	free(model->blocks);
}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gguf.h"
#include "model.h"
#include "perf.h"

#define DEFAULT_PROMPT_LEN 10
#define DEFAULT_MAX_TOKENS 100

void print_help(const char *program_name)
{
	printf("Usage: %s [options]\n\n", program_name);
	printf("Options:\n");
	printf("  --model <file>       Path to the model file\n");

	printf("  --prompt-length <n>  Number of input tokens (default: %d)\n",
	       DEFAULT_PROMPT_LEN);

	printf("  --max-tokens <n>     Maximum number of tokens to generate "
	       "(default: %d)\n",
	       DEFAULT_MAX_TOKENS);

	printf("  --show-gguf          Display GGUF file metadata\n");
	printf("  --show-stats         Display stats info\n");
	printf("  --help               Display this help message and exit\n");
}

void show_stats(transformer_perf_t *stats)
{
	printf("\n=============== STATS ===============\n");
	printf("prefill: %u tok in %.1f ms (%.1f tok/s)\n"
	       "decode : %u tok in %.1f ms (%.1f tok/s)\n",
	       stats->prefill_tokens, stats->prefill_ms,
	       stats->prefill_tokens / (stats->prefill_ms * 1.0e-3f),
	       stats->decode_tokens, stats->decode_ms,
	       stats->decode_tokens / (stats->decode_ms * 1.0e-3f));
}

int main(int argc, char **argv)
{
	int print_gguf	    = 0;
	int print_stats	    = 0;
	uint32_t max_tokens = DEFAULT_MAX_TOKENS;
	uint32_t prompt_len = DEFAULT_PROMPT_LEN;
	char *filename	    = NULL;

	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--show-gguf") == 0)
			print_gguf = 1;
		else if (strcmp(argv[i], "--prompt-length") == 0) {
			prompt_len = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--max-tokens") == 0)
			max_tokens = atoi(argv[++i]);
		else if (strcmp(argv[i], "--model") == 0)
			filename = argv[++i];
		else if (strcmp(argv[i], "--show-stats") == 0)
			print_stats = 1;
		else if (strcmp(argv[i], "--help") == 0) {
			print_help(argv[0]);
			return 0;
		} else {
			printf("Unknown option: %s\n", argv[i]);
			print_help(argv[0]);
			return 1;
		}
	}

	if (filename == NULL) {
		fprintf(stderr, "Error: --model is required\n");
		return 1;
	}

	gguf_file_t file;
	if (gguf_load((char *)filename, &file) != 0) {
		fprintf(stderr, "Error: failed to load the model\n");
		return 1;
	}

	if (print_gguf) {
		gguf_show(&file);
		return 0;
	}

	model_t model;
	model_init(&model, &file);

	if (prompt_len <= 0 || prompt_len > model.n_ctx) {
		fprintf(stderr, "Error: invalid prompt length\n");

		model_free(&model);
		gguf_free(&file);
		return 1;
	}

	uint32_t *prompt_token_ids;
	prompt_token_ids = malloc(prompt_len * sizeof(uint32_t));
	for (uint32_t i = 0; i < prompt_len; ++i)
		prompt_token_ids[i] = (i + 6000) % model.vocab_count;

	transformer_perf_t stats;
	model_run(&model, prompt_token_ids, prompt_len, max_tokens, &stats);

	if (print_stats) show_stats(&stats);

	free(prompt_token_ids);
	model_free(&model);
	gguf_free(&file);

	return 0;
}

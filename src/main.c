#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gguf.h"
#include "model.h"
#include "perf.h"

void print_help(const char *program_name)
{
	printf("Usage: %s [options]\n\n", program_name);
	printf("Options:\n");
	printf("  --model <file>      Path to the model file\n");
	printf("  --max-tokens <num>  Maximum number of tokens to generate\n");
	printf("  --show-gguf         Display GGUF file metadata\n");
	printf("  --show-stats        Display stats info\n");
	printf("  --help              Display this help message and exit\n");
}

void show_stats(transformer_perf_t *stats)
{
	printf("\n=============== STATS ===============\n");
	printf("prefill: %u tok in %.1lf ms (%.1lf tok/s)\n"
	       "decode : %u tok in %.1lf ms (%.1lf tok/s)\n",
	       stats->prefill_tokens, stats->prefill_ns * 1.0e-6,
	       stats->prefill_tokens / (stats->prefill_ns * 1.0e-9),
	       stats->decode_tokens, stats->decode_ns * 1.0e-6,
	       stats->decode_tokens / (stats->decode_ns * 1.0e-9));
}

int main(int argc, char **argv)
{
	int print_gguf	    = 0;
	int print_stats	    = 0;
	uint32_t max_tokens = 100;
	char *filename	    = NULL;

	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--show-gguf") == 0)
			print_gguf = 1;
		else if (strcmp(argv[i], "--max-tokens") == 0)
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

	uint32_t prompt_token_ids[] = {818,   3644, 3783, 11,  257,
				       13934, 2989, 5509, 318, 257,
				       1366,  4645, 810};
	uint32_t prompt_token_count = sizeof(prompt_token_ids) /
				      sizeof(*prompt_token_ids);

	transformer_perf_t stats;
	model_run(&model, prompt_token_ids, prompt_token_count, max_tokens,
		  &stats);

	if (print_stats) show_stats(&stats);

	model_free(&model);
	gguf_free(&file);

	return 0;
}

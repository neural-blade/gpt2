#ifndef __GGUF_H
#define __GGUF_H

#include <stdint.h>
#include <stdbool.h>

typedef enum _ggml_type {
	GGML_TYPE_F32	  = 0,
	GGML_TYPE_F16	  = 1,
	GGML_TYPE_Q4_0	  = 2,
	GGML_TYPE_Q4_1	  = 3,
	// GGML_TYPE_Q4_2 = 4, support has been removed
	// GGML_TYPE_Q4_3 = 5, support has been removed
	GGML_TYPE_Q5_0	  = 6,
	GGML_TYPE_Q5_1	  = 7,
	GGML_TYPE_Q8_0	  = 8,
	GGML_TYPE_Q8_1	  = 9,
	GGML_TYPE_Q2_K	  = 10,
	GGML_TYPE_Q3_K	  = 11,
	GGML_TYPE_Q4_K	  = 12,
	GGML_TYPE_Q5_K	  = 13,
	GGML_TYPE_Q6_K	  = 14,
	GGML_TYPE_Q8_K	  = 15,
	GGML_TYPE_IQ2_XXS = 16,
	GGML_TYPE_IQ2_XS  = 17,
	GGML_TYPE_IQ3_XXS = 18,
	GGML_TYPE_IQ1_S	  = 19,
	GGML_TYPE_IQ4_NL  = 20,
	GGML_TYPE_IQ3_S	  = 21,
	GGML_TYPE_IQ2_S	  = 22,
	GGML_TYPE_IQ4_XS  = 23,
	GGML_TYPE_I8	  = 24,
	GGML_TYPE_I16	  = 25,
	GGML_TYPE_I32	  = 26,
	GGML_TYPE_I64	  = 27,
	GGML_TYPE_F64	  = 28,
	GGML_TYPE_IQ1_M	  = 29,
	GGML_TYPE_BF16	  = 30,
	// GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
	// GGML_TYPE_Q4_0_4_8 = 32,
	// GGML_TYPE_Q4_0_8_8 = 33,
	GGML_TYPE_TQ1_0	  = 34,
	GGML_TYPE_TQ2_0	  = 35,
	// GGML_TYPE_IQ4_NL_4_4 = 36,
	// GGML_TYPE_IQ4_NL_4_8 = 37,
	// GGML_TYPE_IQ4_NL_8_8 = 38,
	GGML_TYPE_MXFP4	  = 39, // MXFP4 (1 block)
	GGML_TYPE_COUNT	  = 40,
} ggml_type;

typedef enum _gguf_metadata_value_type {
	GGUF_METADATA_VALUE_TYPE_UINT8	 = 0,
	GGUF_METADATA_VALUE_TYPE_INT8	 = 1,
	GGUF_METADATA_VALUE_TYPE_UINT16	 = 2,
	GGUF_METADATA_VALUE_TYPE_INT16	 = 3,
	GGUF_METADATA_VALUE_TYPE_UINT32	 = 4,
	GGUF_METADATA_VALUE_TYPE_INT32	 = 5,
	GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
	GGUF_METADATA_VALUE_TYPE_BOOL	 = 7,
	GGUF_METADATA_VALUE_TYPE_STRING	 = 8,
	GGUF_METADATA_VALUE_TYPE_ARRAY	 = 9,
	GGUF_METADATA_VALUE_TYPE_UINT64	 = 10,
	GGUF_METADATA_VALUE_TYPE_INT64	 = 11,
	GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
} gguf_metadata_value_type;

typedef struct _gguf_string_t {
	uint64_t len;
	char *string;
} gguf_string_t;

typedef struct _gguf_array_t {
	gguf_metadata_value_type type;
	uint64_t len;
	union _gguf_metadata_value_t *array;
} gguf_array_t;

typedef union _gguf_metadata_value_t {
	uint8_t uint8;
	int8_t int8;
	uint16_t uint16;
	int16_t int16;
	uint32_t uint32;
	int32_t int32;
	float float32;
	uint64_t uint64;
	int64_t int64;
	double float64;
	bool bool_;
	gguf_string_t string;
	gguf_array_t array;
} gguf_metadata_value_t;

typedef struct _gguf_header_t {
	uint32_t magic;
	uint32_t version;
	uint64_t tensor_count;
	uint64_t metadata_kv_count;
} gguf_header_t;

typedef struct _gguf_metadata_kv_t {
	gguf_string_t key;
	gguf_metadata_value_type value_type;
	gguf_metadata_value_t value;
} gguf_metadata_kv_t;

typedef struct _gguf_tensor_info_t {
	gguf_string_t name;
	uint32_t n_dimensions;
	uint64_t dimensions[4];
	ggml_type type;
	uint64_t offset;
} gguf_tensor_info_t;

typedef struct _gguf_file_t {
	gguf_header_t header;
	gguf_metadata_kv_t *metadata_kv;
	gguf_tensor_info_t *tensor_infos;
	uint8_t *tensor_data;
} gguf_file_t;

int gguf_load(const char *filename, gguf_file_t *file);
int gguf_get_value(gguf_file_t *file, char *key, gguf_metadata_value_t *value);
void gguf_show(gguf_file_t *file);
void gguf_free(gguf_file_t *file);

static inline const float *gguf_get_f32tensor(gguf_file_t *file,
					      uint64_t tensor_id)
{
	return (const float *)(file->tensor_data +
			       file->tensor_infos[tensor_id].offset);
}

#endif /* __GGUF_H */

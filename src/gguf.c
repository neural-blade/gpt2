#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "gguf.h"

static uint8_t read_u8(int fd)
{
	uint8_t val;
	if (read(fd, &val, sizeof(val)) != sizeof(val)) abort();
	return val;
}

static uint16_t read_u16(int fd)
{
	uint16_t val;
	if (read(fd, &val, sizeof(val)) != sizeof(val)) abort();
	return val;
}

static uint32_t read_u32(int fd)
{
	uint32_t val;
	if (read(fd, &val, sizeof(val)) != sizeof(val)) abort();
	return val;
}

static uint64_t read_u64(int fd)
{
	uint64_t val;
	if (read(fd, &val, sizeof(val)) != sizeof(val)) abort();
	return val;
}

static gguf_string_t read_string(int fd)
{
	gguf_string_t str;
	str.len	   = read_u64(fd);
	str.string = malloc(str.len + 1);
	if (read(fd, str.string, str.len) != (ssize_t)str.len) abort();
	str.string[str.len] = '\0';
	return str;
}

static gguf_metadata_value_t read_value(int fd, gguf_metadata_value_type type);

static gguf_array_t read_array(int fd)
{
	gguf_array_t arr;
	arr.type  = read_u32(fd);
	arr.len	  = read_u64(fd);
	arr.array = malloc(arr.len * sizeof(*arr.array));

	for (uint64_t i = 0; i < arr.len; ++i)
		arr.array[i] = read_value(fd, arr.type);

	return arr;
}

static gguf_metadata_value_t read_value(int fd, gguf_metadata_value_type type)
{
	gguf_metadata_value_t value = {0};

	switch (type) {
	case GGUF_METADATA_VALUE_TYPE_UINT64:
	case GGUF_METADATA_VALUE_TYPE_FLOAT64:
	case GGUF_METADATA_VALUE_TYPE_INT64:   value.uint64 = read_u64(fd); break;
	case GGUF_METADATA_VALUE_TYPE_UINT32:
	case GGUF_METADATA_VALUE_TYPE_FLOAT32:
	case GGUF_METADATA_VALUE_TYPE_INT32:   value.uint32 = read_u32(fd); break;
	case GGUF_METADATA_VALUE_TYPE_UINT16:
	case GGUF_METADATA_VALUE_TYPE_INT16:   value.uint16 = read_u16(fd); break;
	case GGUF_METADATA_VALUE_TYPE_UINT8:
	case GGUF_METADATA_VALUE_TYPE_INT8:
	case GGUF_METADATA_VALUE_TYPE_BOOL:    value.uint8 = read_u8(fd); break;
	case GGUF_METADATA_VALUE_TYPE_STRING:
		value.string = read_string(fd);
		break;
	case GGUF_METADATA_VALUE_TYPE_ARRAY:
		value.array = read_array(fd);
		break;
	}

	return value;
}

static gguf_metadata_kv_t read_kv(int fd)
{
	gguf_metadata_kv_t kv;
	kv.key	      = read_string(fd);
	kv.value_type = read_u32(fd);
	kv.value      = read_value(fd, kv.value_type);
	return kv;
}

static gguf_tensor_info_t read_tensor_info(int fd)
{
	gguf_tensor_info_t t_info;
	t_info.name	    = read_string(fd);
	t_info.n_dimensions = read_u32(fd);
	for (uint32_t i = 0; i < t_info.n_dimensions; ++i)
		t_info.dimensions[i] = read_u64(fd);
	t_info.type   = read_u32(fd);
	t_info.offset = read_u64(fd);

	return t_info;
}

static inline uint64_t align_offset(uint64_t offset, uint64_t alignment)
{
	uint64_t rem = offset % alignment;
	return rem == 0 ? offset : offset + alignment - rem;
}

int gguf_get_value(const gguf_file_t *file, const char *key,
		   gguf_metadata_value_t *value)
{
	for (uint64_t i = 0; i < file->header.metadata_kv_count; ++i) {
		if (strcmp(key, file->metadata_kv[i].key.string) == 0) {
			*value = file->metadata_kv[i].value;
			return 0;
		}
	}

	return -1;
}

int gguf_load(const char *filename, gguf_file_t *file)
{
	int fd = open(filename, O_RDONLY);
	if (fd == -1) return -1;

	if (read(fd, &file->header, sizeof(file->header)) !=
	    sizeof(file->header)) {
		close(fd);
		return -2;
	}
	if (memcmp(&file->header.magic, "GGUF", 4) != 0) {
		close(fd);
		return -3;
	}

	file->metadata_kv  = malloc(file->header.metadata_kv_count *
				    sizeof(*file->metadata_kv));
	file->tensor_infos = malloc(file->header.tensor_count *
				    sizeof(*file->tensor_infos));

	for (uint64_t i = 0; i < file->header.metadata_kv_count; ++i)
		file->metadata_kv[i] = read_kv(fd);

	for (uint64_t i = 0; i < file->header.tensor_count; ++i)
		file->tensor_infos[i] = read_tensor_info(fd);

	gguf_metadata_value_t align = {0};
	align.uint32		    = 32;
	gguf_get_value(file, "general.alignment", &align);

	uint64_t data_start = align_offset(lseek(fd, 0, SEEK_CUR),
					   align.uint32);
	off_t file_size	    = lseek(fd, 0, SEEK_END);
	if (lseek(fd, data_start, SEEK_SET) == -1) {
		close(fd);
		return -4;
	}
	size_t tensor_data_size = file_size - data_start;
	file->tensor_data	= malloc(tensor_data_size);
	if (read(fd, file->tensor_data, tensor_data_size) !=
	    (ssize_t)tensor_data_size) {
		close(fd);
		return -5;
	}

	close(fd);

	return 0;
}

static void show_value(gguf_metadata_value_type type,
		       gguf_metadata_value_t value);

static void show_array(gguf_array_t array)
{
	printf("[ ");
	uint64_t limit = array.len > 32 ? 31 : array.len;
	uint64_t i     = 0;
	show_value(array.type, array.array[i++]);
	for (; i < limit; ++i) {
		printf(", ");
		show_value(array.type, array.array[i]);
	}
	if (array.len > limit) {
		printf(", ... ");
		show_value(array.type, array.array[array.len - 1]);
	}
	printf(" ]");
}

static void show_value(gguf_metadata_value_type type,
		       gguf_metadata_value_t value)
{
	switch (type) {
	case GGUF_METADATA_VALUE_TYPE_UINT64:
		printf("%" PRIu64, value.uint64);
		break;
	case GGUF_METADATA_VALUE_TYPE_FLOAT64:
		printf("%lf", value.float64);
		break;
	case GGUF_METADATA_VALUE_TYPE_INT64:
		printf("%" PRIi64, value.int64);
		break;
	case GGUF_METADATA_VALUE_TYPE_UINT32:
		printf("%" PRIu32, value.uint32);
		break;
	case GGUF_METADATA_VALUE_TYPE_FLOAT32:
		printf("%f", value.float32);
		break;
	case GGUF_METADATA_VALUE_TYPE_INT32:
		printf("%" PRIi32, value.int32);
		break;
	case GGUF_METADATA_VALUE_TYPE_UINT16:
		printf("%" PRIu16, value.uint16);
		break;
	case GGUF_METADATA_VALUE_TYPE_INT16:
		printf("%" PRIi16, value.int16);
		break;
	case GGUF_METADATA_VALUE_TYPE_UINT8:
		printf("%" PRIu8, value.uint8);
		break;
	case GGUF_METADATA_VALUE_TYPE_INT8:
		printf("%" PRIi8, value.int8);
		break;
	case GGUF_METADATA_VALUE_TYPE_BOOL:
		printf("%s", value.bool_ ? "true" : "false");
		break;
	case GGUF_METADATA_VALUE_TYPE_STRING:
		printf("%s", value.string.string);
		break;
	case GGUF_METADATA_VALUE_TYPE_ARRAY: show_array(value.array); break;
	}
}

static void show_tensor_info(gguf_tensor_info_t t_info)
{
	printf("%s: [ ", t_info.name.string);
	uint32_t i = 0;
	printf("%" PRIu64, t_info.dimensions[i++]);
	for (; i < t_info.n_dimensions; ++i)
		printf(", %" PRIu64, t_info.dimensions[i]);
	printf(" ] ");
	printf("@%" PRIu64 "\n", t_info.offset);
}

void gguf_show(const gguf_file_t *file)
{
	printf("\n================== GGUF ==================\n");
	printf("GGUF Version: %" PRIu32 "\n", file->header.version);
	printf("Tensor Count: %" PRIu64 "\n", file->header.tensor_count);
	printf("Metadata KV Count: %" PRIu64 "\n",
	       file->header.metadata_kv_count);

	printf("\n================ METADATA ================\n");
	for (uint64_t i = 0; i < file->header.metadata_kv_count; ++i) {
		printf("%s: ", file->metadata_kv[i].key.string);
		show_value(file->metadata_kv[i].value_type,
			   file->metadata_kv[i].value);
		printf("\n");
	}

	printf("\n============== TENSOR INFOS ==============\n");
	for (uint64_t i = 0; i < file->header.tensor_count; ++i)
		show_tensor_info(file->tensor_infos[i]);
}

static void free_value(gguf_metadata_value_t *value,
		       gguf_metadata_value_type type);

static void free_array(gguf_array_t *array)
{
	for (uint64_t i = 0; i < array->len; ++i)
		free_value(&array->array[i], array->type);

	free(array->array);
}

static void free_value(gguf_metadata_value_t *value,
		       gguf_metadata_value_type type)
{
	switch (type) {
	case GGUF_METADATA_VALUE_TYPE_STRING: free(value->string.string); break;
	case GGUF_METADATA_VALUE_TYPE_ARRAY:  free_array(&value->array); break;
	default:			      break;
	}
}

static void free_kv(gguf_metadata_kv_t *metadata_kv)
{
	free(metadata_kv->key.string);
	free_value(&metadata_kv->value, metadata_kv->value_type);
}

void gguf_free(gguf_file_t *file)
{
	free(file->tensor_data);
	for (uint64_t i = 0; i < file->header.metadata_kv_count; ++i)
		free_kv(&file->metadata_kv[i]);
	free(file->metadata_kv);
	for (uint64_t i = 0; i < file->header.tensor_count; ++i)
		free(file->tensor_infos[i].name.string);
	free(file->tensor_infos);
}

const gguf_tensor_info_t *gguf_find_tensor_info(const gguf_file_t *file,
						const char *t_name)
{
	for (uint64_t i = 0; i < file->header.tensor_count; ++i)
		if (strcmp(t_name, file->tensor_infos[i].name.string) == 0)
			return &file->tensor_infos[i];

	return NULL;
}

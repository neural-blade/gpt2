#include <stdint.h>

void add_f32v(const float *restrict a, const float *restrict b,
	      float *restrict c, uint64_t len)
{
	for (uint64_t i = 0; i < len; ++i) c[i] = a[i] + b[i];
}

void add_f32v_inplace(float *a, const float *restrict b, float alpha,
		      uint64_t len)
{
	for (uint64_t i = 0; i < len; ++i) a[i] += alpha * b[i];
}

float sum_f32v(const float *restrict a, uint64_t len)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i];
	return sum;
}

float dot_f32v(const float *restrict a, const float *restrict b, uint64_t len)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i] * b[i];
	return sum;
}

void gemm_f32(const float *restrict a, const float *restrict b,
	      float *restrict c, float alpha, uint64_t m, uint64_t n,
	      uint64_t k)
{
	for (uint64_t i = 0; i < m; ++i)
		for (uint64_t j = 0; j < n; ++j)
			c[i * n + j] = dot_f32v(&a[i * k], &b[j * k], k) *
				       alpha;
}

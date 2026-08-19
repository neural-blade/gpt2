#ifndef __LINALG_H
#define __LINALG_H

#include <stdint.h>

void add_f32v(const float *restrict a, const float *restrict b,
	      float *restrict c, uint64_t len);
void add_f32v_inplace(float *a, const float *restrict b, float alpha,
		      uint64_t len);
float sum_f32v(const float *restrict a, uint64_t len);
float dot_f32v(const float *restrict a, const float *restrict b, uint64_t len);
void gemm_f32(const float *restrict a, const float *restrict b,
	      float *restrict c, float alpha, uint64_t m, uint64_t n,
	      uint64_t k);

#endif /* __LINALG_H */

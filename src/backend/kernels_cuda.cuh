#ifndef __KERNELS_CUDA_CUH
#define __KERNELS_CUDA_CUH

#include <stdint.h>

#define THREADS_PER_BLOCK 256
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void add_f32v_kernel(float *__restrict__ a,
				const float *__restrict__ b, float alpha,
				uint64_t len);

__global__ void gemm_f32_kernel(const float *__restrict__ a,
				const float *__restrict__ b,
				float *__restrict__ c, float alpha, uint64_t m,
				uint64_t n, uint64_t k);

__global__ void token_embd_kernel(const uint32_t *token_ids,
				  const float *__restrict__ token_embd_w,
				  float *__restrict__ embd,
				  uint32_t hidden_dim);

__global__ void layer_norm_kernel(const float *__restrict__ in,
				  const float *__restrict__ weight,
				  const float *__restrict__ bias,
				  float *__restrict__ out, uint32_t seq_len,
				  uint32_t hidden_dim, float eps);

__global__ void attn_scores_kernel(const float *__restrict__ q,
				   const float *__restrict__ k,
				   float *__restrict__ scores, uint64_t q_len,
				   uint64_t k_len, uint64_t heads_count,
				   uint64_t head_len);

__global__ void argmax_f32v_kernel(const float *v, uint64_t len, uint32_t *out);

__global__ void softmax_kernel(float *__restrict__ scores, uint64_t n_head,
			       uint64_t initial_token, uint64_t n_token);

__global__ void attn_v_weighted_sum_kernel(const float *__restrict__ p,
					   const float *__restrict__ v,
					   float *__restrict__ out,
					   uint64_t p_len, uint64_t v_len,
					   uint64_t n_head, uint64_t head_len);

__global__ void gelu_actv_kernel(float *__restrict__ x, uint64_t len);

#endif /* __KERNELS_CUDA_CUH */

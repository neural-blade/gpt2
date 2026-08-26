#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define CUDA_API_CHK(err)                                                      \
	do {                                                                   \
		if ((err) != cudaSuccess) {                                    \
			fprintf(stderr, "%s: %s at %s:%d in %s()\n",           \
				cudaGetErrorName(err),                         \
				cudaGetErrorString(err), __FILE__, __LINE__,   \
				__func__);                                     \
			exit(1);                                               \
		}                                                              \
	} while (0);

#ifdef __cplusplus
extern "C" {
#endif

void backend_malloc_host(void **ptr, size_t size)
{
	CUDA_API_CHK(cudaMallocHost(ptr, size));
}

void backend_malloc_device(void **ptr, size_t size)
{
	CUDA_API_CHK(cudaMalloc(ptr, size));
}

void backend_h2d(void *dst, const void *src, size_t count)
{
	CUDA_API_CHK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void backend_d2h(void *dst, const void *src, size_t count)
{
	CUDA_API_CHK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void backend_d2d(void *dst, const void *src, size_t count)
{
	CUDA_API_CHK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void backend_free_host(void *ptr)
{
	CUDA_API_CHK(cudaFreeHost(ptr));
}

void backend_free_device(void *ptr)
{
	CUDA_API_CHK(cudaFree(ptr));
}

void backend_move_h2d(void **dst, const void *src, size_t count)
{
	CUDA_API_CHK(cudaMalloc(dst, count));
	CUDA_API_CHK(cudaMemcpy(*dst, src, count, cudaMemcpyHostToDevice));
}

void add_f32v(float *__restrict__ a, const float *__restrict__ b, float alpha,
	      uint64_t len)
{
}

void sum_f32v(const float *__restrict__ a, uint64_t len)
{
}

void dot_f32v(const float *__restrict__ a, const float *__restrict__ b,
	      uint64_t len)
{
}

void gemm_f32(const float *__restrict__ a, const float *__restrict__ b,
	      float *__restrict__ c, float alpha, uint64_t m, uint64_t n,
	      uint64_t k)
{
}

void token_embd(const uint32_t *token_ids,
		const float *__restrict__ token_embd_w,
		float *__restrict__ embd, uint32_t token_count,
		uint32_t hidden_dim)
{
}

void pos_embd(float *__restrict__ embd, const float *__restrict__ pos_embd_w,
	      uint32_t token_count, uint32_t initial_token, uint32_t hidden_dim)
{
}

void layer_norm(const float *__restrict__ in, const float *__restrict__ weight,
		const float *__restrict__ bias, float *__restrict__ out,
		uint32_t seq_len, uint32_t hidden_dim, float eps)
{
}

void proj(const float *__restrict__ in, const float *__restrict__ weight,
	  const float *__restrict__ bias, float *__restrict__ out,
	  uint32_t seq_len, uint32_t hidden_dim, uint32_t out_dim)
{
}

void attn_scores(const float *__restrict__ q, const float *__restrict__ k,
		 float *__restrict__ scores, uint64_t q_len, uint64_t k_len,
		 uint64_t heads_count, uint64_t head_len)
{
}

void argmax_f32v(const float *__restrict__ v, uint64_t len)
{
}

void softmax(float *__restrict__ scores, uint64_t n_head,
	     uint64_t initial_token, uint64_t n_token)
{
}

void attn_v_weighted_sum(const float *__restrict__ p,
			 const float *__restrict__ v, float *__restrict__ out,
			 uint64_t p_len, uint64_t v_len, uint64_t n_head,
			 uint64_t head_len)
{
}

void gelu_actv(float *__restrict__ x, uint64_t len)
{
}

#ifdef __cplusplus
}
#endif

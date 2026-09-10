#include "../backend.h"
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "kernels_cuda.cuh"

#define CHECK(err)                                                             \
	do {                                                                   \
		cudaError_t _err = (err);                                      \
		if (_err != cudaSuccess) {                                     \
			fprintf(stderr, "%s: %s at %s:%d in %s()\n",           \
				cudaGetErrorName(_err),                        \
				cudaGetErrorString(_err), __FILE__, __LINE__,  \
				__func__);                                     \
			exit(1);                                               \
		}                                                              \
	} while (0);

static cudaEvent_t time_start, time_stop;

void backend_init(void)
{
	CHECK(cudaEventCreate(&time_start));
	CHECK(cudaEventCreate(&time_stop));
}

void backend_destroy(void)
{
	CHECK(cudaEventDestroy(time_start));
	CHECK(cudaEventDestroy(time_stop));
}

void backend_malloc_host(void **ptr, size_t size)
{
	CHECK(cudaMallocHost(ptr, size));
}

void backend_malloc_device(void **ptr, size_t size)
{
	CHECK(cudaMalloc(ptr, size));
}

void backend_h2d(void *dst, const void *src, size_t count)
{
	CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void backend_d2h(void *dst, const void *src, size_t count)
{
	CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void backend_d2d(void *dst, const void *src, size_t count)
{
	CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void backend_free_host(void *ptr) { CHECK(cudaFreeHost(ptr)); }

void backend_free_device(void *ptr) { CHECK(cudaFree(ptr)); }

void backend_move_h2d(void **dst, const void *src, size_t count)
{
	CHECK(cudaMalloc(dst, count));
	CHECK(cudaMemcpy(*dst, src, count, cudaMemcpyHostToDevice));
}

void backend_time_start(void) { CHECK(cudaEventRecord(time_start)); }

void backend_time_stop(void) { CHECK(cudaEventRecord(time_stop)); }

void backend_time_elaps(float *ms)
{
	CHECK(cudaEventSynchronize(time_stop));
	CHECK(cudaEventElapsedTime(ms, time_start, time_stop));
}

static inline void add_f32v(float *__restrict__ a, const float *__restrict__ b,
			    float alpha, uint64_t len, uint64_t b_dim)
{
	dim3 block_dim = 64;
	dim3 grid_dim(CEIL_DIV(len, block_dim.x));
	add_f32v_kernel<<<grid_dim, block_dim>>>(a, b, alpha, len, b_dim);
	CHECK(cudaGetLastError());
}

static inline void gemm_f32(const float *__restrict__ a,
			    const float *__restrict__ b, float *__restrict__ c,
			    float alpha, uint64_t m, uint64_t n, uint64_t k)
{
	if (m == 1) {
		dim3 block_dim(32, 16);
		dim3 grid_dim(CEIL_DIV(n, block_dim.y));
		gemv_f32_kernel<<<grid_dim, block_dim>>>(a, b, c, alpha, n, k);
	} else {
		dim3 block_dim(16, 16);
		dim3 grid_dim(CEIL_DIV(n, block_dim.x),
			      CEIL_DIV(m, block_dim.y));
		gemm_f32_kernel<<<grid_dim, block_dim>>>(a, b, c, alpha, m, n,
							 k);
	}
	CHECK(cudaGetLastError());
}

static inline void argmax_f32v(const float *__restrict__ v, uint64_t len,
			       uint32_t *__restrict__ out)
{
	argmax_f32v_kernel<<<1, THREADS_PER_BLOCK>>>(v, len, out);
	CHECK(cudaGetLastError());
}

void token_embd(const uint32_t *token_ids,
		const float *__restrict__ token_embd_w,
		float *__restrict__ embd, uint32_t token_count,
		uint32_t hidden_dim)
{
	dim3 grid_dim(CEIL_DIV(hidden_dim, THREADS_PER_BLOCK), token_count);
	token_embd_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(
	    token_ids, token_embd_w, embd, hidden_dim);
	CHECK(cudaGetLastError());
}

void pos_embd(float *__restrict__ embd, const float *__restrict__ pos_embd_w,
	      uint32_t token_count, uint32_t initial_token, uint32_t hidden_dim)
{
	add_f32v(embd, &pos_embd_w[initial_token * hidden_dim], 1.0f,
		 token_count * hidden_dim, token_count * hidden_dim);
}

void layer_norm(const float *__restrict__ in, const float *__restrict__ weight,
		const float *__restrict__ bias, float *__restrict__ out,
		uint32_t seq_len, uint32_t hidden_dim, float eps)
{
	dim3 block_dim(32, 2);
	dim3 grid_dim(1, CEIL_DIV(seq_len, block_dim.y));
	layer_norm_kernel<<<grid_dim, block_dim>>>(in, weight, bias, out,
						   seq_len, hidden_dim, eps);
	CHECK(cudaGetLastError());
}

void proj(const float *__restrict__ in, const float *__restrict__ weight,
	  const float *__restrict__ bias, float *__restrict__ out,
	  uint32_t seq_len, uint32_t hidden_dim, uint32_t out_dim)
{
	gemm_f32(in, weight, out, 1.0f, seq_len, out_dim, hidden_dim);
	add_f32v(out, bias, 1.0f, out_dim * seq_len, out_dim);
}

void attn_scores(const float *__restrict__ q, const float *__restrict__ k,
		 float *__restrict__ scores, uint64_t q_len, uint64_t k_len,
		 uint64_t heads_count, uint64_t head_len)
{
	dim3 block_dim(32, 8);
	dim3 grid_dim(CEIL_DIV(k_len + q_len, block_dim.x),
		      CEIL_DIV(heads_count, block_dim.y),
		      CEIL_DIV(q_len, block_dim.z));
	attn_scores_kernel<<<grid_dim, block_dim>>>(q, k, scores, q_len, k_len,
						    heads_count, head_len);
	CHECK(cudaGetLastError());
}

void softmax(float *__restrict__ scores, uint64_t n_head,
	     uint64_t initial_token, uint64_t n_token)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim(CEIL_DIV(n_head, block_dim.x),
		      CEIL_DIV(n_token, block_dim.y));

	softmax_kernel<<<grid_dim, block_dim>>>(scores, n_head, initial_token,
						n_token);
	CHECK(cudaGetLastError());
}

void attn_v_weighted_sum(const float *__restrict__ p,
			 const float *__restrict__ v, float *__restrict__ out,
			 uint64_t p_len, uint64_t v_len, uint64_t n_head,
			 uint64_t head_len)
{
	dim3 block_dim(32, 8);
	dim3 grid_dim(CEIL_DIV(head_len, block_dim.x),
		      CEIL_DIV(n_head, block_dim.y),
		      CEIL_DIV(p_len, block_dim.z));

	attn_v_weighted_sum_kernel<<<grid_dim, block_dim>>>(
	    p, v, out, p_len, v_len, n_head, head_len);

	CHECK(cudaGetLastError());
}

void resid(float *__restrict__ a, const float *__restrict__ b, uint64_t len)
{
	add_f32v(a, b, 1.0f, len, len);
}

void gelu_actv(float *__restrict__ x, uint64_t len)
{
	dim3 grid_dim(CEIL_DIV(len, THREADS_PER_BLOCK));
	gelu_actv_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(x, len);
	CHECK(cudaGetLastError());
}

void out_proj(const float *__restrict__ in, const float *__restrict__ weight,
	      float *__restrict__ out, uint32_t hidden_dim, uint32_t out_dim)
{
	gemm_f32(in, weight, out, 1.0f, 1, hidden_dim, out_dim);
}

void greedy_decode(const float *__restrict__ v, uint64_t len,
		   uint32_t *__restrict__ out)
{
	argmax_f32v(v, len, out);
}

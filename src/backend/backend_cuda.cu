#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

#define CUDA_API_CHK(err)                                                      \
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

#define THREAD_IDX(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__device__ inline void dot_f32v_device(const float *__restrict__ a,
				       const float *__restrict__ b,
				       uint64_t len, float *__restrict__ out)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i] * b[i];
	*out = sum;
}

__device__ inline void sum_f32v_device(const float *__restrict__ a,
				       uint64_t len, float *__restrict__ out)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i];
	*out = sum;
}

__device__ inline void argmax_f32v_device(const float *__restrict__ v,
					  uint64_t len,
					  uint32_t *__restrict__ out)
{
	float max   = v[0];
	uint32_t id = 0;
	for (uint32_t i = 1; i < len; ++i)
		if (v[i] > max) {
			max = v[i];
			id  = i;
		}
	*out = id;
}

__global__ void add_f32v_kernel(float *__restrict__ a,
				const float *__restrict__ b, float alpha,
				uint64_t len)
{
	uint64_t idx = THREAD_IDX(x);
	if (idx < len) a[idx] += alpha * b[idx];
}

__global__ void gemm_f32_kernel(const float *__restrict__ a,
				const float *__restrict__ b,
				float *__restrict__ c, float alpha, uint64_t m,
				uint64_t n, uint64_t k)
{
	uint64_t j = THREAD_IDX(x);
	uint64_t i = THREAD_IDX(y);
	if (i < m && j < n) {
		dot_f32v_device(&a[i * k], &b[j * k], k, &c[i * n + j]);
		c[i * n + j] *= alpha;
	}
}

__global__ void token_embd_kernel(const uint32_t *token_ids,
				  const float *__restrict__ token_embd_w,
				  float *__restrict__ embd, uint32_t hidden_dim)
{
	uint32_t i	   = THREAD_IDX(x);
	uint32_t token_idx = blockIdx.y;

	if (i < hidden_dim)
		embd[token_idx * hidden_dim + i] =
		    token_embd_w[token_ids[token_idx] * hidden_dim + i];
}

__global__ void layer_norm_kernel(const float *__restrict__ in,
				  const float *__restrict__ weight,
				  const float *__restrict__ bias,
				  float *__restrict__ out, uint32_t seq_len,
				  uint32_t hidden_dim, float eps)
{
	uint32_t i = THREAD_IDX(x);

	if (i < seq_len) {
		uint32_t t_offset    = i * hidden_dim;
		float inv_hidden_dim = 1.0f / hidden_dim;

		float sum	     = 0.0f;
		sum_f32v_device(&in[t_offset], hidden_dim, &sum);
		float mean   = sum * inv_hidden_dim;

		float sum_sq = 0.0f;
		for (uint32_t j = 0; j < hidden_dim; ++j)
			sum_sq += (in[t_offset + j] - mean) *
				  (in[t_offset + j] - mean);
		float var     = sum_sq * inv_hidden_dim;

		float inv_std = rsqrtf(var + eps);

		for (uint32_t j = 0; j < hidden_dim; ++j)
			out[t_offset + j] = (in[t_offset + j] - mean) *
						inv_std * weight[j] +
					    bias[j];
	}
}

__global__ void attn_scores_kernel(const float *__restrict__ q,
				   const float *__restrict__ k,
				   float *__restrict__ scores, uint64_t q_len,
				   uint64_t k_len, uint64_t heads_count,
				   uint64_t head_len)
{
	uint64_t t = THREAD_IDX(x);
	uint64_t j = THREAD_IDX(y);
	uint64_t i = THREAD_IDX(z);

	if (i < q_len && j < heads_count) {
		uint64_t n_keys = k_len + i + 1;
		if (t < n_keys) {
			uint64_t q_stride = heads_count * 3 * head_len;
			uint64_t k_stride = heads_count * head_len;

			float res;
			dot_f32v_device(&q[i * q_stride + j * head_len],
					&k[t * k_stride + j * head_len],
					head_len, &res);

			float inv_sqrt	  = rsqrtf((float)head_len);

			uint64_t row_base = heads_count * (i * (k_len + 1) +
							   (i * (i - 1)) / 2);
			scores[row_base + j * n_keys + t] = res * inv_sqrt;
		}
	}
}

__global__ void argmax_f32v_kernel(const float *v, uint64_t len, uint32_t *out)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
		argmax_f32v_device(v, len, out);
}

__global__ void softmax_kernel(float *__restrict__ scores, uint64_t n_head,
			       uint64_t initial_token, uint64_t n_token)
{
	uint64_t j = THREAD_IDX(x);
	uint64_t i = THREAD_IDX(y);

	if (i < n_token && j < n_head) {
		uint64_t n_keys	  = initial_token + 1 + i;
		uint64_t row_base = n_head * (i * (initial_token + 1) +
					      (i * (i - 1)) / 2);
		float *row	  = &scores[row_base + j * n_keys];

		uint32_t id;
		argmax_f32v_device(row, n_keys, &id);
		float max_val = row[id];

		float sum     = 0.0f;
		for (uint64_t k = 0; k < n_keys; ++k) {
			row[k] = expf(row[k] - max_val);
			sum += row[k];
		}

		float inv_sum = 1.0f / sum;
		for (uint64_t k = 0; k < n_keys; ++k) {
			row[k] *= inv_sum;
		}
	}
}

__global__ void attn_v_weighted_sum_kernel(const float *__restrict__ p,
					   const float *__restrict__ v,
					   float *__restrict__ out,
					   uint64_t p_len, uint64_t v_len,
					   uint64_t n_head, uint64_t head_len)
{
	uint64_t d = THREAD_IDX(x);
	uint64_t j = THREAD_IDX(y);
	uint64_t i = THREAD_IDX(z);

	if (i < p_len && j < n_head && d < head_len) {
		uint64_t v_stride = n_head * head_len;
		uint64_t n_keys	  = v_len + 1 + i;

		uint64_t row_base = n_head *
				    (i * (v_len + 1) + (i * (i - 1)) / 2);
		uint64_t p_head_offset = row_base + j * n_keys;
		uint64_t head_offset   = j * head_len + d;

		float sum	       = 0.0f;
		for (uint64_t k = 0; k < n_keys; ++k) {
			float weight = p[p_head_offset + k];
			float val    = v[k * v_stride + head_offset];
			sum += weight * val;
		}

		out[i * v_stride + head_offset] = sum;
	}
}

__global__ void gelu_actv_kernel(float *__restrict__ x, uint64_t len)
{
	uint64_t i = THREAD_IDX(x);
	if (i < len)
		x[i] = 0.5 * x[i] *
		       (1 + tanhf(0.7978845608f * x[i] +
				  0.0356774081f * x[i] * x[i] * x[i]));
}

static cudaEvent_t time_start, time_stop;

#ifdef __cplusplus
extern "C" {
#endif

void backend_init(void)
{
	CUDA_API_CHK(cudaEventCreate(&time_start));
	CUDA_API_CHK(cudaEventCreate(&time_stop));
}

void backend_destroy(void)
{
	CUDA_API_CHK(cudaEventDestroy(time_start));
	CUDA_API_CHK(cudaEventDestroy(time_stop));
}

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

void backend_time_start(void)
{
	CUDA_API_CHK(cudaEventRecord(time_start));
}

void backend_time_stop(void)
{
	CUDA_API_CHK(cudaEventRecord(time_stop));
}

void backend_time_elaps(float *ms)
{
	CUDA_API_CHK(cudaEventSynchronize(time_stop));
	CUDA_API_CHK(cudaEventElapsedTime(ms, time_start, time_stop));
}

void add_f32v(float *__restrict__ a, const float *__restrict__ b, float alpha,
	      uint64_t len)
{
	dim3 grid_dim(CEIL_DIV(len, THREADS_PER_BLOCK));
	add_f32v_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(a, b, alpha, len);
	CUDA_API_CHK(cudaGetLastError());
}

void gemm_f32(const float *__restrict__ a, const float *__restrict__ b,
	      float *__restrict__ c, float alpha, uint64_t m, uint64_t n,
	      uint64_t k)
{
	dim3 block_dim(32, 8);
	dim3 grid_dim(CEIL_DIV(n, block_dim.x), CEIL_DIV(m, block_dim.y));
	gemm_f32_kernel<<<grid_dim, block_dim>>>(a, b, c, alpha, m, n, k);
	CUDA_API_CHK(cudaGetLastError());
}

void token_embd(const uint32_t *token_ids,
		const float *__restrict__ token_embd_w,
		float *__restrict__ embd, uint32_t token_count,
		uint32_t hidden_dim)
{
	dim3 grid_dim(CEIL_DIV(hidden_dim, THREADS_PER_BLOCK), token_count);
	token_embd_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(
	    token_ids, token_embd_w, embd, hidden_dim);
	CUDA_API_CHK(cudaGetLastError());
}

void pos_embd(float *__restrict__ embd, const float *__restrict__ pos_embd_w,
	      uint32_t token_count, uint32_t initial_token, uint32_t hidden_dim)
{
	for (uint32_t i = 0; i < token_count; ++i)
		add_f32v(&embd[i * hidden_dim],
			 &pos_embd_w[(initial_token + i) * hidden_dim], 1.0f,
			 hidden_dim);
}

void layer_norm(const float *__restrict__ in, const float *__restrict__ weight,
		const float *__restrict__ bias, float *__restrict__ out,
		uint32_t seq_len, uint32_t hidden_dim, float eps)
{
	dim3 grid_dim(CEIL_DIV(seq_len, THREADS_PER_BLOCK));
	layer_norm_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(
	    in, weight, bias, out, seq_len, hidden_dim, eps);
	CUDA_API_CHK(cudaGetLastError());
}

void proj(const float *__restrict__ in, const float *__restrict__ weight,
	  const float *__restrict__ bias, float *__restrict__ out,
	  uint32_t seq_len, uint32_t hidden_dim, uint32_t out_dim)
{
	gemm_f32(in, weight, out, 1.0f, seq_len, out_dim, hidden_dim);
	for (uint64_t i = 0; i < seq_len; ++i)
		add_f32v(&out[i * out_dim], bias, 1.0f, out_dim);
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
	CUDA_API_CHK(cudaGetLastError());
}

void argmax_f32v(const float *__restrict__ v, uint64_t len,
		 uint32_t *__restrict__ out)
{
	argmax_f32v_kernel<<<1, 1>>>(v, len, out);
	CUDA_API_CHK(cudaGetLastError());
}

void softmax(float *__restrict__ scores, uint64_t n_head,
	     uint64_t initial_token, uint64_t n_token)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim(CEIL_DIV(n_head, block_dim.x),
		      CEIL_DIV(n_token, block_dim.y));

	softmax_kernel<<<grid_dim, block_dim>>>(scores, n_head, initial_token,
						n_token);
	CUDA_API_CHK(cudaGetLastError());
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

	CUDA_API_CHK(cudaGetLastError());
}

void gelu_actv(float *__restrict__ x, uint64_t len)
{
	dim3 grid_dim(CEIL_DIV(len, THREADS_PER_BLOCK));
	gelu_actv_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(x, len);
}

#ifdef __cplusplus
}
#endif

#include "kernels_cuda.cuh"

#define THREAD_IDX(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)

static __device__ inline void dot_f32v_device(const float *__restrict__ a,
					      const float *__restrict__ b,
					      uint64_t len,
					      float *__restrict__ out)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i] * b[i];
	*out = sum;
}

static __device__ inline void sum_f32v_device(const float *__restrict__ a,
					      uint64_t len,
					      float *__restrict__ out)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i];
	*out = sum;
}

static __device__ inline void argmax_f32v_device(const float *__restrict__ v,
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

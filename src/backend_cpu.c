#include <stdint.h>
#include <math.h>
#include <string.h>

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

void token_embd(const uint32_t *token_ids, const float *restrict token_embd_w,
		float *restrict embd, uint32_t token_count, uint32_t hidden_dim)
{
	for (uint32_t i = 0; i < token_count; ++i)
		memcpy(&embd[i * hidden_dim],
		       &token_embd_w[token_ids[i] * hidden_dim],
		       hidden_dim * sizeof(*embd));
}

void pos_embd(float *restrict embd, const float *restrict pos_embd_w,
	      uint32_t token_count, uint32_t initial_token, uint32_t hidden_dim)
{
	for (uint32_t i = 0; i < token_count; ++i)
		add_f32v_inplace(&embd[i * hidden_dim],
				 &pos_embd_w[(initial_token + i) * hidden_dim],
				 1.0f, hidden_dim);
}

void layer_norm(const float *restrict in, const float *restrict weight,
		const float *restrict bias, float *restrict out,
		uint32_t seq_len, uint32_t hidden_dim, float eps)
{
	float inv_hidden_dim = 1.0f / hidden_dim;

	for (uint32_t i = 0; i < seq_len; ++i) {
		uint32_t t_offset = i * hidden_dim;
		float sum	  = sum_f32v(&in[t_offset], hidden_dim);
		float mean	  = sum * inv_hidden_dim;

		float sum_sq	  = 0.0f;
		for (uint32_t j = 0; j < hidden_dim; ++j)
			sum_sq += (in[t_offset + j] - mean) *
				  (in[t_offset + j] - mean);
		float var     = sum_sq * inv_hidden_dim;

		float inv_std = 1.0f / sqrtf(var + eps);

		for (uint32_t j = 0; j < hidden_dim; ++j)
			out[t_offset + j] = (in[t_offset + j] - mean) *
						inv_std * weight[j] +
					    bias[j];
	}
}

void proj(const float *restrict in, const float *restrict weight,
	  const float *restrict bias, float *restrict out, uint32_t seq_len,
	  uint32_t hidden_dim, uint32_t out_dim)
{
	gemm_f32(in, weight, out, 1.0f, seq_len, out_dim, hidden_dim);
	for (uint64_t i = 0; i < seq_len; ++i)
		add_f32v_inplace(&out[i * out_dim], bias, 1.0f, out_dim);
}

void attn_scores(const float *restrict q, const float *restrict k,
		 float *restrict scores, uint64_t q_len, uint64_t k_len,
		 uint64_t heads_count, uint64_t head_len)
{
	float inv_sqrt	  = 1.0f / sqrtf((float)head_len);
	uint64_t q_stride = heads_count * 3 * head_len;
	uint64_t k_stride = heads_count * head_len;
	uint64_t n_keys	  = k_len + 1;
	uint64_t row_base = 0;

	for (uint64_t i = 0; i < q_len; ++i) {
		for (uint64_t j = 0; j < heads_count; ++j) {
			uint64_t head_base = row_base + n_keys * j;

			for (uint64_t t = 0; t < n_keys; ++t)
				scores[head_base + t] =
				    dot_f32v(&q[i * q_stride + j * head_len],
					     &k[t * k_stride + j * head_len],
					     head_len) *
				    inv_sqrt;
		}

		row_base += n_keys * heads_count;
		++n_keys;
	}
}

uint32_t argmax_f32v(const float *restrict v, uint64_t len)
{
	float max   = v[0];
	uint32_t id = 0;
	for (uint32_t i = 1; i < len; ++i)
		if (v[i] > max) {
			max = v[i];
			id  = i;
		}
	return id;
}

void softmax(float *restrict scores, uint64_t n_head, uint64_t initial_token,
	     uint64_t n_token)
{
	uint64_t n_keys	  = initial_token + 1;
	uint64_t row_base = 0;

	for (uint64_t i = 0; i < n_token; ++i) {
		for (uint64_t j = 0; j < n_head; ++j) {
			uint64_t head_base  = j * n_keys;
			float *restrict row = &scores[row_base + head_base];

			float max_val	    = row[argmax_f32v(row, n_keys)];
			float sum	    = 0.0f;
			for (uint64_t k = 0; k < n_keys; ++k) {
				row[k] = expf(row[k] - max_val);
				sum += row[k];
			}

			float inv_sum = 1.0f / sum;
			for (uint64_t k = 0; k < n_keys; ++k) row[k] *= inv_sum;
		}

		row_base += n_keys * n_head;
		++n_keys;
	}
}

void attn_v_weighted_sum(const float *restrict p, const float *restrict v,
			 float *restrict out, uint64_t p_len, uint64_t v_len,
			 uint64_t n_head, uint64_t head_len)
{
	uint64_t v_stride = n_head * head_len;
	uint64_t n_keys	  = v_len + 1;
	uint64_t row_base = 0;

	for (uint64_t i = 0; i < p_len * v_stride; ++i) out[i] = 0.0f;

	for (uint64_t i = 0; i < p_len; ++i) {
		for (uint64_t j = 0; j < n_head; ++j) {
			uint64_t head_base = j * head_len;

			for (uint64_t k = 0; k < n_keys; ++k) {
				add_f32v_inplace(&out[i * v_stride + head_base],
						 &v[k * v_stride + head_base],
						 p[row_base + j * n_keys + k],
						 head_len);
			}
		}

		row_base += n_keys * n_head;
		++n_keys;
	}
}

void gelu_actv(float *restrict x, uint64_t len)
{
	for (uint64_t i = 0; i < len; ++i)
		x[i] = 0.5 * x[i] *
		       (1 + tanhf(0.7978845608f * x[i] +
				  0.0356774081f * x[i] * x[i] * x[i]));
}

#ifndef __BACKEND_H
#define __BACKEND_H

#include <stdint.h>

void add_f32v(float *restrict a, const float *restrict b, float alpha,
	      uint64_t len);
float sum_f32v(const float *restrict a, uint64_t len);
float dot_f32v(const float *restrict a, const float *restrict b, uint64_t len);
void gemm_f32(const float *restrict a, const float *restrict b,
	      float *restrict c, float alpha, uint64_t m, uint64_t n,
	      uint64_t k);
void layer_norm(const float *restrict in, const float *restrict weight,
		const float *restrict bias, float *restrict out,
		uint32_t seq_len, uint32_t hidden_dim, float eps);
void proj(const float *restrict in, const float *restrict weight,
	  const float *restrict bias, float *restrict out, uint32_t seq_len,
	  uint32_t hidden_dim, uint32_t out_dim);
void attn_scores(const float *restrict q, const float *restrict k,
		 float *restrict scores, uint64_t q_len, uint64_t k_len,
		 uint64_t heads_count, uint64_t head_len);
uint32_t argmax_f32v(const float *restrict v, uint64_t len);
void softmax(float *restrict scores, uint64_t n_head, uint64_t initial_token,
	     uint64_t n_token);
void attn_v_weighted_sum(const float *restrict p, const float *restrict v,
			 float *restrict out, uint64_t p_len, uint64_t v_len,
			 uint64_t n_head, uint64_t head_len);
void softmax(float *restrict scores, uint64_t n_head, uint64_t initial_token,
	     uint64_t n_token);
void gelu_actv(float *restrict x, uint64_t len);
void token_embd(const uint32_t *restrict token_ids,
		const float *restrict token_embd_w, float *restrict embd,
		uint32_t token_count, uint32_t hidden_dim);
void pos_embd(float *restrict embd, const float *restrict pos_embd_w,
	      uint32_t token_count, uint32_t initial_token,
	      uint32_t hidden_dim);

#endif /* __BACKEND_H */

#ifndef __BACKEND_H
#define __BACKEND_H

#include <stdint.h>
#include <stddef.h>

void backend_init(void);
void backend_destroy(void);
void backend_malloc_host(void **ptr, size_t size);
void backend_malloc_device(void **ptr, size_t size);
void backend_h2d(void *dst, const void *src, size_t count);
void backend_d2h(void *dst, const void *src, size_t count);
void backend_d2d(void *dst, const void *src, size_t count);
void backend_free_host(void *ptr);
void backend_free_device(void *ptr);
void backend_move_h2d(void **dst, const void *src, size_t count);
void backend_time_start(void);
void backend_time_stop(void);
void backend_time_elaps(float *ms);

void add_f32v(float *a, const float *b, float alpha, uint64_t len);
void sum_f32v(const float *a, uint64_t len);
void dot_f32v(const float *a, const float *b, uint64_t len);
void gemm_f32(const float *a, const float *b, float *c, float alpha, uint64_t m,
	      uint64_t n, uint64_t k);
void layer_norm(const float *in, const float *weight, const float *bias,
		float *out, uint32_t seq_len, uint32_t hidden_dim, float eps);
void proj(const float *in, const float *weight, const float *bias, float *out,
	  uint32_t seq_len, uint32_t hidden_dim, uint32_t out_dim);
void attn_scores(const float *q, const float *k, float *scores, uint64_t q_len,
		 uint64_t k_len, uint64_t heads_count, uint64_t head_len);
void argmax_f32v(const float *v, uint64_t len, uint32_t *out);
void softmax(float *scores, uint64_t n_head, uint64_t initial_token,
	     uint64_t n_token);
void attn_v_weighted_sum(const float *p, const float *v, float *out,
			 uint64_t p_len, uint64_t v_len, uint64_t n_head,
			 uint64_t head_len);
void softmax(float *scores, uint64_t n_head, uint64_t initial_token,
	     uint64_t n_token);
void gelu_actv(float *x, uint64_t len);
void token_embd(const uint32_t *token_ids, const float *token_embd_w,
		float *embd, uint32_t token_count, uint32_t hidden_dim);
void pos_embd(float *embd, const float *pos_embd_w, uint32_t token_count,
	      uint32_t initial_token, uint32_t hidden_dim);

#endif /* __BACKEND_H */

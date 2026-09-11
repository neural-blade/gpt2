#define THREAD_IDX(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)

static __device__ inline void sum_f32v_device(const float *__restrict__ a,
					      uint64_t len,
					      float *__restrict__ out)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < len; ++i) sum += a[i];
	*out = sum;
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

#define FULL_MASK 0xffffffffu

__global__ void layer_norm_kernel_optimized(const float *__restrict__ in,
					    const float *__restrict__ weight,
					    const float *__restrict__ bias,
					    float *__restrict__ out,
					    int seq_len, int hidden_dim,
					    float eps)
{
	int lane	     = threadIdx.x % warpSize;
	int token	     = THREAD_IDX(y);
	int t_offset	     = token * hidden_dim;

	float mean	     = 0.0f;
	float sum_sq	     = 0.0f;
	float inv_hidden_dim = 1.0f / hidden_dim;

	if (token < seq_len) {
		for (int i = lane; i < hidden_dim; i += warpSize)
			mean += in[t_offset + i] * inv_hidden_dim;

		for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
			mean += __shfl_xor_sync(FULL_MASK, mean, offset);

		for (int i = lane; i < hidden_dim; i += warpSize)
			sum_sq += (in[t_offset + i] - mean) *
				  (in[t_offset + i] - mean);

		for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
			sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);

		float inv_std = rsqrtf(sum_sq * inv_hidden_dim + eps);
		for (int i = lane; i < hidden_dim; i += warpSize)
			out[t_offset + i] = (in[t_offset + i] - mean) *
						inv_std * weight[i] +
					    bias[i];
	}
}

#define THREAD_IDX(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)

__global__ void add_f32v_kernel(float *__restrict__ a,
				const float *__restrict__ b, float alpha,
				uint64_t len)
{
	uint64_t idx = THREAD_IDX(x);
	if (idx < len) a[idx] += alpha * b[idx];
}

__global__ void add_f32v_kernel_optimized(float *__restrict__ a,
					  const float *__restrict__ b,
					  float alpha, int len, int b_dim)
{
	int idx = THREAD_IDX(x);
	if (idx < len) a[idx] += alpha * b[idx % b_dim];
}

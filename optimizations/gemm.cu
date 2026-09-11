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

#define TILE_SIZE 16

__global__ void gemm_f32_kernel_optimized(const float *__restrict__ a,
					  const float *__restrict__ b,
					  float *__restrict__ c, float alpha,
					  int m, int n, int k)
{
	__shared__ float a_tile[TILE_SIZE][TILE_SIZE];
	__shared__ float b_tile[TILE_SIZE][TILE_SIZE + 1];

	int tid_x = THREAD_IDX(x);
	int tid_y = THREAD_IDX(y);
	int tx	  = threadIdx.x;
	int ty	  = threadIdx.y;
	int b_col = blockIdx.x * TILE_SIZE + ty;

	float acc = 0.0f;
	for (int i = 0; i < k; i += TILE_SIZE) {
		int tile_offset = i + tx;
		if (tid_y < m && tile_offset < k)
			a_tile[ty][tx] = a[tid_y * k + tile_offset];
		else
			a_tile[ty][tx] = 0.0f;

		if (tile_offset < k && b_col < n)
			b_tile[tx][ty] = b[b_col * k + tile_offset];
		else
			b_tile[tx][ty] = 0.0f;

		__syncthreads();

		for (int j = 0; j < TILE_SIZE; ++j)
			acc += a_tile[ty][j] * b_tile[j][tx];

		__syncthreads();
	}

	if (tid_y < m && tid_x < n) c[tid_y * n + tid_x] = alpha * acc;
}

#define FULL_MASK 0xffffffffu

__global__ void gemv_f32_kernel(const float *__restrict__ a,
				const float *__restrict__ b,
				float *__restrict__ c, float alpha, int n,
				int k)
{
	int lane  = threadIdx.x % warpSize;
	int b_col = blockIdx.x * blockDim.y + threadIdx.y;
	float acc = 0.0f;

	if (b_col < n) {
		for (int i = 0; i < k; i += warpSize) {
			int a_col   = i + lane;

			float val_a = a_col < k ? a[a_col] : 0.0f;
			float val_b = a_col < k ? b[b_col * k + a_col] : 0.0f;

			acc += val_a * val_b;
		}

		for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
			acc += __shfl_down_sync(FULL_MASK, acc, offset);
		}

		if (threadIdx.x == 0) c[b_col] = alpha * acc;
	}
}

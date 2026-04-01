
template <typename T>
struct RowView {
	T* rowData_;
	__host__ __device__ RowView(T* data) : rowData_(data) {}
	__host__ __device__ T& operator[](int j) { return rowData_[j]; }
	__host__ __device__ const T& operator[](int j) const { return rowData_[j]; }
};

template <typename T>
struct MatrixView {
	T* data_;
	int rows_,cols_;
	__host__ __device__ MatrixView(T* data, int r, int c) : data_(data), rows_(r), cols_(c) {}
	__host__ __device__ RowView<T> operator[](int i) { return RowView<T>(data_ + (i * cols_)); }
	__host__ __device__ const RowView<T> operator[](int i) const { return RowView<T>(data_ + (i * cols_)); }
};

template <int BLOCKSIZE>
__global__ void matrix_multiplication_kernel(const float* A_ptr, const float* B_ptr, float* MULT_ptr, int M, int N, int K, bool transposeB) {
    MatrixView<float> A((float*)A_ptr,M,N);
    MatrixView<float> B((float*)B_ptr,
                        transposeB ? K : N,   // rows: K (transposed) or N (normal)
                        transposeB ? N : K);  // cols: N (transposed) or K (normal)
    MatrixView<float> MULT(MULT_ptr,M,K);

    //cache
    __shared__ float tempA[BLOCKSIZE][BLOCKSIZE], tempB[BLOCKSIZE][BLOCKSIZE+1];

    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0;
    for (int i = 0; i < (N + BLOCKSIZE - 1)/BLOCKSIZE; i++) {
        int a_col = i * BLOCKSIZE + tx;
        int b_row = i * BLOCKSIZE + ty;

        //load data into shared memory
        tempA[ty][tx] = (row < M && a_col < N) ? A[row][a_col] : 0.0f;
        //tempB[ty][tx] = (b_row < N && col < K) ? B[b_row][col] : 0.0f;
        //load B as transpose for coalesced access
        //tempB[tx][ty] = (b_row < N && col < K) ? B[b_row][col] : 0.0f;
		if (transposeB) {
			// B is (K x N), computing A * B^T: access B[col][b_row]
			// col is a row index into B → < K; b_row is a col index into B → < N
			tempB[tx][ty] =
				(col < K && b_row < N) ? B[col][b_row] : 0.0f;
		} else {
			// Standard GEMM: B is (N x K)
			// b_row is a row index into B → < N; col is a col index into B → < K
			tempB[tx][ty] =
				(b_row < N && col < K) ? B[b_row][col] : 0.0f;
		}


        //synchronize
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BLOCKSIZE; j++) {
            sum += tempA[ty][j] * tempB[tx][j];
        }
        //synchronize
        __syncthreads();
    }
    if (row < M && col < K) {
        MULT[row][col] = sum;
    }
}

__device__ __forceinline__
void warp_reduce_online(float &sum, float &mx) {
    unsigned mask = 0xFFFFFFFFU;

    for (int s = warpSize / 2; s > 0; s >>= 1) {
        float sum_other = __shfl_down_sync(mask, sum, s);
        float max_other = __shfl_down_sync(mask, mx, s);

        float new_max = fmaxf(mx, max_other);

        sum = sum * expf(mx - new_max)
            + sum_other * expf(max_other - new_max);

        mx = new_max;
    }
}

template <int BLOCKSIZE>
__global__ void stage2_norm_softmax(const float* sptr, int M, int N, int d) {
    //each block is responsible for one row of the matrix
    MatrixView<float> score((float*)sptr,M,N);

    const float invSqrtD = 1.0f / sqrtf(d * 1.0f);

    int tid = threadIdx.x;
    int stride = blockDim.x ;

    for (int i = tid ; i < N; i += stride) {
        score[blockIdx.x][i] *= invSqrtD;
    }
    __syncthreads();


    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    //block-stride loop
    //this loop runs
    //each row has N elements so compute ceil(N/256) max and sum values. This is done parallely in each thread
    for (int i = tid ; i < N; i += stride) {
        float val = score[blockIdx.x][i];

        float prev_max = local_max;
        local_max = fmaxf(prev_max, val);

        local_sum = local_sum * expf(prev_max - local_max) + expf(val - local_max);
    }

    //now use warp primitives to compute max and sum
    // warp reduction
    warp_reduce_online(local_sum, local_max);

    //now we have 256 / 32 values that we need to reduce
    //reduce using shared memory

    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    int num_warps = blockDim.x / warpSize;

    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];

    if (lane == 0) {
        smem_max[warp_id] = local_max;
        smem_sum[warp_id] = local_sum;
    }
    __syncthreads();

    //block reduction
    float b_max = -FLT_MAX;
    float b_sum = 0.0f;
    if (warp_id == 0) {
        b_max = (lane < num_warps) ? smem_max[lane] : -FLT_MAX;
        b_sum = (lane < num_warps) ? smem_sum[lane] : 0.0f;

        warp_reduce_online(b_sum, b_max);
        if (lane == 0) {
            smem_max[0] = b_max;
            smem_sum[0] = b_sum;
        }
    }
    __syncthreads();
    b_max = smem_max[0];
    b_sum = smem_sum[0];
    for (int i = tid ; i < N; i += stride) {
        float val = score[blockIdx.x][i];
        val = exp(val - b_max)/b_sum;
        score[blockIdx.x][i] = val;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
	dim3 threadsPerBlock(32,32);
	dim3 blocksPerGridQK((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(M + threadsPerBlock.y - 1) / threadsPerBlock.y);
	dim3 blocksPerGridOut((d + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float *sptr;
    cudaMalloc(&sptr,M*N*sizeof(float));

	matrix_multiplication_kernel<32><<<blocksPerGridQK, threadsPerBlock>>>(Q, K, sptr, M, d, N, true);
	cudaError_t err = cudaGetLastError();
	assert(err == cudaSuccess);
	cudaDeviceSynchronize();

    dim3 threadsPerBlock2(256);
    dim3 blocksPerGrid2(M);
    stage2_norm_softmax<256><<<blocksPerGrid2, threadsPerBlock2>>>(sptr,M,N,d);
	err = cudaGetLastError();
	assert(err == cudaSuccess);
	cudaDeviceSynchronize();

	matrix_multiplication_kernel<32><<<blocksPerGridOut, threadsPerBlock>>>(sptr, V, output, M, N, d, false);
	err = cudaGetLastError();
	assert(err == cudaSuccess);
	cudaDeviceSynchronize();

    cudaFree(sptr);
}

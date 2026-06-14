#include "my_cuda_kernels.cuh"

__global__ void relu_forward_kernel(float* in, float* out, int sz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sz) return;
    out[idx] = (in[idx] > 0.0f) ? in[idx] : 0.0f;
}

__global__ void relu_backward_kernel(float* in, float* grad_out, float* grad_in, int sz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sz) return;
    grad_in[idx] = (in[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

void relu_forward(float *in, float* out, int sz) {
    dim3 blockSize(256);
    dim3 gridSize((sz + blockSize.x - 1)/blockSize.x);

    relu_forward_kernel<<<gridSize,blockSize>>>(in,out,sz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu_backward(float *in, float* grad_out, float* grad_in, int sz) {
    dim3 blockSize(256);
    dim3 gridSize((sz + blockSize.x - 1)/blockSize.x);

    relu_backward_kernel<<<gridSize,blockSize>>>(in,grad_out,grad_in,sz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// C[M,N] = A[M,K] * B[K,N], tiled GEMM ported from leetgpu/medium/matrix_mult.cu.
// B is loaded transposed into shared memory for coalesced reads; tB is padded by
// one column to avoid bank conflicts.
template <int TILE>
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE + 1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        tA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tB[tx][ty] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            sum += tA[ty][k] * tB[tx][k];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<TILE><<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}


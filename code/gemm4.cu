#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cfloat>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cooperative_groups.h>

#include "ndev_cuda_utility.hpp"
#include "memory_alloc.h"
#include "cuda_timer.h"

using namespace std;

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
__global__ void gemm_kernel( const half* A_ptr, const half* B_ptr, half* C_ptr, int M, int N, int K, float alpha, float beta) {

    MatrixView<half> A((half*)A_ptr,M,K);
    MatrixView<half> B((half*)B_ptr,K,N);
    MatrixView<half> C(C_ptr,M,N);

    __shared__ half tempA[BLOCKSIZE][BLOCKSIZE];
    __shared__ half tempB[BLOCKSIZE][BLOCKSIZE + 1]; // avoid bank conflicts

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCKSIZE + ty * 2;
    int col = blockIdx.x * BLOCKSIZE + tx * 2;

    int tx2 = tx + BLOCKSIZE/2;
    int ty2 = ty + BLOCKSIZE/2;

    float sum00, sum01, sum10, sum11;
    sum00 = sum01 = sum10 = sum11 = 0.0f;

    for (int tile = 0; tile < (K + BLOCKSIZE - 1) / BLOCKSIZE; tile++) {

        int a_col0 = tile * BLOCKSIZE + tx;
        int a_col1 = tile * BLOCKSIZE + tx + BLOCKSIZE/2;

        tempA[ty][tx] =
            (row < M && a_col0 < K) ? A[row][a_col0] : __float2half(0.0f);

        tempA[ty][tx + BLOCKSIZE/2] =
            (row < M && a_col1 < K) ? A[row][a_col1] : __float2half(0.0f);

        tempA[ty2][tx] =
            (row + 1 < M && a_col0 < K) ? A[row + 1][a_col0] : __float2half(0.0f);

        tempA[ty2][tx + BLOCKSIZE/2] =
            (row + 1 < M && a_col1 < K) ? A[row + 1][a_col1] : __float2half(0.0f);


        int b_row0 = tile * BLOCKSIZE + ty;
        int b_row1 = tile * BLOCKSIZE + ty + BLOCKSIZE/2;

        tempB[ty][tx] =
            (b_row0 < K && col < N) ? B[b_row0][col] : __float2half(0.0f);

        tempB[ty + BLOCKSIZE/2][tx] =
            (b_row1 < K && col < N) ? B[b_row1][col] : __float2half(0.0f);

        tempB[ty][tx2] =
            (b_row0 < K && col + 1 < N) ? B[b_row0][col + 1] : __float2half(0.0f);

        tempB[ty + BLOCKSIZE/2][tx2] =
            (b_row1 < K && col + 1 < N) ? B[b_row1][col + 1] : __float2half(0.0f);

        __syncthreads();

#pragma unroll
        for (int j = 0; j < BLOCKSIZE; j++) {
            float a0 = __half2float(tempA[ty][j]);
            float a1 = __half2float(tempA[ty2][j]);

            float b0 = __half2float(tempB[j][tx]);
            float b1 = __half2float(tempB[j][tx2]);

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float c00 = __half2float(C[row][col]);
        C[row][col] = __float2half(alpha * sum00 + beta * c00);
    }

    if (row < M && col + 1 < N) {
        float c01 = __half2float(C[row][col + 1]);
        C[row][col + 1] = __float2half(alpha * sum01 + beta * c01);
    }

    if (row + 1 < M && col < N) {
        float c10 = __half2float(C[row + 1][col]);
        C[row + 1][col] = __float2half(alpha * sum10 + beta * c10);
    }

    if (row + 1 < M && col + 1 < N) {
        float c11 = __half2float(C[row + 1][col + 1]);
        C[row + 1][col + 1] = __float2half(alpha * sum11 + beta * c11);
    }
}

extern "C" void solve(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    float alpha, float beta
) {
    dim3 block(8, 8);
    dim3 grid( (N + 2 * block.x - 1) / (2 * block.x), (M + 2 * block.y - 1) / (2 * block.y));

    gemm_kernel<16><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

int main() {
    MemoryManager mmgr;
    //const int n = (1 << 4) + 7;
    const int M = 1024, K = 1024, N = 1024;
    auto A = mmgr.allocate_host_and_device<half>(M * K);
    auto B = mmgr.allocate_host_and_device<half>(K * N);
    auto C = mmgr.allocate_host_and_device<half>(M * N);
    fillArrayRandom(A.hostPtr,M * K ,true, 331);
    fillArrayRandom(B.hostPtr,K * N ,true, 331);
    fillArrayRandom(C.hostPtr,M * N ,true, 331);

    A.copy_to_device(M * K);
    B.copy_to_device(K * N);
    C.copy_to_device(M * N);

    const float alpha = 3.1415;
    const float beta  = 2.7183;

    CudaTimer timer;
    timer.start();
    solve(A.devicePtr, B.devicePtr, C.devicePtr, M, N, K, alpha, beta);
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU reverse array is " << runtime << "us" << endl;
}

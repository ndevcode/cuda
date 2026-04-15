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
#include "ndev_math_utility.hpp"

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

template <int TILE, int TT>
__global__ __launch_bounds__(256, 2)
void gemm_kernel(const half* A_ptr, const half* B_ptr, half* C_ptr,
                 int M, int N, int K, float alpha, float beta) {
    constexpr int REG     = TILE / TT;
    constexpr int THREADS = TT * TT;
    constexpr int H       = THREADS / TILE;
    constexpr int LOADS   = TILE / H;

    MatrixView<const half> A((half*)A_ptr, M, K);
    MatrixView<const half> B((half*)B_ptr, K, N);
    MatrixView<half>       C(C_ptr, M, N);

    __shared__ half sA[TILE][TILE + 8];
    __shared__ half sB[TILE][TILE + 8];

    // lo accumulates even-k products, hi accumulates odd-k products
    float rC[REG][REG] = {};

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;
    int tidx = threadIdx.x % TT;
    int tidy = threadIdx.x / TT;

    for (int rk = 0; rk < (K + TILE - 1) / TILE; ++rk) {
                { // begin cooperative loads
                        int ltx = threadIdx.x % TILE;
                        int lty = threadIdx.x / TILE;
                        for (int i = 0; i < LOADS; ++i) {
                                int local_row = i * H + lty;
                                int a_row = block_row + local_row;
                                int a_col = rk * TILE + ltx;
                                sA[local_row][ltx] =
                                        (a_row < M && a_col < K) ? __ldg(&A[a_row][a_col]) : __float2half(0.0f);

                                int b_row = rk * TILE + local_row;
                                int b_col = block_col + ltx;
                                half b_val = (b_row < K && b_col < N) ? __ldg(&B[b_row][b_col]) : __float2half(0.0f);
                                sB[ltx][local_row] = b_val;           // B is loaded transposed
                        }
                } // end cooperative loads
                __syncthreads();

                // outer-product, k in steps of 2 via half2, vectorized k-loads but float accumulation
                #pragma unroll
                for (int ik = 0; ik < TILE; ik += 2) {
                        half2 rB_h2[REG];
                        #pragma unroll
                        for (int i = 0; i < REG; ++i) {
                                rB_h2[i] = *(const half2*)&sB[i * TT + tidx][ik];
                        }
                        #pragma unroll
                        for (int i = 0; i < REG; ++i) {
                half2 a_h2 = *(const half2*)&sA[i * TT + tidy][ik];
                                float a0 = __half2float(__low2half(a_h2));
                                float a1 = __half2float(__high2half(a_h2));
                                #pragma unroll
                                for (int j = 0; j < REG; ++j) {
                    float b0 = __half2float(__low2half(rB_h2[j]));
                    float b1 = __half2float(__high2half(rB_h2[j]));
                    rC[i][j] = __fmaf_rn(a0, b0, rC[i][j]);
                    rC[i][j] = __fmaf_rn(a1, b1, rC[i][j]);
                }
                        }
                }
                __syncthreads();
        }

    // write back: apply alpha/beta ----
    #pragma unroll
    for (int i = 0; i < REG; ++i) {
        #pragma unroll
        for (int j = 0; j < REG; ++j) {
            int out_row = block_row + i * TT + tidy;
            int out_col = block_col + j * TT + tidx;
            if (out_row < M && out_col < N) {
                float c_val = __half2float(C[out_row][out_col]);
                C[out_row][out_col] = __float2half(alpha * rC[i][j] + beta * c_val);
            }
        }
    }
}

extern "C" void solve(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta)
{
    constexpr int TILE = 64;
    constexpr int TT   = 16;
    dim3 block(TT * TT);   // 256 threads, 1D
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<TILE, TT><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
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
    cout << "Runtime for GPU GEMM is " << runtime << "us" << endl;
}

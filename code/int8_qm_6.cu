//nvcc -arch=sm_89 -x cu int8_qm_6.cu
/* The optimizations that got #1, in order of impact:

64×64 tile with 4×4 register tile (was 32×32 with 2×2) — 4× better compute:load ratio
dp4a intrinsic — 4 MACs per instruction, 4× fewer inner loop iterations
Transposed sB layout — enabled contiguous k-packing for dp4a
sB padding by 4 — eliminated 16-way shared memory bank conflicts
Coalesced stores via sC staging — doubled store throughput
__launch_bounds__(256, 4) — forced register budget, tripled occupancy
*/

Hoisted zero-point correction — removed 2 subtracts per MAC from inner loop
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
    int rows_, cols_;
    __host__ __device__ MatrixView(T* data, int r, int c) : data_(data), rows_(r), cols_(c) {}
    __host__ __device__ RowView<T> operator[](int i) { return RowView<T>(data_ + (i * cols_)); }
    __host__ __device__ const RowView<T> operator[](int i) const { return RowView<T>(data_ + (i * cols_)); }
};

__global__ void compute_colsum(const int8_t* B, int* colsum, int K, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    int sum = 0;
    for (int k = 0; k < K; k++)
        sum += B[k * N + j];
    colsum[j] = sum;
}

template <int TILE, int TT>
__global__ __launch_bounds__(256, 4)
void kernel_1(const int8_t* A_ptr, const int8_t* B_ptr, int8_t* C_ptr,
                         int M, int N, int K,
                         float scale_A, float scale_B, float scale_C,
                         int zA, int zB, int zC,
                         const int* colsum)
{
        constexpr int REG     = TILE / TT;
        constexpr int THREADS = TT * TT;
        constexpr int H       = THREADS / TILE;
        constexpr int LOADS   = TILE / H;

        MatrixView<const int8_t> A((int8_t*)A_ptr, M, K);
        MatrixView<const int8_t> B((int8_t*)B_ptr, K, N);
        MatrixView<int8_t>       C(C_ptr, M, N);

        // sA[row][k]  - k contiguous (natural layout)
        // sB[col][k]  - k contiguous (transposed from B[k][col])
        __shared__ int8_t sA[TILE][TILE];
        __shared__ int8_t sB[TILE][TILE+4];

        int rC[REG][REG] = {};
        int rowsum[REG]  = {};

        int block_row = blockIdx.y * TILE;
        int block_col = blockIdx.x * TILE;

        int tidx = threadIdx.x % TT;
        int tidy = threadIdx.x / TT;

        for (int rk = 0; rk < (K + TILE - 1) / TILE; ++rk)
        {
                // ---- cooperative load ----
                {
                        int ltx = threadIdx.x % TILE;
                        int lty = threadIdx.x / TILE;

                        for (int i = 0; i < LOADS; ++i)
                        {
                                int local_row = i * H + lty;

                                // A: store naturally as sA[row][k]
                                int a_row = block_row + local_row;
                                int a_col = rk * TILE + ltx;
                                sA[local_row][ltx] =
                                        (a_row < M && a_col < K) ? A[a_row][a_col] : 0;

                                // B: load B[k][col], store TRANSPOSED as sB[col][k]
                                int b_row = rk * TILE + local_row;  // k index
                                int b_col = block_col + ltx;         // col index
                                int8_t b_val = (b_row < K && b_col < N) ? B[b_row][b_col] : 0;
                                sB[ltx][local_row] = b_val;          // transposed!
                        }
                }

                __syncthreads();

                // ---- dp4a outer-product, k steps by 4 ----
                for (int ik = 0; ik < TILE; ik += 4)
                {
                        int rA[REG], rB[REG];  // each holds 4 packed int8 values

#pragma unroll
                        for (int i = 0; i < REG; ++i)
                        {
                                // 4 consecutive k-values for this row of A
                                rA[i] = *reinterpret_cast<const int*>(&sA[i * TT + tidy][ik]);
                                // 4 consecutive k-values for this col of B (transposed)
                                rB[i] = *reinterpret_cast<const int*>(&sB[i * TT + tidx][ik]);
                        }

#pragma unroll
                        for (int i = 0; i < REG; ++i)
                        {
                                // rowsum: dot product with {1,1,1,1} packed as 0x01010101
                                rowsum[i] = __dp4a(rA[i], 0x01010101, rowsum[i]);

#pragma unroll
                                for (int j = 0; j < REG; ++j)
                                        rC[i][j] = __dp4a(rA[i], rB[j], rC[i][j]);
                        }
                }

                __syncthreads();
        }

        // ---- zero-point correction + quantize + coalesced store ----
        //int correction = K * zA * zB;

        __shared__ int8_t sC_data[TILE * TILE];
        MatrixView<int8_t> sC(sC_data, TILE, TILE);

        for (int i = 0; i < REG; ++i)
    {
        for (int j = 0; j < REG; ++j)
        {
            int out_col = block_col + j * TT + tidx;
            if (out_col < N)
            {
                // recompute correction inline rather than storing
                int val = rC[i][j] - zB * rowsum[i]
                    - zA * colsum[out_col]
                    + K * zA * zB;
                int iv = __float2int_rn((float)val * scale_A * scale_B / scale_C) + zC;
                sC[i * TT + tidy][j * TT + tidx] =
                    (int8_t)(iv < -128 ? -128 : (iv > 127 ? 127 : iv));
            }
        }
    }

        __syncthreads();

        {
                int ltx = threadIdx.x % TILE;
                int lty = threadIdx.x / TILE;

                for (int i = 0; i < LOADS; ++i)
                {
                        int out_row = block_row + i * H + lty;
                        int out_col = block_col + ltx;
                        if (out_row < M && out_col < N)
                                C[out_row][out_col] = sC[i * H + lty][ltx];
                }
        }
}

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C,
                      int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C)
{
    constexpr int TILE = 64;
    constexpr int TT   = 16;

    int* d_colsum;
    cudaMalloc(&d_colsum, N * sizeof(int));
    compute_colsum<<<(N + 255) / 256, 256>>>(B, d_colsum, K, N);

    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    kernel_1<TILE, TT><<<grid, TT * TT>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C, d_colsum);

    cudaDeviceSynchronize();
    cudaFree(d_colsum);
}

int main() {
    MemoryManager mmgr;
    //const int n = (1 << 4) + 7;
    const int M = 8192, K = 2048, N = 4096;
    //const int M = 1024, K = 1024, N = 1024;
    auto A = mmgr.allocate_host_and_device<int8_t>(M * K);
    auto B = mmgr.allocate_host_and_device<int8_t>(K * N);
    auto C = mmgr.allocate_host_and_device<int8_t>(M * N);
    fillArrayRandom(A.hostPtr,M * K ,true, 127);
    fillArrayRandom(B.hostPtr,K * N ,true, 127);

    int zero_point_A = -111;
    int zero_point_B = -128;
    int zero_point_C = 127;

    float scale_A = 3.1415;
    float scale_B = 3.1415;
    float scale_C = 3.1415;

    CudaTimer timer;
    timer.start();
    solve(A.devicePtr, B.devicePtr, C.devicePtr, M, N, K,
                      scale_A, scale_B, scale_C,
                      zero_point_A, zero_point_B, zero_point_C);
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU batched matrix mult is " << runtime << "us" << endl;
}

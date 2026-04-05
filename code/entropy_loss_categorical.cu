#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cfloat>

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

__device__ __forceinline__
void warp_reduce_online(float &sum, float &mx) {
    unsigned mask = __activemask();

    for (int s = warpSize / 2; s > 0; s >>= 1) {
        float sum_other = __shfl_down_sync(mask, sum, s);
        float max_other = __shfl_down_sync(mask, mx, s);

        float new_max = fmaxf(mx, max_other);

        sum = sum * __expf(mx - new_max)
            + sum_other * __expf(max_other - new_max);

        mx = new_max;
    }
}

__global__ void kernel_1_block_reduce(const float* logits_ptr, const int* true_labels, float* loss, int N, int C) {
        MatrixView<float> logits((float*)logits_ptr,N,C);
        int tid = threadIdx.x;

        //formula :
        //float loss_i = -z_i_y + m_i + logf(sum_exp);

        float local_max  = -FLT_MAX;
        float local_exp_sum  = 0.0f;

        float local_max2 = -FLT_MAX;
        float local_exp_sum2 = 0.0f;

        int stride = blockDim.x;

#pragma unroll 4
        for (int i = tid; i < C; i += 2 * stride) {

                // ---- first element ----
                if (i < C) {
                        float val = logits[blockIdx.x][i];

                        float prev_max = local_max;
                        local_max = fmaxf(prev_max, val);

                        local_exp_sum = local_exp_sum * __expf(prev_max - local_max)
                                + __expf(val - local_max);
                }

                // ---- second element (independent stream) ----
                int j = i + stride;
                if (j < C) {
                        float val2 = logits[blockIdx.x][j];

                        float prev_max2 = local_max2;
                        local_max2 = fmaxf(prev_max2, val2);

                        local_exp_sum2 = local_exp_sum2 * __expf(prev_max2 - local_max2)
                                + __expf(val2 - local_max2);
                }
        }

        // ---- merge the two streams ----
        float new_max = fmaxf(local_max, local_max2);

        float merged_sum =
                local_exp_sum  * __expf(local_max  - new_max) +
                local_exp_sum2 * __expf(local_max2 - new_max);

        local_max = new_max;
        local_exp_sum = merged_sum;

        // warp reduction
        warp_reduce_online(local_exp_sum, local_max);

        int lane = tid & (warpSize - 1);
        int warp_id = tid / warpSize;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;

        __shared__ float smem_max[32];
        __shared__ float smem_exp_sum[32];

        if (lane == 0) {
                smem_max[warp_id] = local_max;
                smem_exp_sum[warp_id] = local_exp_sum;
        }

        __syncthreads();

        if (warp_id == 0) {
                float b_max = (lane < num_warps) ? smem_max[lane] : -FLT_MAX;
                float b_exp_sum = (lane < num_warps) ? smem_exp_sum[lane] : 0.0f;

                warp_reduce_online(b_exp_sum, b_max);

                if (lane == 0) {
                        int y = true_labels[blockIdx.x];
                        float z_i_y = logits[blockIdx.x][y];

                        float loss_i = -z_i_y + b_max + logf(b_exp_sum);
                        atomicAdd(&loss[0],loss_i/N);
                }
        }
}

extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    dim3 block(128);
    dim3 grid(N);

    cudaMemset(loss,0,sizeof(float));

    kernel_1_block_reduce<<<grid,block>>>(logits,true_labels,loss,N,C);
    cudaDeviceSynchronize();

}

int main() {
    MemoryManager mmgr;
    //const int n = (1 << 4) + 7;
    const int N = 1e4;
    const int C = 1e3;
    auto logits = mmgr.allocate_host_and_device<float>(N * C);
    auto true_labels = mmgr.allocate_host_and_device<int>(N);
    auto loss = mmgr.allocate_host_and_device<float>(1);
    fillArrayRandom(logits.hostPtr,N*C,true, 331);
    fillArrayRandom(true_labels.hostPtr,N,true, C);

    logits.copy_to_device(N*C);
    true_labels.copy_to_device(N);

    CudaTimer timer;
    timer.start();
    solve(logits.devicePtr,true_labels.devicePtr,loss.devicePtr,N,C);
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU categorical cross entropy loss is " << runtime << "us" << endl;
}

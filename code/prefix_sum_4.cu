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
struct AddOp {
    __device__ static T identity() { return T(0); }

    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

template <typename T>
struct MulOp {
    __device__ static T identity() { return T(1); }

    __device__ T operator()(T a, T b) const {
        return a * b;
    }
};

template <typename T>
struct MaxOp {
    __device__ static T identity() { return -FLT_MAX; }

    __device__ T operator()(T a, T b) const {
        return fmaxf(a, b);
    }
};

//for array 1,2,3,4
//computes 1,3,6,10
template <typename T, typename Op>
__device__ T warp_scan(T val, Op op) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31;

    for (int offset = 1; offset < 32; offset <<= 1) {
        T n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) {
            val = op(n, val);   // associative combine
        }
    }
    return val;
}

template <typename T, typename Op>
__global__ void kernel_1_scan(const T* input, T* parr, T* output, int N) {

        Op op;

        int tid = threadIdx.x;
        int gid = blockDim.x * blockIdx.x + tid;

        T val = (gid < N) ? input[gid] : Op::identity();

        // warp scan
        T prefix = warp_scan<T, Op>(val, op);

        __shared__ T shmem[32];

        int lane = tid & 31;
        int warp_id = tid / 32;
        int num_warps = (blockDim.x + 31) / 32;

        // last lane of each warp writes warp sum
        if (lane == 31) {
                shmem[warp_id] = prefix;
        }
        __syncthreads();

        // scan warp sums
        if (warp_id == 0) {
                T warp_val = (lane < num_warps) ? shmem[lane] : Op::identity();
                T scan = warp_scan<T, Op>(warp_val, op);
                if (lane < num_warps) {
                        shmem[lane] = scan;
                }
        }
        __syncthreads();

        // add previous warp contribution
        if (warp_id > 0) {
                prefix = op(shmem[warp_id - 1], prefix);
        }

        if (gid < N) {
                output[gid] = prefix;
        }

        // last element of block → partial sum
        if (tid == blockDim.x - 1 || gid == N - 1) {
                parr[blockIdx.x] = prefix;
        }
}

template <typename T, typename Op>
__global__ void kernel_2_scan(T* parr, int num_blocks) {

        extern __shared__ T shmem[];
        Op op;

        int tid = threadIdx.x;

        shmem[tid] = (tid < num_blocks) ? parr[tid] : Op::identity();
        __syncthreads();

        for (int offset = 1; offset < num_blocks; offset <<= 1) {
                T val = (tid >= offset) ? shmem[tid - offset] : Op::identity();
                __syncthreads();
                shmem[tid] = op(shmem[tid], val);
                __syncthreads();
        }

        if (tid < num_blocks) {
                parr[tid] = shmem[tid];
        }
}

template <typename T, typename Op>
__global__ void kernel_3_broadcast(const T* parr, T* output, int N) {

    Op op;

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    if (gid < N && blockIdx.x > 0) {
        output[gid] = op(parr[blockIdx.x - 1], output[gid]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {

        using T = float;
        using Op = AddOp<T>;   // swap here: AddOp / MulOp / MaxOp

        dim3 block(1024);
        dim3 grid((N + block.x - 1) / block.x);

        T* parr;
        cudaMalloc(&parr, grid.x * sizeof(T));

        // Phase 1
        kernel_1_scan<T, Op><<<grid, block>>>(input, parr, output, N);
        cudaDeviceSynchronize();

        // next power of 2 threads
        int num_threads = 1;
        while (num_threads < grid.x) num_threads <<= 1;

        // Phase 2
        kernel_2_scan<T, Op><<<1, num_threads, num_threads * sizeof(T)>>>(parr, grid.x);
        cudaDeviceSynchronize();

        // Phase 3
        kernel_3_broadcast<T, Op><<<grid, block>>>(parr, output, N);

        cudaFree(parr);
}

int main() {
    MemoryManager mmgr;
    //const int n = (1 << 4) + 7;
    const int n = 250000;
    auto input = mmgr.allocate_host_and_device<float>(n);
    auto output = mmgr.allocate_host_and_device<float>(n);
    fillArrayRandom(input.hostPtr,n,true, 331);

    input.copy_to_device(n);

    CudaTimer timer;
    timer.start();
    solve(input.devicePtr,output.devicePtr,n);
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU prefix sum is " << runtime << "us" << endl;
}

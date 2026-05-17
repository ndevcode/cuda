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

struct Add {
    template <typename T>
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

struct Max {
    template <typename T>
    __device__ T operator()(T a, T b) const {
        return max(a,b);
    }
};

//for array 1,2,3,4
//computes 1,3,6,10
template <typename T, typename Op>
__device__ T warp_scan_inclusive(T val, Op op) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    for (int offset = 1; offset < warpSize ; offset <<= 1) {
        T n = __shfl_up_sync(mask,val,offset);
        if (lane >= offset) {
            val = op(val,n);
        }
    }
    return val;
}

template <typename T, typename Op>
__device__ T warp_scan_exclusive(T val, Op op, T identity) {
    T inclusive = warp_scan_inclusive(val,op);
    int lane = threadIdx.x & (warpSize - 1);
    return lane == 0 ? identity :  __shfl_up_sync(0xffffffff, inclusive, 1);
}

template <typename T, typename Op, int BLOCKSIZE>
__device__ T block_scan_inclusive(T val, Op op, T* smem) {
    int num_warps = (BLOCKSIZE + warpSize - 1)/warpSize;

    int tid = threadIdx.x;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    T scan = warp_scan_inclusive(val,op);

    if (lane == warpSize - 1) {
        smem[warp_id] = scan;
    }
    __syncthreads();

    if (warp_id == 0) {
        T warp_sum = (lane < num_warps) ? smem[lane] : T{};
        warp_sum = warp_scan_inclusive(warp_sum, op);
        if (lane < num_warps) {
            smem[lane] = warp_sum;
        }
    }
    __syncthreads();

    if (warp_id > 0) {
        scan = op(scan, smem[warp_id - 1]);
    }
    return scan;
}

template <typename T, typename Op, int BLOCKSIZE>
__device__ T block_scan_exclusive(T val, Op op, T identity, T* smem) {
    T inclusive = block_scan_inclusive<T,Op,BLOCKSIZE>( val, op, smem);

    int tid = threadIdx.x;

    __syncthreads();

    smem[tid] = inclusive;

    __syncthreads();

    if (tid == 0)
        return identity;

    return smem[tid - 1];
}

template <typename T, typename Op, int BLOCKSIZE>
__global__ void scan_kernel(const T* input, T* output, T* block_sums, int N, Op op, T identity) {
    __shared__ T smem[BLOCKSIZE];
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    T val = (gid < N) ? input[gid] : identity;

    T scan = block_scan_inclusive<T, Op, BLOCKSIZE> (val, op, smem);

    if (gid < N) {
        output[gid] = scan;
    }

    int remaining = N - blockIdx.x * blockDim.x;

    int valid = min(BLOCKSIZE, remaining);

    if (block_sums != nullptr && threadIdx.x == valid - 1) {
        block_sums[blockIdx.x] = scan;
    }
}

template <typename T, typename Op, int BLOCKSIZE>
__global__ void uniform_add(const T* block_prefix, T* output, int N, Op op) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= N || blockIdx.x == 0) {
        return ;
    }

    output[gid] = op(output[gid],block_prefix[blockIdx.x-1]);
}

template <typename T, typename Op, int BLOCKSIZE>
void device_scan(const T* input, T* output, int N, Op op, T identity) {
    int num_blocks = (N + BLOCKSIZE - 1)/BLOCKSIZE;

    if (num_blocks <= 1) {
        scan_kernel<T,Op,BLOCKSIZE><<<1,BLOCKSIZE>>>(input, output, nullptr, N, op, identity);
        return;
    }

    T* block_sums;
    T* scanned_block_sums;

    cudaMalloc(&block_sums, num_blocks * sizeof(T));
    cudaMalloc(&scanned_block_sums, num_blocks * sizeof(T));

    scan_kernel<T,Op,BLOCKSIZE><<<num_blocks,BLOCKSIZE>>>(input, output, block_sums, N, op, identity);

    //recursive
    device_scan<T,Op,BLOCKSIZE>(block_sums, scanned_block_sums, num_blocks, op, identity);

    uniform_add<T,Op,BLOCKSIZE><<<num_blocks, BLOCKSIZE>>>(scanned_block_sums, output, N, op);

    cudaFree(block_sums);
    cudaFree(scanned_block_sums);
}

extern "C" void solve(const float* input, float* output, int N) {
    constexpr int BLOCKSIZE = 256;
    device_scan<float,Add,BLOCKSIZE>(input,output,N,Add{},0.0f);
    cudaDeviceSynchronize();
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
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int BLOCKSIZE = 256;
    device_scan<float,Add,BLOCKSIZE>(input,output,N,Add{},0.0f);
    cudaDeviceSynchronize();
}

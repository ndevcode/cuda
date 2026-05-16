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
__device__ float warp_scan_exclusive(T val, Op op, T identity) {
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

template <int BLOCKSIZE>
__global__ void scan_blocks(const float* input, float* output, float* block_sums, int N) {
    __shared__ float smem[BLOCKSIZE];
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    float val = (gid < N) ? input[gid] : 0.0f;

    float scan = block_scan_inclusive<float, Add, BLOCKSIZE> (val, Add{}, smem);

    if (gid < N) {
        output[gid] = scan;
    }

    int last = min(blockDim.x , N - blockIdx.x * blockDim.x) - 1;

    if (threadIdx.x == last) {
        block_sums[blockIdx.x] = scan;
    }
}

template <int BLOCKSIZE>
__global__ void uniform_add(const float* block_prefix, float* output, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= N || blockIdx.x == 0) {
        return ;
    }

    output[gid] += block_prefix[blockIdx.x-1];
}

__global__ void kernel_2_hillis_steele(float* parr, int num_blocks) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;

    shmem[tid] = (tid < num_blocks) ? parr[tid] : 0.0f;
    __syncthreads();

    //Hillis-Steele scan
    for (int offset = 1; offset < num_blocks; offset <<= 1) {
        float val = (tid >= offset) ? shmem[tid - offset] : 0.0f;
        __syncthreads();
        shmem[tid] += val;
        __syncthreads();
    }
    if (tid < num_blocks) {
        parr[tid] = shmem[tid];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    constexpr int BLOCKSIZE = 256;

    dim3 block(BLOCKSIZE);
    dim3 grid((N + block.x - 1)/block.x);

    float *parr;
    cudaMalloc(&parr,grid.x*sizeof(float));

    //phase 1 : block reduction
    scan_blocks<BLOCKSIZE><<<grid,block>>>(input,output,parr,N);
    cudaDeviceSynchronize();

    int num_threads = 1;
    for (; num_threads < grid.x; num_threads <<= 1);

    //phase 2 : overall prefix sum
    kernel_2_hillis_steele<<<1,num_threads,num_threads*sizeof(float)>>>(parr,grid.x);
    cudaDeviceSynchronize();

    //phase 3 : send this data to the blocks
    uniform_add<BLOCKSIZE><<<grid,block>>>(parr,output,N);

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

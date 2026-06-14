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

struct AddOp {
    __device__ float operator()(float a, float b) const {
        return a + b;
    }
};

template <typename T, typename Op>
__device__ T warp_reduce(T x, Op op) {
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset >= 1; offset >>= 1) {
        T y = __shfl_down_sync(mask, x, offset);
        x = op(x,y);
    }
    return x;
}

// Coalesced batch norm with warp_reduce reused via a shared-memory transpose.
//
//   block = (32, 32):  threadIdx.x == channel-within-block (lane / contiguous),
//                      threadIdx.y == batch partition (the reduction axis).
//
// Global accesses keep the channel on threadIdx.x so a warp reads 32 consecutive
// columns input[i*C + j], j = base..base+31  -> fully coalesced.  The reduction,
// however, is over threadIdx.y, which is NOT the lane axis -- so we stage the
// per-thread partials in padded shared memory and read them back TRANSPOSED, so
// that warp w owns one channel and its 32 lanes hold that channel's row-partials.
// Then the existing warp_reduce collapses the column in log2(32) shuffles.
__global__ void batch_norm_warp(const float* input, const float* gamma, const float* beta, float* output,
                                int N, int C, float eps) {
    const int tx = threadIdx.x;                       // channel within block (lane)
    const int ty = threadIdx.y;                       // batch partition / warp id
    const int j  = blockIdx.x * blockDim.x + tx;      // global channel index
    const bool active = (j < C);

    // +1 column of padding so the transposed column read is bank-conflict free.
    __shared__ float sSum[32][33];
    __shared__ float sSq [32][33];
    __shared__ float sMu [32];
    __shared__ float sInv[32];

    // --- Pass 1: coalesced accumulation.  channel on tx, batch strided by ty.
    float sum = 0.0f, sumsq = 0.0f;
    if (active) {
        for (int i = ty; i < N; i += blockDim.y) {
            float x = input[i * C + j];               // warp -> 32 consecutive j: coalesced
            sum   += x;
            sumsq += x * x;
        }
    }
    sSum[ty][tx] = sum;
    sSq [ty][tx] = sumsq;
    __syncthreads();

    // --- Pass 2: reduce over the batch by REUSING warp_reduce on a transposed view.
    // Warp w == ty owns channel (base + w); lane == tx indexes the 32 row-partials.
    // Reading sSum[tx][ty] walks a padded column -> distinct banks per lane.
    float s  = warp_reduce(sSum[tx][ty], AddOp{});
    float s2 = warp_reduce(sSq [tx][ty], AddOp{});
    if (tx == 0) {                                    // lane 0 holds the channel total
        float mu  = s  / N;
        float var = fmaxf(s2 / N - mu * mu, 0.0f);    // guard catastrophic cancellation
        sMu [ty] = mu;                                // index by channel-within-block
        sInv[ty] = rsqrtf(var + eps);
    }
    __syncthreads();

    // --- Pass 3: normalize.  channel back on tx -> coalesced loads and stores.
    if (active) {
        float mu  = sMu [tx];
        float inv = sInv[tx];
        float g   = gamma[j];
        float b   = beta[j];
        for (int i = ty; i < N; i += blockDim.y) {
            int idx = i * C + j;
            output[idx] = g * ((input[idx] - mu) * inv) + b;   // centered form
        }
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    dim3 block(32, 32);                               // 32 channels x 32 batch partitions
    dim3 grid((C + block.x - 1) / block.x, 1);
    batch_norm_warp<<<grid, block>>>(input, gamma, beta, output, N, C, eps);
    CUDA_CHECK(cudaPeekAtLastError());
}

int main() {
    MemoryManager mmgr;

    const int N = 1e4;
    const int C = 1 << 10;
    const float eps = 1e-5;

    auto input  = mmgr.allocate_host_and_device<float>(N * C);
    fillArrayRandom(input.hostPtr,N * C ,true, 101);
    input.copy_to_device(N * C);

    auto gamma  = mmgr.allocate_host_and_device<float>(C);
    fillArrayRandom(gamma.hostPtr,C,false,11);
    gamma.copy_to_device(C);

    auto beta   = mmgr.allocate_host_and_device<float>(C);
    fillArrayRandom(beta.hostPtr,C,true,11);
    beta.copy_to_device(C);

    auto output = mmgr.allocate_host_and_device<float>(N * C);

    CudaTimer timer;
    timer.start();

    solve(input.devicePtr,gamma.devicePtr,beta.devicePtr,output.devicePtr,N,C,eps);

    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU batch normalization is " << runtime << "us" << endl;
}

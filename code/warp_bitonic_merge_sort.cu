#include <algorithm>
#include <cfloat>
#include <iostream>

#include <cuda_runtime.h>

#include "cuda_timer.h"
#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"

using namespace std;

__device__ __forceinline__ float bitonic_warp_sort_val(float x) {
    const int lane = threadIdx.x & (warpSize - 1);
    const unsigned mask = __activemask();

    for (int size = 2; size <= warpSize; size <<= 1) {
        const bool dir = ((lane & size) == 0);

        for (int stride = size >> 1; stride >= 1; stride >>= 1) {
            const float y = __shfl_xor_sync(mask, x, stride);
            const float minv = fminf(x, y);
            const float maxv = fmaxf(x, y);

            if ((lane & stride) == 0) {
                x = dir ? minv : maxv;
            } else {
                x = dir ? maxv : minv;
            }
        }
    }

    return x;
}

__device__ __forceinline__ int binary_search_partition(
    const float* A,
    const float* B,
    int sizeA,
    int sizeB,
    int k) {
    int left = max(0, k - sizeB);
    int right = min(k, sizeA);

    while (left < right) {
        const int mid = (left + right) >> 1;
        const int j = k - mid;

        const float leftA = (mid > 0) ? A[mid - 1] : -FLT_MAX;
        const float rightA = (mid < sizeA) ? A[mid] : FLT_MAX;
        const float leftB = (j > 0) ? B[j - 1] : -FLT_MAX;
        const float rightB = (j < sizeB) ? B[j] : FLT_MAX;

        if (leftA > rightB) {
            right = mid;
        } else if (leftB > rightA) {
            left = mid + 1;
        } else {
            return mid;
        }
    }

    return left;
}

__device__ float* merge_sort_block(float* input, float* output) {
    const int tid = threadIdx.x;

    for (int S = warpSize * 2; S < blockDim.x; S <<= 1) {
        const int segment = tid / (S << 1);
        const int local_k = tid & ((S << 1) - 1);
        const int start = segment * (S << 1);

        const float* A = &input[start];
        const float* B = &input[start + S];

        const int i = binary_search_partition(A, B, S, S, local_k);
        const int j = local_k - i;

        const float a_val = (i < S) ? A[i] : FLT_MAX;
        const float b_val = (j < S) ? B[j] : FLT_MAX;

        output[tid] = fminf(a_val, b_val);
        __syncthreads();

        float* tmp = input;
        input = output;
        output = tmp;
        __syncthreads();
    }

    return input;
}

__device__ __forceinline__ float warp_merge_32_32(float* shared) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);

    const int pair = warp >> 1;
    const bool is_upper = warp & 1;
    const int base = pair * 64;

    const float* A = &shared[base];
    const float* B = &shared[base + 32];

    const int k = lane + (is_upper ? 32 : 0);
    const int i = binary_search_partition(A, B, 32, 32, k);
    const int j = k - i;

    const float a_val = (i < 32) ? A[i] : FLT_MAX;
    const float b_val = (j < 32) ? B[j] : FLT_MAX;

    return fminf(a_val, b_val);
}

__global__ void block_sort_kernel(const float* input, float* output, int N) {
    extern __shared__ float shmem[];

    float* bufA = shmem;
    float* bufB = shmem + blockDim.x;

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float x = FLT_MAX;
    if (gid < N) {
        x = input[gid];
    }

    bufA[tid] = x;
    __syncthreads();

    x = bitonic_warp_sort_val(x);
    bufA[tid] = x;
    __syncthreads();

    x = warp_merge_32_32(bufA);
    bufA[tid] = x;
    __syncthreads();

    float* final_buf = merge_sort_block(bufA, bufB);

    if (gid < N) {
        output[gid] = final_buf[tid];
    }
}

__global__ void merge_kernel(const float* input, float* output, int N, int S) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) {
        return;
    }

    const int segment = tid / (2 * S);
    const int local_k = tid % (2 * S);
    const int start = segment * (2 * S);

    if (start >= N) {
        return;
    }

    const int sizeA = min(S, N - start);
    const int sizeB = max(0, min(S, N - start - sizeA));

    if (local_k >= sizeA + sizeB) {
        return;
    }

    const float* A = &input[start];
    const float* B = &input[start + sizeA];

    const int i = binary_search_partition(A, B, sizeA, sizeB, local_k);
    const int j = local_k - i;

    const float a_val = (i < sizeA) ? A[i] : FLT_MAX;
    const float b_val = (j < sizeB) ? B[j] : FLT_MAX;

    output[start + local_k] = fminf(a_val, b_val);
}

extern "C" void solve(float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    const dim3 block(256);
    const dim3 grid((N + block.x - 1) / block.x);

    const size_t shmem_size = 2 * block.x * sizeof(float);

    block_sort_kernel<<<grid, block, shmem_size>>>(input, output, N);
    cudaDeviceSynchronize();

    float* in = output;
    float* out = input;
    int S = block.x;

    while (S < N) {
        merge_kernel<<<grid, block>>>(in, out, N, S);
        cudaDeviceSynchronize();
        std::swap(in, out);
        S <<= 1;
    }

    if (in != output) {
        cudaMemcpy(output, in, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

int main() {
    MemoryManager mmgr;
    const int n = (1 << 16) + 37;

    auto input = mmgr.allocate_host_and_device<float>(n);
    auto output = mmgr.allocate_host_and_device<float>(n);

    fillArrayRandom(input.hostPtr, n, true, 331);
    cout << "First 8 input values: ";
    for (int i = 0; i < min(8, n); ++i) {
        cout << input.hostPtr[i] << (i + 1 == min(8, n) ? '\n' : ' ');
    }
    input.copy_to_device(n);

    CudaTimer timer;
    timer.start();
    solve(input.devicePtr, output.devicePtr, n);
    timer.stop();

    output.copy_to_host(n);

    cout << "Runtime for GPU sort is " << timer.elapsedTimeUs() << "us" << endl;
    cout << "First 8 sorted values: ";
    for (int i = 0; i < min(8, n); ++i) {
        cout << output.hostPtr[i] << (i + 1 == min(8, n) ? '\n' : ' ');
    }
}

//nvcc -x cu parallel_blelloch.cpp -o pb
#include <iostream>
#include <chrono>
#include <device_launch_parameters.h>
#include "memory_alloc.h"
#include "utility.hpp"

const int DSIZE = 8;
const int THREADS_PER_BLOCK = DSIZE;

using namespace std;

__global__ void blellochPrefixSum(float* arr, int N, int logN) {
    __shared__ float temp[THREADS_PER_BLOCK];
    int tid = threadIdx.x;

    //load input into shared memory
    temp[tid] = arr[tid];
    __syncthreads();

    //upsweep (reduce) phase
    for (int d = 0; d < logN; d++) {
        int step = 1 << (d + 1);
        int idx = (tid + 1) * step - 1;
        if (idx < N) {
            temp[idx] += temp[idx - (step >> 1)];
        }
        __syncthreads();
    }
    if (tid == 0) temp[N-1] = 0;
    __syncthreads();

    //downward phase
    for (int d = logN-1; d >= 0; d--) {
        int step = 1 << (d + 1);
        int idx = (tid + 1) * step - 1;
        if (idx < N) {
            float t = temp[idx - (step >> 1)];
            temp[idx - (step >> 1)] = temp[idx];
            temp[idx] += t;
        }
        __syncthreads();
    }

    //write result back to global memory
    arr[tid] = temp[tid];

}

int main() {
    MemoryManager mmgr;
    auto A = mmgr.allocate_host_and_device<float>(DSIZE);
    srand(42);
    for (int i = 0; i < DSIZE-1; i++) {
        A.hostPtr[i] = rand()%DSIZE;
    }
    A.hostPtr[DSIZE-1] = 0;

    printArr(A.hostPtr,DSIZE);

    auto log2 = [](unsigned int x) -> int {
        int res = 0;
        while (x >>= 1) ++res;
        return res;
    };

    dim3 grid(1);
    dim3 block(THREADS_PER_BLOCK);
    {
        auto start = std::chrono::high_resolution_clock::now();

        A.copy_to_device(DSIZE);

        blellochPrefixSum <<<grid,block>>> (A.devicePtr,DSIZE,log2(DSIZE));
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        A.copy_to_host(DSIZE);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cout << endl;
        cout << "Time taken : " << duration.count() << " us" << endl;
        cout << endl;
    }
    printArr(A.hostPtr,DSIZE);
}

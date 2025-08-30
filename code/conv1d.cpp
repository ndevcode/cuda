//nvcc -x cu conv1d.cpp -o conv1d

#include <iostream>
#include <cassert>
#include <chrono>
#include <device_launch_parameters.h>
#include "memory_alloc.h"
#include "utility.hpp"

using namespace std;

const int DSIZE = 1 << 15;
const int THREADS_PER_BLOCK = 1 << 7; 

__global__ void conv1d_algo1(int N, float* input, float *output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float result = 0;
    for (int i = idx; i < min(idx + 3,N); i++) {
        result += input[i];
    }
    output[idx] = result / 3.0;
}


__global__ void conv1d_algo2(int N, float *input, float *output) {
    __shared__ float support[THREADS_PER_BLOCK+2];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    support[threadIdx.x] = input[idx];
    if (threadIdx.x < 2) {
        support[THREADS_PER_BLOCK + threadIdx.x] = input[THREADS_PER_BLOCK + idx];
    }
    __syncthreads();
    float result = 0;
    for (int i = threadIdx.x; i < min(threadIdx.x + 3,N); i++) {
        result += support[i];
    }
    output[idx] = result / 3.0;
}

int main() {
    MemoryManager mmgr;

    auto A = mmgr.allocate_host_and_device<float>(DSIZE+2);
    auto B = mmgr.allocate_host_and_device<float>(DSIZE+2);
    auto Y = mmgr.allocate_host_and_device<float>(DSIZE);
    auto Z = mmgr.allocate_host_and_device<float>(DSIZE);
    srand(42);

    for (int i = 0; i < DSIZE+2; i++) {
        A.hostPtr[i] = B.hostPtr[i] = rand()%DSIZE;
    } 

    dim3 grid(DSIZE/THREADS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);

    {
        auto start = std::chrono::high_resolution_clock::now();
        A.copy_to_device(DSIZE);

        conv1d_algo1 <<<grid,block>>> (DSIZE,A.devicePtr,Y.devicePtr);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        Y.copy_to_host(DSIZE);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        cout << endl;
        cout << "Time taken : " << duration.count() << " us" << endl;
        cout << endl;
        //printArr(A.hostPtr,DSIZE+2);
        //cout << endl;
        //printArr(Y.hostPtr,DSIZE);
    }
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        B.copy_to_device(DSIZE);
        conv1d_algo2 <<<grid,block>>> (DSIZE,B.devicePtr,Z.devicePtr);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        Z.copy_to_host(DSIZE);

        auto stop1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
        cout << endl;
        cout << "Time taken : " << duration1.count() << " us" << endl;
        cout << endl;
        //printArr(B.hostPtr,DSIZE+2);
        //cout << endl;
        //printArr(Z.hostPtr,DSIZE);
    }

    assert(areEqual(Y.hostPtr,Z.hostPtr,DSIZE));


}

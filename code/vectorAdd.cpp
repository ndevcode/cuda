//nvcc -x cu vadd.cpp -o vadd
#include <iostream>
#include <chrono>
#include <cassert>
#include <device_launch_parameters.h>
#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"

using namespace std;

const int NUM_BLOCKS = 4096;
const int BLOCK_SIZE = 256;

void serialAdd(const float* A, const float* B, float* SUM, int sz) {
    for (unsigned gid = 0; gid < sz; gid++) {
        SUM[gid] = A[gid] + B[gid];
    }
}

__global__ void vectorAdd (const float* A, const float* B, float* SUM, int sz) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < sz) {
        SUM[gid] = A[gid] + B[gid];
    }
}

__global__ void vectorAddGridStride (const float* A, const float* B, float* SUM, int sz) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < sz; i += blockDim.x * gridDim.x) {
        SUM[i] = A[i] + B[i];
    }
}

int main() {
    MemoryManager mmgr;
    auto A = mmgr.allocate_host_and_device<float>(NUM_BLOCKS*BLOCK_SIZE);
    auto B = mmgr.allocate_host_and_device<float>(NUM_BLOCKS*BLOCK_SIZE);
    auto SUM = mmgr.allocate_host_and_device<float>(NUM_BLOCKS*BLOCK_SIZE);
    auto SUM_serial = mmgr.allocate_only_host<float>(NUM_BLOCKS*BLOCK_SIZE);

    srand(42);
    for (int i = 0; i < NUM_BLOCKS*BLOCK_SIZE; i++) {
        A.hostPtr[i] = rand()%(NUM_BLOCKS*BLOCK_SIZE);
        B.hostPtr[i] = rand()%(NUM_BLOCKS*BLOCK_SIZE);
        SUM.hostPtr[i] = 0;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();

        serialAdd(A.hostPtr,B.hostPtr,SUM_serial, NUM_BLOCKS*BLOCK_SIZE);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cout << endl;
        cout << "Time taken for serial add is : " << duration.count() << " us" << endl;
        cout << endl;

    }

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);

    A.copy_to_device(NUM_BLOCKS*BLOCK_SIZE);
    B.copy_to_device(NUM_BLOCKS*BLOCK_SIZE);

    {
        auto start = std::chrono::high_resolution_clock::now();

        vectorAdd <<<grid,block>>> (A.devicePtr, B.devicePtr, SUM.devicePtr, NUM_BLOCKS*BLOCK_SIZE);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        SUM.copy_to_host(NUM_BLOCKS*BLOCK_SIZE);

        assert(areEqual(SUM.hostPtr,SUM_serial,NUM_BLOCKS*BLOCK_SIZE));

        cout << endl;
        cout << "Time taken for parallel add is : " << duration.count() << " us" << endl;
        cout << endl;
    }

    for (int i = 0; i < NUM_BLOCKS*BLOCK_SIZE; i++) {
        SUM.hostPtr[i] = 0;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();

        vectorAddGridStride <<<grid,block>>> (A.devicePtr, B.devicePtr, SUM.devicePtr, NUM_BLOCKS*BLOCK_SIZE);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        SUM.copy_to_host(NUM_BLOCKS*BLOCK_SIZE);

        assert(areEqual(SUM.hostPtr,SUM_serial,NUM_BLOCKS*BLOCK_SIZE));

        cout << endl;
        cout << "Time taken for grid stride parallel add is : " << duration.count() << " us" << endl;
        cout << endl;
    }

}

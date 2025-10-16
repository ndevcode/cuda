#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>
#include <cassert>

#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"
#include "cuda_timer.h"

using namespace std;

#define BLOCKSIZE 16

__global__ void matrix_transpose_kernel(const float *input, float *output, int rows, int cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int oindex = col * rows + row;
    int iindex = row * cols + col;
    if (oindex < rows * cols && iindex < rows * cols) {
        output[oindex] = input[iindex];
    }
}

__global__ void matrix_transpose_kernel_shared(const float *input, float *output, int rows, int cols) {

    __shared__ float shmem[BLOCKSIZE][BLOCKSIZE+1];

    int tdx = threadIdx.x, tdy = threadIdx.y;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < rows && col < cols) {
        shmem[tdy][tdx] = input[row * cols + col];
    }
    __syncthreads();

    int ocol = blockDim.y * blockIdx.y + threadIdx.x;
    int orow = blockDim.x * blockIdx.x + threadIdx.y;
    int orows = cols, ocols = rows;

    if (orow < orows && ocol < ocols) {
        output[orow * ocols + ocol] = shmem[tdx][tdy];
    }
}

//CPU
void matrix_transpose(const float *input, float *output, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

int main() {
    MemoryManager mmgr;
    srand(42);
    const int sz = 1 << 24;
    int m = 1 << 10;
    int n = sz / m;
    auto input = mmgr.allocate_host_and_device<float>(sz);
    float *arr = mmgr.allocate_only_host<float>(sz);
    auto output = mmgr.allocate_host_and_device<float>(sz);

    fillArrayRandom(input.hostPtr,sz,256);

    {
        auto start = std::chrono::high_resolution_clock::now();
        matrix_transpose(input.hostPtr,arr,m,n);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        cout << "Time taken for serial transpose is : " << duration.count() << " us" << endl;
    }

    input.copy_to_device(sz);

    {
        dim3 block(BLOCKSIZE,BLOCKSIZE);
        dim3 grid((n + block.x - 1)/block.x,(m + block.y - 1)/block.y);

        CudaTimer timer;
        timer.start();
        matrix_transpose_kernel<<<grid,block>>>(input.devicePtr,output.devicePtr,m,n);
        timer.stop();
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cout << "Time taken for GPU transpose is : " << timer.elapsedTimeUs() << " us" << endl;

        output.copy_to_host(sz);
        assert(areEqual(arr,output.hostPtr,sz));
    }

    {
        dim3 block(BLOCKSIZE,BLOCKSIZE);
        dim3 grid((n + block.x - 1)/block.x,(m + block.y - 1)/block.y);

        CudaTimer timer;
        timer.start();
        matrix_transpose_kernel_shared<<<grid,block>>>(input.devicePtr,output.devicePtr,m,n);
        //matrix_transpose_kernel_smem_pad<<<grid,block>>>(input.devicePtr,output.devicePtr,m,n);
        timer.stop();
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cout << "Time taken for GPU transpose is : " << timer.elapsedTimeUs() << " us" << endl;

        output.copy_to_host(sz);
        assert(areEqual(arr,output.hostPtr,sz));
    }

}


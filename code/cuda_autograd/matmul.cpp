//nvcc -x cu matmul.cpp -o matmul
#include <iostream>
#include <chrono>
#include <functional>
#include <device_launch_parameters.h>
#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"

using namespace std;

//const int DSIZE = 8192;
const int DSIZE = 512;
const int block_size = 32;

const float A_val = 1.0;
const float B_val = 1.0;

struct MatmulNaive {
    __device__ float operator()(const float* A, const float* B, int n, int row, int col) const {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        return sum;
    }
};

template <typename ComputeFn>
__global__ void matmul(const float* A, const float* B, float* MULT, int n, ComputeFn compute) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < n && idy < n) {
        MULT[idy * n + idx] = compute(A, B, n, idy, idx);
    }
}

template <int BLOCK_SIZE>
struct MatmulTiled {
    __device__ float operator()(float tempA[BLOCK_SIZE][BLOCK_SIZE], float tempB[BLOCK_SIZE][BLOCK_SIZE + 1], int row, int col) const {
        float sum = 0;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += tempA[row][j] * tempB[j][col];
        }
        return sum;
    }
};

template <int BLOCK_SIZE, typename ComputeFn>
__global__ void matmul_cache(const float* A, const float* B, float* MULT, int a_ncols, ComputeFn compute) {
    //cache
    __shared__ float tempA[block_size][block_size], tempB[block_size][block_size+1];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int localIdx = threadIdx.x, localIdy = threadIdx.y;

    if (idx < a_ncols && idy < a_ncols) {
        float sum = 0;
        for (int i = 0; i < a_ncols/block_size; i++) {
            //load data into shared memory
            tempA[localIdy][localIdx] = A[a_ncols * idy + (i * block_size + localIdx)];
            tempB[localIdy][localIdx] = B[(i * block_size + localIdy) * a_ncols + idx];

            //synchronize
            __syncthreads();
            sum += compute(tempA,tempB,localIdy,localIdx);
            //synchronize

            __syncthreads();
        }
        MULT[a_ncols * idy + idx] = sum;
    }
}

void matmul_serial(const float* A, const float* B, float *MULT, int a_ncols) {
    for (int i = 0; i < a_ncols; i++) {
        for (int j = 0; j < a_ncols; j++) {
            int sum = 0;
            for (int k = 0; k < a_ncols; k++) {
                sum += A[i * a_ncols + k] * B[k * a_ncols + j];
            }
            MULT[i * a_ncols + j] = sum;
        }
    }
}

void printMatrix(float *arr, int sz) {
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz-1; j++) {
            cout << arr[i*sz+j] << ", " ;
        }
        cout << arr[i*sz+sz-1] << endl;
        cout << endl;
    }
}

int main() {
    MemoryManager mmgr;
    {
        auto A = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
        auto B = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
        auto MULT = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
        srand(42);
        for (int i = 0; i < DSIZE*DSIZE; i++) {
            //A.hostPtr[i] = rand()%31;
            //B.hostPtr[i] = rand()%31;
            A.hostPtr[i] = A_val;
            B.hostPtr[i] = B_val;
            MULT.hostPtr[i] = 0;
        }

        A.copy_to_device(DSIZE*DSIZE);
        B.copy_to_device(DSIZE*DSIZE);

        dim3 block(block_size,block_size);
        dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);

        auto start = std::chrono::high_resolution_clock::now();

        matmul <<<grid,block>>> (A.devicePtr, B.devicePtr, MULT.devicePtr, DSIZE, MatmulNaive{});
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        MULT.copy_to_host(DSIZE*DSIZE);

        cout << "Time taken : " << duration.count() << " us" << endl;
    }

    {
        auto A1 = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
        auto B1 = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
        auto MULT1 = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
        srand(42);
        for (int i = 0; i < DSIZE*DSIZE; i++) {
            //A.hostPtr[i] = rand()%31;
            //B.hostPtr[i] = rand()%31;
            A1.hostPtr[i] = A_val;
            B1.hostPtr[i] = B_val;
            MULT1.hostPtr[i] = 0;
        }

        A1.copy_to_device(DSIZE*DSIZE);
        B1.copy_to_device(DSIZE*DSIZE);

        dim3 block(block_size,block_size);
        dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);

        auto start1 = std::chrono::high_resolution_clock::now();

        matmul_cache <block_size> <<<grid,block>>> (A1.devicePtr, B1.devicePtr, MULT1.devicePtr, DSIZE, MatmulTiled<block_size>{});
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        auto stop1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);

        MULT1.copy_to_host(DSIZE*DSIZE);

        cout << "Time taken : " << duration1.count() << " us" << endl;
    }

    {
        auto A2 = mmgr.allocate_only_host<float>(DSIZE*DSIZE);
        auto B2 = mmgr.allocate_only_host<float>(DSIZE*DSIZE);
        auto MULT2 = mmgr.allocate_only_host<float>(DSIZE*DSIZE);

        for (int i = 0; i < DSIZE*DSIZE; i++) {
            A2[i] = A_val;
            B2[i] = B_val;
            MULT2[i] = 0;
        }

        auto start2 = std::chrono::high_resolution_clock::now();

        matmul_serial(A2,B2,MULT2,DSIZE);

        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);

        cout << "Time taken : " << duration2.count() << " us" << endl;
    }
    //printMatrix(A.hostPtr,DSIZE);
    //printMatrix(B.hostPtr,DSIZE);
    //printMatrix(MULT.hostPtr,DSIZE);

}

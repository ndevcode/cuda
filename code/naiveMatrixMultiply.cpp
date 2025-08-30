//nvcc -x cu matmul.cpp -o matmul
#include <iostream>
#include <device_launch_parameters.h>
#include "memory_alloc.h"
#include "utility.hpp"

using namespace std;

__global__ void matmul (const float* A, const float* B, float* MULT, int a_ncols ) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;

  if (idx < a_ncols && idy < a_ncols) {
      float sum = 0;
      for (int i = 0; i < a_ncols; i++) {
          sum += A[a_ncols * idy + i] * B[i * a_ncols + idx];
      }
      MULT[a_ncols * idy + idx] = sum;
  }

}

const int DSIZE = 64;
const int block_size = 4;

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
    auto A = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
    auto B = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
    auto MULT = mmgr.allocate_host_and_device<float>(DSIZE*DSIZE);
    srand(42);
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        A.hostPtr[i] = rand()%31;
        B.hostPtr[i] = rand()%31;
        //A.hostPtr[i] = 1;
        //B.hostPtr[i] = 2;
        MULT.hostPtr[i] = 0;
    }
    A.copy_to_device(DSIZE*DSIZE);
    B.copy_to_device(DSIZE*DSIZE);

    dim3 block(block_size,block_size);
    dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);

    matmul <<<grid,block>>> (A.devicePtr, B.devicePtr, MULT.devicePtr, DSIZE);
    CUDA_CHECK(cudaPeekAtLastError()); // launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    MULT.copy_to_host(DSIZE*DSIZE);

    printMatrix(A.hostPtr,DSIZE);
    printMatrix(B.hostPtr,DSIZE);
    printMatrix(MULT.hostPtr,DSIZE);

}

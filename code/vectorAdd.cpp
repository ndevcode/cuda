//nvcc -x cu vadd.cpp -o vadd
#include <iostream>
#include <device_launch_parameters.h>
#include "memory_alloc.h"

using namespace std;

__global__ void vectorAdd (const float* A, const float* B, float* SUM, int sz) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < sz) {
        SUM[gid] = A[gid] + B[gid];
    }
}

const int DSIZE = 64;
const int block_size = 4;

void printArr(float *arr, int sz) {
    for (int i = 0; i < sz-1; i++) {
        cout << arr[i] << ", ";
    }
    cout << arr[sz-1] << endl;
}

int main() {
    MemoryManager mmgr;
    auto A = mmgr.allocate_host_and_device<float>(DSIZE);
    auto B = mmgr.allocate_host_and_device<float>(DSIZE);
    auto SUM = mmgr.allocate_host_and_device<float>(DSIZE);
    srand(42);
    for (int i = 0; i < DSIZE; i++) {
        A.hostPtr[i] = rand()%DSIZE;
        B.hostPtr[i] = rand()%DSIZE;
        SUM.hostPtr[i] = 0;
    }
    A.copy_to_device(DSIZE);
    B.copy_to_device(DSIZE);

    dim3 grid((DSIZE+block_size-1)/block_size);
    dim3 block(block_size);

    vectorAdd <<<grid,block>>> (A.devicePtr, B.devicePtr, SUM.devicePtr, DSIZE);

    SUM.copy_to_host(DSIZE);

    printArr(A.hostPtr,DSIZE);
    printArr(B.hostPtr,DSIZE);
    printArr(SUM.hostPtr,DSIZE);
}

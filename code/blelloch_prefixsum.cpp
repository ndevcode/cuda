//nvcc -x cu one_thread_blelloch.cu -o one_thread_blelloch
#include <iostream>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>

#include "ndev_cuda_utility.hpp"
#include "memory_alloc.h"
#include "cuda_timer.h"

using namespace std;

__global__ void blelloch_shared_memory(float *arr, int n, float *res) {
    extern __shared__ float shmem[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= n) return;

    shmem[tid] = arr[gid];
    __syncthreads();

    //upsweep
    for (int d = 1; d < blockDim.x; d <<= 1) {
        int idx = (tid + 1) * d * 2 - 1;
        if (idx < blockDim.x) {
            shmem[idx]  +=  shmem[idx-d];
        }
        __syncthreads();
    }

    //copy over last element to result array
    if (tid == 0) {
        //res[blockIdx.x] = shmem[blockDim.x-1];
        shmem[blockDim.x-1] = 0;
    }
    __syncthreads();

    //downsweep
    for (int d = blockDim.x/2; d > 0; d >>= 1) {
        int idx = (tid + 1) * d * 2 - 1;
        if (idx < blockDim.x) {
            float tmp = shmem[idx];
            shmem[idx] += shmem[idx-d];
            shmem[idx-d] = tmp;
        }
        __syncthreads();
    }

    arr[gid] = shmem[tid];
    __syncthreads();
}

void partial_sum_CUDA(vector<float>& orig) {
    MemoryManager mmgr;

    int n = orig.size();
    int n2 = 1;
    for (; n2 < n; n2 <<= 1);


    float *arr = mmgr.allocate_only_device<float>(n2);

    CUDA_CHECK(cudaMemcpy(arr,orig.data(),n*sizeof(float),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(arr+n,0,(n2-n)*sizeof(float)));

    dim3 blockSize(256);

    blelloch_shared_memory<<<1,blockSize,blockSize.x*sizeof(float)>>>(arr,n,arr);

    CUDA_CHECK(cudaMemcpy(orig.data(),arr,n*sizeof(float),cudaMemcpyDeviceToHost));
}

int main() {
    int numElements = 256;

    vector<float> orig(numElements);
    iota(begin(orig),end(orig),1);

    partial_sum_CUDA(orig);

    copy(begin(orig),end(orig),ostream_iterator<float>(cout,", "));
    cout << endl;
}

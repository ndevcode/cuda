//nvcc -x cu max_kernel.cpp -o max_kernel
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <chrono>

#include "ndev_cuda_utility.hpp"
#include "memory_alloc.h"
#include "cuda_timer.h"

using namespace std;

struct ReduceOper {
    __host__ __device__ float operator()(float a, float b) const {
        const bool a_is_nan = isnan(a);
        const bool b_is_nan = isnan(b);
        if (a_is_nan) {
            return b_is_nan ? -FLT_MAX : b;
        }
        if (b_is_nan) {
            return a;
        }
        return a > b ? a : b;
    }
};

template <typename Reducer>
__global__ void maxKernelCUDA(Reducer& reducer, float *arr, float *out, int n) {
    __shared__ float shmem[1024];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int tid = threadIdx.x;

    float localMax = -FLT_MAX;

    while (gid < n) {
        localMax = reducer(localMax,arr[gid]);
        gid += gridDim.x * blockDim.x ;
    }
    shmem[tid] = localMax;

    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] = reducer(shmem[tid],shmem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shmem[0];
    }
}

template <typename Reducer>
float maxElement(Reducer& reducer, HostDevicePair<float>& arr, HostDevicePair<float>& res, int n) {
    dim3 blockSize(256);

    float runtime = 0.0f;

    bool swapNeeded = false;

    while (n > 1) {
        dim3 gridSize((n + blockSize.x - 1)/ blockSize.x);

        CudaTimer timer;
        maxKernelCUDA<<<gridSize,blockSize>>>(reducer,arr.devicePtr,res.devicePtr,n);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();

        runtime += timer.elapsedTimeUs();

        float *tmp = arr.devicePtr;
        arr.devicePtr = res.devicePtr;
        res.devicePtr = tmp;

        swapNeeded ^= true;

        n = gridSize.x;
    }

    if (swapNeeded) {
        CUDA_CHECK(cudaMemcpy(res.devicePtr,arr.devicePtr,sizeof(float),cudaMemcpyDeviceToDevice));
    }

    return runtime;

}

int main() {
    srand(42);
    MemoryManager mmgr;

    int numElements = (1<<25) + 7;
    //int numElements = 4;

    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1)/ blockSize.x);

    auto arr = mmgr.allocate_host_and_device<float>(numElements);
    auto res = mmgr.allocate_host_and_device<float>(gridSize.x);

    fillArrayRandom(arr.hostPtr,numElements);
    std::transform(arr.hostPtr,arr.hostPtr+numElements,arr.hostPtr,[](float x) { return -1.0 * x; });
    //float tmpArr[] = {-3,-7,NAN,-1};
    //copy(tmpArr,tmpArr+4,arr.hostPtr);

    {
        auto start = std::chrono::high_resolution_clock::now();

        float maxValue = *std::max_element(arr.hostPtr,arr.hostPtr + numElements);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cout << "Max value std::max_element is " << maxValue << " computed in " << duration.count() << " us" << endl;
    }

    arr.copy_to_device(numElements);
    res.copy_to_device(gridSize.x);

    ReduceOper oper;

    float runtime = maxElement(oper,arr,res,numElements);

    res.copy_to_host(1);

    cout << "Max value CUDA is " << res.hostPtr[0] << " computed in " << runtime << " us" << endl;


}


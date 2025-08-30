#include <cuda_runtime.h>
#include <iostream>

#include "ndev_cuda_utility.hpp"

struct CudaTimer {
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
        CUDA_CHECK(cudaEventRecord(start_));
    }
    virtual ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    float elapsedTimeUs() const {
        float ms = 0.0;
        CUDA_CHECK(cudaEventElapsedTime(&ms,start_,stop_));
        return ms * 1000.0;
    }
private :
    cudaEvent_t start_,stop_;
};

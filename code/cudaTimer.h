#include <cuda_runtime.h>
#include <iostream>

struct CudaTimer {
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }
    virtual ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }
    float elapsedTimeUs() const {
        float ms = 0.0;
        cudaEventElapsedTime(&ms,start_,stop_);
        return ms * 1000.0;
    }
private :
    cudaEvent_t start_,stop_;
};

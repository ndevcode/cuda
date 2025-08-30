#include <vector>
#include <cstdlib> 
#include <iostream>
#include <cuda_runtime.h> 

#include "ndev_cuda_utility.hpp"

template <typename T>
struct HostDevicePair {
    T* hostPtr = nullptr;
    T* devicePtr = nullptr;

    void copy_to_device(size_t count) const {
        CUDA_CHECK(cudaMemcpy(devicePtr,hostPtr,count * sizeof(T),cudaMemcpyHostToDevice));
    }

    void copy_to_host(size_t count) const {
        CUDA_CHECK(cudaMemcpy(hostPtr,devicePtr,count * sizeof(T),cudaMemcpyDeviceToHost));
    }
};

struct MemoryManager {
    MemoryManager() = default;

    virtual ~MemoryManager() {
        for (void* ptr : host_allocations_) {
            std::free(ptr);
        }
        for (void* dptr : device_allocations_) {
            cudaFree(dptr);
        }
    }

    template <typename T>
    T* allocate_only_host(unsigned int sz) {
        T* ptr = static_cast<T*>(std::malloc(sz * sizeof(T)));
        if (!ptr) {
            throw std::bad_alloc();
        }
        host_allocations_.push_back(ptr);
        return ptr;
    }

    template <typename T>
    T* allocate_only_device(unsigned int sz) {
        T* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, sz * sizeof(T))); 
        device_allocations_.push_back(d_ptr);

        return d_ptr;
    }

    template <typename T>
    HostDevicePair<T> allocate_host_and_device(unsigned int sz) {
        T* h_ptr = static_cast<T*>(std::malloc(sz * sizeof(T)));
        if (!h_ptr) {
            throw std::bad_alloc();
        }
        host_allocations_.push_back(h_ptr);

        T* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, sz * sizeof(T)));
        device_allocations_.push_back(d_ptr);

        return HostDevicePair<T>{h_ptr,d_ptr};
    }

private:
    std::vector<void*> host_allocations_;
    std::vector<void*> device_allocations_;
};


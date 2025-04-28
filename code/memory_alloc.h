#include <vector>
#include <cstdlib> 
#include <iostream>
#include <cuda_runtime.h> 

template <typename T>
struct HostDevicePair {
    T* hostPtr = nullptr;
    T* devicePtr = nullptr;

    void copy_to_device(size_t count) const {
        cudaMemcpy(devicePtr,hostPtr,count * sizeof(T),cudaMemcpyHostToDevice);
    }

    void copy_to_host(size_t count) const {
        cudaMemcpy(hostPtr,devicePtr,count * sizeof(T),cudaMemcpyDeviceToHost);
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
    HostDevicePair<T> allocate_host_and_device(unsigned int sz) {
        T* h_ptr = static_cast<T*>(std::malloc(sz * sizeof(T)));
        if (!h_ptr) {
            throw std::bad_alloc();
        }
        host_allocations_.push_back(h_ptr);

        T* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, sz * sizeof(T));
        if (err != cudaSuccess) {
            std::free(h_ptr); 
            throw std::runtime_error("cudaMalloc failed");
        }
        device_allocations_.push_back(d_ptr);

        return HostDevicePair<T>{h_ptr,d_ptr};
    }

private:
    std::vector<void*> host_allocations_;
    std::vector<void*> device_allocations_;
};


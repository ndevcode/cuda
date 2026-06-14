#include <unordered_map>
#include <iostream>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include "ndev_cuda_utility.hpp"

struct CudaTimer {
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    virtual ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(stop_));
        CUDA_CHECK(cudaEventDestroy(start_));
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

class TimerManager {
    public :
        TimerManager() = default;
        virtual ~TimerManager() {
            name2timer_.clear();
        }
        void addTimer(const std::string& name) {
            if (name2timer_.find(name) == name2timer_.end()) {
                name2timer_[name] = std::make_unique<CudaTimer>();
            }
        }
        CudaTimer* getTimer(const std::string &name) {
            auto it = name2timer_.find(name);
            return it == name2timer_.end() ? nullptr : it->second.get();
        }
        void start(const std::string& name) {
            addTimer(name);
            name2timer_[name]->start();
        }
        void stop(const std::string& name, bool print = true) {
            auto *t = getTimer(name);
            if (!t) return;
            t->stop();
            if (print) {
                std::cout << "[Timer] " << name << " : " << t->elapsedTimeUs() / 1000.f << "ms " << std::endl;
            }
        }
        
    private :
        TimerManager(const TimerManager&) = delete;
        TimerManager& operator = (const TimerManager&) = delete;

        std::unordered_map<std::string,std::unique_ptr<CudaTimer>> name2timer_;
        
};

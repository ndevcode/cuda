//nvcc -x cu canny.cpp -o canny
#include <iostream>

#include <vector>
#include <array>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <algorithm>

#include <device_launch_parameters.h>

#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"
#include "cuda_timer.h"

using namespace std;

#define TILE_SIZE 32
/*******************************************************************************/
namespace SobelFilter {
    __constant__ int X_dev[3][3];
    __constant__ int Y_dev[3][3];

    constexpr int X_host[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    constexpr int Y_host[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    inline void uploadToDevice() {
        CUDA_CHECK(cudaMemcpyToSymbol(X_dev, X_host, sizeof(X_host)));
        CUDA_CHECK(cudaMemcpyToSymbol(Y_dev, Y_host, sizeof(Y_host)));
    }
}

__host__ __device__ inline
void sobelAtPixel(const float *img, int i, int j,
    int m, int n,
    const int X[3][3], const int Y[3][3],
    float &vi, float &vj) {
    vi = vj = 0.0f;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
            float pixel = img[ni * n + nj];
            vi += pixel * X[di+1][dj+1];
            vj += pixel * Y[di+1][dj+1];
        }
    }
}

/*******************************************************************************/
__global__ void gaussianBlurHorizontalCudaOne(const float* input, float* output, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || j < 0 || i >= m || j >= n) return;
    float kernel[5] = {0.06136,0.24477,0.38774,0.24477,0.06136};
    int ksize = 2;

    float sum = 0;
    for (int k = -ksize; k <= ksize; k++) {
        int xx = max(0, min(j+k, n-1));
        sum += input[i*n+xx] * kernel[k+ksize];
    }
    output[i*n+j] = sum;
}

__global__ void gaussianBlurVerticalCudaOne(const float* input, float* output, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || j < 0 || i >= m || j >= n) return;
    float kernel[5] = {0.06136,0.24477,0.38774,0.24477,0.06136};
    int ksize = 2;

    float sum = 0;
    for (int k = -ksize; k <= ksize; k++) {
        int yy = max(0, min(i+k, m-1));
        sum += input[yy*n + j] * kernel[k+ksize];
    }
    output[i*n+j] = sum;
}

__global__ void sobelKernelOne(
    const float* img,
    float *magnitude,
    float *direction,
    int m,
    int n) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || j < 0 || i >= m || j >= n) return;

    float vi = 0.0f, vj = 0.0f;
    sobelAtPixel(img,i,j,m,n,SobelFilter::X_dev, SobelFilter::Y_dev,vi,vj);

    int idx = i * n + j;
    magnitude[idx] = sqrtf(vi * vi + vj * vj);
    direction[idx] = atan2f(vj, vi);  // Output in radians (convert later on CPU if needed)
}

__global__ void doNMSKernelOne(
    const float* magnitude,
    const float* direction,
    int m,
    int n,
    float *nms) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <= 0 || j <= 0 || i >= m-1 || j >= n-1) return;
    constexpr float PI = 3.14159265358979323846f;

    int idx = i * n + j;
    float angle = direction[idx] * 180.0 / PI;
    if (angle < 0) angle += 180;

    float mag1 = 0, mag2 = 0;

    //Quantize orientation to 0,45,90,135
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
        mag1 = magnitude[idx-1];
        mag2 = magnitude[idx+1];
    } else if (angle >= 22.5 && angle < 67.5) {
        mag1 = magnitude[(i-1) * n + (j+1)];
        mag2 = magnitude[(i+1) * n + (j-1)];
    } else if (angle >= 67.5 && angle < 112.5) {
        mag1 = magnitude[(i-1) * n + j];
        mag2 = magnitude[(i+1) * n + j];
    } else if (angle >= 112.5 && angle < 157.5) {
        mag1 = magnitude[(i-1) * n + (j-1)];
        mag2 = magnitude[(i+1) * n + (j+1)];
    }


    float mag = magnitude[idx];
    nms[idx] = (mag >= mag1 && mag >= mag2) ? mag : 0;
}

__global__ void segmented_max_reduction_kernel(const float* input, float* output, int length) {
    extern __shared__ float sharedInput[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // load into shared memory
    float val = -FLT_MAX;
    if (idx < length) val = input[idx];
    sharedInput[tid] = val;
    __syncthreads();

    // reduction in shared memory
    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedInput[tid] = fmaxf(sharedInput[tid], sharedInput[tid + stride]);
        }
        __syncthreads();
    }

    // first thread of block writes its block max
    if (tid == 0) {
		atomicExch(output, fmax(*output,sharedInput[0]));
    }
}

template <typename MaxMagnitudeFind>
float getMaxMagnitude(MemoryManager& mmgr, MaxMagnitudeFind& reducer, const float* nms, int m, int n) {
    int numPixels = m * n;
    dim3 blockSize(TILE_SIZE * TILE_SIZE);
    dim3 gridSize((numPixels + blockSize.x - 1)/blockSize.x);

    auto dd_max = mmgr.allocate_host_and_device<float>(1);

    float init = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(dd_max.devicePtr,&init,sizeof(float),cudaMemcpyHostToDevice));

    reducer<<<gridSize,blockSize,blockSize.x*sizeof(float)>>>(nms,dd_max.devicePtr,numPixels);
    CUDA_CHECK(cudaPeekAtLastError()); // launch errors

    dd_max.copy_to_host(1);

    return dd_max.hostPtr[0];
}

__global__ void doDoubleThresholdKernelOne(const float *nms,
    int m,
    int n,
    float highThreshold,
    float lowThreshold,
    float *edges) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || j < 0 || i >= m || j >= n) return;

    int idx = i * n + j;

    if (nms[idx] >= highThreshold) {
        edges[idx] = 2;
    } else if (nms[idx] >= lowThreshold) {
        edges[idx] = 1;
    } else {
        edges[idx] = 0;
    }
}

__global__ void seedFrontierKernel(const float* edges, int* frontier, int *frontier_size, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= (m*n)) return;

    if (edges[idx] == 2) {
        int pos = atomicAdd(frontier_size,1);
        frontier[pos] = idx;
    }
}

__device__ int atomicCAS_floatInt(float* addr, int compare, int val) {
    int old_bits = atomicCAS(reinterpret_cast<int*>(addr),
                             __float_as_int(static_cast<float>(compare)),
                             __float_as_int(static_cast<float>(val)));
    // Return the *numeric* previous value (0/1/2) as int:
    float old_val = __int_as_float(old_bits);
    return static_cast<int>(old_val);
}

__global__ void expandFrontierKernel(float* edges,
                const int* frontier, int frontier_size,
                int* next_frontier, int* next_size,
                int m, int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= frontier_size) return;

    int idx = frontier[tid];
    int i = idx / n, j = idx % n;

    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = i + di, nj = j + dj;
            if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
            int nidx = ni * n + nj;
            int old = atomicCAS_floatInt(&edges[nidx],1,2);
            if (old == 1) {
                int pos = atomicAdd(next_size,1);
                next_frontier[pos] = nidx;
            }
        }
    }
}

__global__ void cleanupWeakKernel(float *edges, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    if (edges[tid] == 1) {
        edges[tid] = 0;
    }
}

template <
        typename HysteresisSeed,
        typename HysteresisBFS,
        typename HysteresisCleanup
        >
void hysteresis_bfs_cuda(
    MemoryManager& mmgr,
    HysteresisSeed& seeder,
    HysteresisBFS&  bfs,
    HysteresisCleanup& cleaner,
    float *edges,
    int m,
    int n) {

    int numPixels = m * n;
    dim3 blockSize(TILE_SIZE * TILE_SIZE);
    dim3 gridSize((numPixels + blockSize.x - 1)/blockSize.x);

    auto frontier = mmgr.allocate_only_device<int>(numPixels);
    auto next_frontier = mmgr.allocate_only_device<int>(numPixels);
    auto frontier_size = mmgr.allocate_only_device<int>(1);
    auto next_size = mmgr.allocate_only_device<int>(1);

    CUDA_CHECK(cudaMemset(frontier_size,0,sizeof(int)));
    seeder<<<gridSize, blockSize>>>(edges,frontier,frontier_size,m,n);
    CUDA_CHECK(cudaPeekAtLastError());

    int sz = 0;
    CUDA_CHECK(cudaMemcpy(&sz,frontier_size,sizeof(int),cudaMemcpyDeviceToHost));

    while (sz > 0) {
        CUDA_CHECK(cudaMemset(next_size,0,sizeof(int)));
        dim3 gridSize2((sz + blockSize.x - 1)/blockSize.x);
        bfs<<<gridSize2,blockSize>>>(edges,frontier,sz,next_frontier,next_size,m,n);
        CUDA_CHECK(cudaPeekAtLastError());

        //swap frontiers
        std::swap(frontier,next_frontier);
        std::swap(frontier_size,next_size);

        //read new frontier size
        CUDA_CHECK(cudaMemcpy(&sz,frontier_size,sizeof(int),cudaMemcpyDeviceToHost));
    }

    cleaner<<<gridSize,blockSize>>>(edges,numPixels);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <
    typename GaussianBlurHorizontal, 
    typename GaussianBlurVertical,
    typename SobelKernel,
    typename NMSFind,
    typename MaxMagnitudeFind,
    typename EdgeThin,
    typename HysteresisSeed,
    typename HysteresisBFS,
    typename HysteresisCleanup
    >
std::array<float*,3>
runCudaKernelExperiments(MemoryManager& mmgr,
                int width, int height,
                GaussianBlurHorizontal& hBlurrer,
                GaussianBlurVertical& vBlurrer,
                SobelKernel& kernel,
                NMSFind& finder,
                MaxMagnitudeFind& reducer,
                EdgeThin& thinner,
                HysteresisSeed& seeder,
                HysteresisBFS& bfs,
                HysteresisCleanup& cleaner) {

    int numPixels = width * height;
    auto input = mmgr.allocate_host_and_device<float>(numPixels);
    auto blurred = mmgr.allocate_host_and_device<float>(numPixels);
    auto nms = mmgr.allocate_host_and_device<float>(numPixels);
    auto edges = mmgr.allocate_host_and_device<float>(numPixels);
    auto magnitude = mmgr.allocate_host_and_device<float>(numPixels);
    auto direction = mmgr.allocate_host_and_device<float>(numPixels);

    float maxMagnitude = 0.0f;

    srand(42);
    for (int i = 0; i < numPixels; i++) {
        input.hostPtr[i] = rand()%255;
    }

    float totalTime = 0.0f;

    //Step 0 : copy from host to device memory
    {
        CudaTimer timer; 
        input.copy_to_device(numPixels);
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        cout << "H2D copy times : " << tt << " us" << endl;
    }

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 1 : horizontal Gaussian blurring
    {
        CudaTimer timer;
        hBlurrer<<<gridSize, blockSize>>>(
                input.devicePtr,
                edges.devicePtr,
                height,
                width
                );
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors

        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Canny Horizontal Gaussian blur time: " << tt << " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 2 : vertical Gaussian blurring
    {
        CudaTimer timer;
        vBlurrer<<<gridSize, blockSize>>>(
                edges.devicePtr,
                blurred.devicePtr,
                height,
                width
                );
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors

        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Canny Vertical Gaussian blur time: " << tt << " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 3 : find magnitude and orientation via Sobel
    {
        CudaTimer timer;
        kernel<<<gridSize, blockSize>>>(
                blurred.devicePtr,
                magnitude.devicePtr,
                direction.devicePtr,
                height,
                width
                );
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Sobel kernel magnitude and direction finding time: " << tt <<  " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 4 : do non maximum suppression 
    {
        CudaTimer timer;
        finder<<<gridSize, blockSize>>>(
                magnitude.devicePtr,
                direction.devicePtr,
                height,
                width,
                nms.devicePtr
                );
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Non maximum suppression finding time: " << tt << " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 5 : find maximum magnitude
    {
        CudaTimer timer;
        maxMagnitude = getMaxMagnitude(mmgr,reducer,nms.devicePtr,height,width);
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Canny maximum magnitude finding time: " << tt << " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 6 : edge thinning
    {
        float highThreshold = 0.2f * maxMagnitude, lowThreshold = 0.1f * maxMagnitude;
        CudaTimer timer;
        thinner<<<gridSize, blockSize>>>(
                nms.devicePtr,
                height,
                width,
                highThreshold,
                lowThreshold,
                edges.devicePtr
                );
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Double threshold/edge thinning time: " << tt << " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 7 : hysteresis
    {
        CudaTimer timer;
        hysteresis_bfs_cuda(mmgr,seeder,bfs,cleaner,edges.devicePtr,height,width);
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;
        std::cout << "Hysteresis time: " << tt << " us\n";
    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    //Step 8 :  copy memory from device to host
    {
        CudaTimer timer;
        edges.copy_to_host(numPixels);
        magnitude.copy_to_host(numPixels);
        direction.copy_to_host(numPixels);
        timer.stop();
        float tt = timer.elapsedTimeUs();
        totalTime += tt;

        cout << "D2H copy times : " << tt << " us" << endl;

    }

    //synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    cout << "Total CUDA runtime for Canny algorithm is " << totalTime << " us" << endl;

    return {edges.hostPtr, magnitude.hostPtr, direction.hostPtr};
} 

/*******************************************************************************/
int main() {
    int width = 512;
    int height = 512;

    MemoryManager mmgr;

    SobelFilter::uploadToDevice();

    auto [kernel_one_edges, kernel_one_magnitude, kernel_one_direction] = runCudaKernelExperiments(
                                        mmgr,
                                        width,
                                        height,
                                        gaussianBlurHorizontalCudaOne,
                                        gaussianBlurVerticalCudaOne,
                                        sobelKernelOne,
                                        doNMSKernelOne,
                                        segmented_max_reduction_kernel,
                                        doDoubleThresholdKernelOne,
                                        seedFrontierKernel,
                                        expandFrontierKernel,
                                        cleanupWeakKernel);

}

//nvcc -x cu sobel.cpp -o sobel
#include <iostream>
#include <cassert>
#include <chrono>
#include <utility>
#include <tuple>
#include <device_launch_parameters.h>

#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"
#include "cuda_timer.h"

using namespace std;

#define TILE_SIZE 32 

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
void sobelAtPixel(const float* img, int i, int j,
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

void sobelKernelBaseline(
    const float* img,
    float *magnitude,
    float *direction,
    int m,
    int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            float vi = 0.0f, vj = 0.0f;
            sobelAtPixel(img,i,j,m,n,SobelFilter::X_host, SobelFilter::Y_host,vi,vj);
            magnitude[idx] = sqrtf(vi * vi + vj * vj);
            direction[idx] = atan2f(vj, vi);  // Output in radians (convert later on CPU if needed)
        }
    }
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

__global__ void sobelKernelTwoStrides(
    const float *img,
    float *magnitude,
    float *direction,
    int m,
    int n) {

    int i0 = blockIdx.y * blockDim.y + threadIdx.y;
    int j0 = blockIdx.x * blockDim.x + threadIdx.x;

    int strideI = blockDim.y * gridDim.y;
    int strideJ = blockDim.x * gridDim.x;

    for (int i = i0; i < m; i += strideI) {
        for (int j = j0; j < n; j += strideJ) {
            if (i < 0 || j < 0 || i >= m || j >= n) continue;

            float vi = 0.0f, vj = 0.0f;
            sobelAtPixel(img,i,j,m,n,SobelFilter::X_dev, SobelFilter::Y_dev,vi,vj);

            int idx = i * n + j;
            magnitude[idx] = sqrtf(vi * vi + vj * vj);
            direction[idx] = atan2f(vj, vi);  // Output in radians (convert later on CPU if needed)
        }
    }
}

__global__ void sobelKernelThreeShared(
    const float *img,
    float *magnitude,
    float *direction,
    int m,
    int n) {

    //shared memory tile with halo
    __shared__ float tile[TILE_SIZE+2][TILE_SIZE+2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + tx;
    int i = blockIdx.y * TILE_SIZE + ty;

    for (int dy = ty; dy < TILE_SIZE + 2; dy += blockDim.y) {
        for (int dx = tx; dx < TILE_SIZE + 2; dx += blockDim.x) {
            int global_i = blockIdx.y * TILE_SIZE + dy - 1;
            int global_j = blockIdx.x * TILE_SIZE + dx - 1;
            if (global_i >= 0 && global_i < m && global_j >= 0 && global_j < n) {
                tile[dy][dx] = img[global_i * n + global_j];
            } else {
                tile[dy][dx] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (i < 0 || j < 0 || i >= m || j >= n || tx >= TILE_SIZE || ty >= TILE_SIZE) return;

    float vi = 0.0f, vj = 0.0f;
    sobelAtPixel(&tile[0][0],ty+1,tx+1,TILE_SIZE+2,TILE_SIZE+2,SobelFilter::X_dev, SobelFilter::Y_dev,vi,vj);

    int idx = i * n + j;
    magnitude[idx] = sqrtf(vi * vi + vj * vj);
    direction[idx] = atan2f(vj, vi);  // Output in radians (convert later on CPU if needed)

}

__global__ void sobelKernelThreeDotOneShared(
    const float *img,
    float *magnitude,
    float *direction,
    int m,
    int n) {

    //shared memory tile with halo
    __shared__ float tile[TILE_SIZE+2][TILE_SIZE+2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + tx;
    int i = blockIdx.y * TILE_SIZE + ty;

    int tilePitch = TILE_SIZE + 2;
    int totalTileElements = tilePitch * tilePitch;

    //flatten thread indices in the block
    int threadId = ty * blockDim.x + tx;
    int threadsPerBlock = blockDim.x * blockDim.y;

    for (int idx = threadId; idx < totalTileElements; idx += threadsPerBlock) {
        int localY = idx / tilePitch, localX = idx % tilePitch;

        int gi = blockIdx.y * TILE_SIZE + localY - 1;
        int gj = blockIdx.x * TILE_SIZE + localX - 1;

        //clamp global coords to valid range
        int ci = max(0,min(m-1,gi));
        int cj = max(0,min(n-1,gj));

        float val = img[ci * n + cj];

        //mask to zero out, out of range loads
        int inb = (gi >= 0 && gi < m && gj >= 0 and gj < n);
        tile[localY][localX] = inb ? val : 0.0f;
    }

    __syncthreads();

    if (i < 0 || j < 0 || i >= m || j >= n || tx >= TILE_SIZE || ty >= TILE_SIZE) return;

    float vi = 0.0f, vj = 0.0f;
    sobelAtPixel(&tile[0][0],ty+1,tx+1,TILE_SIZE+2,TILE_SIZE+2,SobelFilter::X_dev, SobelFilter::Y_dev,vi,vj);

    int idx = i * n + j;
    magnitude[idx] = sqrtf(vi * vi + vj * vj);
    direction[idx] = atan2f(vj, vi);  // Output in radians (convert later on CPU if needed)

}

std::pair<float*,float*> runBaseline(MemoryManager& mmgr, int width, int height) {
    int numPixels = width * height;
    auto input = mmgr.allocate_only_host<float>(numPixels); 
    auto magnitude_host = mmgr.allocate_only_host<float>(numPixels);
    auto direction_host = mmgr.allocate_only_host<float>(numPixels);

    srand(42);
    for (int i = 0; i < numPixels; i++) {
        input[i] = rand()%255;
    }

    {
        auto start = std::chrono::high_resolution_clock::now();

        sobelKernelBaseline(input,magnitude_host,direction_host,height,width);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cout << endl;
        cout << "Time taken for serial sobel is : " << duration.count() << " us" << endl;
        cout << endl;
    }
    return {magnitude_host,direction_host};
}

template <typename KernelFunc, typename... Args>
std::pair<float*, float*> runCudaKernelExperiments(MemoryManager& mmgr,
                                           int width, int height,
                                           KernelFunc kernel,
                                           Args... args) {
    int numPixels = width * height;
    auto input = mmgr.allocate_host_and_device<float>(numPixels);
    auto magnitude = mmgr.allocate_host_and_device<float>(numPixels);
    auto direction = mmgr.allocate_host_and_device<float>(numPixels);

    cout << endl;

    srand(42);
    for (int i = 0; i < numPixels; i++) {
        input.hostPtr[i] = rand()%255;
    }

    {
        CudaTimer timer; 
        input.copy_to_device(numPixels);
        timer.stop();
        cout << "H2D copy times : " << timer.elapsedTimeUs() << " us" << endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    {
        CudaTimer timer;

        kernel<<<gridSize, blockSize>>>(
                input.devicePtr,
                magnitude.devicePtr,
                direction.devicePtr,
                height,
                width
                );
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors

        timer.stop();
        std::cout << "Sobel kernel time: " << timer.elapsedTimeUs() << " us\n";
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    {
        CudaTimer timer;
        magnitude.copy_to_host(numPixels);
        direction.copy_to_host(numPixels);

        timer.stop();
        cout << "D2H copy times : " << timer.elapsedTimeUs() << " us" << endl;

    }

    CUDA_CHECK(cudaDeviceSynchronize());

    return {magnitude.hostPtr, direction.hostPtr};
}

int main() {
    int width = 512;
    int height = 512;

    MemoryManager mmgr;

    SobelFilter::uploadToDevice();

    auto [baseline_magnitude, baseline_direction] = runBaseline(mmgr,width,height);

    auto [kernel_one_magnitude, kernel_one_direction] = runCudaKernelExperiments(mmgr,width,height,sobelKernelOne);
    assert(areEqual(baseline_magnitude,kernel_one_magnitude,width*height));
    assert(areEqual(baseline_direction,kernel_one_direction,width*height));

    auto [kernel_two_magnitude, kernel_two_direction] = runCudaKernelExperiments(mmgr,width,height,sobelKernelTwoStrides);
    assert(areEqual(baseline_magnitude,kernel_two_magnitude,width*height));
    assert(areEqual(baseline_direction,kernel_two_direction,width*height));
    
    auto [kernel_three_magnitude, kernel_three_direction] = runCudaKernelExperiments(mmgr,width,height,sobelKernelThreeShared);
    assert(areEqual(baseline_magnitude,kernel_three_magnitude,width*height));
    assert(areEqual(baseline_direction,kernel_three_direction,width*height));
    
    auto [kernel_four_magnitude, kernel_four_direction] = runCudaKernelExperiments(mmgr,width,height,sobelKernelThreeDotOneShared);
    assert(areEqual(baseline_magnitude,kernel_four_magnitude,width*height));
    assert(areEqual(baseline_direction,kernel_four_direction,width*height));
    
}


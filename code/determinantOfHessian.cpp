//nvcc -x cu doh.cpp -o doh 
#include <iostream>
#include <tuple>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <chrono>

#include "cuda_timer.h"
#include "ndev_cuda_utility.hpp"
#include "memory_alloc.h"   

using namespace std;

#define AT(img, i, j, n) (img[(i)*(n) + (j)])

#define TILE_SIZE 32 


// ---------- kernels (DoH components) ----------

__device__ __host__ float gaussianSecondDerivativeX(float x, float y, float sigma) {
    float sigma4 = sigma * sigma;
    sigma4 = sigma4 * sigma4;
    float norm = -1.0f / (M_PI * sigma4);
    float factor = 1.0f - (x*x) / (sigma*sigma);
    return norm * factor * std::exp(-(x*x + y*y) / (2.0f * sigma*sigma));
}

__device__ __host__ float gaussianSecondDerivativeY(float x, float y, float sigma) {
    float sigma4 = sigma * sigma;
    sigma4 = sigma4 * sigma4;
    float norm = -1.0f / (M_PI * sigma4);
    float factor = 1.0f - (y*y) / (sigma*sigma);
    return norm * factor * std::exp(-(x*x + y*y) / (2.0f * sigma*sigma));
}

__device__ __host__ float gaussianSecondDerivativeXY(float x, float y, float sigma) {
    float sigma4 = sigma * sigma;
    sigma4 = sigma4 * sigma4;
    sigma4 *= (sigma * sigma);
    float norm = 1.0f / (M_PI * sigma4);
    return norm * (x * y) * std::exp(-(x*x + y*y) / (2.0f * sigma*sigma));
}

// ---------- generate kernel into preallocated buffer ----------
template <typename Func>
__device__ __host__ void generateKernel(float* kernelBuf, int ksize, float sigma,  Func func) {
    int half = ksize/2;
    for (int di = -half; di <= half; ++di) {
        for (int dj = -half; dj <= half; ++dj) {
            kernelBuf[(di+half)*ksize + (dj+half)] = func((float)dj,(float)di,sigma);
        }
    }
}

// ---------- convolution at a pixel (bounds-checked) ----------
__device__ __host__ float convolveAtPixel(const float* img, int m, int n, const float* kernel, int ksize, int i, int j) {
    int half = ksize/2;
    float sum = 0.0f;
    for (int di=-half; di<=half; ++di) {
        for (int dj=-half; dj<=half; ++dj) {
            int ni = i + di;
            int nj = j + dj;
            if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                sum += AT(img,ni,nj,n) * kernel[(di+half)*ksize + (dj+half)];
            }
        }
    }
    return sum;
}

// ---------- determinant of hessian (normalized) at pixel, for DoH ----------
__device__ __host__ float detHAtPixelFromImages(const float* Lxx, const float* Lyy, const float* Lxy, int idx, float sigma) {
    float val = Lxx[idx] * Lyy[idx] - Lxy[idx] * Lxy[idx];
    // scale normalization: sigma^4 => multiply by sigma^4; but earlier code used sigma^2 in some places.
    // Following Lindeberg: normalized determinant scales with sigma^4.
    float sigma4 = sigma*sigma; sigma4 = sigma4*sigma4;
    return val * sigma4;
}

// ---------- spatial+scale NMS (3x3x3) checker ----------
__device__ __host__ bool isScaleSpaceMax(const float* detHs, int m, int n, int numScales,
                     int i, int j, int s, float threshold)
{
    int numPixels = m*n;
    const float center = detHs[s * numPixels + i * n + j];
    if (center < threshold) return false;

    int smin = max(0, s-1), smax = min(numScales-1, s+1);
    for (int ss = smin; ss <= smax; ++ss) {
        for (int di=-1; di<=1; ++di) {
            for (int dj=-1; dj<=1; ++dj) {
                if (ss==s && dj==0 && di==0) continue;
                int ni = i + di, nj = j + dj;
                if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
                float v = detHs[ss * numPixels + ni * n + nj];
                if (v >= center) return false;
            }
        }
    }
    return true;
}

/**************************************************************************************************/

// ---------- convolve entire image into out (out preallocated) ----------
void convolveImage(const float* img, float* out, int m, int n, const float* kernel, int ksize) {
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            AT(out,i,j,n) = convolveAtPixel(img, m, n, kernel, ksize, i, j);
}

int getMaxKernelSize(const float* scales, int numScales) {
    int maxKernelSize = 1;
    for (int si = 0; si < numScales; ++si) {
        float sigma = scales[si];
        int k1 = 2 * static_cast<int>(3 * sigma) + 1;
        maxKernelSize = max(maxKernelSize, k1);
    }
    if ((maxKernelSize & 1) == 0) {
        maxKernelSize++;
    }
    return maxKernelSize;
}

// ---------- main detection routine (CPU, shared memory) ----------
vector<tuple<int,int,float,float>> detectBlobsScaleSpaceShared(MemoryManager &mmgr,
        const float* img, int m, int n,
        const float* scales, int numScales, float threshold) {

    auto start = std::chrono::high_resolution_clock::now();
    int numPixels = m * n;

    int maxKernelSize = getMaxKernelSize(scales,numScales);

    // Allocate shared kernels with max size
    float* Kxx = mmgr.allocate_only_host<float>(maxKernelSize * maxKernelSize);
    float* Kyy = mmgr.allocate_only_host<float>(maxKernelSize * maxKernelSize);
    float* Kxy = mmgr.allocate_only_host<float>(maxKernelSize * maxKernelSize);

    // Allocate shared image response buffers 
    float* Lxx = mmgr.allocate_only_host<float>(numPixels);
    float* Lyy = mmgr.allocate_only_host<float>(numPixels);
    float* Lxy = mmgr.allocate_only_host<float>(numPixels);

    // Allocate detHs buffer: numScales contiguous blocks
    float* detHs = mmgr.allocate_only_host<float>(numPixels * numScales);
    std::fill(detHs, detHs + (numPixels * numScales), 0.0f);

    // Generate responses per scale into detHs[s*numPixels + ...]
    for (int s = 0; s < numScales; s++) {
        float sigma = scales[s];

        float *detS = detHs + s * numPixels;

        // generate gaussian 2nd-derivative kernels 
        generateKernel(Kxx, maxKernelSize, sigma, gaussianSecondDerivativeX);
        generateKernel(Kyy, maxKernelSize, sigma, gaussianSecondDerivativeY);
        generateKernel(Kxy, maxKernelSize, sigma, gaussianSecondDerivativeXY);

        // convolve image -> Lxx,Lyy,Lxy (reuse allocated buffers)
        convolveImage(img, Lxx, m, n, Kxx, maxKernelSize);
        convolveImage(img, Lyy, m, n, Kyy, maxKernelSize);
        convolveImage(img, Lxy, m, n, Kxy, maxKernelSize);

        // compute normalized determinant into detHs[s*numPixels + idx]
        for (int idx = 0; idx < numPixels; idx++) {
            detS[idx] = detHAtPixelFromImages(Lxx, Lyy, Lxy, idx, sigma);
        }
    }

    // Now perform scale-space NMS and collect blobs
    vector<tuple<int,int,float,float>> blobs; // x,y,sigma,response
    for (int s = 0; s < numScales; ++s) {
        float sigma = scales[s];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isScaleSpaceMax(detHs, m, n, numScales, i, j, s, threshold)) {
                    float resp = detHs[s * numPixels + i*n + j];
                    blobs.emplace_back(i,j,sigma,resp);
                }
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Time taken for serial DoH is : " << duration.count() << " us" << endl;
    return blobs;
}

/**************************************************************************************************/
__global__ void convolveImageCUDA(const float* img, float* out, int m, int n, const float* kernel, int ksize) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int half = ksize / 2;
    if (i < half || j < half || i >= m - half || j >= n - half) return;

    AT(out,i,j,n) = convolveAtPixel(img, m, n, kernel, ksize, i, j);
}

__global__ void computeDeterminantCUDA(float* detS, float* Lxx, float* Lyy, float* Lxy, float sigma, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;
    detS[idx] = detHAtPixelFromImages(Lxx, Lyy, Lxy, idx, sigma);
}

__global__ void findScaleSpaceMaxCUDA(const float* detHs, bool* response, int m, int n, int numScales, int s, float threshold) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1 || j < 1 || i >= m-1 || j >= n-1) return;

    AT(response,i,j,n) = isScaleSpaceMax(detHs,m,n,numScales,i,j,s,threshold);

}

vector<tuple<int,int,float,float>> 
doNMSCUDABase(int numScales, const float *scales, int m, int n, HostDevicePair<bool>& response, HostDevicePair<float>& detHs, float threshold, float& runtime) {
    vector<tuple<int,int,float,float>> blobs; // x,y,sigma,response
    int numPixels = m * n;

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    for (int s = 0; s < numScales; ++s) {
        float sigma = scales[s];
        {
            CudaTimer timer; 
            std::fill(response.hostPtr,response.hostPtr + numPixels, false);

            findScaleSpaceMaxCUDA<<<gridSize,blockSize>>>(detHs.devicePtr, response.devicePtr, m, n, numScales, s, threshold);

            response.copy_to_host(numPixels);
            CUDA_CHECK(cudaPeekAtLastError()); // launch errors
            CUDA_CHECK(cudaDeviceSynchronize());

            timer.stop();
            runtime += timer.elapsedTimeUs();
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int idx = i * n + j;
                    if (response.hostPtr[idx] == true) {
                        float resp = detHs.hostPtr[s * numPixels + idx];
                        blobs.emplace_back(i,j,sigma,resp);
                    }
                }
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            runtime += duration.count();
        }
    }
    return blobs;
}

__global__ void fusedDeterminantKernel(const float* img, const float* Kxx, const float* Kyy, const float* Kxy, 
    float *Lxx, float *Lyy, float* Lxy, float* detS, int m, int n, int ksize, float sigma) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int half = ksize / 2;
    if (i < half || j < half || i >= m - half || j >= n - half) return;

    AT(Lxx,i,j,n) = convolveAtPixel(img, m, n, Kxx, ksize, i, j);
    AT(Lyy,i,j,n) = convolveAtPixel(img, m, n, Kyy, ksize, i, j);
    AT(Lxy,i,j,n) = convolveAtPixel(img, m, n, Kxy, ksize, i, j);
    AT(detS,i,j,n) = detHAtPixelFromImages(Lxx, Lyy, Lxy, i*n+j, sigma);
}

__global__ void fusedDeterminantShared(const float* img, const float* Kxx, const float* Kyy, const float* Kxy, 
    float* detS, int m, int n, int ksize, float sigma) {

    extern __shared__ float tile[];
    //shared memory as 2D tile with halo
    int radius = ksize/2;
    int tileM, tileN;
    tileM = tileN = TILE_SIZE + 2 * radius;

    float* tile2D = tile;

    int ti = threadIdx.y, tj = threadIdx.x;
    int j = blockIdx.x * TILE_SIZE + tj;
    int i = blockIdx.y * TILE_SIZE + ti;

    //cooperative load
    for (int di = ti; di < tileM; di += blockDim.y) {
        for (int dj = tj; dj < tileN; dj += blockDim.x) {
            int global_i = blockIdx.y * TILE_SIZE + di  - radius;
            int global_j = blockIdx.x * TILE_SIZE + dj  - radius;
            if (global_i >= 0 && global_i < m && global_j >= 0 && global_j < n) {
                tile2D[di * tileN + dj] = img[global_i * n + global_j];
            } else {
                tile2D[di * tileN + dj] = 0.0f;
            }
        }
    }

    __syncthreads();
    if (i >= m || j >= n || tj >= TILE_SIZE || ti >= TILE_SIZE) {
        return;
    }
    float Lxx, Lyy, Lxy;
    Lxx = Lyy = Lxy = 0.0f;

    for (int di = -radius; di <= radius; di++) {
        for (int dj = -radius; dj <= radius; dj++) {
            float pixel = tile2D[(ti + di + radius) * tileN  + (tj + dj + radius)];
            int kidx = (di + radius) * ksize + (dj + radius);
            Lxx += pixel * Kxx[kidx];
            Lyy += pixel * Kyy[kidx];
            Lxy += pixel * Kxy[kidx];
        }
    }

    float s4 = sigma * sigma;
    s4 = s4 * s4;
    float det = (Lxx * Lyy - Lxy * Lxy) * s4;

    detS[i * n + j] = det;
}

// -----------------------------
// Warp-per-pixel NMS reducer
// -----------------------------

__inline__ __device__ float warpReduceMax32(float val) {
    unsigned fullMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(fullMask, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__global__ void warpPerPixelNMSKernel(const float* detHs, bool* response, int m, int n, int numScales, int s, float threshold)
{
    const int WARP_SIZE = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int p = tid / WARP_SIZE;
    int numPixels = m * n;

    if (p >= numPixels) return;

    const int i = p / n;
    const int j = p % n;

    const float centerVal = detHs[s * numPixels + i * n + j];
    
    // Lane 0 checks the threshold for the center pixel
    if (lane == 0 && centerVal < threshold) {
        response[p] = false;
        return;
    }

    // Map the 26 neighbors to the first 26 lanes of the warp.
    // The remaining 6 lanes will be unused.
    float neighborVal = -INFINITY;
    if (lane < 26) {
        int k = lane;
        // The mapping is more complex than a simple linear assignment.
        // It needs to correctly map 0-25 to the 26 neighbors
        // (3 scales * 3x3 grid - 1 center pixel).
        int neighbor_idx = k;
        if (neighbor_idx >= 13) neighbor_idx++; // skip center (at index 13)

        int ss_off = neighbor_idx / 9 - 1;
        int rem = neighbor_idx % 9;
        int i_off = rem / 3 - 1;
        int j_off = rem % 3 - 1;

        int ni = i + i_off;
        int nj = j + j_off;
        int ns = s + ss_off;
        
        // This check ensures we don't access out-of-bounds memory.
        if (ni >= 0 && ni < m && nj >= 0 && nj < n && ns >= 0 && ns < numScales) {
            neighborVal = detHs[ns * numPixels + ni * n + nj];
        }
    }

    // Find the maximum value among the 26 neighbors using warp reduction.
    float maxNeighbor = warpReduceMax32(neighborVal);
    
    // Broadcast the result to all threads in the warp.
    unsigned fullMask = 0xffffffffu;
    maxNeighbor = __shfl_sync(fullMask, maxNeighbor, 0);

    // Only one thread per warp needs to write the result.
    if (lane == 0) {
        // Compare the center value with the maximum neighbor value.
        // For a strict local maximum, centerVal must be strictly greater.
        response[p] = (centerVal > maxNeighbor);
    }
}

// drop-in reducer function: one-warp-per-pixel NMS
vector<tuple<int,int,float,float>> 
doNMSCUDAWarpPerPixel(int numScales, const float *scales, int m, int n,
                      HostDevicePair<bool>& response, HostDevicePair<float>& detHs,
                      float threshold, float& runtime)
{
    vector<tuple<int,int,float,float>> blobs;
    int numPixels = m * n;

    // Launch config: we'll allocate numPixels * WARP_SIZE threads, where each warp maps to one pixel.
    const int WARP_SIZE = 32;
    int totalThreads = numPixels * WARP_SIZE;
    int threadsPerBlock = 256; // typical, must be multiple of warp size
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // kernel uses detHs.devicePtr already populated for all scales
    for (int s = 0; s < numScales; ++s) {
        float sigma = scales[s];

        {
            CudaTimer timer;
            // zero response on host before kernel (optional)
            //std::fill(response.hostPtr, response.hostPtr + numPixels, false);
            // zero device response too
            response.copy_to_device(numPixels);

            // launch kernel
            warpPerPixelNMSKernel<<<blocks, threadsPerBlock>>>(detHs.devicePtr, response.devicePtr, m, n, numScales, s, threshold);

            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            runtime += timer.elapsedTimeUs();
        }

        // copy boolean mask back and collect blobs
        {
            CudaTimer timer;
            response.copy_to_host(numPixels);
            timer.stop();
            runtime += timer.elapsedTimeUs();
        }

        // gather survivors on host reading detHs.hostPtr (detHs host buffer must have been copied already)
        for (int p = 0; p < numPixels; ++p) {
            if (response.hostPtr[p]) {
                int i = p / n;
                int j = p % n;
                float resp = detHs.hostPtr[s * numPixels + p];
                blobs.emplace_back(i, j, sigma, resp);
            }
        }
    }
    return blobs;
}

float findDeterminantCUDAFused(const float* img, const float* Kxx, const float* Kyy, const float* Kxy, 
    float *Lxx, float *Lyy, float* Lxy, float* detS, int m, int n, int maxKernelSize, float sigma) {

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    float runtime = 0.0f;

    CudaTimer timer; 

    fusedDeterminantKernel<<<gridSize,blockSize>>>(img,Kxx,Kyy,Kxy,Lxx,Lyy,Lxy,detS,m,n,maxKernelSize,sigma);

    timer.stop();
    runtime += timer.elapsedTimeUs();

    return runtime;

}

float findDeterminantCUDASharedFused(const float* img, const float* Kxx, const float* Kyy, const float* Kxy, 
    float *Lxx, float *Lyy, float* Lxy, float* detS, int m, int n, int maxKernelSize, float sigma) {

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMem = (TILE_SIZE + maxKernelSize - 1) * (TILE_SIZE + maxKernelSize - 1) * sizeof(float);

    float runtime = 0.0f;

    CudaTimer timer; 

    fusedDeterminantShared<<<gridSize,blockSize,sharedMem>>>(img,Kxx,Kyy,Kxy,detS,m,n,maxKernelSize,sigma);

    timer.stop();
    runtime += timer.elapsedTimeUs();

    return runtime;

}

float findDeterminantCUDABase(const float* img, const float* Kxx, const float* Kyy, const float* Kxy, 
    float *Lxx, float *Lyy, float* Lxy, float* detS, int m, int n, int maxKernelSize, float sigma) {

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    int numPixels = m * n;

    float runtime = 0.0f;

    {
        CudaTimer timer; 

        convolveImageCUDA<<<gridSize,blockSize>>>(img, Lxx, m, n, Kxx, maxKernelSize);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        runtime += timer.elapsedTimeUs();
    }

    {
        CudaTimer timer; 

        convolveImageCUDA<<<gridSize,blockSize>>>(img, Lyy, m, n, Kyy, maxKernelSize);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        runtime += timer.elapsedTimeUs();
    }

    {
        CudaTimer timer; 

        convolveImageCUDA<<<gridSize,blockSize>>>(img, Lxy, m, n, Kxy, maxKernelSize);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        runtime += timer.elapsedTimeUs();
    }

    dim3 blockSize2(TILE_SIZE * TILE_SIZE);
    dim3 gridSize2((numPixels + blockSize2.x - 1)/blockSize2.x);

    // compute normalized determinant into detHs[s*numPixels + idx]

    {
        CudaTimer timer; 

        computeDeterminantCUDA<<<gridSize2,blockSize2>>>(detS,Lxx,Lyy,Lxy,sigma,numPixels);
        CUDA_CHECK(cudaPeekAtLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        runtime += timer.elapsedTimeUs();
    }
    return runtime;
} // find determinant

template <typename DeterminantComputer, typename NMSReducer>
vector<tuple<int,int,float,float>> detectBlobsScaleSpaceSharedCUDA(MemoryManager &mmgr,
        const HostDevicePair<float>& img, int m, int n,
        const float* scales, int numScales, float threshold, DeterminantComputer& computer, NMSReducer& reducer ) {

    cout << endl;
    float runtime = 0.0f;
    int numPixels = 0, maxKernelSize = 1;

    //Step 0 : get kernel size
    {
        numPixels = m * n;
        maxKernelSize = getMaxKernelSize(scales,numScales);
    }

    HostDevicePair<float> Kxx, Kyy, Kxy, Lxx, Lyy, Lxy, detHs;
    HostDevicePair<bool> response;

    //Step 1 : allocate memory
    {
        // Allocate shared kernels with max size
        Kxx = mmgr.allocate_host_and_device<float>(maxKernelSize * maxKernelSize);
        Kyy = mmgr.allocate_host_and_device<float>(maxKernelSize * maxKernelSize);
        Kxy = mmgr.allocate_host_and_device<float>(maxKernelSize * maxKernelSize);

        // Allocate shared image response buffers 
        Lxx = mmgr.allocate_host_and_device<float>(numPixels);
        Lyy = mmgr.allocate_host_and_device<float>(numPixels);
        Lxy = mmgr.allocate_host_and_device<float>(numPixels);

        // Allocate detHs buffer: numScales contiguous blocks
        detHs = mmgr.allocate_host_and_device<float>(numPixels * numScales);
        std::fill(detHs.hostPtr, detHs.hostPtr + (numPixels * numScales), 0.0f);

        // NMS response buffer
        response = mmgr.allocate_host_and_device<bool>(numPixels);
        std::fill(response.hostPtr,response.hostPtr + numPixels, false);
    }

    //Step 2 : copy img, detHs,Lxx,... to device
    {
        CudaTimer timer; 

        img.copy_to_device(numPixels);
        detHs.copy_to_device(numPixels * numScales);

        //copy Lxx,... to device
        Lxx.copy_to_device(numPixels);
        Lyy.copy_to_device(numPixels);
        Lxy.copy_to_device(numPixels);

        //copy response
        response.copy_to_device(numPixels);

        timer.stop();
        runtime += timer.elapsedTimeUs();
    }

    //Step 3 : generate kernels and responses
    {
        // Generate responses per scale into detHs[s*numPixels + ...]
        for (int s = 0; s < numScales; s++) {
            float sigma = scales[s];

            float *detS = detHs.devicePtr + s * numPixels;

            // generate gaussian 2nd-derivative kernels 
            generateKernel(Kxx.hostPtr, maxKernelSize, sigma, gaussianSecondDerivativeX);
            generateKernel(Kyy.hostPtr, maxKernelSize, sigma, gaussianSecondDerivativeY);
            generateKernel(Kxy.hostPtr, maxKernelSize, sigma, gaussianSecondDerivativeXY);

            // convolve image -> Lxx,Lyy,Lxy (reuse allocated buffers)
            {
                CudaTimer timer; 

                //copy Kxx,... to device
                Kxx.copy_to_device(maxKernelSize * maxKernelSize);
                Kyy.copy_to_device(maxKernelSize * maxKernelSize);
                Kxy.copy_to_device(maxKernelSize * maxKernelSize);

                timer.stop();
                runtime += timer.elapsedTimeUs();
            }

            runtime += computer(img.devicePtr, Kxx.devicePtr, Kyy.devicePtr, Kxy.devicePtr, 
                        Lxx.devicePtr, Lyy.devicePtr, Lxy.devicePtr, detS, m, n, maxKernelSize, sigma);
        }
    }

    //Step 4 : copy detHs back to host
    {
            CudaTimer timer; 

            detHs.copy_to_host(numPixels * numScales);

            timer.stop();
            runtime += timer.elapsedTimeUs();
    }

    //Step 5 : do scale-space NMS and collect blobs
    auto blobs = reducer(numScales,scales,m,n,response,detHs,threshold, runtime);
    cout << "Time taken for CUDA DoH is : " << runtime << " us" << endl;
    return blobs;
}

/**************************************************************************************************/

bool approxEqual(const std::tuple<int,int,float,float>& a,
                 const std::tuple<int,int,float,float>& b,
                 float eps = 1e-6f) 
{
    return std::get<0>(a) == std::get<0>(b) &&
           std::get<1>(a) == std::get<1>(b) &&
           std::fabs(std::get<2>(a) - std::get<2>(b)) < eps &&
           std::fabs(std::get<3>(a) - std::get<3>(b)) < eps;
}

int main() {
    MemoryManager mmgr;
    int height = 64, width = 64;
    int numPixels = height * width;

    auto img = mmgr.allocate_host_and_device<float>(numPixels);
    std::fill(img.hostPtr, img.hostPtr+numPixels, 0.0f);

    // sample blobs
    AT(img.hostPtr, height/2, width/2, width) = 255.0f;
    AT(img.hostPtr, height/4, width/4, width) = 200.0f;
    AT(img.hostPtr, 3*height/4, 3*width/4, width) = 180.0f;

    float scales_arr[] = {1.0f, 1.6f, 2.5f, 4.0f};
    int numScales = sizeof(scales_arr)/sizeof(scales_arr[0]);

    // Run DoH Serial
    auto blobs_doh = detectBlobsScaleSpaceShared(mmgr, img.hostPtr, height, width, scales_arr, numScales, 1e-2f);

    cout << "DoH blobs, Serial :\n";
    for (auto &b : blobs_doh) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    // Run DoH Parallel
    auto blobs_doh_cuda = detectBlobsScaleSpaceSharedCUDA(mmgr, img, height, width, scales_arr, numScales, 1e-2f, findDeterminantCUDABase, doNMSCUDABase);
    cout << "DoH blobs, CUDA baseline:\n";
    for (auto &b : blobs_doh_cuda) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    assert(blobs_doh.size() == blobs_doh_cuda.size());
    for (int i = 0; i < blobs_doh.size(); i++) {
        auto bs = blobs_doh[i], bc = blobs_doh_cuda[i];
        assert(approxEqual(bs,bc,1e-3f));
    }

    // Run DoH Parallel - Warp primitives
    auto blobs_doh_cuda2 = detectBlobsScaleSpaceSharedCUDA(mmgr, img, height, width, scales_arr, numScales, 1e-2f, findDeterminantCUDABase, doNMSCUDAWarpPerPixel);
    cout << "DoH blobs, CUDA warp intrinsics:\n";
    for (auto &b : blobs_doh_cuda2) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    assert(blobs_doh.size() == blobs_doh_cuda2.size());
    for (int i = 0; i < blobs_doh.size(); i++) {
        auto bs = blobs_doh[i], bc = blobs_doh_cuda2[i];
        assert(approxEqual(bs,bc,1e-3f));
    } 

    // Run DoH Parallel - Fused Kernel, Warp primitives
    auto blobs_doh_cuda3 = detectBlobsScaleSpaceSharedCUDA(mmgr, img, height, width, scales_arr, numScales, 1e-2f, findDeterminantCUDASharedFused, doNMSCUDAWarpPerPixel);
    cout << "DoH blobs, Fused Kernel CUDA warp intrinsics:\n";
    for (auto &b : blobs_doh_cuda3) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    assert(blobs_doh.size() == blobs_doh_cuda3.size());
    for (int i = 0; i < blobs_doh.size(); i++) {
        auto bs = blobs_doh[i], bc = blobs_doh_cuda3[i];
        assert(approxEqual(bs,bc,1e-3f));
    } 
    return 0;
}


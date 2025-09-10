//nvcc -x cu ncc_template.cpp -o ncc
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <type_traits>

#include "memory_alloc.h"

using namespace std;

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

void fillMatrixRandom(float* matrix, int m, int n) {
    for (int i = 0; i < m * n ; i++) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

//1. Naive Cross Correlation
__host__ __device__ inline
float ccAtPixel(const float *img, int i, int j, int m, int n, const int *K, int k1, int k2) {
    float value = 0.0f;
    int off_i = k1/2, off_j = k2/2;
    for (int di = -off_i; di <= off_i ; di++) {
        for (int dj = -off_j ; dj <= off_j ; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
            float pixel = img[ni * n + nj];
            value += pixel * K[(di + off_i) * k2 + (dj + off_j)];
        }
    }
    return value;
}

void naive_cc(const float *img, float *res, int m, int n, int k1, int k2) {
    int N = n - k2 + 1;
    int off_i = k1/2, off_j = k2/2;
    for (int i = 0; i < m-k1+1; i++) {
        for (int j = 0; j < n-k2+1; j++) {
            int idx = i * N + j;
            res[idx] = ccAtPixel(img,i+off_i,j+off_j,m,n,&SobelFilter::X_host[0][0],k1,k2);
        }
    }
}

//2. Normalized Cross Correlation
//Mean and standard deviation by Welford's algorithm
template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
__host__ __device__ inline
void mean_M2(const T* img, int i, int j, int m, int n, int k1, int k2, float& mean, float& M2) {
    int count = 0;
    M2 = 0.0f;
    mean = 0.0f;
    int off_i = k1/2, off_j = k2/2;
    for (int di = -off_i ; di <= off_i  ; di++) {
        for (int dj = -off_j ; dj <= off_j ; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
            float pixel = static_cast<float>(img[ni * n + nj]);
            count++;
            float delta = pixel - mean;
            mean += delta/count;
            float delta2 = pixel - mean;
            M2 += delta * delta2;
        }
    }
}

void normalized_cc(const float *img, float *res, int m, int n, int k1, int k2) {
    //1. Compute Kernel Mean and Stddev
    float kernel_mean = 0.0f, kernel_M2 = 0.0f;
    mean_M2(&SobelFilter::X_host[0][0],1,1,k1,k2,k1,k2,kernel_mean,kernel_M2); 

    //2. Compute Normalized value
    int N = n - k2 + 1;
    int off_i = k1/2, off_j = k2/2;
    for (int i = 0; i < m-k1+1; i++) {
        for (int j = 0; j < n-k2+1; j++) {
            int idx = i * N + j;
            int ci = i + off_i, cj = j + off_j;
            float mean = 0.0f, M2 = 0.0f;
            mean_M2(img,ci,cj,m,n,k1,k2,mean,M2); 
            float numerator = 0.0f;
            for (int di = -off_i ; di <= off_i ; di++) {
                for (int dj = -off_j ; dj <= off_j ; dj++) {
                    int ni = i + di + off_i , nj = j + dj + off_j;
                    if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
                    float pixel = img[ni * n + nj];
                    float kernel_val = static_cast<float>(SobelFilter::X_host[di + off_i][dj + off_j]);
                    numerator += (pixel - mean) * (kernel_val - kernel_mean);
                }
            }
            float denom = std::sqrt(M2 * kernel_M2) + 1e-12f;
            res[idx] = numerator / denom;
        }
    }
}

//3. Sum of Squared Differences formulation
template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
__host__ __device__ inline void computeIntegralImage(const T* img, float* result, int m, int n) {
    int W = n + 1;
    for (int j = 0; j < W; j++) {
        result[j] = 0.0f;
    }
    for (int i = 1; i <= m; i++) {
        result[i * W + 0] = 0.0f;
    }
    for (int i = 1; i <= m; i++) {
        float rowSumSq = 0.0f;
        for (int j = 1; j <= n; j++) {
            float val = img[(i-1) * n + (j-1)];
            rowSumSq += val * val;
            result[i * W + j] = result[(i-1)*W+j] + rowSumSq;
        }
    }
}

template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
__host__ __device__ inline float patchEnergyAtPixel(const T* img,  int i, int j, int m, int n, int k1, int k2) {
    float value = 0;
    int off_i  = k1/2, off_j = k2/2;
    for (int di = -off_i ; di <= off_i ; di++) {
        for (int dj = -off_j ; dj <=  off_j ; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
            float pixel = img[ni * n + nj];
            value += (pixel * pixel);
        }
    }
    return value;
}

__host__ __device__ inline
float rectSumSqPadded(const float* integral_padded, int n, int x1, int y1, int x2, int y2) {
    // convert to padded indices (shift by +1)
    int W = n + 1;
    int r1 = x1 + 1, c1 = y1 + 1;
    int r2 = x2 + 1, c2 = y2 + 1;
    return integral_padded[r2 * W + c2]
         - integral_padded[(r1 - 1) * W + c2]
         - integral_padded[r2 * W + (c1 - 1)]
         + integral_padded[(r1 - 1) * W + (c1 - 1)];
}

void ssd(const float *img, const float* integral_image, float *res, int m, int n, int k1, int k2) {
    float templateEnergy = patchEnergyAtPixel(&SobelFilter::X_host[0][0],1,1,k1,k2,k1,k2);
    int N = n - k2 + 1;
    int off_i = k1/2, off_j = k2/2;
    for (int i = 0; i < m-k1+1; i++) {
        for (int j = 0; j < n-k2+1; j++) {
            int idx = i * N + j;
            int ci = i + off_i, cj = j + off_j;
            //float patchEnergy = patchEnergyAtPixel(img,i,j,m,n,3,3);
            float patchEnergy = rectSumSqPadded(integral_image,n,ci-off_i,cj-off_j,ci+off_i,cj+off_j);
            float correlation = ccAtPixel(img,ci,cj,m,n,&SobelFilter::X_host[0][0],k1,k2);
            res[idx] = patchEnergy + templateEnergy - 2 * correlation;
        }
    }
}

int main() {
    srand(42);
    int height = 64, width = 64;
    MemoryManager mmgr;

    float *matrix = mmgr.allocate_only_host<float>(height * width);
    fillMatrixRandom(matrix,height,width);

    float *result = mmgr.allocate_only_host<float>((height-2) * (width-2));

    //for SSD
    float *integral_image = mmgr.allocate_only_host<float>((height + 1) * (width + 1));

    {
        auto start = std::chrono::high_resolution_clock::now();

        naive_cc(matrix,result,height,width,3,3);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cout << "Time taken for naive cc is : " << duration.count() << " us" << endl;
    }
    {
        auto start = std::chrono::high_resolution_clock::now();

        normalized_cc(matrix,result,height,width,3,3);
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        for (int i = 0; i < (height - 2) * (width - 2); i++) {
            assert((result[i] >= -1) && (result[i] <= 1));
        }

        cout << "Time taken for normalized cc is : " << duration.count() << " us" << endl;
    }
    {
        auto start = std::chrono::high_resolution_clock::now();

        computeIntegralImage(matrix,integral_image,height,width);
        ssd(matrix,integral_image,result,height,width,3,3);
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cout << "Time taken for ssd is : " << duration.count() << " us" << endl;
    }
}

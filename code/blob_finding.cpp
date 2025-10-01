//nvcc -x cu doh.cpp -o doh 
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include "memory_alloc.h"   

using namespace std;

#define AT(img, i, j, n) (img[(i)*(n) + (j)])

// ---------- kernels (LoG & DoG & DoH components) ----------
float gaussian(float x, float y, float sigma) {
    float norm = 1.0f / (2.0f * M_PI * sigma * sigma);
    return norm * std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
}

float laplacianOfGaussian(float x, float y, float sigma) {
    float r2 = x*x + y*y;
    float norm = -1.0f / (M_PI * pow(sigma, 4));
    float factor = 1.0f - (r2 / (2.0f * sigma * sigma));
    return norm * factor * std::exp(-r2 / (2.0f * sigma * sigma));
}

float gaussianSecondDerivativeX(float x, float y, float sigma) {
    float norm = -1.0f / (M_PI * pow(sigma, 4));
    float factor = 1.0f - (x*x) / (sigma*sigma);
    return norm * factor * std::exp(-(x*x + y*y) / (2.0f * sigma*sigma));
}

float gaussianSecondDerivativeY(float x, float y, float sigma) {
    float norm = -1.0f / (M_PI * pow(sigma, 4));
    float factor = 1.0f - (y*y) / (sigma*sigma);
    return norm * factor * std::exp(-(x*x + y*y) / (2.0f * sigma*sigma));
}

float gaussianSecondDerivativeXY(float x, float y, float sigma) {
    float norm = 1.0f / (M_PI * pow(sigma, 6));
    return norm * (x * y) * std::exp(-(x*x + y*y) / (2.0f * sigma*sigma));
}

// ---------- generate kernel into preallocated buffer ----------
void generateKernel(float* kernelBuf, int ksize, float sigma, std::function<float(float,float,float)> func) {
    int half = ksize/2;
    for (int di = -half; di <= half; ++di) {
        for (int dj = -half; dj <= half; ++dj) {
            kernelBuf[(di+half)*ksize + (dj+half)] = func((float)dj,(float)di,sigma);
        }
    }
}

// ---------- convolution at a pixel (bounds-checked) ----------
inline float convolveAtPixel(const float* img, int m, int n, const float* kernel, int ksize, int i, int j) {
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

// ---------- convolve entire image into out (out preallocated) ----------
void convolveImage(const float* img, float* out, int m, int n, const float* kernel, int ksize) {
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            AT(out,i,j,n) = convolveAtPixel(img, m, n, kernel, ksize, i, j);
}

// ---------- determinant of hessian (normalized) at pixel, for DoH ----------
inline float detHAtPixelFromImages(const float* Lxx, const float* Lyy, const float* Lxy, int idx, float sigma) {
    float val = Lxx[idx] * Lyy[idx] - Lxy[idx] * Lxy[idx];
    // scale normalization: sigma^4 => multiply by sigma^4; but earlier code used sigma^2 in some places.
    // Following Lindeberg: normalized determinant scales with sigma^4.
    float sigma4 = sigma*sigma; sigma4 = sigma4*sigma4;
    return val * sigma4;
}

// ---------- spatial+scale NMS (3x3x3) checker ----------
bool isScaleSpaceMax(const float* detHs, int m, int n, int numScales,
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

// ---------- detector enum ----------
enum DetectorType { DOH=0, LOG=1, DOG = 2 };

// ---------- main detection routine (CPU, shared memory) ----------
vector<tuple<int,int,float,float>> detectBlobsScaleSpaceShared(MemoryManager &mmgr,
        const float* img, int m, int n,
        const float* scales, int numScales, float threshold,
        DetectorType detector)
{
    int numPixels = m * n;

    int maxKernelSize = 1;
    float siftFactor = 1.6f;
    for (int si = 0; si < numScales; ++si) {
        float sigma = scales[si];
        int k1 = 2 * static_cast<int>(3 * sigma) + 1;
        float ksigma = sigma * siftFactor;
        int k2 = 2 * static_cast<int>(3 * ksigma) + 1;
        maxKernelSize = max(maxKernelSize, max(k1, k2));
    }
	if ((maxKernelSize & 1) == 0) {
		maxKernelSize++;
	}

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
        int ksize = 2 * static_cast<int>(3 * sigma) + 1;
        ksize = max(3,ksize);
        if ((ksize & 1) == 0) {
            ksize++;
        }

        float *detS = detHs + s * numPixels;

        if (detector == DOH) {
            // generate gaussian 2nd-derivative kernels 
            generateKernel(Kxx, ksize, sigma, gaussianSecondDerivativeX);
            generateKernel(Kyy, ksize, sigma, gaussianSecondDerivativeY);
            generateKernel(Kxy, ksize, sigma, gaussianSecondDerivativeXY);

            // convolve image -> Lxx,Lyy,Lxy (reuse allocated buffers)
            convolveImage(img, Lxx, m, n, Kxx, ksize);
            convolveImage(img, Lyy, m, n, Kyy, ksize);
            convolveImage(img, Lxy, m, n, Kxy, ksize);

            // compute normalized determinant into detHs[s*numPixels + idx]
            for (int idx = 0; idx < numPixels; idx++) {
                detS[idx] = detHAtPixelFromImages(Lxx, Lyy, Lxy, idx, sigma);
            }

        } else if (detector == LOG) { // LOG
            // generate LoG kernel 
            generateKernel(Kxx, ksize, sigma, laplacianOfGaussian);
            // convolve into Lxx
            convolveImage(img, Lxx, m, n, Kxx, ksize);

            // normalize by sigma^2 (scale-normalized LoG)
            float sigma2 = sigma * sigma;
            for (int idx = 0; idx < numPixels; idx++) {
                detS[idx] = Lxx[idx] * sigma2;
            }
        } else {

            // generate gaussian kernel 
            generateKernel(Kxx, ksize, sigma, gaussian);
            // convolve into Lxx
            convolveImage(img, Lxx, m, n, Kxx, ksize);

            // generate gaussian kernel 
            generateKernel(Kyy, ksize, sigma * siftFactor, gaussian);
            // convolve into Lyy
            convolveImage(img, Lyy, m, n, Kyy, ksize);

            for (int idx = 0; idx < numPixels; idx++) {
                detS[idx] = Lyy[idx] - Lxx[idx];
            }
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

    return blobs;
}

int main() {
    MemoryManager mmgr;
    int height = 64, width = 64;
    int numPixels = height * width;

    float* img = mmgr.allocate_only_host<float>(numPixels);
    std::fill(img, img+numPixels, 0.0f);

    // sample blobs
    AT(img, height/2, width/2, width) = 255.0f;
    AT(img, height/4, width/4, width) = 200.0f;
    AT(img, 3*height/4, 3*width/4, width) = 180.0f;

    float scales_arr[] = {1.0f, 1.6f, 2.5f, 4.0f};
    int numScales = sizeof(scales_arr)/sizeof(scales_arr[0]);

    // Run DoH
    auto blobs_doh = detectBlobsScaleSpaceShared(mmgr, img, height, width, scales_arr, numScales, 1e-2f, DOH);
    cout << "DoH blobs:\n";
    for (auto &b : blobs_doh) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    // Run LoG 
    auto blobs_log = detectBlobsScaleSpaceShared(mmgr, img, height, width, scales_arr, numScales, 1e-2f, LOG);
    cout << "LoG blobs:\n";
    for (auto &b : blobs_log) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    // Run DoG 
    auto blobs_dog = detectBlobsScaleSpaceShared(mmgr, img, height, width, scales_arr, numScales, 1e-2f, DOG);
    cout << "DoG blobs:\n";
    for (auto &b : blobs_dog) {
        cout << " x=" << get<0>(b) << " y=" << get<1>(b) << " sigma=" << get<2>(b) << " resp=" << get<3>(b) << "\n";
    }

    return 0;
}


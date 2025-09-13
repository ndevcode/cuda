//nvcc -x cu dft.cpp -o dft
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <chrono>

#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"
#include "cuda_timer.h"

using namespace std;

void fillArrayRandom(float *arr, int N)  {
    for (int i = 0; i < 2 * N; i += 2) {
        arr[i] = static_cast<float>(rand() % 255);
        arr[i+1] = 0.0f;
    }
}

void discreteFourierTransform(const float *X_n, float *X_k, int N) {
    //this is an O(N^2) algorithm
    constexpr float PI = 3.14159265358979323846f;
    for (int k = 0; k < N; k++) { //outer loop for X_k
        float real = 0.0f, imag = 0.0f;
        for (int n = 0; n < N; n++) { //iterate over all X_n
            float angle = (-2 * PI * k * n) / N ;
            float cos_t = std::cos(angle);
            float sin_t = std::sin(angle);
            float a = X_n[2 * n];
            float b = X_n[2 * n + 1];
            real += a * cos_t - b * sin_t;
            imag += a * sin_t + b * cos_t;
        }
        X_k[2 * k]     = real;
        X_k[2 * k + 1] = imag;
    } 
}

void discreteFourierTransform2(const float *X_n, float *X_k, int N) {
    constexpr float PI = 3.14159265358979323846f;

    for (int k = 0; k < N; k++) { // for each frequency bin
        float real = 0.0f, imag = 0.0f;

        // Precompute the angle increment for this frequency bin
        float angle_step = -2.0f * PI * k / N;
        float cos_step   = std::cos(angle_step);
        float sin_step   = std::sin(angle_step);

        // Start with W = 1 (cos(0) + i*sin(0))
        float cos_t = 1.0f;
        float sin_t = 0.0f;

        for (int n = 0; n < N; n++) {
            float a = X_n[2 * n];     // real part of x[n]
            float b = X_n[2 * n + 1]; // imag part of x[n]

            // Multiply x[n] by the current twiddle factor
            real += a * cos_t - b * sin_t;
            imag += a * sin_t + b * cos_t;

            // Rotate twiddle factor: (cos_t + i sin_t) *= (cos_step + i sin_step)
            float tmp_cos_t = cos_t * cos_step - sin_t * sin_step;
            float tmp_sin_t = cos_t * sin_step + sin_t * cos_step;
            cos_t           = tmp_cos_t;
            sin_t           = tmp_sin_t;
        }

        X_k[2 * k]     = real;
        X_k[2 * k + 1] = imag;
    }
}

void bitReverse(float *X_n, int N) {

    auto reverseBits = [](int n, int N) {
        unsigned int rev = 0;
        int numBits = static_cast<int>(std::log2(N));
        for (int i = 0; i < numBits; i++) {
            rev <<= 1;
            rev |= (n & 1);
            n >>= 1;
        }
        return rev;
    };

    for (int i = 0; i < N; i++) {
        int rev = reverseBits(i,N);
        if (rev > i) {
            swap(X_n[2*i]  ,X_n[2*rev]);
            swap(X_n[2*i+1],X_n[2*rev+1]);
        }
    }
}

__global__ void bitReverseCUDA(float *X_n, int N, int logN) {
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    if (i >= N) return;

    auto reverseBits = [](int n, int logN) {
        unsigned int rev = 0;
        int numBits = logN;
        for (int i = 0; i < numBits; i++) {
            rev <<= 1;
            rev |= (n & 1);
            n >>= 1;
        }
        return rev;
    };

    int rev = reverseBits(i,logN);
    if (rev > i) {
        float tmp = X_n[2*i];
        X_n[2*i]  = X_n[2*rev];
        X_n[2*rev] = tmp;

        tmp = X_n[2*i+1];
        X_n[2*i+1]  = X_n[2*rev+1];
        X_n[2*rev+1] = tmp;
    }
}

__global__ void fftstage(float *X_n, int N, int m, const float PI) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int butterflies = N / 2;
    if (tid >= butterflies) return;

    int group = tid / (m / 2);
    int j     = tid % (m / 2);
    int k     = group * m;

    float theta = -2 * PI * j / m;
    float w_real = cosf(theta);
    float w_imag = sinf(theta);

    int left_idx = 2 * (k + j);
    int right_idx = 2 * (k + j + m/2);

    float u_real = X_n[left_idx];
    float u_imag = X_n[left_idx + 1];

    float v_real = X_n[right_idx] * w_real - X_n[right_idx + 1] * w_imag;
    float v_imag = X_n[right_idx] * w_imag + X_n[right_idx + 1] * w_real;

    X_n[left_idx]       = u_real + v_real;
    X_n[left_idx + 1]   = u_imag + v_imag;
    X_n[right_idx]      = u_real - v_real;
    X_n[right_idx + 1]  = u_imag - v_imag;

}

float fastFFTCUDA(HostDevicePair<float>& X_n, int N) {
    float runtime = 0.0f;

    //1. H2D copy
    {
        CudaTimer timer;

        X_n.copy_to_device(2*N);

        timer.stop();

        float stepTime = timer.elapsedTimeUs();

        cout << "H2D copy time " << stepTime << endl;
        runtime += stepTime;
    }

    //2. bit reverse
    {
        CudaTimer timer;
        int logN = 0; while ((1 << logN) < N) logN++;

        dim3 blockSize(256);
        dim3 gridSize((N + blockSize.x - 1)/blockSize.x);

        bitReverseCUDA<<<gridSize,blockSize>>>(X_n.devicePtr,N,logN);

        CUDA_CHECK(cudaPeekAtLastError()); // launch errors

        timer.stop();

        float stepTime = timer.elapsedTimeUs();

        cout << "bit reverse time CUDA " << stepTime << endl;
        runtime += stepTime;
    }

    //3. FFT Algo
    constexpr float PI = 3.14159265358979323846f;

    int butterfiles = N/2;
    int numThreads = 256;

    float algoTime = 0.0f;
    for (int m = 2; m <= N; m *= 2) {
        dim3 blockSize(numThreads);
        dim3 gridSize((butterfiles + blockSize.x - 1)/blockSize.x);

        CudaTimer timer;

        fftstage<<<gridSize,blockSize>>>(X_n.devicePtr,N,m,PI);

        CUDA_CHECK(cudaPeekAtLastError()); // launch errors

        timer.stop();

        algoTime += timer.elapsedTimeUs();
    }

    cout << "Algo time " << algoTime << endl;
    runtime += algoTime;

    //4. D2H copy time
    {
        CudaTimer timer;

        X_n.copy_to_host(2*N);

        timer.stop();

        float stepTime = timer.elapsedTimeUs();

        cout << "D2H copy time " << stepTime << endl;
        runtime += stepTime;
    }

    return runtime;
}

float fastFourierTransform(float *X_n, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    bitReverse(X_n,N);

    constexpr float PI = 3.14159265358979323846f;

    for (int m = 2; m <= N; m *= 2) {

        float theta = -2 * PI / m;
        float w_m_real = std::cos(theta);
        float w_m_imag = std::sin(theta);

        for (int k = 0; k < N; k += m) {

            float w_real = 1.0f, w_imag = 0.0f;

            for (int j = 0; j < m/2; j++) {

                int left_idx  = 2 * (k + j);
                int right_idx = 2 * (k + j + m/2);

                float u_real = X_n[left_idx];
                float u_imag = X_n[left_idx + 1];

                float v_real = X_n[right_idx] * w_real - X_n[right_idx + 1] * w_imag;
                float v_imag = X_n[right_idx] * w_imag + X_n[right_idx + 1] * w_real;

                X_n[left_idx]       = u_real + v_real;
                X_n[left_idx + 1]   = u_imag + v_imag;
                X_n[right_idx]      = u_real - v_real;
                X_n[right_idx + 1]  = u_imag - v_imag;

                float tmp_real = w_real * w_m_real  - w_imag * w_m_imag;
                w_imag = w_real * w_m_imag + w_imag * w_m_real;
                w_real = tmp_real;
            }

        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return duration.count();
}

template <typename FFT>
void inverseFFT(FFT theFFT, float *X_k, int N) {
    for (int i = 0; i < 2*N; i += 2) {
        X_k[i+1] = -X_k[i+1]; //conjugate
    }

    theFFT(X_k,N);

    for (int i = 0; i < 2*N; i += 2) {
        X_k[i+1] = -X_k[i+1]; //conjugate
    }

    for (int i = 0; i < 2*N; i++) {
        X_k[i] /= N;
    }
}

int main() {
	srand(42);

	int n = (1 << 15) + 7 ;
	int N = 1; while (N < n) { N <<= 1; }

	MemoryManager mmgr;

	auto  X_n = mmgr.allocate_host_and_device<float>(N*2);
	float *X_k = mmgr.allocate_only_host<float>(N*2);
	float *X_n_copy = mmgr.allocate_only_host<float>(N*2);

	std::fill(X_n.hostPtr,X_n.hostPtr + 2*N, 0.0f);
	std::fill(X_k,X_k + 2*N, 0.0f);

	fillArrayRandom(X_n.hostPtr,n);
	std::copy(X_n.hostPtr,X_n.hostPtr+2*N,X_n_copy);

    {
        auto start = std::chrono::high_resolution_clock::now();
        discreteFourierTransform2(X_n.hostPtr,X_k,N);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        cout << "Time taken for DFT is : " << duration.count() << " us" << endl;
    }

	//Find FFT, then IFFT and the compare
    {
        float duration = fastFourierTransform(X_n.hostPtr,N);
        cout << "Time taken for FFT is : " << duration << " us" << endl;
    }

	//compare DFT and FFT output
	double l2_err = 0.0;
	double l2_ref = 0.0;

	for (int i = 0; i < 2*n; ++i) {
		double diff = X_k[i] - X_n.hostPtr[i];
		l2_err += diff * diff;
		l2_ref += X_k[i] * X_k[i];
	}
	l2_err = std::sqrt(l2_err);
	l2_ref = std::sqrt(l2_ref);

	double rel_l2 = l2_err / (l2_ref > 0 ? l2_ref : 1.0);

	assert(rel_l2 < 1e-3);

    //find IFFT and then compare

	inverseFFT(fastFourierTransform, X_n.hostPtr,N);
	int numErrors = 0;
	for (int k = 0; k < n; k++) {
		float orig = X_n_copy[2*k];
		float ifft = X_n.hostPtr[2*k];

		float err_real = std::abs(orig - ifft);

		if (err_real > 1e-1) {
			numErrors++;
		}
	} 
	assert(numErrors == 0);

    {
        float runtime = 0.0f;

        runtime = fastFFTCUDA(X_n,N);

        cout << "Time taken for FFT CUDA is : " << runtime << " us" << endl;
    }

	//compare DFT and FFT output
	l2_err = 0.0;
	l2_ref = 0.0;

	for (int i = 0; i < 2*n; ++i) {
		double diff = X_k[i] - X_n.hostPtr[i];
		l2_err += diff * diff;
		l2_ref += X_k[i] * X_k[i];
	}
	l2_err = std::sqrt(l2_err);
	l2_ref = std::sqrt(l2_ref);

	rel_l2 = l2_err / (l2_ref > 0 ? l2_ref : 1.0);

	assert(rel_l2 < 1e-3);

}

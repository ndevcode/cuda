#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cfloat>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "ndev_cuda_utility.hpp"
#include "memory_alloc.h"
#include "cuda_timer.h"

using namespace std;

template <typename T>
struct RowView {
        T* rowData_;
        __host__ __device__ RowView(T* data) : rowData_(data) {}
        __host__ __device__ T& operator[](int j) { return rowData_[j]; }
        __host__ __device__ const T& operator[](int j) const { return rowData_[j]; }
};

template <typename T>
struct MatrixView {
        T* data_;
        int rows_,cols_;
        __host__ __device__ MatrixView(T* data, int r, int c) : data_(data), rows_(r), cols_(c) {}
        __host__ __device__ RowView<T> operator[](int i) { return RowView<T>(data_ + (i * cols_)); }
        __host__ __device__ const RowView<T> operator[](int i) const { return RowView<T>(data_ + (i * cols_)); }
};

// Solves (H + damping*I) * delta = grad via Gaussian elimination with
// partial pivoting. `A` is a [d x (d+1)] workspace in global memory.
// `grad` and `delta` may alias.
__device__ void solve_linear(const float* H_ptr, const float* grad, float* delta,
                             float* A_ptr, int d, float damping) {
        const int W = d + 1;
    MatrixView<const float> H(H_ptr, d, d + 1);
    MatrixView<float> A(A_ptr, d, d);

        // Augmented matrix [H + damping*I | grad]
        for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                        A[i * W + j] = H[i * d + j];
                }
                A[i * W + i] += damping;
                A[i * W + d]  = grad[i];
        }

        // Forward elimination with partial pivoting
        for (int i = 0; i < d; i++) {

                // Pivot
                int pivot = i;
                float max_val = fabsf(A[i * W + i]);

                for (int k = i + 1; k < d; k++) {
                        float val = fabsf(A[k * W + i]);
                        if (val > max_val) {
                                max_val = val;
                                pivot = k;
                        }
                }

                // Swap rows
                if (pivot != i) {
                        for (int j = i; j <= d; j++) {
                                float tmp = A[i * W + j];
                                A[i * W + j] = A[pivot * W + j];
                                A[pivot * W + j] = tmp;
                        }
                }

                // Avoid division by zero (very crude safeguard)
                float diag = A[i * W + i];
                if (fabsf(diag) < 1e-12f) diag = (diag >= 0.0f ? 1e-12f : -1e-12f);

                // Normalize row
                float inv = 1.0f / diag;
                for (int j = i; j <= d; j++) {
                        A[i * W + j] *= inv;
                }

                // Eliminate below
                for (int k = i + 1; k < d; k++) {
                        float factor = A[k * W + i];

                        for (int j = i; j <= d; j++) {
                                A[k * W + j] -= factor * A[i * W + j];
                        }
                }
        }

        // Back substitution
        for (int i = d - 1; i >= 0; i--) {
                float val = A[i * W + d];

                for (int j = i + 1; j < d; j++) {
                        val -= A[i * W + j] * delta[j];
                }

                delta[i] = val;
        }
}

__global__ void grad_hessian_kernel(
    const float* X_ptr, const float* y, const float* beta,
    float* grad, float* H_out,
    int n, int d, float invN)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        MatrixView<const float> X(X_ptr, n, d);

        float z = 0.0f;
        for (int j = 0; j < d; j++) {
                z += X[i][j] * beta[j];
        }

        float yhat = 1.0f / (1.0f + expf(-z));

        float diff = yhat - y[i];
        float w    = yhat * (1.0f - yhat);

        for (int j = 0; j < d; j++) {
                atomicAdd(&grad[j], X[i][j] * diff * invN);

                MatrixView<float> H(H_out, d, d);
                for (int k = 0; k < d; k++) {
                        atomicAdd(&H[j][k], X[i][j] * X[i][k] * w * invN);
                }
        }
}

__global__ void solve_kernel(
    const float* H, const float* beta, float* grad_delta, float* A,
    int d, float damping, float lambda_reg)
{
        if (blockIdx.x != 0 || threadIdx.x != 0) return;

        // L2 regularization on the gradient: grad += lambda_reg * beta
        for (int i = 0; i < d; i++) grad_delta[i] += lambda_reg * beta[i];

        solve_linear(H, grad_delta, grad_delta, A, d, damping);
}

__global__ void update_beta_kernel(float* beta, const float* delta, int d) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < d) beta[i] -= delta[i];
}

extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {
        assert(n_samples > 0   && "n_samples must be positive");
        assert(n_features > 0  && "n_features must be positive");

        const int   max_iter   = 30;
        const float invN       = 1.0f / static_cast<float>(n_samples);
        const float lambda_reg = 1e-6f * invN;
        const float damping    = lambda_reg + 1e-6f;

        float *d_grad = nullptr, *d_H = nullptr, *d_A = nullptr;
        CUDA_CHECK(cudaMalloc(&d_grad, n_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_H,    n_features * n_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A,    n_features * (n_features + 1) * sizeof(float)));

        CUDA_CHECK(cudaMemset(beta, 0, n_features * sizeof(float)));

        const int BLOCK = 256;
        const dim3 gridSamples ((n_samples  + BLOCK - 1) / BLOCK);
        const dim3 gridFeatures((n_features + BLOCK - 1) / BLOCK);
        const dim3 block(BLOCK);

        for (int iter = 0; iter < max_iter; iter++) {
                CUDA_CHECK(cudaMemset(d_grad, 0, n_features * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_H,    0, n_features * n_features * sizeof(float)));

                grad_hessian_kernel<<<gridSamples, block>>>(
                    X, y, beta, d_grad, d_H, n_samples, n_features, invN);

                solve_kernel<<<1, 1>>>(d_H, beta, d_grad, d_A, n_features, damping, lambda_reg);

                update_beta_kernel<<<gridFeatures, block>>>(beta, d_grad, n_features);
        }

        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_H));
        CUDA_CHECK(cudaFree(d_A));
}

int main() {
    MemoryManager mmgr;
    const int n_samples = 16;
    const int n_features = 8;
    auto X = mmgr.allocate_host_and_device<float>(n_samples * n_features);
    auto y = mmgr.allocate_host_and_device<float>(n_samples);
    auto beta = mmgr.allocate_host_and_device<float>(n_features);
    fillArrayRandom(X.hostPtr,n_samples * n_features ,true, 1000);
    fillArrayRandom(y.hostPtr,n_samples,true, 1000);

    X.copy_to_device(n_samples * n_features);
    y.copy_to_device(n_samples);

    CudaTimer timer;
    timer.start();
    solve(X.devicePtr , y.devicePtr, beta.devicePtr, n_samples, n_features) ;
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU logistic regression is " << runtime << "us" << endl;
}

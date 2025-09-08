#include "my_cuda_kernels.cuh"

__global__ void relu_forward_kernel(float* in, float* out, int sz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sz) return;
    out[idx] = (in[idx] > 0.0f) ? in[idx] : 0.0f;
}

__global__ void relu_backward_kernel(float* in, float* grad_out, float* grad_in, int sz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sz) return;
    grad_in[idx] += (in[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

void relu_forward(float *in, float* out, int sz) {
    dim3 blockSize(256);
    dim3 gridSize((sz + blockSize.x - 1)/blockSize.x);

    relu_forward_kernel<<<blockSize,gridSize>>>(in,out,sz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu_backward(float *in, float* grad_out, float* grad_in, int sz) {
    dim3 blockSize(256);
    dim3 gridSize((sz + blockSize.x - 1)/blockSize.x);

    relu_backward_kernel<<<blockSize,gridSize>>>(in,grad_out,grad_in,sz);
    CUDA_CHECK(cudaDeviceSynchronize());
}


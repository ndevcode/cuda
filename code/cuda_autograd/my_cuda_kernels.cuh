#pragma once

#include <cuda_runtime.h>
#include "ndev_cuda_utility.hpp"

void relu_forward(float* in, float* out, int sz);
void relu_backward(float* in, float* grad_out, float* grad_in, int sz);
void matmul(const float* A, const float* B, float* C, int M, int K, int N);


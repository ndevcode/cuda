
# CUDA Projects Portfolio

This repository contains a collection of CUDA and C++ projects that I have implemented to explore GPU programming, parallel algorithms, and performance optimization.  
The projects are grouped by category for clarity.
---

##  Deep Learning
- [Transformer Softmax Attention](https://github.com/ndevcode/cuda/blob/main/code/softmax_attention.cu)  
  GPU-based softmax attention computation, illustrating use of matrix multiplication and per query softmax computation.

- [Categorical cross entropy loss](https://github.com/ndevcode/cuda/blob/main/code/entropy_loss_categorical.cu)
  For a MLP output of shape N X C, where is N is batch size and C is the number of categories and true labels of shape N X 1 find the cross entropy loss.

- [INT8 Quantized Matmul](https://github.com/ndevcode/cuda/blob/main/code/int8_qm_6.cu)
  Implement a quantized matrix multiplication program for 8-bit signed integer matrices.
---

##  Reduction Algorithms
- [max_kernel.cu](https://github.com/ndevcode/cuda/blob/main/code/max_kernel.cu)  
  Custom CUDA kernel to compute maximum values across arrays, illustrating warp-level parallelism and reduction strategies.

- [blelloch_prefixsum.cpp](https://github.com/ndevcode/cuda/blob/main/code/blelloch_prefixsum.cpp)  
  Parallel prefix-sum (scan) implementation using the Blelloch upsweep/downsweep algorithm with multi-block CUDA support.
---

## Prefix scan 
- [prefix_sum_4.cu](https://github.com/ndevcode/cuda/blob/main/code/prefix_sum_4.cu)
  prefix scan to compute prefix sums/products etc.

- [prefix_sum_generic.cu](https://github.com/ndevcode/cuda/blob/main/code/prefix_scan_generic.cu)
  Generic prefix scan to compute prefix sums
---

##  Matrix Multiplication
- [naiveMatrixMultiply.cpp](https://github.com/ndevcode/cuda/blob/main/code/naiveMatrixMultiply.cpp)  
  GPU-based naive matrix multiplication for baseline comparison.

- [matmul.cpp](https://github.com/ndevcode/cuda/blob/main/code/matmul.cpp)  
  CUDA shared memory based matrix multiplication

- [gemm4.cu](https://github.com/ndevcode/cuda/blob/main/code/gemm4.cu)
  Matrix multiplication where each thread does 4 multiplications instead of 1 with the same shared memory

- [gemm_outer_product.cu](https://github.com/ndevcode/cuda/blob/main/code/gemm_outer_product.cu)
  Matrix multiplication but compute outer product in a register tile rather than computing dot products.
---

## Fast Fourier Transform (FFT)
- [fft.cpp](https://github.com/ndevcode/cuda/blob/main/code/fft.cpp)  
  Fast Fourier Transform implementation using CUDA, leveraging parallel butterfly operations for signal transforms.

---

## Numerical Computing
- [logistic_regression.cu](https://github.com/ndevcode/cuda/blob/main/code/logisticRegression.cu)
  Solve logistic regression for a small problem where the linear solve can be done on a single thread.
---

##  Image Processing & Computer Vision
- [sobel.cpp](https://github.com/ndevcode/cuda/blob/main/code/sobel.cpp)  
  Edge detection using Sobel filters on GPU.

- [canny.cpp](https://github.com/ndevcode/cuda/blob/main/code/canny.cpp)  
  Full Canny edge detection pipeline accelerated with CUDA.

- [conv1d.cpp](https://github.com/ndevcode/cuda/blob/main/code/conv1d.cpp)  
  1D convolution operator implemented in CUDA.

- [template_matching.cpp](https://github.com/ndevcode/cuda/blob/main/code/template_matching.cpp)  
  Normalized cross-correlation (NCC)–based template matching using CUDA.

- [determinantOfHessian.cpp](https://github.com/ndevcode/cuda/blob/main/code/determinantOfHessian.cpp)  
  CUDA-based computation of the Hessian determinant, used for blob detection (DoH) and scale-space interest point detection.

- [blob_finding.cpp](https://github.com/ndevcode/cuda/blob/main/code/blob_finding.cpp)  
  CPU-based implementation of blob detection using Determinant of Hessian (DoH), Laplacian of Gaussian (LoG), and Difference of Gaussian (DoG) with scale-space non-maximum suppression.
---


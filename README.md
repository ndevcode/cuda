# CUDA Projects Portfolio

This repository contains a collection of CUDA and C++ projects that I have implemented to explore GPU programming, parallel algorithms, and performance optimization.  
The projects are grouped by category for clarity.

---

## Core CUDA Fundamentals
- **vectorAdd.cpp / vectorAdd2.cpp**  
  Basic CUDA vector addition examples, demonstrating memory transfers, grid/block indexing, and kernel launches.

- **my_cuda_kernels.cuh**  
  Header file collecting reusable custom CUDA kernels for modular development.

---

## Reduction Algorithms
- **max_kernel.cu**  
  Custom CUDA kernel to compute maximum values across arrays, illustrating warp-level parallelism and reduction strategies.

- **blelloch_prefixsum.cpp**  
  Parallel prefix-sum (scan) implementation using the Blelloch upsweep/downsweep algorithm with multi-block CUDA support.

---

## Matrix Multiplication
- **naiveMatrixMultiply.cpp**  
  CPU-based naive matrix multiplication for baseline comparison.

- **matmul.cpp**  
  Optimized CUDA matrix multiplication (GEMM-like), exploring shared memory, tiling techniques, and memory coalescing.

---

## Fast Fourier Transform (FFT)
- **fft.cpp**  
  Fast Fourier Transform implementation using CUDA, leveraging parallel butterfly operations for signal transforms.

---

## Image Processing & Computer Vision
- **sobel.cpp**  
  Edge detection using Sobel filters on GPU.

- **canny.cpp**  
  Full Canny edge detection pipeline accelerated with CUDA.

- **conv1d.cpp**  
  1D convolution operator implemented in CUDA.

- **template_matching.cpp**  
  Normalized cross-correlation (NCC)–based template matching using CUDA.

- **determinantOfHessian.cpp**  
  CUDA-based computation of the Hessian determinant, used for blob detection (DoH) and scale-space interest point detection.

---

## Utilities & Infrastructure
- **cudaTimer.h**  
  Simple CUDA timing utility for profiling GPU kernel execution.

- **memory_alloc.h**  
  Helper functions for CUDA memory allocation and management.

- **utility.hpp**  
  Common utility functions and abstractions used across multiple CUDA projects.

- **cuda_autograd**  
  Experimental beginnings of an automatic differentiation system for CUDA, exploring GPU-based gradient computation.

---

## Notes
- Projects range from **fundamental CUDA exercises** (vector addition, prefix sum) to **advanced CV algorithms** (Canny, DoH, template matching).
- Each project highlights **parallel programming principles**:  
  - memory coalescing  
  - shared memory tiling  
  - warp divergence avoidance  
  - profiling and benchmarking with custom timers  

---


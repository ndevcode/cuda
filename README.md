
# CUDA Projects Portfolio

This repository contains a collection of CUDA and C++ projects that I have implemented to explore GPU programming, parallel algorithms, and performance optimization.  
The projects are grouped by category for clarity.

---

##  Reduction Algorithms
- [max_kernel.cu](https://github.com/ndevcode/cuda/blob/main/code/max_kernel.cu)  
  Custom CUDA kernel to compute maximum values across arrays, illustrating warp-level parallelism and reduction strategies.

- [blelloch_prefixsum.cpp](https://github.com/ndevcode/cuda/blob/main/code/blelloch_prefixsum.cpp)  
  Parallel prefix-sum (scan) implementation using the Blelloch upsweep/downsweep algorithm with multi-block CUDA support.

---

##  Matrix Multiplication
- [naiveMatrixMultiply.cpp](https://github.com/ndevcode/cuda/blob/main/code/naiveMatrixMultiply.cpp)  
  GPU-based naive matrix multiplication for baseline comparison.

- [matmul.cpp](https://github.com/ndevcode/cuda/blob/main/code/matmul.cpp)  
  CUDA shared memory based matrix multiplication 

---

## Fast Fourier Transform (FFT)
- [fft.cpp](https://github.com/ndevcode/cuda/blob/main/code/fft.cpp)  
  Fast Fourier Transform implementation using CUDA, leveraging parallel butterfly operations for signal transforms.

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

---

##  Utilities & Infrastructure
- [cudaTimer.h](https://github.com/ndevcode/cuda/blob/main/code/cudaTimer.h)  
  Simple CUDA timing utility for profiling GPU kernel execution.

- [memory_alloc.h](https://github.com/ndevcode/cuda/blob/main/code/memory_alloc.h)  
  Helper functions for CUDA memory allocation and management.

- [utility.hpp](https://github.com/ndevcode/cuda/blob/main/code/utility.hpp)  
  Common utility functions and abstractions used across multiple CUDA projects.


---

##  Notes
- Projects range from **fundamental CUDA exercises** (vector addition, prefix sum) to **advanced CV algorithms** (Canny, DoH, template matching).
- Each project highlights **parallel programming principles**:  
  - memory coalescing  
  - shared memory tiling  
  - warp divergence avoidance  
  - profiling and benchmarking with custom timers  

---

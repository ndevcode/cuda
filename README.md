# CUDA Projects Portfolio

This repository is a collection of CUDA and C++ programs that I have written to explore GPU
programming, parallel algorithms, and performance optimization. The work ranges from
warp-level reductions and tiled matrix multiplication to quantized matmul, parallel scan,
spatial search, and full image-processing pipelines. Several problems are presented as a
progression from a naive version to a tuned one, so the optimization steps are visible. The
projects are grouped by category below.

**Building.** Each program is a single self-contained file with a `main()` that times the
kernel, and many also check the GPU result against a CPU reference. The shared headers live
next to the sources, so build from inside `code/`:

```sh
nvcc -O3 -arch=sm_89 int8_qm_7.cu -o run && ./run
nvcc -O3 -arch=sm_89 -x cu canny.cpp -o run && ./run
```

The default target is the Ada architecture (`sm_89`); adjust `-arch` for your GPU.

---

## Deep Learning

- [INT8 Quantized Matmul](https://github.com/ndevcode/cuda/blob/main/code/int8_qm_7.cu) -
  An 8-bit signed integer matrix multiplication using `__dp4a` accumulation with zero-point
  correction, computing the per-row sum in the same pass by taking a dot product against
  packed ones to avoid a separate reduction. The kernel is tiled and register-blocked, and the
  B tile is stored transposed in shared memory.

- [Categorical Cross-Entropy Loss](https://github.com/ndevcode/cuda/blob/main/code/entropy_loss_categorical.cu) -
  Mean cross-entropy over an N X C logit matrix and N integer labels, computed in a single
  fused pass with an online softmax that keeps a running maximum and sum and rescales on the
  fly, so the logits are read only once.

- [Softmax Attention](https://github.com/ndevcode/cuda/blob/main/code/softmax_attention.cu) -
  Scaled dot-product attention computed as a tiled QK^T matmul, a per-row online softmax, and a
  second matmul with V. This is a correct, readable baseline; a fused FlashAttention version
  that avoids materializing the N X N score matrix is left as future work.

- [ALiBi Attention](https://github.com/ndevcode/cuda/blob/main/code/alibi_attention_2.cu) -
  Scaled dot-product attention with an ALiBi linear positional bias added to the scores before
  the softmax, so position is encoded without any learned embeddings. Each block owns one query
  row: the QK^T scores are formed with a warp reduction over the head dimension, the bias and
  1/sqrt(d) scaling are applied in shared memory, and the softmax is a streaming reduction in
  which a single warp shuffle carries the running maximum and sum together and rescales on the
  fly, so the score row never needs a separate max pass. The weighted sum with V uses coalesced
  lane-per-column reads reduced across the key warps without atomics, and the tile sizes are
  template parameters.

---

## Parallel Scan and Reduction

- [Generic Prefix Scan](https://github.com/ndevcode/cuda/blob/main/code/prefix_scan_generic.cu) -
  A multi-level recursive device scan templated on the element type and a binary operator with
  identity (addition, multiplication, maximum). Each level performs a warp scan and a block
  scan, then recurses on the block sums and adds them back, so it handles an arbitrary N.

- [Prefix Scan (three-phase)](https://github.com/ndevcode/cuda/blob/main/code/prefix_sum_4.cu) -
  An operator-templated three-phase scan: a per-block scan, a scan of the block totals, and a
  uniform add that broadcasts the offsets.

- [Blelloch Scan](https://github.com/ndevcode/cuda/blob/main/code/blelloch_prefixsum.cpp) -
  A work-efficient exclusive scan using the Blelloch upsweep and downsweep in shared memory.

- [Max Reduction](https://github.com/ndevcode/cuda/blob/main/code/max_kernel.cu) -
  A custom kernel that computes the maximum of an array with a NaN-aware reducer, illustrating
  the block reduction pattern, and validated against `std::max_element`.

---

## Sorting and Search

- [Warp Bitonic and Merge-Path Sort](https://github.com/ndevcode/cuda/blob/main/code/warp_bitonic_merge_sort.cu) -
  A full sort that begins with a warp-level bitonic sort using `__shfl_xor`, merges within a
  block using binary-search (merge-path) partitioning, and finishes with iterative global merge
  passes.

- [Approximate Nearest Neighbour](https://github.com/ndevcode/cuda/blob/main/code/nearest_neighbour_2.cu) -
  Approximate nearest-neighbour search that encodes each point with a Morton (Z-order) code,
  radix-sorts the points along the space-filling curve, and searches only a small window of
  neighbours, reducing the cost from O(N^2) to roughly O(N).

---

## Matrix Multiplication and Data Movement

These are presented as a progression that shows the optimization path:

- [Naive Matmul](https://github.com/ndevcode/cuda/blob/main/code/naiveMatrixMultiply.cpp) -
  GPU matrix multiplication with one thread per output element and direct global reads, kept as
  a baseline for comparison.

- [Shared-Memory Tiled Matmul](https://github.com/ndevcode/cuda/blob/main/code/matmul.cpp) -
  Tiled matrix multiplication that loads square blocks of A and B into shared memory and reuses
  each loaded value across the tile.

- [Register-Blocked GEMM](https://github.com/ndevcode/cuda/blob/main/code/gemm4.cu) -
  A tiled GEMM in which each thread computes a 2 X 2 block of outputs from the same
  shared-memory traffic, raising the work done per memory access.

- [Outer-Product GEMM](https://github.com/ndevcode/cuda/blob/main/code/gemm_outer_product.cu) -
  A GEMM that accumulates outer products in a per-thread register tile instead of computing dot
  products, using vectorized shared-memory loads and a transposed B tile.

- [Matrix Transpose](https://github.com/ndevcode/cuda/blob/main/code/transpose.cu) -
  Naive and shared-memory transposes, with the shared tile padded by one column to avoid bank
  conflicts, illustrating coalescing and bank-conflict avoidance.

---

## Image Processing and Computer Vision

GPU implementations:

- [Sobel Edge Detection](https://github.com/ndevcode/cuda/blob/main/code/sobel.cpp) -
  Sobel edge detection developed as a progression of kernels, from a serial baseline to a naive
  global version, a grid-stride version, and a shared-memory version with a halo, with the
  filters held in constant memory and every stage checked against the CPU baseline.

- [Canny Edge Detection](https://github.com/ndevcode/cuda/blob/main/code/canny.cpp) -
  A complete Canny pipeline on the GPU: separable Gaussian blur, Sobel gradients, non-maximum
  suppression, a segmented maximum reduction, double thresholding, and a hysteresis stage
  parallelized as a breadth-first frontier expansion using atomic compare-and-swap. It is
  validated end to end against a CPU baseline.

- [Determinant of Hessian](https://github.com/ndevcode/cuda/blob/main/code/determinantOfHessian.cpp) -
  Scale-space blob detection using the determinant of the Hessian, with fused
  second-derivative-of-Gaussian convolutions and a warp-per-pixel non-maximum suppression across
  the scale stack.

- [1D Convolution](https://github.com/ndevcode/cuda/blob/main/code/conv1d.cpp) -
  A one-dimensional box convolution implemented both with direct global reads and with a
  shared-memory halo, cross-checked between the two kernels.

CPU reference implementations, used as correctness baselines for the GPU versions:

- [Blob Detection](https://github.com/ndevcode/cuda/blob/main/code/blob_finding.cpp) -
  A CPU implementation of blob detection using the determinant of Hessian, Laplacian of
  Gaussian, and difference of Gaussian, with scale-space non-maximum suppression. It is the
  reference for the GPU determinant-of-Hessian detector.

- [Template Matching](https://github.com/ndevcode/cuda/blob/main/code/template_matching.cpp) -
  A CPU implementation of template matching, computing normalized cross-correlation with
  Welford's algorithm for the mean and variance, and sum of squared differences using an
  integral image.

---

## Signal Processing

- [Fast Fourier Transform](https://github.com/ndevcode/cuda/blob/main/code/fft.cpp) -
  A radix-2 iterative FFT on the GPU using bit-reversal and butterfly stages, validated against
  a CPU O(N^2) discrete Fourier transform.

---

## Numerical Computing

- [Logistic Regression](https://github.com/ndevcode/cuda/blob/main/code/logisticRegression.cu) -
  Logistic regression by Newton's method, accumulating the gradient and Hessian in parallel on
  the GPU and solving the small dense system on a single thread.

---

## CUDA Extensions and Machine Learning

- [CUDA Autograd and MNIST MLP](https://github.com/ndevcode/cuda/tree/main/code/cuda_autograd) -
  A small reverse-mode autograd engine that trains a multilayer perceptron on MNIST, with the
  ReLU forward and backward passes offloaded to a custom CUDA kernel exposed to Python through
  pybind11.

---

## Basics

- [Vector Add](https://github.com/ndevcode/cuda/blob/main/code/vectorAdd2.cpp) -
  Introductory vector-addition kernels, a one-thread-per-element version and a grid-stride
  version, each checked against a serial baseline.

---


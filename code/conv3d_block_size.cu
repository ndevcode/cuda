//nvcc -O3 -arch=sm_89 -x cu conv3d_block_size.cu -o conv3d_block_size
//
// 3D convolution with the same kernel launched at two block sizes, plus a CPU reference.
//
// On an RTX 4060 (48 warp slots per SM) a 32x32 block is 32 warps, so only one block fits on
// an SM and 16 slots stay idle (66% occupancy). A 16x16 block is 8 warps, so six blocks fit
// and all 48 slots are busy. The work is identical, but the 16x16 launch is faster.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_timer.h"
#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"

using namespace std;

const int MAX_KERNEL_SIZE = 256;

__constant__ float kernel_const[MAX_KERNEL_SIZE];

template <typename T>
struct RowView {
    T* rowData_;
    __host__ __device__ RowView(T* data) : rowData_(data) {}
    __host__ __device__ T& operator[](int j) { return rowData_[j]; }
    __host__ __device__ const T& operator[](int j) const { return rowData_[j]; }
};

template <typename T>
struct TwoDim {
    T* data_;
    int rows_, cols_;
    __host__ __device__ TwoDim(T* data, int r, int c) : data_(data), rows_(r), cols_(c) {}
    __host__ __device__ RowView<T> operator[](int i) { return RowView<T>(data_ + (i * cols_)); }
    __host__ __device__ const RowView<T> operator[](int i) const { return RowView<T>(data_ + (i * cols_)); }
};

template <typename T>
struct MatrixView {
    T* data_;
    int rows_, cols_, depth_;
    __host__ __device__ MatrixView(T* data, int r, int c, int d) : data_(data), rows_(r), cols_(c), depth_(d) {}
    __host__ __device__ TwoDim<T> operator[](int i) { return TwoDim<T>(data_ + (i * (cols_ * depth_)), cols_, depth_); }
    __host__ __device__ const TwoDim<T> operator[](int i) const { return TwoDim<T>(data_ + (i * (cols_ * depth_)), cols_, depth_); }
};

struct Shape {
    int input_depth, input_rows, input_cols;
    int kernel_depth, kernel_rows, kernel_cols;

    int outputDepth() const { return input_depth - kernel_depth + 1; }
    int outputRows() const { return input_rows - kernel_rows + 1; }
    int outputCols() const { return input_cols - kernel_cols + 1; }
    int inputSize() const { return input_depth * input_rows * input_cols; }
    int kernelSize() const { return kernel_depth * kernel_rows * kernel_cols; }
    int outputSize() const { return outputDepth() * outputRows() * outputCols(); }
};

__global__ void conv3d_kernel(const float* input_ptr, float* output_ptr, int input_depth, int input_rows, int input_cols,
                              int kernel_depth, int kernel_rows, int kernel_cols) {
    extern __shared__ float shmem_ptr[];

    MatrixView<float> input((float*)input_ptr, input_depth, input_rows, input_cols);
    MatrixView<float> kernel((float*)kernel_const, kernel_depth, kernel_rows, kernel_cols);

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    int image_rows = blockDim.y + kernel_rows - 1;
    int image_cols = blockDim.x + kernel_cols - 1 + 1; // +1 breaks bank conflicts

    TwoDim<float> shmem(shmem_ptr, image_rows, image_cols);

    int gx0 = bx * blockDim.x, gy0 = by * blockDim.y, gz0 = bz;

    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    MatrixView<float> output(output_ptr, output_depth, output_rows, output_cols);

    int y = gy0 + ty;
    int x = gx0 + tx;
    bool valid_output = (gz0 < output_depth && y < output_rows && x < output_cols);
    float sum = 0.0f;

    for (int kz = 0; kz < kernel_depth; kz++) {
        int current_z = gz0 + kz;

        for (int yy = ty; yy < image_rows; yy += blockDim.y) {
            for (int xx = tx; xx < image_cols; xx += blockDim.x) {
                int gz = current_z;
                int gy = gy0 + yy, gx = gx0 + xx;
                shmem[yy][xx] = (gz < input_depth && gy < input_rows && gx < input_cols) ? input[gz][gy][gx] : 0.0f;
            }
        }

        __syncthreads();

        if (valid_output) {
            for (int ky = 0; ky < kernel_rows; ky++) {
                for (int kx = 0; kx < kernel_cols; kx++) {
                    sum += shmem[ty + ky][tx + kx] * kernel[kz][ky][kx];
                }
            }
        }

        __syncthreads();
    }

    if (valid_output) {
        output[gz0][y][x] = sum;
    }
}

// kernel must already be in kernel_const; returns the kernel's runtime in microseconds
float conv3dGpu(const float* input, float* output, const Shape& s, dim3 block) {
    assert(block.x > 0 && block.y > 0 && "block dimensions must be positive");

    int image_rows = block.y + s.kernel_rows - 1;
    int image_cols = block.x + s.kernel_cols - 1 + 1; // +1 breaks bank conflicts
    size_t shmem_size = image_rows * image_cols * sizeof(float);

    dim3 grid((s.outputCols() + block.x - 1) / block.x, (s.outputRows() + block.y - 1) / block.y, s.outputDepth());

    CudaTimer timer;
    timer.start();
    conv3d_kernel<<<grid, block, shmem_size>>>(input, output, s.input_depth, s.input_rows, s.input_cols,
                                               s.kernel_depth, s.kernel_rows, s.kernel_cols);
    timer.stop();
    CUDA_CHECK(cudaPeekAtLastError());
    return timer.elapsedTimeUs();
}

// golden reference: accumulates in double so the GPU's rounding error is what gets measured
void conv3dCpu(const float* input, const float* kernel, float* output, const Shape& s) {
    for (int z = 0; z < s.outputDepth(); z++) {
        for (int y = 0; y < s.outputRows(); y++) {
            for (int x = 0; x < s.outputCols(); x++) {
                double acc = 0.0;
                for (int kz = 0; kz < s.kernel_depth; kz++) {
                    for (int ky = 0; ky < s.kernel_rows; ky++) {
                        for (int kx = 0; kx < s.kernel_cols; kx++) {
                            int in = ((z + kz) * s.input_rows + (y + ky)) * s.input_cols + (x + kx);
                            int k = (kz * s.kernel_rows + ky) * s.kernel_cols + kx;
                            acc += (double)input[in] * kernel[k];
                        }
                    }
                }
                output[(z * s.outputRows() + y) * s.outputCols() + x] = (float)acc;
            }
        }
    }
}

void fillNormal(float* data, int size, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<float> dist(0.0f, 1.0f);
    generate(data, data + size, [&]() { return dist(gen); });
}

float maxAbsError(const float* a, const float* b, int size) {
    float err = 0.0f;
    for (int i = 0; i < size; i++) {
        err = max(err, fabsf(a[i] - b[i]));
    }
    return err;
}

float median(vector<float> values) {
    assert(!values.empty() && "median needs at least one value");
    sort(begin(values), end(values));
    return values[values.size() / 2];
}

// Step 1 : both block sizes must match the CPU reference, including shapes that are not a
// multiple of the block size. Step 2 : time both, interleaved so that clock drift affects them equally.
void run(const Shape& s, int reps, bool timeCpu) {
    assert(s.kernelSize() <= MAX_KERNEL_SIZE && "kernel does not fit in kernel_const");

    MemoryManager mmgr;
    auto input = mmgr.allocate_host_and_device<float>(s.inputSize());
    auto output = mmgr.allocate_host_and_device<float>(s.outputSize());
    vector<float> kernel(s.kernelSize());
    vector<float> reference(s.outputSize());

    fillNormal(input.hostPtr, s.inputSize(), 1);
    fillNormal(kernel.data(), s.kernelSize(), 2);
    input.copy_to_device(s.inputSize());
    CUDA_CHECK(cudaMemcpyToSymbol(kernel_const, kernel.data(), s.kernelSize() * sizeof(float)));

    auto cpuStart = chrono::steady_clock::now();
    conv3dCpu(input.hostPtr, kernel.data(), reference.data(), s);
    float cpuMs = chrono::duration<float, milli>(chrono::steady_clock::now() - cpuStart).count();

    const dim3 blocks[] = {dim3(16, 16), dim3(32, 32)};
    cout << "input " << s.input_depth << "x" << s.input_rows << "x" << s.input_cols << endl;

    for (const auto& block : blocks) {
        conv3dGpu(input.devicePtr, output.devicePtr, s, block);
        output.copy_to_host(s.outputSize());
        float err = maxAbsError(output.hostPtr, reference.data(), s.outputSize());
        assert(err < 1e-3f && "GPU output does not match the CPU reference");
        cout << "  block " << block.x << "x" << block.y << " max abs error vs CPU = " << err << endl;
    }

    if (reps > 0) {
        vector<float> times16, times32;
        for (int i = 0; i < reps; i++) {
            times16.push_back(conv3dGpu(input.devicePtr, output.devicePtr, s, blocks[0]));
            times32.push_back(conv3dGpu(input.devicePtr, output.devicePtr, s, blocks[1]));
        }
        cout << "  block 16x16 median = " << median(times16) / 1000.0f << " ms" << endl;
        cout << "  block 32x32 median = " << median(times32) / 1000.0f << " ms" << endl;
    }
    if (timeCpu) {
        cout << "  CPU (single thread) = " << cpuMs << " ms" << endl;
    }
}

int main() {
    run({19, 53, 61, 5, 5, 5}, 0, false);
    run({32, 64, 64, 5, 5, 5}, 0, false);
    run({256, 256, 256, 5, 5, 5}, 30, true);
}

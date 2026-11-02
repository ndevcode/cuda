//nvcc -O3 -arch=sm_89 -x cu conv3d_block_size.cu -o conv3d_block_size
//compare 16x16 and 32x32 blocks for 3d convolution
#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>

#include "memory_alloc.h"
#include "ndev_cuda_utility.hpp"
#include "cuda_timer.h"

using namespace std;

const int MAX_KERNEL_SIZE = 1 << 8;

__constant__ float kernel_dev[MAX_KERNEL_SIZE];

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

template <typename T>
struct MatrixView3D {
        T* data_;
        int rows_,cols_,depth_;
        __host__ __device__ MatrixView3D(T* data, int r, int c, int d) : data_(data), rows_(r), cols_(c), depth_(d) {}
        __host__ __device__ MatrixView<T> operator[](int i) { return MatrixView<T>(data_ + (i * (cols_ * depth_)), cols_, depth_); }
        __host__ __device__ const MatrixView<T> operator[](int i) const { return MatrixView<T>(data_ + (i * (cols_ * depth_)), cols_, depth_); }
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

__global__ void conv3dCUDA(const float *input_ptr, float *output_ptr, int input_depth, int input_rows, int input_cols,
                          int kernel_depth, int kernel_rows, int kernel_cols) {
    extern __shared__ float shmem_ptr[];

    MatrixView3D<float> input((float*)input_ptr, input_depth, input_rows, input_cols);
    MatrixView3D<float> kernel((float*)kernel_dev, kernel_depth, kernel_rows, kernel_cols);

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    int image_rows = blockDim.y + kernel_rows - 1;
    int image_cols = blockDim.x + kernel_cols - 1 + 1; //one extra column to avoid bank conflicts

    MatrixView<float> shmem(shmem_ptr, image_rows, image_cols);

    int gx0 = bx * blockDim.x, gy0 = by * blockDim.y, gz0 = bz;

    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    MatrixView3D<float> output(output_ptr, output_depth, output_rows, output_cols);

    int y = gy0 + ty;
    int x = gx0 + tx;
    bool valid_output = gz0 < output_depth && y < output_rows && x < output_cols;
    float sum = 0.0f;

    for (int kz = 0; kz < kernel_depth; kz++) {
        int current_z = gz0 + kz;

        for (int yy = ty; yy < image_rows; yy += blockDim.y) {
            for (int xx = tx; xx < image_cols; xx += blockDim.x) {
                int gz = current_z;
                int gy = gy0 + yy,gx = gx0 + xx;
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

float conv3dGPU(const float *input, float *output, const Shape& s, dim3 block) {
    assert(block.x > 0 && block.y > 0);

    int image_rows = block.y + s.kernel_rows - 1;
    int image_cols = block.x + s.kernel_cols - 1 + 1;
    size_t shmem_size = image_rows * image_cols * sizeof(float);

    dim3 grid((s.outputCols() + block.x - 1)/block.x,
              (s.outputRows() + block.y - 1)/block.y,s.outputDepth());

    CudaTimer timer;
    timer.start();
    conv3dCUDA<<<grid,block,shmem_size>>>(input,output,s.input_depth,s.input_rows,s.input_cols,
                                          s.kernel_depth,s.kernel_rows,s.kernel_cols);
    timer.stop();
    CUDA_CHECK(cudaPeekAtLastError()); //launch errors
    return timer.elapsedTimeUs();
}

//CPU reference. accumulate in double to check the fp32 error
void conv3dCPU(const float *input, const float *kernel, float *output, const Shape& s) {
    for (int z = 0; z < s.outputDepth(); z++) {
        for (int y = 0; y < s.outputRows(); y++) {
            for (int x = 0; x < s.outputCols(); x++) {
                double acc = 0.0;
                for (int kz = 0; kz < s.kernel_depth; kz++) {
                    for (int ky = 0; ky < s.kernel_rows; ky++) {
                        for (int kx = 0; kx < s.kernel_cols; kx++) {
                            int in = ((z + kz) * s.input_rows + (y + ky)) * s.input_cols + (x + kx);
                            int k = (kz * s.kernel_rows + ky) * s.kernel_cols + kx;
                            acc += static_cast<double>(input[in]) * kernel[k];
                        }
                    }
                }
                output[(z * s.outputRows() + y) * s.outputCols() + x] = static_cast<float>(acc);
            }
        }
    }
}

void fillNormal(float *data, int n, int seed) {
    mt19937 gen(seed);
    normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}

float maxError(const float *a, const float *b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
        err = max(err, fabsf(a[i] - b[i]));
    }
    return err;
}

float getMedian(vector<float> values) {
    assert(!values.empty());
    sort(begin(values), end(values));
    return values[values.size() / 2];
}

void testConv3d(const Shape& s, int iterations, bool timeCPU) {
    assert(s.kernelSize() <= MAX_KERNEL_SIZE);

    MemoryManager mmgr;
    auto input = mmgr.allocate_host_and_device<float>(s.inputSize());
    auto output = mmgr.allocate_host_and_device<float>(s.outputSize());
    vector<float> kernel(s.kernelSize());
    vector<float> reference(s.outputSize());

    fillNormal(input.hostPtr, s.inputSize(), 1);
    fillNormal(kernel.data(), s.kernelSize(), 2);
    input.copy_to_device(s.inputSize());
    CUDA_CHECK(cudaMemcpyToSymbol(kernel_dev,kernel.data(),s.kernelSize() * sizeof(float)));

    auto start = chrono::high_resolution_clock::now();
    conv3dCPU(input.hostPtr,kernel.data(),reference.data(),s);
    auto stop = chrono::high_resolution_clock::now();
    auto cpuTime = chrono::duration_cast<chrono::microseconds>(stop-start);

    const dim3 blocks[] = {dim3(16, 16), dim3(32, 32)};
    cout << "input " << s.input_depth << "x" << s.input_rows << "x" << s.input_cols << endl;

    for (const auto& block : blocks) {
        conv3dGPU(input.devicePtr,output.devicePtr,s,block);
        output.copy_to_host(s.outputSize());
        float err = maxError(output.hostPtr,reference.data(),s.outputSize());
        assert(err < 1e-3f);
        cout << "Block " << block.x << "x" << block.y << " max error vs CPU = " << err << endl;
    }

    if (iterations > 0) {
        vector<float> time16,time32;
        for (int i = 0; i < iterations; i++) {
            //interleave so the GPU clocks affect both approximately the same
            time16.push_back(conv3dGPU(input.devicePtr,output.devicePtr,s,blocks[0]));
            time32.push_back(conv3dGPU(input.devicePtr,output.devicePtr,s,blocks[1]));
        }
        cout << "Block 16x16 median time = " << getMedian(time16)/1000.0f << " ms" << endl;
        cout << "Block 32x32 median time = " << getMedian(time32)/1000.0f << " ms" << endl;
    }
    if (timeCPU) {
        cout << "CPU single thread time = " << cpuTime.count()/1000.0f << " ms" << endl;
    }
    cout << endl;
}

int main() {
    //sizes which are not a multiple of either block size
    testConv3d({19,53,61,5,5,5},0,false);
    testConv3d({32,64,64,5,5,5},0,false);
    //testConv3d({8,8,8,3,3,3},0,false);

    testConv3d({256,256,256,5,5,5},30,true);
}

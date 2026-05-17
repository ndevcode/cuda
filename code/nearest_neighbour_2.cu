#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cooperative_groups.h>

#include "ndev_cuda_utility.hpp"
#include "memory_alloc.h"
#include "cuda_timer.h"

using namespace std;

struct Add {
    template <typename T>
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

struct Max {
    template <typename T>
    __device__ T operator()(T a, T b) const {
        return max(a,b);
    }
};

//for array 1,2,3,4
//computes 1,3,6,10
template <typename T, typename Op>
__device__ T warp_scan_inclusive(T val, Op op) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    for (int offset = 1; offset < warpSize ; offset <<= 1) {
        T n = __shfl_up_sync(mask,val,offset);
        if (lane >= offset) {
            val = op(val,n);
        }
    }
    return val;
}

template <typename T, typename Op>
__device__ T warp_scan_exclusive(T val, Op op, T identity) {
    T inclusive = warp_scan_inclusive(val,op);
    int lane = threadIdx.x & (warpSize - 1);
    return lane == 0 ? identity :  __shfl_up_sync(0xffffffff, inclusive, 1);
}

template <typename T, typename Op, int BLOCKSIZE>
__device__ T block_scan_inclusive(T val, Op op, T* smem) {
    int num_warps = (BLOCKSIZE + warpSize - 1)/warpSize;

    int tid = threadIdx.x;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    T scan = warp_scan_inclusive(val,op);

    if (lane == warpSize - 1) {
        smem[warp_id] = scan;
    }
    __syncthreads();

    if (warp_id == 0) {
        T warp_sum = (lane < num_warps) ? smem[lane] : T{};
        warp_sum = warp_scan_inclusive(warp_sum, op);
        if (lane < num_warps) {
            smem[lane] = warp_sum;
        }
    }
    __syncthreads();

    if (warp_id > 0) {
        scan = op(scan, smem[warp_id - 1]);
    }
    return scan;
}

template <typename T, typename Op, int BLOCKSIZE>
__device__ T block_scan_exclusive(T val, Op op, T identity, T* smem) {
    T inclusive = block_scan_inclusive<T,Op,BLOCKSIZE>( val, op, smem);

    int tid = threadIdx.x;

    __syncthreads();

    smem[tid] = inclusive;

    __syncthreads();

    if (tid == 0)
        return identity;

    return smem[tid - 1];
}

template <typename T, typename Op, int BLOCKSIZE>
__global__ void scan_kernel(const T* input, T* output, T* block_sums, int N, Op op, T identity) {
    __shared__ T smem[BLOCKSIZE];
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    T val = (gid < N) ? input[gid] : identity;

    T scan = block_scan_inclusive<T, Op, BLOCKSIZE> (val, op, smem);

    if (gid < N) {
        output[gid] = scan;
    }

    int remaining = N - blockIdx.x * blockDim.x;

    int valid = min(BLOCKSIZE, remaining);

    if (block_sums != nullptr && threadIdx.x == valid - 1) {
        block_sums[blockIdx.x] = scan;
    }
}

template <typename T, typename Op, int BLOCKSIZE>
__global__ void uniform_add(const T* block_prefix, T* output, int N, Op op) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= N || blockIdx.x == 0) {
        return ;
    }

    output[gid] = op(output[gid],block_prefix[blockIdx.x-1]);
}

template <typename T, typename Op, int BLOCKSIZE>
void device_scan(const T* input, T* output, int N, Op op, T identity) {
        int num_blocks = (N + BLOCKSIZE - 1)/BLOCKSIZE;
        if (num_blocks <= 1) {
                scan_kernel<T,Op,BLOCKSIZE><<<1,BLOCKSIZE>>>(input, output, nullptr, N, op, identity);
                return;
        }

        T* block_sums; T* scanned_block_sums;

        cudaMalloc(&block_sums, num_blocks * sizeof(T));
        cudaMalloc(&scanned_block_sums, num_blocks * sizeof(T));

        scan_kernel<T,Op,BLOCKSIZE><<<num_blocks,BLOCKSIZE>>>(input, output, block_sums, N, op, identity);

        //recursive
        device_scan<T,Op,BLOCKSIZE>(block_sums, scanned_block_sums, num_blocks, op, identity);

        uniform_add<T,Op,BLOCKSIZE><<<num_blocks, BLOCKSIZE>>>(scanned_block_sums, output, N, op);

        cudaFree(block_sums);
        cudaFree(scanned_block_sums);
}

__global__ void predicate_kernel(const float* A, int* predicate, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int i = gid; i < N; i += stride) {
        predicate[i] = (A[i] > 0);
    }
}

__global__ void inclusive_to_exclusive(const int* inclusive, int* exclusive, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int i = gid; i < N; i += stride) {
        exclusive[i] = (i == 0) ? 0 : inclusive[i-1];
    }
}

__global__ void scatter_kernel(const float* A, const int* predicate, const int* positions, float* out, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int i = gid; i < N; i += stride) {
        if (predicate[i]) {
            int dst = positions[i];
            out[dst] = A[i];
        }
    }
}

template <int BLOCKSIZE, int ITEMS_PER_THREAD>
__global__ void compact_kernel( const float* A, float* out, int* block_counts, const int* block_offsets, int N) {
    constexpr int ELEMENTS_PER_BLOCK = BLOCKSIZE * ITEMS_PER_THREAD;

    __shared__ int scan_smem[BLOCKSIZE];
    __shared__ float compacted[ELEMENTS_PER_BLOCK];
    __shared__ int block_total_sh;

    int tid = threadIdx.x;

    int block_start = blockIdx.x * ELEMENTS_PER_BLOCK;

    int base = block_start + tid * ITEMS_PER_THREAD;

    // Load values + predicates

    float vals[ITEMS_PER_THREAD];
    int flags[ITEMS_PER_THREAD];

    int local_count = 0;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = base + i;
        float x = (idx < N) ? A[idx] : 0.0f;
        vals[i] = x;
        int pred = (x > 0.0f);
        flags[i] = pred;
        local_count += pred;
    }

    // Block exclusive scan

    int local_offset = block_scan_exclusive< int, Add, BLOCKSIZE>( local_count, Add{}, 0, scan_smem);

    // Compact into shared memory

    int dst = local_offset;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (flags[i]) {
            compacted[dst++] = vals[i];
        }
    }

    // Write block count

    int block_total = local_offset + local_count;

    if (tid == BLOCKSIZE - 1) {
        block_total_sh = block_total;
        if (block_counts != nullptr) {
            block_counts[blockIdx.x] = block_total;
        }
    }

    __syncthreads();

    // Scatter to global output

    if ((block_offsets != nullptr) || (gridDim.x == 1)) {
        int block_offset = (block_offsets == nullptr) ? 0 : block_offsets[blockIdx.x];
        for (int i = tid; i < block_total_sh; i += BLOCKSIZE) {
            out[block_offset + i] = compacted[i];
        }
    }
}

__device__ uint32_t expandBits(uint32_t v) {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
}

__device__ uint32_t morton3D(
    float x, float y, float z,
    float minx, float miny, float minz,
    float maxx, float maxy, float maxz) {
    float nx = (x - minx) / (maxx - minx + 1e-20f);
    float ny = (y - miny) / (maxy - miny + 1e-20f);
    float nz = (z - minz) / (maxz - minz + 1e-20f);

    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f);

    uint32_t xx = expandBits((uint32_t)(nx * 1023.0f));
    uint32_t yy = expandBits((uint32_t)(ny * 1023.0f));
    uint32_t zz = expandBits((uint32_t)(nz * 1023.0f));

    return (xx << 2) | (yy << 1) | zz;
}

__global__ void form_morton_points(const float* points, int* morton_points, int N,
        float minx, float miny, float minz,
        float maxx, float maxy, float maxz) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) {
        float xi = points[3*i], yi = points[3*i + 1], zi = points[3*i + 2];
        morton_points[i] = morton3D(xi,yi,zi,minx,miny,minz,maxx,maxy,maxz);
    }
}

__global__ void form_indices(int* indices, int N) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += stride) {
        indices[i] = i;
    }
}

__global__ void build_predicate(const int* morton_points, int* predicate, int N, int bit) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += stride) {
        uint32_t b = (morton_points[i] >> bit) & 1u;
        predicate[i] = (b == 0);
    }
}

__global__ void radix_scatter(const int* keys, const int* values, int* out_keys, int* out_values, const int* predicate, const int* position,
        int num_zeros, int N, int bit) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int i = gid; i < N; i += stride) {
        int key = keys[i];
        int dst;
        if (predicate[i]) {
            dst = position[i];
        } else {
            dst = num_zeros + (i - position[i]);
        }
        out_keys[dst] = key;
        out_values[dst] = values[i];
    }
}

__global__ void gather_points(const float* points, const int* sorted_indices, float* sorted_points, int N) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) {
        int src = sorted_indices[i];
        sorted_points[3*i]     = points[3*src];
        sorted_points[3*i + 1] = points[3*src + 1];
        sorted_points[3*i + 2] = points[3*src + 2];
    }
}

template <int WINDOW>
__global__ void nn_kernel(const float* sorted_points, const int* sorted_indices, int* output_indices, int N) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += stride) {
        float max_dist = FLT_MAX;
        int index = -1;
        float xi = sorted_points[3*i], yi = sorted_points[3*i + 1], zi = sorted_points[3*i + 2];
        int begin = max(0, i - WINDOW);
        int end   = min(N, i + WINDOW + 1);
        for (int j = begin; j < end; j++) {
            if (i == j) continue;
            float xj = sorted_points[3*j], yj = sorted_points[3*j + 1], zj = sorted_points[3*j + 2];
            float sqdist = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)  + (zi - zj) * (zi - zj);
            int cand = sorted_indices[j];
            if (sqdist < max_dist || (sqdist == max_dist && cand < index)) {
                max_dist = sqdist;
                index = cand;
            }
        }
        output_indices[sorted_indices[i]] = index;
    }
}

extern "C" void solve(const float* points, int* indices, int N) {
    std::vector<float> host(3*N);
    cudaMemcpy(host.data(), points, 3*N*sizeof(float),
               cudaMemcpyDeviceToHost);

    float minx=FLT_MAX, miny=FLT_MAX, minz=FLT_MAX;
    float maxx=-FLT_MAX, maxy=-FLT_MAX, maxz=-FLT_MAX;

    for (int i = 0; i < N; i++) {
        float x = host[3*i];
        float y = host[3*i+1];
        float z = host[3*i+2];

        minx = std::min(minx, x);
        miny = std::min(miny, y);
        minz = std::min(minz, z);

        maxx = std::max(maxx, x);
        maxy = std::max(maxy, y);
        maxz = std::max(maxz, z);
    }

    int* morton_points;
    cudaMalloc(&morton_points, N * sizeof(int));

    int* temp_keys;
    cudaMalloc(&temp_keys, N * sizeof(int));

    int* temp_values;
    cudaMalloc(&temp_values, N * sizeof(int));

    int* predicate;
    cudaMalloc(&predicate, N * sizeof(int));

    int* positions;
    cudaMalloc(&positions, N * sizeof(int));

    int* exclusive_positions;
    cudaMalloc(&exclusive_positions, N * sizeof(int));

    dim3 block(256);
    dim3 grid((N + block.x - 1)/block.x);

    form_morton_points<<<grid,block>>>(points, morton_points, N,
        minx, miny, minz, maxx, maxy, maxz);

    form_indices<<<grid,block>>>(indices, N);

    int* curr_keys   = morton_points;
    int* curr_values = indices;
    int* next_keys   = temp_keys;
    int* next_values = temp_values;

    for (int bit = 0; bit < 32; bit++) {
        build_predicate<<<grid,block>>>(curr_keys, predicate, N, bit);

        device_scan<int, Add, 256>(predicate, positions, N, Add{}, 0);

        inclusive_to_exclusive<<<grid,block>>>(positions, exclusive_positions, N);

        int last_pred, last_excl;
        cudaMemcpy(&last_pred, predicate + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_excl, exclusive_positions + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
        int num_zeros = last_pred + last_excl;

        radix_scatter<<<grid,block>>>(curr_keys, curr_values, next_keys, next_values, predicate, exclusive_positions, num_zeros, N, bit);

        int* tmp;
        tmp = curr_keys;   curr_keys   = next_keys;   next_keys   = tmp;
        tmp = curr_values; curr_values = next_values; next_values = tmp;
    }

    // curr_values holds sorted_indices (permutation in morton order)
    // next_values is unused after the loop -- repurpose as nn output buffer

    float* sorted_points;
    cudaMalloc(&sorted_points, 3 * N * sizeof(float));

    gather_points<<<grid,block>>>(points, curr_values, sorted_points, N);

    constexpr int WINDOW = 32;
    nn_kernel<WINDOW><<<grid,block>>>(sorted_points, curr_values, next_values, N);

    cudaMemcpy(indices, next_values, N * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    cudaFree(sorted_points);
    cudaFree(morton_points);
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(predicate);
    cudaFree(positions);
    cudaFree(exclusive_positions);
}

int main() {
    MemoryManager mmgr;
    //const int n = (1 << 4) + 7;
    const int n = 1e4;
    auto points = mmgr.allocate_host_and_device<float>(3*n);
    auto indices = mmgr.allocate_host_and_device<int>(n);
    for (int i = 0; i < n; i++) {
        indices.hostPtr[i] = -1;
    }
    fillArrayRandom(points.hostPtr,3*n,true, 1001);

    points.copy_to_device(3*n);
    indices.copy_to_device(n);

    CudaTimer timer;
    timer.start();
    solve(points.devicePtr,indices.devicePtr,n);
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    cout << "Runtime for GPU reverse array is " << runtime << "us" << endl;
}

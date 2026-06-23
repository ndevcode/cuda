#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cooperative_groups.h>

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

struct AddOp {
    __device__ float operator()(float a, float b) const {
        return a + b;
    }
};

struct MaxOp {
    __device__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

__device__ __forceinline__
void warp_reduce_online(float &sum, float &mx) {
    unsigned mask = 0xFFFFFFFFU;

    for (int s = warpSize / 2; s > 0; s >>= 1) {
        float sum_other = __shfl_down_sync(mask, sum, s);
        float max_other = __shfl_down_sync(mask, mx, s);

        float new_max = fmaxf(mx, max_other);

        sum = sum * expf(mx - new_max)
            + sum_other * expf(max_other - new_max);

        mx = new_max;
    }
}

template <typename T, typename Op>
__device__ T warp_reduce(T x, Op op) {
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset >= 1; offset >>= 1) {
        T y = __shfl_down_sync(mask, x, offset);
        x = op(x,y);
    }
    return x;
}

template <typename T, typename Op>
__device__ T block_reduce(T val, Op op, T identity) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    val = warp_reduce(val, op);

    __shared__ T shmem[32];
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    int num_warps = (blockDim.x * blockDim.y + warpSize - 1) / warpSize;

    if (lane == 0) {
        shmem[warp_id] = val;
    }
    __syncthreads();
    T block_val = identity;
    if (warp_id == 0) {
        block_val = (lane < num_warps) ? shmem[lane] : identity;
        block_val = warp_reduce(block_val, op);
        if (lane == 0) {
            shmem[0] = block_val;
        }
    }
    __syncthreads();
    return shmem[0];
}

template <int Mb, int Nb, int db>
__device__ void attention_v2_query_block(const float* Q_ptr, const float* K_ptr, const float* V_ptr, float* output_ptr, int M, int N, int d, float alpha) {
    //__shared__ float Kb[Nb][db];
    //__shared__ float Vb[Nb][db];
                //Vb[row][col] = V[key_row_start + row][key_col_start + col];
                //Vb[row][col] = 0.0f;
            /* 
            float scoreb[Nb];

            if ((key_row_start + row < key_row_end) && (key_col_start + col < key_col_end)) {
                Kb[row][col] = K[key_row_start + row][key_col_start + col];
            } else {
                Kb[row][col] = 0.0f;
            } 
            __syncthreads();
            for (int kj = 0; kj < Nb; kj++) {
                float score = 0.0f;
                for (int dk = 0; dk < db; dk++) {
                    score += qv * Kb[kj][dk];
                }
                scoreb[kj] = score;
            }
            */
}

template <int Nb, int db>
__device__ void attention_v2_query_row(const float* Q_ptr, const float* K_ptr, const float* V_ptr, float* output_ptr, float* scoreb, int qi, int M, int N, int d, float dinvsqrt, float alpha) {
    RowView<const float>    Q(Q_ptr); //of length d
    RowView<float>          attention_row(output_ptr); //of length d
    MatrixView<const float> K(K_ptr,N,d);
    MatrixView<const float> V(V_ptr,N,d);

    int row = threadIdx.y, col = threadIdx.x;
    // scoreb is the kernel's dynamic shared buffer (size N), passed in

    //Step 1 : compute score
    for (int ni = 0; ni < N; ni += Nb) {
        int key_row_start   = ni ;
        int key_row_end     = min(key_row_start + Nb, N);
        float score = 0.0f;
        for (int di = 0; di < d; di += db) {
            int key_col_start   = di ;
            int key_col_end     = min(key_col_start + db, d);

            float key_value = (key_row_start + row < key_row_end) && (key_col_start + col < key_col_end) ? K[key_row_start + row][key_col_start + col] : 0.0f;
            float query_value = (key_col_start + col < key_col_end) ? Q[key_col_start + col] : 0.0f;
            score += query_value * key_value;
        }
        score = warp_reduce(score, AddOp {});
        if (col == 0 && ni + row < key_row_end) {
            scoreb[ni+row] = score;
        } 
    }
    __syncthreads();

    //step 1.5 : normalize the score with dinvsqrt and compute alibi factor
    int tid = threadIdx.y * blockDim.x + threadIdx.x; //linearize thread index
    int stride = blockDim.x * blockDim.y;

    for (int i = tid; i < N; i += stride) {
        scoreb[i] *= dinvsqrt;
        float dist = float(qi) - float(i);
        scoreb[i] += (alpha * dist);
    }

    __syncthreads();

    //Step 2 : find max, exp, sum and softmax
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float val = scoreb[i];

        float prev_max = local_max;
        local_max = fmaxf(prev_max, val);

        local_sum = local_sum * expf(prev_max - local_max) + expf(val - local_max);
    }

    //now use warp primitives to compute max and sum
    // warp reduction
    warp_reduce_online(local_sum, local_max);

    //now we have 256 / 32 values that we need to reduce
    //reduce using shared memory

    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    int num_warps = (blockDim.x * blockDim.y) / warpSize;

    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];

    if (lane == 0) {
        smem_max[warp_id] = local_max;
        smem_sum[warp_id] = local_sum;
    }
    __syncthreads();

    //block reduction
    float b_max = -FLT_MAX;
    float b_sum = 0.0f;
    if (warp_id == 0) {
        b_max = (lane < num_warps) ? smem_max[lane] : -FLT_MAX;
        b_sum = (lane < num_warps) ? smem_sum[lane] : 0.0f;

        warp_reduce_online(b_sum, b_max);
        if (lane == 0) {
            smem_max[0] = b_max;
            smem_sum[0] = b_sum;
        }
    }
    __syncthreads();
    b_max = smem_max[0];
    b_sum = smem_sum[0];
    for (int i = tid ; i < N; i += stride) {
        float val = scoreb[i];
        val = expf(val - b_max)/b_sum;
        scoreb[i] = val;
    }
    __syncthreads();

    //Step 3 : attention[c] = sum_key softmax[key] * V[key][c]
    //  lane(col) -> output column c = di+col  (coalesced V reads, stride 1 across lanes)
    //  warp(row) -> a strided subset of keys; partial summed in a register over the ni loop,
    //  then reduced across the Nb warps via shared memory (no atomics).
    __shared__ float red[Nb][db];
    for (int di = 0; di < d; di += db) {
        int outcol = di + col;
        float acc = 0.0f;
        for (int ni = 0; ni < N; ni += Nb) {
            int key_index = ni + row;
            float p = (key_index < N) ? scoreb[key_index] : 0.0f;
            float v = (key_index < N && outcol < d) ? V[key_index][outcol] : 0.0f;
            acc += p * v;
        }
        red[row][col] = acc;
        __syncthreads();
        if (row == 0) {
            float s = 0.0f;
            for (int r = 0; r < Nb; r++) s += red[r][col];
            if (outcol < d) attention_row[outcol] = s;
        }
        __syncthreads();   // also guards scoreb reuse by the next serial query row
    }
}

template <int Mb, int Nb, int db>
__global__ void attention_v2_gpu(const float* Q_ptr, const float* K_ptr, const float* V_ptr, float* output_ptr, int M, int N, int d, float dinvsqrt, float alpha) {
    extern __shared__ float scoreb[]; // size N, shared across the Mb serial rows
    MatrixView<const float> Q(Q_ptr,M,d);
    MatrixView<float>       output(output_ptr,M,d);

    int query_row_start = blockIdx.x * Mb;
    int query_row_end   = min(query_row_start + Mb, M);

    for (int qi = query_row_start; qi < query_row_end ; qi++) {
        //each block is responsible for generating Mb output rows, one row at a time
        attention_v2_query_row<Nb,db>(Q[qi].rowData_, K_ptr, V_ptr, output[qi].rowData_, scoreb, qi, M, N, d, dinvsqrt, alpha);
    }

}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    // blockDim.x must stay warpSize (lane == threadIdx.x; db is the lane-reduction width)
    // blockDim.y must equal Nb (row=threadIdx.y covers the Nb keys of a tile)
    // Mb=1 -> one query row per block -> grid has M blocks (max occupancy, no serial row loop)
    const int Mb = 1, Nb = 16, db = 32;
	dim3 block(32,16); // 512 threads: 96 regs/thread (debug build) * 512 = 49152 < 65536
    dim3 grid((M + Mb - 1)/Mb);

    const float dinvsqrt = 1.0 / sqrtf(d * 1.0f);

    attention_v2_gpu<Mb,Nb,db><<<grid,block,N*sizeof(float)>>>(Q, K, V, output, M, N, d, dinvsqrt, alpha);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("sync: %s\n", cudaGetErrorString(err));
}

void copy2DToDevice(std::vector<std::vector<float>> src, float *hostPtr, int rows, int cols) {
	// sanity checks
	assert((int)src.size() == rows);
	for (int i = 0; i < rows; i++) {
		assert((int)src[i].size() == cols);
	}

	// flatten (row-major)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			hostPtr[i * cols + j] = src[i][j];
		}
	}
}

int main() {
    MemoryManager mmgr;
    // realistic size for profiling (toggle SMALL for the golden fixture)
    const bool SMALL = false;
    const int M = SMALL ? 2    : 2048;
    const int N = SMALL ? 3    : 2048;
    const int d = SMALL ? 4    : 1024;
    const float alpha = 0.5;

    auto Q_ptr = mmgr.allocate_host_and_device<float>(M * d);
    auto K_ptr = mmgr.allocate_host_and_device<float>(N * d);
    auto V_ptr = mmgr.allocate_host_and_device<float>(N * d);
    auto output_ptr = mmgr.allocate_host_and_device<float>(M * d);

    if (SMALL) {
        copy2DToDevice({{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}},Q_ptr.hostPtr,2,4);
        copy2DToDevice({{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}},K_ptr.hostPtr,3,4);
        copy2DToDevice({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}},V_ptr.hostPtr,3,4);
    } else {
        fillArrayRandom(Q_ptr.hostPtr,M*d);
        fillArrayRandom(K_ptr.hostPtr,N*d);
        fillArrayRandom(V_ptr.hostPtr,N*d);
    }

    Q_ptr.copy_to_device(M*d);
    K_ptr.copy_to_device(N*d);
    V_ptr.copy_to_device(N*d);

    CudaTimer timer;
    timer.start();
    solve(Q_ptr.devicePtr,K_ptr.devicePtr,V_ptr.devicePtr,output_ptr.devicePtr,M,N,d,alpha);
    timer.stop();
    float runtime = timer.elapsedTimeUs();
    output_ptr.copy_to_host(M * d);
    cout << "Runtime for GPU attention is " << runtime << "us" << endl;
    int rowsToPrint = SMALL ? M : min(M, 2);
    /* 
    for (int i = 0; i < rowsToPrint; i++) {
        for (int j = 0; j < d; j++) {
            cout << output_ptr.hostPtr[i * d + j] << (j + 1 == d ? '\n' : ' ');
        }
    }
    */
}

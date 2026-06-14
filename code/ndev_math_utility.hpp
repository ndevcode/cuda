#pragma once
#include <cuda_fp16.h>
#include <type_traits>

// ------------------------------
// TYPE NORMALIZATION
// ------------------------------

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(double x) { return static_cast<float>(x); }
__device__ __forceinline__ float to_float(int x) { return static_cast<float>(x); }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

// ------------------------------
// BACK TO HALF
// ------------------------------

__device__ __forceinline__ half to_half(float x) {
    return (half)__float2half(x);
}

// ------------------------------
// MULTIPLY (FAST PATH ENABLED)
// ------------------------------

template <typename A, typename B>
__device__ __forceinline__ half hmul(A a, B b) {

    if constexpr (std::is_same_v<A, __half> && std::is_same_v<B, __half>) {
        return (half)__hmul(a, b);
    } else {
        float af = to_float(a);
        float bf = to_float(b);
        return to_half(af * bf);
    }
}

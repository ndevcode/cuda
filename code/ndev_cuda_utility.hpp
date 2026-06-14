#include <iostream>
#include <type_traits>
#include <cmath>
#include <cstdlib>

#include <cuda_fp16.h>

#pragma once

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)


template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
void printArr(T* arr, unsigned int sz) {
    for (int i = 0; i < sz-1; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << arr[sz-1] << std::endl;
}

template <typename T, typename U,
          typename = typename std::enable_if<
              std::is_arithmetic<T>::value && std::is_arithmetic<U>::value &&
              !(std::is_floating_point<T>::value || std::is_floating_point<U>::value)>::type>
bool areEqual(T* X, U* Y, unsigned int sz) {
    for (unsigned int i = 0; i < sz; ++i) {
        if (X[i] != Y[i]) return false;
    }
    return true;
}

template <typename T, typename U,
          typename = typename std::enable_if<
              std::is_floating_point<T>::value || std::is_floating_point<U>::value>::type>
bool areEqual(T* X, U* Y, unsigned int sz,
              typename std::common_type<T, U>::type epsilon = static_cast<typename std::common_type<T, U>::type>(1e-5)) {
    
    using Common = typename std::common_type<T, U>::type;

    for (unsigned int i = 0; i < sz; ++i) {
        if (std::abs(static_cast<Common>(X[i]) - static_cast<Common>(Y[i])) > epsilon)
            return false;
    }
    return true;
}

template <typename T, typename U,
          typename = typename std::enable_if<
            std::is_arithmetic<T>::value && std::is_arithmetic<U>::value &&
            !(std::is_floating_point<T>::value || std::is_floating_point<U>::value)>::type>
bool areEqualWithReport(const T* X, const U* Y, unsigned int sz) {
    for (unsigned int i = 0; i < sz; ++i) {
        if (X[i] != Y[i]) {
            fprintf(stderr,"Mismatch at index %d X = %f , Y = %f \n",i, X[i], Y[i]);
            return false;
        }
    }
    return true;
}

template <typename T, typename U,
          typename = typename std::enable_if<
              std::is_floating_point<T>::value || std::is_floating_point<U>::value>::type>
bool areEqualWithReport(T* X, U* Y, unsigned int sz,
              typename std::common_type<T, U>::type epsilon = static_cast<typename std::common_type<T, U>::type>(1e-5)) {
    
    using Common = typename std::common_type<T, U>::type;

    for (unsigned int i = 0; i < sz; ++i) {
        if (std::abs(static_cast<Common>(X[i]) - static_cast<Common>(Y[i])) > epsilon)
            fprintf(stderr,"Mismatch at index %d X = %f , Y = %f \n",i, X[i], Y[i]);
            return false;
    }
    return true;
}

template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
void fillArrayRandom(T *array, int sz, bool negate = false, int modulus = RAND_MAX, int seed = 42) {
    srand(seed);
    for (int i = 0; i < sz; i++) {
        array[i] = static_cast<T>(rand()%modulus);
        if (negate) {
            bool donegate = rand()%2;
            array[i] *= (donegate ? -1 : 1);
        }
    }
}

void fillArrayRandom(half *array, int sz, bool negate = false, int modulus = RAND_MAX, int seed = 42) {
    srand(seed);
    for (int i = 0; i < sz; i++) {
        float val = static_cast<float>(rand() % modulus);
        if (negate && (rand() % 2)) val *= -1.0f;
        array[i] = __float2half(val);
    }
}

struct KernelCommand {
    virtual void run(cudaStream_t stream) = 0;
    virtual ~KernelCommand();
};

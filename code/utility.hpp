#include <iostream>
#include <type_traits>
#include <cmath>

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

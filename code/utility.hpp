#include <iostream>

template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
void printArr(T* arr, unsigned int sz) {
    for (int i = 0; i < sz-1; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << arr[sz-1] << std::endl;
}

template <typename T, typename U,
          typename = typename std::enable_if<std::is_arithmetic<T>::value && std::is_arithmetic<T>::value>::type>
bool areEqual(T* X, U* Y, unsigned int sz) {
    for (int i = 0; i < sz; i++) {
        if (X[i] != Y[i]) return false;
    }
    return true;
}

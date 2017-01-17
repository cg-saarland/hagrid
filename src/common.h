#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cmath>

#ifdef __NVCC__
#include <iostream>
#endif

namespace hagrid {

/// Rounds the division by an integer so that round_div(i, j) * j > i
HOST DEVICE inline int round_div(int i, int j) {
    return i / j + (i % j ? 1 : 0);
}

/// Computes the minimum between two values
template <typename T> HOST DEVICE T min(T a, T b) { return a < b ? a : b; }
/// Computes the maximum between two values
template <typename T> HOST DEVICE T max(T a, T b) { return a > b ? a : b; }
/// Clamps the first value in the range defined by the last two arguments
template <typename T> HOST DEVICE T clamp(T a, T b, T c) { return min(c, max(b, a)); }

/// Reinterprets a values as unsigned int
template <typename U, typename T>
HOST DEVICE U as(T t) {
    union { T t; U u; } v;
    v.t = t;
    return v.u;
}

/// Returns x with the sign of x * y
HOST DEVICE inline float safe_rcp(float x) {
    return x != 0 ? 1.0f / x : copysign(as<float>(0x7f800000u), x);
}

/// Returns x with the sign of x * y
HOST DEVICE inline float prodsign(float x, float y) {
    return as<float>(as<uint32_t>(x) ^ (as<uint32_t>(y) & 0x80000000));
}

/// Converts a float to an ordered float
HOST DEVICE inline uint32_t float_to_ordered(float f) {
    auto u = as<uint32_t>(f);
    auto mask = -(int)(u >> 31u) | 0x80000000u;
    return u ^ mask;
}

/// Converts back an ordered integer to float
HOST DEVICE inline float ordered_to_float(uint32_t u) {
    auto mask = ((u >> 31u) - 1u) | 0x80000000u;
    return as<float>(u ^ mask);
}

/// Computes the cubic root of an integer
HOST DEVICE inline int icbrt(int x) {
    unsigned y = 0;
    for (int s = 30; s >= 0; s = s - 3) {
        y = 2 * y;
        const unsigned b = (3 * y * (y + 1) + 1) << s;
        if (x >= b) {
            x = x - b;
            y = y + 1;
        }
    }
    return y;
}

#ifdef __NVCC__
#ifndef NDEBUG
#define DEBUG_SYNC() CHECK_CUDA_CALL(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC() do{} while(0)
#endif
#define CHECK_CUDA_CALL(x) check_cuda_call(x, __FILE__, __LINE__)

__host__ static void check_cuda_call(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << file << "(" << line << "): " << cudaGetErrorString(err) << std::endl;
        abort();
    }
}

template <typename T>
__host__ void set_global(T& symbol, const T* ptr) {
    size_t size;
    CHECK_CUDA_CALL(cudaGetSymbolSize(&size, symbol));
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(symbol, ptr, size));
}
#endif // __NVCC__

} // namespace hagrid

#endif

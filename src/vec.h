#ifndef VEC_H
#define VEC_H

#include "common.h"

namespace hagrid {

template <typename T>
struct tvec2 {
    T x, y;
    HOST DEVICE tvec2() {}
    HOST DEVICE tvec2(T xy) : x(xy), y(xy) {}
    HOST DEVICE tvec2(T x, T y) : x(x), y(y) {}
    template <typename U>
    HOST DEVICE explicit tvec2(const tvec2<U>& xy) : x(xy.x), y(xy.y) {}

    HOST DEVICE tvec2& operator += (const tvec2& other) { *this = *this + other; return *this; }
    HOST DEVICE tvec2& operator -= (const tvec2& other) { *this = *this - other; return *this; }
    HOST DEVICE tvec2& operator *= (const tvec2& other) { *this = *this * other; return *this; }
    HOST DEVICE tvec2& operator /= (const tvec2& other) { *this = *this / other; return *this; }

    HOST DEVICE tvec2& operator *= (T t) { *this = *this * t; return *this; }
    HOST DEVICE tvec2& operator /= (T t) { *this = *this / t; return *this; }
};

template <typename T> HOST DEVICE tvec2<T> operator + (const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(a.x + b.x, a.y + b.y); }
template <typename T> HOST DEVICE tvec2<T> operator - (const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(a.x - b.x, a.y - b.y); }
template <typename T> HOST DEVICE tvec2<T> operator * (const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(a.x * b.x, a.y * b.y); }
template <typename T> HOST DEVICE tvec2<T> operator / (const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(a.x / b.x, a.y / b.y); }
template <typename T> HOST DEVICE tvec2<T> operator * (const tvec2<T>& a, T b) { return tvec2<T>(a.x * b, a.y * b); }
template <typename T> HOST DEVICE tvec2<T> operator / (const tvec2<T>& a, T b) { return tvec2<T>(a.x / b, a.y / b); }
template <typename T> HOST DEVICE tvec2<T> operator * (T a, const tvec2<T>& b) { return tvec2<T>(a * b.x, a * b.y); }
template <typename T> HOST DEVICE tvec2<T> operator / (T a, const tvec2<T>& b) { return tvec2<T>(a / b.x, a / b.y); }
template <typename T> HOST DEVICE tvec2<T> min(const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(min(a.x, b.x), min(a.y, b.y)); }
template <typename T> HOST DEVICE tvec2<T> max(const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(max(a.x, b.x), max(a.y, b.y)); }
template <typename T> HOST DEVICE T dot(const tvec2<T>& a, const tvec2<T>& b) { return a.x * b.x + a.y * b.y; }

template <typename T> HOST DEVICE tvec2<T> operator << (const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(a.x << b.x, a.y << b.y); }
template <typename T> HOST DEVICE tvec2<T> operator >> (const tvec2<T>& a, const tvec2<T>& b) { return tvec2<T>(a.x >> b.x, a.y >> b.y); }
template <typename T> HOST DEVICE tvec2<T> operator << (const tvec2<T>& a, T b) { return tvec2<T>(a.x << b, a.y << b); }
template <typename T> HOST DEVICE tvec2<T> operator >> (const tvec2<T>& a, T b) { return tvec2<T>(a.x >> b, a.y >> b); }
template <typename T> HOST DEVICE tvec2<T> operator << (T a, const tvec2<T>& b) { return tvec2<T>(a << b.x, a << b.y); }
template <typename T> HOST DEVICE tvec2<T> operator >> (T a, const tvec2<T>& b) { return tvec2<T>(a >> b.x, a >> b.y); }

template <typename T> HOST DEVICE tvec2<T> clamp(const tvec2<T>& a, T b, T c) { return tvec2<T>(min(max(a.x, b), c), min(max(a.y, b), c)); }

template <typename T>
struct tvec3 {
    union { T x; T r; };
    union { T y; T g; };
    union { T z; T b; };
    HOST DEVICE tvec3() {}
    HOST DEVICE tvec3(T xyz) : x(xyz), y(xyz), z(xyz) {}
    HOST DEVICE tvec3(T x, T y, T z) : x(x), y(y), z(z) {}
    template <typename U>
    HOST DEVICE explicit tvec3(const tvec3<U>& xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}

    HOST DEVICE tvec3& operator += (const tvec3& other) { *this = *this + other; return *this; }
    HOST DEVICE tvec3& operator -= (const tvec3& other) { *this = *this - other; return *this; }
    HOST DEVICE tvec3& operator *= (const tvec3& other) { *this = *this * other; return *this; }
    HOST DEVICE tvec3& operator /= (const tvec3& other) { *this = *this / other; return *this; }

    HOST DEVICE tvec3& operator *= (T t) { *this = *this * t; return *this; }
    HOST DEVICE tvec3& operator /= (T t) { *this = *this / t; return *this; }
};

template <typename T> HOST DEVICE tvec3<T> operator + (const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(a.x + b.x, a.y + b.y, a.z + b.z); }
template <typename T> HOST DEVICE tvec3<T> operator - (const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(a.x - b.x, a.y - b.y, a.z - b.z); }
template <typename T> HOST DEVICE tvec3<T> operator * (const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(a.x * b.x, a.y * b.y, a.z * b.z); }
template <typename T> HOST DEVICE tvec3<T> operator / (const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(a.x / b.x, a.y / b.y, a.z / b.z); }
template <typename T> HOST DEVICE tvec3<T> operator * (const tvec3<T>& a, T b) { return tvec3<T>(a.x * b, a.y * b, a.z * b); }
template <typename T> HOST DEVICE tvec3<T> operator / (const tvec3<T>& a, T b) { return tvec3<T>(a.x / b, a.y / b, a.z / b); }
template <typename T> HOST DEVICE tvec3<T> operator * (T a, const tvec3<T>& b) { return tvec3<T>(a * b.x, a * b.y, a * b.z); }
template <typename T> HOST DEVICE tvec3<T> operator / (T a, const tvec3<T>& b) { return tvec3<T>(a / b.x, a / b.y, a / b.z); }
template <typename T> HOST DEVICE tvec3<T> min(const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
template <typename T> HOST DEVICE tvec3<T> max(const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
template <typename T> HOST DEVICE T dot(const tvec3<T>& a, const tvec3<T>& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

template <typename T>
HOST DEVICE tvec3<T> cross(const tvec3<T>& a, const tvec3<T>& b) {
    return tvec3<T>(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
}

template <typename T> HOST DEVICE tvec3<T> operator << (const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(a.x << b.x, a.y << b.y, a.z << b.z); }
template <typename T> HOST DEVICE tvec3<T> operator >> (const tvec3<T>& a, const tvec3<T>& b) { return tvec3<T>(a.x >> b.x, a.y >> b.y, a.z >> b.z); }
template <typename T> HOST DEVICE tvec3<T> operator << (const tvec3<T>& a, T b) { return tvec3<T>(a.x << b, a.y << b, a.z << b); }
template <typename T> HOST DEVICE tvec3<T> operator >> (const tvec3<T>& a, T b) { return tvec3<T>(a.x >> b, a.y >> b, a.z >> b); }
template <typename T> HOST DEVICE tvec3<T> operator << (T a, const tvec3<T>& b) { return tvec3<T>(a << b.x, a << b.y, a << b.z); }
template <typename T> HOST DEVICE tvec3<T> operator >> (T a, const tvec3<T>& b) { return tvec3<T>(a >> b.x, a >> b.y, a >> b.z); }

template <typename T> HOST DEVICE tvec3<T> clamp(const tvec3<T>& a, T b, T c) { return tvec3<T>(min(max(a.x, b), c), min(max(a.y, b), c), min(max(a.z, b), c)); }

typedef tvec2<float> vec2;
typedef tvec2<int>   ivec2;
typedef tvec3<float> vec3;
typedef tvec3<int>   ivec3;

} // namespace hagrid

#endif // VEC_H

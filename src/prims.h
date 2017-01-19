#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <cmath>
#include "vec.h"
#include "bbox.h"
#include "ray.h"

namespace hagrid {

/// Triangle (point + edges + normal)
struct Tri {
    vec3 v0; float nx;
    vec3 e1; float ny;
    vec3 e2; float nz;

    HOST DEVICE Tri() {}
    HOST DEVICE Tri(const vec3& v0, float nx,
                    const vec3& e1, float ny,
                    const vec3& e2, float nz)
        : v0(v0), nx(nx)
        , e1(e1), ny(ny)
        , e2(e2), nz(nz)
    {}

    HOST DEVICE BBox bbox() const {
        auto v1 = v0 - e1;
        auto v2 = v0 + e2;
        return BBox(min(v0, min(v1, v2)), max(v0, max(v1, v2)));    
    }

    HOST DEVICE vec3 normal() const {
        return vec3(nx, ny, nz);
    }
};

HOST DEVICE inline bool plane_overlap_box(const vec3& n, float d, const vec3& min, const vec3& max) {
    auto first = vec3(n.x > 0 ? min.x : max.x,
                      n.y > 0 ? min.y : max.y,
                      n.z > 0 ? min.z : max.z);

    auto last = vec3(n.x <= 0 ? min.x : max.x,
                     n.y <= 0 ? min.y : max.y,
                     n.z <= 0 ? min.z : max.z);

    auto d0 = dot(n, first) - d;
    auto d1 = dot(n, last)  - d;
#if __CUDACC_VER_MAJOR__ == 7
    union { int i; float f; } u0 = { .f = d0 };
    union { int i; float f; } u1 = { .f = d1 };
    // Equivalent to d1 * d0 <= 0.0f (CUDA 7.0 bug)
    return (((u0.i ^ u1.i) & 0x80000000) | (d0 == 0.0f) | (d1 == 0.0f)) != 0;
#else
    return d1 * d0 <= 0.0f;
#endif
}

HOST DEVICE inline bool axis_test_x(const vec3& half_size,
                                    const vec3& e, const vec3& f,
                                    const vec3& v0, const vec3& v1) {
    auto p0 = e.y * v0.z - e.z * v0.y;
    auto p1 = e.y * v1.z - e.z * v1.y;
    auto rad = f.z * half_size.y + f.y * half_size.z;
    return fmin(p0, p1) > rad | fmax(p0, p1) < -rad;
}

HOST DEVICE inline bool axis_test_y(const vec3& half_size,
                                    const vec3& e, const vec3& f,
                                    const vec3& v0, const vec3& v1) {
    auto p0 = e.z * v0.x - e.x * v0.z;
    auto p1 = e.z * v1.x - e.x * v1.z;
    auto rad = f.z * half_size.x + f.x * half_size.z;
    return fmin(p0, p1) > rad | fmax(p0, p1) < -rad;
}

HOST DEVICE inline bool axis_test_z(const vec3& half_size,
                                    const vec3& e, const vec3& f,
                                    const vec3& v0, const vec3& v1) {
    auto p0 = e.x * v0.y - e.y * v0.x;
    auto p1 = e.x * v1.y - e.y * v1.x;
    auto rad = f.y * half_size.x + f.x * half_size.y;
    return fmin(p0, p1) > rad | fmax(p0, p1) < -rad;
}

template <bool bounds_check, bool cross_axes>
HOST DEVICE inline bool intersect_tri_box(const vec3& v0, const vec3& e1, const vec3& e2, const vec3& n, const vec3& min, const vec3& max) {
    if (!plane_overlap_box(n, dot(v0, n), min, max))
        return false;

    auto v1 = v0 - e1;
    auto v2 = v0 + e2;
    if (bounds_check) {
        auto min_x = fmin(v0.x, fmin(v1.x, v2.x));
        auto max_x = fmax(v0.x, fmax(v1.x, v2.x));
        if (min_x > max.x | max_x < min.x) return false;

        auto min_y = fmin(v0.y, fmin(v1.y, v2.y));
        auto max_y = fmax(v0.y, fmax(v1.y, v2.y));
        if (min_y > max.y | max_y < min.y) return false;

        auto min_z = fmin(v0.z, fmin(v1.z, v2.z));
        auto max_z = fmax(v0.z, fmax(v1.z, v2.z));
        if (min_z > max.z | max_z < min.z) return false;
    }

    if (cross_axes) {
        auto center    = (max + min) * 0.5f;
        auto half_size = (max - min) * 0.5f;

        auto w0 = v0 - center;
        auto w1 = v1 - center;
        auto w2 = v2 - center;

        auto f1 = vec3(fabs(e1.x), fabs(e1.y), fabs(e1.z));
        if (axis_test_x(half_size, e1, f1, w0, w2) ||
            axis_test_y(half_size, e1, f1, w0, w2) ||
            axis_test_z(half_size, e1, f1, w1, w2))
            return false;

        auto f2 = vec3(fabs(e2.x), fabs(e2.y), fabs(e2.z));
        if (axis_test_x(half_size, e2, f2, w0, w1) ||
            axis_test_y(half_size, e2, f2, w0, w1) ||
            axis_test_z(half_size, e2, f2, w1, w2))
            return false;

        auto e3 = e1 + e2;

        auto f3 = vec3(fabs(e3.x), fabs(e3.y), fabs(e3.z));
        if (axis_test_x(half_size, e3, f3, w0, w2) ||
            axis_test_y(half_size, e3, f3, w0, w2) ||
            axis_test_z(half_size, e3, f3, w0, w1))
            return false;
    }

    return true;
}

HOST DEVICE inline bool intersect_prim_box(const Tri& tri, const BBox& bbox) {
    return intersect_tri_box<true, true>(tri.v0, tri.e1, tri.e2, tri.normal(), bbox.min, bbox.max);
}

HOST DEVICE inline bool intersect_prim_ray(const Tri& tri, const Ray& ray, int id, Hit& hit) {
    // Moeller Trumbore
    auto n = tri.normal();

    auto c = tri.v0 - ray.org;
    auto r = cross(ray.dir, c);
    auto det = dot(n, ray.dir);
    auto abs_det = fabs(det);

    auto u = prodsign(dot(r, tri.e2), det);
    auto v = prodsign(dot(r, tri.e1), det);
    auto w = abs_det - u - v;

    auto eps = 1e-9f;
    if (u >= -eps && v >= -eps && w >= -eps) {
        auto t = prodsign(dot(n, c), det);
        if (t >= abs_det * ray.tmin && abs_det * ray.tmax > t) {
            auto inv_det = 1.0f / abs_det;
            hit.t = t * inv_det;
#ifdef COMPUTE_UVS
            hit.u = u * inv_det;
            hit.v = v * inv_det;
#endif
            hit.id = id;
            return true;
        }
    }

    return false;
}

#ifdef __NVCC__
__device__ __forceinline__ Tri load_prim(const Tri* tri_ptr) {
    const float4* ptr = (const float4*)tri_ptr;
    auto tri0 = ptr[0];
    auto tri1 = ptr[1];
    auto tri2 = ptr[2];
    return Tri(vec3(tri0.x, tri0.y, tri0.z), tri0.w,
               vec3(tri1.x, tri1.y, tri1.z), tri1.w,
               vec3(tri2.x, tri2.y, tri2.z), tri2.w);
}
#endif

} // namespace hagrid

#endif // PRIMITIVES_H

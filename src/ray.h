#ifndef RAY_H
#define RAY_H

#include "vec.h"

namespace hagrid {

/// Ray, defined as org + t * dir with t in [tmin, tmax]
struct Ray {
    vec3 org;
    float tmin;
    vec3 dir;
    float tmax;

    HOST DEVICE Ray() {}
    HOST DEVICE Ray(const vec3& org, float tmin,
                    const vec3& dir, float tmax)
        : org(org), tmin(tmin), dir(dir), tmax(tmax)
    {}
};

/// Result of a hit (id is -1 if there is no hit)
struct Hit {
    int id;
    float t;
    float u;
    float v;

    HOST DEVICE Hit() {}
    HOST DEVICE Hit(int id, float t, float u, float v)
        : id(id), t(t), u(u), v(v)
    {}
};

#ifdef __NVCC__
__device__ __forceinline__ Ray load_ray(const Ray* ray_ptr) {
    const float4* ptr = (const float4*)ray_ptr;
    auto ray0 = ptr[0];
    auto ray1 = ptr[1];
    return Ray(vec3(ray0.x, ray0.y, ray0.z), ray0.w,
               vec3(ray1.x, ray1.y, ray1.z), ray1.w);
}

__device__ __forceinline__ void store_hit(Hit* hit_ptr, const Hit& hit) {
    float4* ptr = (float4*)hit_ptr;
    ptr[0] = make_float4(__int_as_float(hit.id), hit.t, hit.u, hit.v);
}
#endif

} // namespace hagrid

#endif // RAY_H

#ifndef BBOX_H
#define BBOX_H

#include <cfloat>
#include <algorithm>
#include "vec.h"

namespace hagrid {

struct BBox {
    vec3 min;
    int pad0;
    vec3 max;
    int pad1;

    HOST DEVICE BBox() {}
    HOST DEVICE BBox(const vec3& v) : min(v), max(v) {}
    HOST DEVICE BBox(const vec3& min, const vec3& max) : min(min), max(max) {}

    HOST DEVICE BBox& extend(const vec3& f) {
        min = hagrid::min(min, f);
        max = hagrid::max(max, f);
        return *this;
    }

    HOST DEVICE BBox& extend(const BBox& bb) {
        min = hagrid::min(min, bb.min);
        max = hagrid::max(max, bb.max);
        return *this;
    }

    HOST DEVICE BBox& overlap(const BBox& bb) {
        min = hagrid::max(min, bb.min);
        max = hagrid::min(max, bb.max);
        return *this;
    }

    HOST DEVICE vec3 extents() const {
        return max - min;
    }

    HOST DEVICE float half_area() const {
        const vec3 len = max - min;
        const float kx = hagrid::max(len.x, 0.0f);
        const float ky = hagrid::max(len.y, 0.0f);
        const float kz = hagrid::max(len.z, 0.0f);
        return kx * (ky + kz) + ky * kz;
    }

    HOST DEVICE bool is_empty() const {
        return min.x > max.x || min.y > max.y || min.z > max.z;
    }

    HOST DEVICE bool is_inside(const vec3& f) const {
        return f.x >= min.x && f.y >= min.y && f.z >= min.z &&
               f.x <= max.x && f.y <= max.y && f.z <= max.z;
    }

    HOST DEVICE bool is_overlapping(const BBox& bb) const {
        return min.x <= bb.max.x && max.x >= bb.min.x &&
               min.y <= bb.max.y && max.y >= bb.min.y &&
               min.z <= bb.max.z && max.z >= bb.min.z;
    }

    HOST DEVICE bool is_included(const BBox& bb) const {
        return min.x >= bb.min.x && max.x <= bb.max.x &&
               min.y >= bb.min.y && max.y <= bb.max.y &&
               min.z >= bb.min.z && max.z <= bb.max.z;
    }

    HOST DEVICE bool is_strictly_included(const BBox& bb) const {
        return is_included(bb) &&
               (min.x > bb.min.x || max.x < bb.max.x ||
                min.y > bb.min.y || max.y < bb.max.y ||
                min.z > bb.min.z || max.z < bb.max.z);
    }

    HOST DEVICE static BBox empty() { return BBox(vec3( FLT_MAX), vec3(-FLT_MAX)); }
    HOST DEVICE static BBox full()  { return BBox(vec3(-FLT_MAX), vec3( FLT_MAX)); }
};

#ifdef __NVCC__
__device__ BBox load_bbox(const BBox* bb_ptr) {
    const float4* ptr = (const float4*)bb_ptr;
    auto bb0 = ptr[0];
    auto bb1 = ptr[1];
    return BBox(vec3(bb0.x, bb0.y, bb0.z),
                vec3(bb1.x, bb1.y, bb1.z));
}

__device__ void store_bbox(BBox* bb_ptr, const BBox& bb) {
    float4* ptr = (float4*)bb_ptr;
    ptr[0] = make_float4(bb.min.x, bb.min.y, bb.min.z, 0);
    ptr[1] = make_float4(bb.max.x, bb.max.y, bb.max.z, 0);
}
#endif // __NVCC__

} // namespace hagrid

#endif // BBOX_H

#include "traverse.h"

namespace hagrid {

static __constant__ ivec3 grid_dims;
static __constant__ vec3  grid_min;
static __constant__ vec3  grid_max;
static __constant__ vec3  cell_size;
static __constant__ vec3  grid_inv;
static __constant__ int   grid_shift;

__device__ __forceinline__ vec2 intersect_ray_box(vec3 org, vec3 inv_dir, vec3 box_min, vec3 box_max) {
    auto tmin = (box_min - org) * inv_dir;
    auto tmax = (box_max - org) * inv_dir;
    auto t0 = min(tmin, tmax);
    auto t1 = max(tmin, tmax);
    return vec2(fmax(t0.x, fmax(t0.y, t0.z)),
                fmin(t1.x, fmin(t1.y, t1.z)));
}

__device__ __forceinline__ vec3 compute_voxel(vec3 org, vec3 dir, float t) {
    return (t * dir + org - grid_min) * grid_inv;
}

template <typename Primitive>
__global__ void traverse_grid(const Entry* __restrict__ entries,
                              const Cell* __restrict__ cells,
                              const int*  __restrict__ ref_ids,
                              const Primitive*  __restrict__ prims,
                              const Ray*  __restrict__ rays,
                              Hit* __restrict__ hits,
                              int num_rays) {
    const int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_rays) return;

    auto ray = load_ray(rays + id);
    auto inv_dir = vec3(safe_rcp(ray.dir.x), safe_rcp(ray.dir.y), safe_rcp(ray.dir.z));

    // Intersect the grid bounding box
    auto tbox = intersect_ray_box(ray.org, inv_dir, grid_min, grid_max);
    auto tstart = fmax(tbox.x, ray.tmin);
    auto tend   = fmin(tbox.y, ray.tmax);

    auto hit = Hit(-1, ray.tmax, 0, 0);
    ivec3 voxel;

    // Early exit if the ray does not hit the grid
    if (tstart > tend) goto exit;

    // Find initial voxel
    voxel = clamp(ivec3(compute_voxel(ray.org, ray.dir, tstart)), ivec3(0, 0, 0), grid_dims - 1);

    while (true) {
        // Lookup entry
        const int entry = lookup_entry(entries, grid_shift, grid_dims, voxel);

        // Lookup the cell associated with this voxel
        auto cell = load_cell(cells + entry);

        // Intersect the farmost planes of the cell bounding box
        auto cell_point = ivec3(ray.dir.x >= 0.0f ? cell.max.x : cell.min.x,
                                ray.dir.y >= 0.0f ? cell.max.y : cell.min.y,
                                ray.dir.z >= 0.0f ? cell.max.z : cell.min.z);
        auto tcell = (vec3(cell_point) * cell_size + grid_min - ray.org) * inv_dir;
        auto texit = fmin(tcell.x, fmin(tcell.y, tcell.z));

        // Move to the next voxel
        auto exit_point = ivec3(compute_voxel(ray.org, ray.dir, texit));
        auto next_voxel = ivec3(texit == tcell.x ? cell_point.x + (ray.dir.x >= 0.0f ? 0 : -1) : exit_point.x,
                                texit == tcell.y ? cell_point.y + (ray.dir.y >= 0.0f ? 0 : -1) : exit_point.y,
                                texit == tcell.z ? cell_point.z + (ray.dir.z >= 0.0f ? 0 : -1) : exit_point.z);
        voxel.x = ray.dir.x >= 0.0f ? max(next_voxel.x, voxel.x) : min(next_voxel.x, voxel.x);
        voxel.y = ray.dir.y >= 0.0f ? max(next_voxel.y, voxel.y) : min(next_voxel.y, voxel.y);
        voxel.z = ray.dir.z >= 0.0f ? max(next_voxel.z, voxel.z) : min(next_voxel.z, voxel.z);

        // Intersect the cell contents and exit if an intersection was found
        for (int i = cell.begin; i < cell.end; i++) {
            const int ref = ref_ids[i];
            auto prim = load_prim(prims + ref);
            intersect_prim_ray(prim, Ray(ray.org, ray.tmin, ray.dir, hit.t), ref, hit);
        }

        if (hit.t <= texit ||
            (voxel.x < 0 | voxel.x >= grid_dims.x |
             voxel.y < 0 | voxel.y >= grid_dims.y |
             voxel.z < 0 | voxel.z >= grid_dims.z))
            break;
    }

exit:
    store_hit(hits + id, hit);
}

void setup_traversal(const Grid& grid) {
    auto extents = grid.bbox.extents();
    auto grid_inv  = vec3(grid.top_dims) / extents;
    auto cell_size = extents / vec3(grid.top_dims);

    set_global(hagrid::grid_dims,  &grid.top_dims);
    set_global(hagrid::grid_min,   &grid.bbox.min);
    set_global(hagrid::grid_max,   &grid.bbox.max);
    set_global(hagrid::cell_size,  &cell_size);
    set_global(hagrid::grid_inv,   &grid_inv);
    set_global(hagrid::grid_shift, &grid.num_levels);
}

void traverse(const Grid& grid, const Tri* tris, const Ray* rays, Hit* hits, int num_rays) {
    traverse_grid<<<round_div(num_rays, 64), 64>>>(grid.entries, grid.cells, grid.ref_ids, tris, rays, hits, num_rays);
}

} // namespace hagrid

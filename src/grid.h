#ifndef GRID_H
#define GRID_H

#include <vector>

#include "vec.h"
#include "bbox.h"

namespace hagrid {

/// Voxel map entry
struct Entry {
    enum {
        LOG_DIM_BITS = 2,
        BEGIN_BITS   = 32 - LOG_DIM_BITS
    };

    uint32_t log_dim : LOG_DIM_BITS;    ///< Logarithm of the dimensions of the entry (0 for leaves)
    uint32_t begin   : BEGIN_BITS;      ///< Next entry index (cell index for leaves)

    HOST DEVICE Entry(uint32_t l, uint32_t b)
        : log_dim(l), begin(b)
    {}
};

/// Cell of the irregular grid
struct Cell {
    ivec3 min;     ///< Minimum bounding box coordinate
    int begin;     ///< Index of the first reference
    ivec3 max;     ///< Maximum bounding box coordinate
    int end;       ///< Past-the-end reference index

    HOST DEVICE Cell() {}
    HOST DEVICE Cell(const ivec3& min, int begin, const ivec3& max, int end)
        : min(min), begin(begin), max(max), end(end)
    {}
};

/// Structure holding an irregular grid
struct Grid {
    Entry* entries;             ///< Voxel map, stored as a contiguous array
    int*   ref_ids;             ///< Array of primitive references
    Cell*  cells;               ///< Cells of the structure

    BBox bbox;                  ///< Bounding box of the scene
    ivec3 dims;                 ///< Top-level dimensions
    int num_cells;              ///< Number of cells
    int num_entries;            ///< Number of elements in the voxel map
    int num_refs;               ///< Number of primitive references
    int shift;                  ///< Amount of bits to shift to get from the deepest level to the top-level
    std::vector<int> offsets;   ///< Offset to each level of the voxel map octree
};

/// A 3D integer range
struct Range {
    int lx, ly, lz;
    int hx, hy, hz;
    HOST DEVICE Range() {}
    HOST DEVICE Range(int lx, int ly, int lz,
                      int hx, int hy, int hz)
        : lx(lx), ly(ly), lz(lz)
        , hx(hx), hy(hy), hz(hz)
    {}
    HOST DEVICE int size() const { return (hx - lx + 1) * (hy - ly + 1) * (hz - lz + 1) ; }
};

/// Computes the range of cells that intersect the given box
HOST DEVICE inline Range compute_range(const ivec3& dims, const BBox& grid_bb, const BBox& obj_bb) {
    auto inv = vec3(dims) / grid_bb.extents();
    int lx = max(int((obj_bb.min.x - grid_bb.min.x) * inv.x), 0);
    int ly = max(int((obj_bb.min.y - grid_bb.min.y) * inv.y), 0);
    int lz = max(int((obj_bb.min.z - grid_bb.min.z) * inv.z), 0);
    int hx = min(int((obj_bb.max.x - grid_bb.min.x) * inv.x), dims.x - 1);
    int hy = min(int((obj_bb.max.y - grid_bb.min.y) * inv.y), dims.y - 1);
    int hz = min(int((obj_bb.max.z - grid_bb.min.z) * inv.z), dims.z - 1);
    return Range(lx, ly, lz, hx, hy, hz);
}

/// Computes grid dimensions based on the formula by Cleary et al.
HOST DEVICE inline ivec3 compute_grid_dims(const BBox& bb, int num_prims, float density) {
    const vec3 extents = bb.extents();
    const float volume = extents.x * extents.y * extents.z;
    const float ratio = cbrtf(density * num_prims / volume);
    return max(ivec3(1), ivec3(extents.x * ratio, extents.y * ratio, extents.z * ratio));
}

HOST DEVICE inline uint32_t lookup_entry(const Entry* entries, int shift, const ivec3& dims, const ivec3& voxel) {
    auto entry = entries[(voxel.x >> shift) + dims.x * ((voxel.y >> shift) + dims.y * (voxel.z >> shift))];
    auto log_dim = entry.log_dim, d = log_dim;
    while (log_dim) {
        auto begin = entry.begin;
        auto mask = (1 << log_dim) - 1;

        auto k = (voxel >> int(shift - d)) & mask;
        entry = entries[begin + k.x + ((k.y + (k.z << log_dim)) << log_dim)];
        log_dim = entry.log_dim;
        d += log_dim;
    }
    return entry.begin;
}

#ifdef __NVCC__
__device__ __forceinline__ Cell load_cell(const Cell* cell_ptr) {
    const int4* ptr = (const int4*)cell_ptr;
    auto cell0 = ptr[0];
    auto cell1 = ptr[1];
    return Cell(ivec3(cell0.x, cell0.y, cell0.z), cell0.w,
                ivec3(cell1.x, cell1.y, cell1.z), cell1.w);
}

__device__ __forceinline__ ivec3 load_cell_min(const Cell* cell_ptr) {
    auto cell0 = ((const int4*)cell_ptr)[0];
    return ivec3(cell0.x, cell0.y, cell0.z);
}

__device__ __forceinline__ void store_cell(Cell* cell_ptr, const Cell& cell) {
    int4* ptr = (int4*)cell_ptr;
    ptr[0] = make_int4(cell.min.x, cell.min.y, cell.min.z, cell.begin);
    ptr[1] = make_int4(cell.max.x, cell.max.y, cell.max.z, cell.end);
}
#endif // __NVCC__

} // namespace hagrid

#endif // GRID_H

#include "build.h"

namespace hagrid {

static __constant__ ivec3 grid_dims;
static __constant__ vec3  grid_min;
static __constant__ vec3  cell_size;
static __constant__ vec3  grid_inv;
static __constant__ int   grid_shift;

/// Returns true if an overlap with a neighboring cell is possible
template <int axis, bool dir>
__device__ bool overlap_possible(const Cell& cell) {
    if (dir)
        return get<axis>(cell.max) < get<axis>(grid_dims);
    else
        return get<axis>(cell.min) > 0;
}

/// Determines if the given range of references is a subset of the other
__device__ __forceinline__ bool is_subset(const int* __restrict__ p0, int c0, const int* __restrict__ p1, int c1) {
    if (c1 > c0) return false;
    if (c1 == 0) return true;

    int i = 0, j = 0;

    do {
        const int a = p0[i];
        const int b = p1[j];
        if (b < a) return false;
        j += (a == b);
        i++;
    } while (i < c0 & j < c1);

    return j == c1;
}

/// Computes the amount of overlap possible for a cell and a given primitive
template <int axis, bool dir, typename Primitive>
__device__ int compute_overlap(const Primitive& prim, const Cell& cell, const BBox& cell_bbox, int d) {
    static constexpr int axis1 = (axis + 1) % 3;
    static constexpr int axis2 = (axis + 2) % 3;
    auto prim_bbox = prim.bbox();

    if (get<axis1>(prim_bbox.min) <= get<axis1>(cell_bbox.max) &&
        get<axis1>(prim_bbox.max) >= get<axis1>(cell_bbox.min) &&
        get<axis2>(prim_bbox.min) <= get<axis2>(cell_bbox.max) &&
        get<axis2>(prim_bbox.max) >= get<axis2>(cell_bbox.min)) {
        // Approximation: use the original bounding box, not the clipped one
        int prim_d = ((dir ? get<axis>(prim_bbox.min) : get<axis>(prim_bbox.max)) - get<axis>(grid_min)) * get<axis>(grid_inv);
        d = dir
            ? min(d, prim_d - get<axis>(cell.max))
            : max(d, prim_d - get<axis>(cell.min) + 1);
        d = dir ? max(d, 0) : min(d, 0);
    }
    return d;
}

/// Finds the maximum overlap possible for one cell
template <int axis, bool dir, bool subset_only, typename Primitive>
__device__ int find_overlap(const Entry* __restrict__ entries,
                            const int* __restrict__ refs,
                            const Primitive* __restrict__ prims,
                            const Cell* cells,
                            const Cell& cell,
                            bool& continue_overlap) {
    constexpr int axis1 = (axis + 1) % 3;
    constexpr int axis2 = (axis + 2) % 3;

    if (!overlap_possible<axis, dir>(cell)) return 0;

    int d = dir ? get<axis>(grid_dims) : -get<axis>(grid_dims);
    int k1, k2 = get<axis2>(grid_dims);
    int i = get<axis1>(cell.min);
    int j = get<axis2>(cell.min);
    int max_d = d;
    while (true) {
        ivec3 next_cell;
        if (axis == 0) next_cell = ivec3(dir ? cell.max.x : cell.min.x - 1, i, j);
        if (axis == 1) next_cell = ivec3(j, dir ? cell.max.y : cell.min.y - 1, i);
        if (axis == 2) next_cell = ivec3(i, j, dir ? cell.max.z : cell.min.z - 1);
        auto entry = lookup_entry(entries, grid_shift, grid_dims >> grid_shift, next_cell);
        auto next = load_cell(cells + entry);

        max_d = dir
            ? min(max_d, get<axis>(next.max) - get<axis>(cell.max))
            : max(max_d, get<axis>(next.min) - get<axis>(cell.min));
        d = dir ? min(d, max_d) : max(d, max_d);

        if (subset_only) {
            if (!is_subset(refs + cell.begin, cell.end - cell.begin,
                           refs + next.begin, next.end - next.begin)) {
                d = 0;
                break;
            }
        } else {
            if (next.begin < next.end) {
                auto cell_bbox = BBox(grid_min + cell_size * vec3(cell.min),
                                      grid_min + cell_size * vec3(cell.max));

                int p1 = cell.begin, p2 = next.begin;
                int ref2 = refs[p2];
                while (true) {
                    // Skip references that are present in the current cell
                    while (p1 < cell.end) {
                        int ref1 = refs[p1];

                        if (ref1  > ref2) break;
                        if (ref1 == ref2) {
                            if (++p2 >= next.end) break;
                            ref2 = refs[p2];
                        }

                        p1++;
                    }

                    if (p2 >= next.end) break;

                    // Process references that are only present in the next cell
                    d = compute_overlap<axis, dir>(load_prim(prims + ref2), cell, cell_bbox, d);
                    if (d == 0 || ++p2 >= next.end) break;
                    ref2 = refs[p2];
                }
            }

            if (d == 0) break;
        }

        k1 = get<axis1>(next.max) - i;
        k2 = min(k2, get<axis2>(next.max) - j);

        i += k1;
        if (i >= get<axis1>(cell.max)) {
            i = get<axis1>(cell.min);
            j += k2;
            k2 = get<axis2>(grid_dims);
            if (j >= get<axis2>(cell.max)) break;
        }
    }

    continue_overlap |= d == max_d;
    return d;
}

template <int axis, typename Primitive>
__global__ void overlap_step(const Entry* __restrict__ entries,
                             const int* __restrict__ refs,
                             const Primitive* __restrict__ prims,
                             const Cell* __restrict__ cells,
                             Cell* __restrict__ new_cells,
                             int* __restrict__ cell_flags,
                             int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells || (cell_flags[id] & (1 << axis)) == 0)
        return;

    auto cell = load_cell(cells + id);
    bool flag = false;
    constexpr bool subset_only = true;
    auto ov1 = find_overlap<axis, false, subset_only>(entries, refs, prims, cells, cell, flag);
    auto ov2 = find_overlap<axis, true,  subset_only>(entries, refs, prims, cells, cell, flag);

    if (axis == 0) {
        cell.min.x += ov1;
        cell.max.x += ov2;
    }

    if (axis == 1) {
        cell.min.y += ov1;
        cell.max.y += ov2;
    }

    if (axis == 2) {
        cell.min.z += ov1;
        cell.max.z += ov2;
    }

    // If the cell has not been expanded, we will not process it next time
    cell_flags[id] = (flag ? 1 << axis : 0) | (cell_flags[id] & ~(1 << axis));

    store_cell(new_cells + id, cell);
}

template <typename Primitive>
void expansion_iter(Grid& grid, const Primitive* prims, Cell*& new_cells, int* cell_flags) {
    overlap_step<0><<<round_div(grid.num_cells, 64), 64>>>(grid.entries, grid.ref_ids, prims, grid.cells, new_cells, cell_flags, grid.num_cells);
    std::swap(new_cells, grid.cells);
    overlap_step<1><<<round_div(grid.num_cells, 64), 64>>>(grid.entries, grid.ref_ids, prims, grid.cells, new_cells, cell_flags, grid.num_cells);
    std::swap(new_cells, grid.cells);
    overlap_step<2><<<round_div(grid.num_cells, 64), 64>>>(grid.entries, grid.ref_ids, prims, grid.cells, new_cells, cell_flags, grid.num_cells);
    std::swap(new_cells, grid.cells);
}

template <typename Primitive>
void expand(MemManager& mem, Grid& grid, const Primitive* prims, int iters) {
    if (iters == 0) return;

    auto new_cells  = mem.alloc<Cell>(grid.num_cells);
    auto cell_flags = mem.alloc<int>(grid.num_cells);

    mem.one(cell_flags, grid.num_cells);
    auto extents = grid.bbox.extents();
    auto dims = grid.dims << grid.shift;
    auto cell_size = extents / vec3(dims);
    auto grid_inv = vec3(dims) / extents;

    set_global(hagrid::grid_dims,  dims);
    set_global(hagrid::grid_min,   grid.bbox.min);
    set_global(hagrid::cell_size,  cell_size);
    set_global(hagrid::grid_inv,   grid_inv);
    set_global(hagrid::grid_shift, grid.shift);

    for (int i = 0; i < iters; i++)
        expansion_iter(grid, prims, new_cells, cell_flags);

    mem.free(cell_flags);
    mem.free(new_cells);
}

void expand_grid(MemManager& mem, Grid& grid, const Tri* tris, int iters) { expand(mem, grid, tris, iters); }

} // namespace hagrid

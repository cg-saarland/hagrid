#include "build.h"
#include "parallel.cuh"

namespace hagrid {

/// Structure that contains buffers used during merging
struct MergeBuffers {
    int* merge_counts;  ///< Contains the number of references in each cell (positive if merged, otherwise negative)
    int* prevs, *nexts; ///< Contains the index of the previous/next neighboring cell on the merging axis (positive if merged, otherwise negative)
    int* ref_counts;    ///< Contains the number of references per cell after merge
    int* cell_flags;    ///< Contains 1 if the cell is kept (it is not a residue), otherwise 0
    int* cell_scan;     ///< Scan over cell_flags (insertion position of the cells into the new cell array)
    int* ref_scan;      ///< Scan over ref_counts (insertion position of the references into the new reference array)
    int* new_cell_ids;  ///< Mapping between the old cell indices and the new cell indices
};

static __constant__ ivec3 grid_dims;
static __constant__ vec3  cell_size;
static __constant__ int   grid_shift;

template <int axis>
__device__ bool aligned(const Cell& cell1, const Cell& cell2) {
    constexpr int axis1 = (axis + 1) % 3;
    constexpr int axis2 = (axis + 2) % 3;

    return get<axis >(cell1.max) == get<axis >(cell2.min) &&
           get<axis1>(cell1.min) == get<axis1>(cell2.min) &&
           get<axis2>(cell1.min) == get<axis2>(cell2.min) &&
           get<axis1>(cell1.max) == get<axis1>(cell2.max) &&
           get<axis2>(cell1.max) == get<axis2>(cell2.max);
}

/// Restricts the merges so that cells are better aligned for the next iteration
__device__ __forceinline__ bool merge_allowed(int empty_mask, int pos) {
    auto top_level_mask = (1 << grid_shift) - 1;
    auto is_shifted   = (pos >> grid_shift) & empty_mask;
    auto is_top_level = !(pos & top_level_mask);
    return !is_shifted || !is_top_level;
}

/// Computes the position of the next cell of the grid on the axis
template <int axis>
__device__ ivec3 next_cell(const ivec3& min, const ivec3& max) {
    return ivec3(axis == 0 ? max.x : min.x,
                 axis == 1 ? max.y : min.y,
                 axis == 2 ? max.z : min.z);
}

/// Computes the position of the previous cell of the grid on the axis
template <int axis>
__device__ ivec3 prev_cell(const ivec3& min) {
    return ivec3(axis == 0 ? min.x - 1 : min.x,
                 axis == 1 ? min.y - 1 : min.y,
                 axis == 2 ? min.z - 1 : min.z);
}

/// Counts the number of elements in the union of two sorted arrays
__device__ __forceinline__ int count_union(const int* __restrict__ p0, int c0,
                                           const int* __restrict__ p1, int c1) {
    int i = 0, j = 0, c = 0;
    while (i < c0 & j < c1) {
        auto a = p0[i];
        auto b = p1[j];
        i += (a <= b);
        j += (a >= b);
        c++;
    }
    return c + (c1 - j) + (c0 - i);
}

/// Merges the two sorted reference arrays
__device__ __forceinline__ void merge_refs(const int* __restrict__ p0, int c0,
                                           const int* __restrict__ p1, int c1,
                                           int* __restrict__ q) {
    int i = 0;
    int j = 0;
    while (i < c0 && j < c1) {
        auto a = p0[i];
        auto b = p1[j];
        *(q++) = (a < b) ? a : b;
        i += (a <= b);
        j += (a >= b);
    }
    auto k = i < c0 ? i  :  j;
    auto c = i < c0 ? c0 : c1;
    auto p = i < c0 ? p0 : p1;
    while (k < c) *(q++) = p[k++];
}

/// Computes the number of references per cell after the merge
template <int axis>
__global__ void compute_merge_counts(const Entry* __restrict__ entries,
                                     const Cell* __restrict__  cells,
                                     const int* __restrict__   refs,
                                     int* __restrict__ merge_counts,
                                     int* __restrict__ nexts,
                                     int* __restrict__ prevs,
                                     int empty_mask,
                                     int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    static constexpr auto unit_cost = 1.0f;

    auto cell1 = load_cell(cells + id);
    auto next_pos = next_cell<axis>(cell1.min, cell1.max);
    int count = -(cell1.end - cell1.begin + 1);
    int next_id = -1;

    if (merge_allowed(empty_mask, get<axis>(cell1.min)) &&
        get<axis>(next_pos) < get<axis>(grid_dims)) {
        next_id = lookup_entry(entries, grid_shift, grid_dims >> grid_shift, next_pos);
        auto cell2 = load_cell(cells + next_id);

        if (aligned<axis>(cell1, cell2)) {
            auto e1 = vec3(cell1.max - cell1.min) * cell_size;
            auto e2 = vec3(cell2.max - cell2.min) * cell_size;
            auto a1 = e1.x * (e1.y + e1.z) + e1.y * e1.z;
            auto a2 = e2.x * (e2.y + e2.z) + e2.y * e2.z;
            auto a  = a1 + a2 - get<(axis + 1) % 3>(e1) * get<(axis + 2) % 3>(e1);

            int n1 = cell1.end - cell1.begin;
            int n2 = cell2.end - cell2.begin;
            auto c1 = a1 * (n1 + unit_cost);
            auto c2 = a2 * (n2 + unit_cost);
            // Early exit test: there is a minimum of max(n1, n2)
            // primitives in the union of the two cells
            if (a * (max(n1, n2) + unit_cost) <= c1 + c2) {
                auto n = count_union(refs + cell1.begin, n1,
                                     refs + cell2.begin, n2);
                auto c = a * (n + unit_cost);
                if (c <= c1 + c2) count = n;
            }
        }
    }
    
    merge_counts[id] = count;

    next_id = count >= 0 ? next_id : -1;
    nexts[id] = next_id;
    if (next_id >= 0) prevs[next_id] = id;
}

/// Traverses the merge chains and mark the cells at odd positions as residue
template <int axis>
__global__ void compute_cell_flags(const int* __restrict__ nexts,
                                   const int* __restrict__ prevs,
                                   int* __restrict__ cell_flags,
                                   int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    // If the previous cell does not exist or does not want to merge with this cell
    if (prevs[id] < 0) {
        int next_id = nexts[id];
        cell_flags[id] = 1;

        // If this cell wants to merge with the next
        if (next_id >= 0) {
            int count = 1;

            // Traverse the merge chain
            do {
                cell_flags[next_id] = count % 2 ? 0 : 1;
                next_id = nexts[next_id];
                count++;
            } while (next_id >= 0);
        }
    }
}

/// Computes the number of new references per cell
__global__ void compute_ref_counts(const int* __restrict__ merge_counts,
                                    const int* __restrict__ cell_flags,
                                    int* __restrict__ ref_counts,
                                    int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    int count = 0;
    if (cell_flags[id]) {
        const int merged = merge_counts[id];
        count = merged >= 0 ? merged : -(merged + 1);
    }
    ref_counts[id] = count;
}

/// Performs the merge
template <int axis>
__global__ void merge(const Entry* __restrict__ entries,
                      const Cell* __restrict__ cells,
                      const int* __restrict__ refs,
                      const int* __restrict__ cell_scan,
                      const int* __restrict__ ref_scan,
                      const int* __restrict__ merge_counts,
                      int* __restrict__ new_cell_ids,
                      Cell* __restrict__ new_cells,
                      int* __restrict__ new_refs,
                      int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    bool valid = id < num_cells;
    int new_id = valid ? cell_scan[id] : 0;
    valid &= cell_scan[id + 1] > new_id;

    int cell_begin = 0, cell_end = 0;
    int next_begin = 0, next_end = 0;
    int new_refs_begin;

    if (valid) {
        auto cell = load_cell(cells + id);
        int merge_count = merge_counts[id];

        new_refs_begin = ref_scan[id];
        new_cell_ids[id] = new_id;
        cell_begin = cell.begin;
        cell_end   = cell.end;

        ivec3 new_min;
        ivec3 new_max;
        int new_refs_end;
        if (merge_count >= 0) {
            // Do the merge and store the references into the new array
            auto next_id = lookup_entry(entries, grid_shift, grid_dims >> grid_shift, next_cell<axis>(cell.min, cell.max));
            auto next_cell = load_cell(cells + next_id);
            next_begin = next_cell.begin;
            next_end   = next_cell.end;

            // Make the next cell point to the merged one
            new_cell_ids[next_id] = new_id;

            new_min = min(next_cell.min, cell.min);
            new_max = max(next_cell.max, cell.max);
            new_refs_end = new_refs_begin + merge_count;
        } else {
           new_min = cell.min;
           new_max = cell.max;
           new_refs_end = new_refs_begin + (cell_end - cell_begin);
        }

        store_cell(new_cells + new_id, Cell(new_min, new_refs_begin,
                                            new_max, new_refs_end));
    }

    // Copy the original references into the new array, using blocking
    bool merge = next_begin < next_end;
    auto blocked = (cell_end - cell_begin) >= 10;
    auto mask = __ballot(!merge & blocked);
    while (mask) {
        int bit = __ffs(mask) - 1;
        mask &= ~(1 << bit);

        auto begin = __shfl(cell_begin, bit);
        auto end   = __shfl(cell_end,   bit);
        auto warp_id = id % 32;

        for (int i = begin + warp_id, j = __shfl(new_refs_begin, bit) + warp_id; i < end; i += 32, j += 32)
            new_refs[j] = refs[i];
    }

    if (!merge & !blocked) {
        for (int i = cell_begin, j = new_refs_begin; i < cell_end; i++, j++)
            new_refs[j] = refs[i];
    }

    // Merge references if required
    if (merge) {
        merge_refs(refs + cell_begin, cell_end - cell_begin,
                   refs + next_begin, next_end - next_begin,
                   new_refs + new_refs_begin);
    }
}

/// Maps the old cell indices in the voxel map to the new ones
__global__ void remap_entries(Entry* __restrict__ entries,
                              const int* __restrict__ new_cell_ids,
                              int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < num_entries) {
        auto entry = entries[id];
        if (entry.log_dim == 0) entries[id] = make_entry(0, new_cell_ids[entry.begin]);
    }
}

template <int axis>
void merge_iteration(MemManager& mem, Grid& grid, Cell*& new_cells, int*& new_refs, int empty_mask, MergeBuffers& bufs) {
    Parallel par(mem);

    int num_cells   = grid.num_cells;
    int num_entries = grid.num_entries;
    auto cells   = grid.cells;
    auto refs    = grid.ref_ids;
    auto entries = grid.entries;

    mem.one(bufs.prevs, num_cells);
    compute_merge_counts<axis><<<round_div(num_cells, 64), 64>>>(entries, cells, refs, bufs.merge_counts, bufs.nexts, bufs.prevs, empty_mask, num_cells);
    DEBUG_SYNC();
    compute_cell_flags<axis><<<round_div(num_cells, 64), 64>>>(bufs.nexts, bufs.prevs, bufs.cell_flags, num_cells);
    DEBUG_SYNC();
    compute_ref_counts<<<round_div(num_cells, 64), 64>>>(bufs.merge_counts, bufs.cell_flags, bufs.ref_counts, num_cells);
    DEBUG_SYNC();

    int num_new_refs  = par.scan(bufs.ref_counts, num_cells + 1, bufs.ref_scan);
    int num_new_cells = par.scan(bufs.cell_flags, num_cells + 1, bufs.cell_scan);

    merge<axis><<<round_div(num_cells, 64), 64>>>(entries, cells, refs,
                                                  bufs.cell_scan, bufs.ref_scan,
                                                  bufs.merge_counts, bufs.new_cell_ids,
                                                  new_cells, new_refs,
                                                  num_cells);
    DEBUG_SYNC();
    remap_entries<<<round_div(num_entries, 64), 64>>>(entries, bufs.new_cell_ids, num_entries);
    DEBUG_SYNC();

    std::swap(new_cells, cells);
    std::swap(new_refs,  refs);

    grid.cells     = cells;
    grid.ref_ids   = refs;
    grid.num_cells = num_new_cells;
    grid.num_refs  = num_new_refs;
}

void merge_grid(MemManager& mem, Grid& grid, float alpha) {
    MergeBuffers bufs;

    auto new_cells = mem.alloc<Cell>(grid.num_cells);
    auto new_refs  = mem.alloc<int> (grid.num_refs);

    size_t buf_size = grid.num_cells + 1;
    buf_size = buf_size % 4 ? buf_size + 4 - buf_size % 4 : buf_size;

    bufs.merge_counts = mem.alloc<int>(buf_size);
    bufs.ref_counts   = mem.alloc<int>(buf_size);
    bufs.cell_flags   = mem.alloc<int>(buf_size);
    bufs.cell_scan    = mem.alloc<int>(buf_size);
    bufs.ref_scan     = mem.alloc<int>(buf_size);
    bufs.new_cell_ids = bufs.cell_flags;
    bufs.prevs        = bufs.cell_scan;
    bufs.nexts        = bufs.ref_scan;

    auto extents = grid.bbox.extents();
    auto dims = grid.dims << grid.shift;
    auto cell_size = extents / vec3(dims);

    set_global(hagrid::grid_dims,  dims);
    set_global(hagrid::cell_size,  cell_size);
    set_global(hagrid::grid_shift, grid.shift);

    if (alpha > 0) {
        int prev_num_cells = 0, iter = 0;
        do {
            prev_num_cells = grid.num_cells;
            auto mask = iter > 3 ? 0 : (1 << (iter + 1)) - 1;
            merge_iteration<0>(mem, grid, new_cells, new_refs, mask, bufs);
            merge_iteration<1>(mem, grid, new_cells, new_refs, mask, bufs);
            merge_iteration<2>(mem, grid, new_cells, new_refs, mask, bufs);
            iter++;
        } while (grid.num_cells < alpha * prev_num_cells);
    }

    mem.free(bufs.merge_counts);
    mem.free(bufs.ref_counts);
    mem.free(bufs.cell_flags);
    mem.free(bufs.cell_scan);
    mem.free(bufs.ref_scan);

    mem.free(new_cells);
    mem.free(new_refs);
}

} // namespace hagrid

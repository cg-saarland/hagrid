#include <cmath>

#include "build.h"
#include "vec.h"
#include "bbox.h"
#include "grid.h"
#include "prims.h"
#include "mem_manager.h"
#include "parallel.cuh"

namespace hagrid {

/// Level of the grid during construction
struct Level {
    int* ref_ids;               ///< Array of primitive indices
    int* cell_ids;              ///< Array of cell indices
    int num_refs;               ///< Number of references in the level
    int num_kept;               ///< Number of references kept (remaining is split)
    Cell* cells;                ///< Array of cells
    Entry* entries;             ///< Array of voxel map entries
    int num_cells;              ///< Number of cells

    Level() {}
    Level(int* ref_ids, int* cell_ids, int num_refs, int num_kept, Cell* cells, Entry* entries, int num_cells)
        : ref_ids(ref_ids)
        , cell_ids(cell_ids)
        , num_refs(num_refs)
        , num_kept(num_kept)
        , cells(cells)
        , entries(entries)
        , num_cells(num_cells)
    {
        assert(num_refs >= num_kept);
    }
};

static __constant__ ivec3 grid_dims;
static __constant__ BBox  grid_bbox;
static __constant__ vec3  cell_size;
static __constant__ int   grid_shift;

/// Compute the bounding box of every primitive
template <typename Primitive>
__global__ void compute_bboxes(const Primitive* __restrict__ prims,
                               BBox* __restrict__ bboxes,
                               int num_prims) {
    const int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_prims)
        return;

    auto prim = load_prim(prims + id);
    store_bbox(bboxes + id, prim.bbox());
}

/// Compute an over-approximation of the number of references
/// that are going to be generated during reference emission
__global__ void count_new_refs(const BBox*  __restrict__ bboxes,
                               int*        __restrict__ counts,
                               int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    auto ref_bb = load_bbox(bboxes + id);
    auto range  = compute_range(grid_dims, grid_bbox, ref_bb);
    counts[id]  = max(0, range.size());
}

/// Emit the new references by inserting existing ones into the sub-levels
__global__ void __launch_bounds__(64)
emit_new_refs(const BBox* __restrict__ bboxes,
              const int* __restrict__ start_emit,
              int* __restrict__ new_ref_ids,
              int* __restrict__ new_cell_ids,
              int num_prims) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    Range range;
    int start = 0, end = 0;

    if (id < num_prims) {
        start = start_emit[id + 0];
        end   = start_emit[id + 1];

        if (start < end) {
            auto ref_bb = load_bbox(bboxes + id);
            range  = compute_range(grid_dims, grid_bbox, ref_bb);
        }
    }

    bool blocked = (end - start) >= 16;
    if (!blocked && start < end) {
        int x = range.lx;
        int y = range.ly;
        int z = range.lz;
        int cur = start;
        while (cur < end) {
            new_ref_ids [cur] = id;
            new_cell_ids[cur] = x + grid_dims.x * (y + grid_dims.y * z);
            cur++;
            x++;
            if (x > range.hx) { x = range.lx; y++; }
            if (y > range.hy) { y = range.ly; z++; }
        }
    }

    int mask = __ballot(blocked);
    while (mask) {
        int bit = __ffs(mask) - 1;
        mask &= ~(1 << bit);

        int warp_start = __shfl(start, bit);
        int warp_end   = __shfl(end,   bit);
        int warp_id    = threadIdx.x - __shfl(threadIdx.x, 0);

        int lx = __shfl(range.lx, bit);
        int ly = __shfl(range.ly, bit);
        int lz = __shfl(range.lz, bit);
        int hx = __shfl(range.hx, bit);
        int hy = __shfl(range.hy, bit);
        int r  = __shfl(id, bit);

        int sx = hx - lx + 1;
        int sy = hy - ly + 1;

        // Split the work on all the threads of the warp
        for (int i = warp_start + warp_id; i < warp_end; i += 32) {
            int k = i - warp_start;
            int x = lx + (k % sx);
            int y = ly + ((k / sx) % sy);
            int z = lz + (k / (sx * sy));
            new_ref_ids[i]  = r;
            new_cell_ids[i] = x + grid_dims.x * (y + grid_dims.y * z);
        }
    }
}

/// Filter out references that do not intersect the cell they are in
template <typename Primitive>
__global__ void filter_refs(int* __restrict__ cell_ids,
                            int* __restrict__ ref_ids,
                            const Primitive* __restrict__ prims,
                            const Cell* __restrict__ cells,
                            int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    auto cell = load_cell(cells + cell_ids[id]);
    auto prim = load_prim(prims +  ref_ids[id]);
    auto bbox = BBox(grid_bbox.min + vec3(cell.min) * cell_size,
                     grid_bbox.min + vec3(cell.max) * cell_size);
    bool intersect = intersect_prim_cell(prim, bbox);
    if (!intersect) {
        cell_ids[id] = -1;
        ref_ids[id]  = -1;
    }
}

/// Compute a mask for each reference which determines which sub-cell is intersected
template <typename Primitive>
__global__ void compute_split_masks(const int* __restrict__ cell_ids,
                                    const int* __restrict__ ref_ids,
                                    const Primitive* __restrict__ prims,
                                    const Cell* __restrict__ cells,
                                    int* split_masks,
                                    int num_split) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_split) return;

    auto cell_id = cell_ids[id];
    if (cell_id < 0) {
        split_masks[id] = 0;
        return;
    }
    auto ref  =  ref_ids[id];
    auto cell = load_cell(cells + cell_id);
    auto prim = load_prim(prims + ref);

    auto cell_min = grid_bbox.min + cell_size * vec3(cell.min);
    auto cell_max = grid_bbox.min + cell_size * vec3(cell.max);
    auto middle = (cell_min + cell_max) * 0.5f;

    int mask = 0xFF;

    // Optimization: Test against half spaces first
    auto ref_bb = prim.bbox();
    if (ref_bb.min.x > cell_max.x ||
        ref_bb.max.x < cell_min.x) mask  = 0;
    if (ref_bb.min.x >   middle.x) mask &= 0xAA;
    if (ref_bb.max.x <   middle.x) mask &= 0x55;
    if (ref_bb.min.y > cell_max.y ||
        ref_bb.max.y < cell_min.y) mask  = 0;
    if (ref_bb.min.y >   middle.y) mask &= 0xCC;
    if (ref_bb.max.y <   middle.y) mask &= 0x33;
    if (ref_bb.min.z > cell_max.z ||
        ref_bb.max.z < cell_min.z) mask  = 0;
    if (ref_bb.min.z >   middle.z) mask &= 0xF0;
    if (ref_bb.max.z <   middle.z) mask &= 0x0F;

    for (int i = __ffs(mask) - 1;;) {
        auto bbox = BBox(vec3(i & 1 ? middle.x : cell_min.x,
                              i & 2 ? middle.y : cell_min.y,
                              i & 4 ? middle.z : cell_min.z),
                         vec3(i & 1 ? cell_max.x : middle.x,
                              i & 2 ? cell_max.y : middle.y,
                              i & 4 ? cell_max.z : middle.z));
        if (!intersect_prim_cell(prim, bbox)) mask &= ~(1 << i);

        // Skip non-intersected children
        int skip = __ffs(mask >> (i + 1));
        if (skip == 0) break;
        i += 1 + (skip - 1);
    }

    split_masks[id] = mask;
}

/// Split references according to the given array of split masks
__global__ void split_refs(const int* __restrict__ cell_ids,
                           const int* __restrict__ ref_ids,
                           const Entry* __restrict__ entries,
                           const int* __restrict__ split_masks,
                           const int* __restrict__ start_split,
                           int* __restrict__ new_cell_ids,
                           int* __restrict__ new_ref_ids,
                           int num_split) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_split) return;

    auto cell_id = cell_ids[id];
    auto ref = ref_ids[id];
    auto begin = entries[cell_id].begin;

    auto mask  = split_masks[id];
    auto start = start_split[id];
    while (mask) {
        int child_id = __ffs(mask) - 1;
        mask &= ~(1 << child_id);
        new_ref_ids [start] = ref;
        new_cell_ids[start] = begin + child_id;
        start++;
    }
}

/// Compute the number of references per cell using atomics
__global__ void count_refs_per_cell(const int* __restrict__ cell_ids,
                                    int* __restrict__ refs_per_cell,
                                    int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;
    int cell_id = cell_ids[id];
    if (cell_id >= 0) atomicAdd(refs_per_cell + cell_id, 1);
}

/// Compute the logarithm of the sub-level resolution for top-level cells
__global__ void compute_log_dims(const int* __restrict__ refs_per_cell,
                                 int* __restrict__ log_dims,
                                 float snd_density,
                                 int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto extents = grid_bbox.extents() / vec3(grid_dims);
    auto bbox = BBox(vec3(0, 0, 0), extents);
    auto dims = compute_grid_dims(bbox, refs_per_cell[id], snd_density);
    auto max_dim = max(dims.x, max(dims.y, dims.z));
    auto log_dim = 31 - __clz(max_dim);
    log_dim = (1 << log_dim) < max_dim ? log_dim + 1 : log_dim;
    log_dims[id] = log_dim;
}

/// Update the logarithm of the sub-level resolution for top-level cells (after a new subdivision level)
__global__ void update_log_dims(int* __restrict__ log_dims, int num_top_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_top_cells) return;

    log_dims[id] = max(0, log_dims[id] - 1);
}

/// Given a position on the virtual grid, return the corresponding top-level cell index
__device__ __forceinline__ int top_level_cell(ivec3 pos) {
    return (pos.x >> grid_shift) + grid_dims.x * ((pos.y >> grid_shift) + grid_dims.y * (pos.z >> grid_shift));
}

/// Count the (sub-)dimensions of each cell, based on the array of references
__global__ void compute_dims(const int*  __restrict__ cell_ids,
                             const Cell* __restrict__ cells,
                             const int*  __restrict__ log_dims,
                             Entry* __restrict__ entries,
                             int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    auto cell_id = cell_ids[id];
    if (cell_id < 0) return;

    auto cell_min = load_cell_min(cells + cell_id);
    auto top_cell_id = top_level_cell(cell_min);
    auto log_dim = log_dims[top_cell_id];

    entries[cell_id] = make_entry(min(log_dim, 1), 0);
}

/// Mark references that are kept so that they can be moved to the beginning of the array
__global__ void mark_kept_refs(const int*   __restrict__ cell_ids,
                               const Entry* __restrict__ entries,
                               int* kept_flags,
                               int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    auto cell_id = cell_ids[id];
    kept_flags[id] = (cell_id >= 0) && (entries[cell_id].log_dim == 0);
}

/// Update the entries for the one level before the current one
__global__ void update_entries(const int* __restrict__ start_cell,
                               Entry* __restrict__ entries,
                               int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto start = start_cell[id];
    auto entry = entries[id];

    // If the cell is subdivided, write the first sub-cell index into the current entry
    entry.begin = entry.log_dim != 0 ? start : id;
    entries[id] = entry;
}

/// Generate cells for the top level
__global__ void emit_top_cells(Cell* __restrict__ new_cells, int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    int x = id % grid_dims.x;
    int y = (id / grid_dims.x) % grid_dims.y;
    int z = id / (grid_dims.x * grid_dims.y);
    int inc = 1 << grid_shift;

    x <<= grid_shift;
    y <<= grid_shift;
    z <<= grid_shift;

    Cell cell;
    cell.min = ivec3(x, y, z);
    cell.max = ivec3(x + inc, y + inc, z + inc);
    cell.begin = 0;
    cell.end   = 0;
    store_cell(new_cells + id, cell);
}

/// Generate new cells based on the previous level
__global__ void emit_new_cells(const Entry* __restrict__ entries,
                               const Cell* __restrict__ cells,
                               Cell* __restrict__ new_cells,
                               int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto entry = entries[id];
    auto log_dim = entry.log_dim;
    if (log_dim == 0) return;

    auto start = entry.begin;
    auto cell = load_cell(cells + id);
    int min_x = cell.min.x;
    int min_y = cell.min.y;
    int min_z = cell.min.z;
    int inc = (cell.max.x - cell.min.x) >> 1;

    for (int i = 0; i < 8; i++) {
        int x = min_x + (i & 1) * inc;
        int y = min_y + ((i >> 1) & 1) * inc;
        int z = min_z + (i >> 2) * inc;

        cell.min = ivec3(x, y, z);
        cell.max = ivec3(x + inc, y + inc, z + inc);
        cell.begin = 0;
        cell.end   = 0;
        store_cell(new_cells + start + i, cell);
    }
}

/// Copy the references with an offset, different for each level
__global__ void copy_refs(const int* __restrict__ cell_ids,
                          int* __restrict__ new_cell_ids,
                          int cell_off,
                          int num_kept) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_kept) return;

    new_cell_ids[id] = cell_ids[id] + cell_off;
}

/// Mark the cells that are used as 'kept'
__global__ void mark_kept_cells(const Entry* __restrict__ entries,
                                int* kept_cells,
                                int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    kept_cells[id] = entries[id].log_dim == 0;
}

/// Copy only the cells that are kept to another array of cells
__global__ void copy_cells(const Cell* __restrict__ cells,
                           const int* __restrict__ start_cell,
                           Cell* new_cells,
                           int cell_off,
                           int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto cell = load_cell(cells + id);
    auto start = start_cell[cell_off + id + 0];
    auto end   = start_cell[cell_off + id + 1];
    if (start < end) store_cell(new_cells + start, cell);
}

/// Copy the voxel map entries and remap kept cells to their correct indices
__global__ void copy_entries(const Entry* __restrict__ entries,
                             const int* __restrict__ start_cell,
                             Entry* __restrict__ new_entries,
                             int cell_off,
                             int next_level_off,
                             int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto entry = entries[id];
    if (entry.log_dim == 0) {
        // Points to a cell
        entry.begin = start_cell[cell_off + entry.begin];
    } else {
        // Points to another entry in the next level
        entry.begin += next_level_off;
    }
    new_entries[id] = entry;
}

/// Remap references so that they map to the correct cells
__global__ void remap_refs(int* __restrict__ cell_ids,
                           const int* __restrict__ start_cell,
                           int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    cell_ids[id] = start_cell[cell_ids[id]];
}

/// Sets the cell ranges once the references are sorted by cell
__global__ void compute_cell_ranges(const int* cell_ids, Cell* cells, int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    int cell_id = cell_ids[id + 0];
    if (id >= num_refs - 1) {
        cells[cell_id].end = id + 1;
        return;
    }
    int next_id = cell_ids[id + 1];

    if (cell_id != next_id) {
        cells[cell_id].end   = id + 1;
        cells[next_id].begin = id + 1;
    }
}

template <typename Primitive>
void first_build_iter(MemManager& mem, float snd_density,
                      const Primitive* prims, int num_prims,
                      const BBox* bboxes, const BBox& grid_bb, const ivec3& dims,
                      int*& log_dims, int& grid_shift, std::vector<Level>& levels) {
    Parallel par(mem);

    int num_top_cells = dims.x * dims.y * dims.z;

    // Emission of the references in 4 passes: count new refs + scan + emission + filtering
    auto start_emit     = mem.alloc<int>(num_prims + 1);
    auto new_ref_counts = mem.alloc<int>(num_prims + 1);
    auto refs_per_cell  = mem.alloc<int>(num_top_cells);
    log_dims            = mem.alloc<int>(num_top_cells + 1);
    count_new_refs<<<round_div(num_prims, 64), 64>>>(bboxes, new_ref_counts, num_prims);
    DEBUG_SYNC();

    int num_new_refs = par.scan(new_ref_counts, num_prims + 1, start_emit);
    mem.free(new_ref_counts);

    auto new_ref_ids  = mem.alloc<int>(2 * num_new_refs);
    auto new_cell_ids = new_ref_ids + num_new_refs;
    emit_new_refs<<<round_div(num_prims, 64), 64>>>(bboxes, start_emit, new_ref_ids, new_cell_ids, num_prims);
    DEBUG_SYNC();

    mem.free(start_emit);

    // Compute the number of references per cell
    mem.zero(refs_per_cell, num_top_cells);
    count_refs_per_cell<<<round_div(num_new_refs, 64), 64>>>(new_cell_ids, refs_per_cell, num_new_refs);
    DEBUG_SYNC();

    // Compute an independent resolution in each of the top-level cells
    compute_log_dims<<<round_div(num_top_cells, 64), 64>>>(refs_per_cell, log_dims, snd_density, num_top_cells);
    DEBUG_SYNC();
    mem.free(refs_per_cell);

    // Find the maximum sub-level resolution
    grid_shift = par.reduce(log_dims, num_top_cells, log_dims + num_top_cells, [] __device__ (int a, int b) { return max(a, b); });
    auto cell_size = grid_bb.extents() / vec3(dims << grid_shift);

    set_global(hagrid::grid_shift, grid_shift);
    set_global(hagrid::cell_size,  cell_size);

    // Emission of the new cells
    auto new_cells   = mem.alloc<Cell >(num_top_cells + 0);
    auto new_entries = mem.alloc<Entry>(num_top_cells + 1);
    emit_top_cells<<<round_div(num_top_cells, 64), 64>>>(new_cells, num_top_cells);
    DEBUG_SYNC();
    mem.zero(new_entries, num_top_cells + 1);

    // Filter out the references that do not intersect the cell they are in
    filter_refs<<<round_div(num_new_refs, 64), 64>>>(new_cell_ids, new_ref_ids, prims, new_cells, num_new_refs);

    levels.emplace_back(new_ref_ids, new_cell_ids, num_new_refs, num_new_refs, new_cells, new_entries, num_top_cells);
}

template <typename Primitive>
bool build_iter(MemManager& mem,
                const Primitive* prims, int num_prims,
                const ivec3& dims, int* log_dims,
                std::vector<Level>& levels) {
    Parallel par(mem);

    int* cell_ids  = levels.back().cell_ids;
    int* ref_ids   = levels.back().ref_ids;
    Cell* cells    = levels.back().cells;
    Entry* entries = levels.back().entries;

    int num_top_cells = dims.x * dims.y * dims.z;
    int num_refs  = levels.back().num_refs;
    int num_cells = levels.back().num_cells;

    int cur_level  = levels.size();

    auto kept_flags = mem.alloc<int>(num_refs + 1);

    // Find out which cell will be split based on whether it is empty or not and the maximum depth
    compute_dims<<<round_div(num_refs, 64), 64>>>(cell_ids, cells, log_dims, entries, num_refs);
    DEBUG_SYNC();
    update_log_dims<<<round_div(num_top_cells, 64), 64>>>(log_dims, num_top_cells);
    DEBUG_SYNC();
    mark_kept_refs<<<round_div(num_refs, 64), 64>>>(cell_ids, entries, kept_flags, num_refs);
    DEBUG_SYNC();

    // Store the sub-cells starting index in the entries
    auto start_cell = mem.alloc<int>(num_cells + 1);
    int num_new_cells = par.scan(par.transform(entries, [] __device__ (Entry e) {
        return e.log_dim == 0 ? 0 : 8;
    }), num_cells + 1, start_cell);
    update_entries<<<round_div(num_cells, 64), 64>>>(start_cell, entries, num_cells);
    DEBUG_SYNC();

    mem.free(start_cell);

    // Partition the set of cells into the sets of those which will be split and those which won't
    auto tmp_ref_ids  = mem.alloc<int>(num_refs * 2);
    auto tmp_cell_ids = tmp_ref_ids + num_refs;
    int num_sel_refs  = par.partition(ref_ids,  tmp_ref_ids,  num_refs, kept_flags);
    int num_sel_cells = par.partition(cell_ids, tmp_cell_ids, num_refs, kept_flags);
    assert(num_sel_refs == num_sel_cells);

    mem.free(kept_flags);

    std::swap(tmp_ref_ids, ref_ids);
    std::swap(tmp_cell_ids, cell_ids);
    mem.free(tmp_ref_ids);

    int num_kept = num_sel_refs;
    levels.back().ref_ids  = ref_ids;
    levels.back().cell_ids = cell_ids;
    levels.back().num_kept = num_kept;

    if (num_new_cells == 0) {
        // Exit here because no new reference will be emitted
        mem.free(log_dims);
        return false;
    }

    int num_split = num_refs - num_kept;

    // Split the references
    auto split_masks = mem.alloc<int>(num_split + 1);
    auto start_split = mem.alloc<int>(num_split + 1);
    compute_split_masks<<<round_div(num_split, 64), 64>>>(cell_ids + num_kept, ref_ids + num_kept, prims, cells, split_masks, num_split);
    DEBUG_SYNC();

    int num_new_refs = par.scan(par.transform(split_masks, [] __device__ (int mask) {
        return __popc(mask);
    }), num_split + 1, start_split);
    assert(num_new_refs <= 8 * num_split);

    auto new_ref_ids = mem.alloc<int>(num_new_refs * 2);
    auto new_cell_ids = new_ref_ids + num_new_refs;
    split_refs<<<round_div(num_split, 64), 64>>>(cell_ids + num_kept, ref_ids + num_kept, entries, split_masks, start_split, new_cell_ids, new_ref_ids, num_split);
    DEBUG_SYNC();

    mem.free(split_masks);
    mem.free(start_split);

    // Emission of the new cells
    auto new_cells   = mem.alloc<Cell >(num_new_cells + 0);
    auto new_entries = mem.alloc<Entry>(num_new_cells + 1);
    emit_new_cells<<<round_div(num_cells, 64), 64>>>(entries, cells, new_cells, num_cells);
    DEBUG_SYNC();
    mem.zero(new_entries, num_new_cells + 1);

    levels.emplace_back(new_ref_ids, new_cell_ids, num_new_refs, num_new_refs, new_cells, new_entries, num_new_cells);
    return true;
}

void concat_levels(MemManager& mem, std::vector<Level>& levels, Grid& grid) {
    Parallel par(mem);
    int num_levels = levels.size();

    // Start with references
    int total_refs = 0;
    int total_cells = 0;
    for (auto& level : levels) {
        total_refs  += level.num_kept;
        total_cells += level.num_cells;
    }

    // Copy primitive references as-is
    auto ref_ids  = mem.alloc<int>(total_refs);
    auto cell_ids = mem.alloc<int>(total_refs);
    for (int i = 0, off = 0; i < num_levels; off += levels[i].num_kept, i++) {
        mem.copy<Copy::DEV_TO_DEV>(ref_ids + off, levels[i].ref_ids, levels[i].num_kept);
    }
    // Copy the cell indices with an offset
    for (int i = 0, off = 0, cell_off = 0; i < num_levels; off += levels[i].num_kept, cell_off += levels[i].num_cells, i++) {
        int num_kept = levels[i].num_kept;
        if (num_kept) {
            copy_refs<<<round_div(num_kept, 64), 64>>>(levels[i].cell_ids, cell_ids + off, cell_off, num_kept);
            DEBUG_SYNC();
        }
        mem.free(levels[i].ref_ids);
    }

    // Mark the cells at the leaves of the structure as kept
    auto kept_cells = mem.alloc<int>(total_cells + 1);
    for (int i = 0, cell_off = 0; i < num_levels; cell_off += levels[i].num_cells, i++) {
        int num_cells = levels[i].num_cells;
        mark_kept_cells<<<round_div(num_cells, 64), 64>>>(levels[i].entries, kept_cells + cell_off, num_cells);
        DEBUG_SYNC();
    }

    // Compute the insertion position of each cell
    auto start_cell = mem.alloc<int>(total_cells + 1);
    int new_total_cells = par.scan(kept_cells, total_cells + 1, start_cell);
    mem.free(kept_cells);

    // Allocate new cells, and copy only the cells that are kept
    auto cells = mem.alloc<Cell>(new_total_cells);
    for (int i = 0, cell_off = 0; i < num_levels; cell_off += levels[i].num_cells, i++) {
        int num_cells = levels[i].num_cells;
        copy_cells<<<round_div(num_cells, 64), 64>>>(levels[i].cells, start_cell, cells, cell_off, num_cells);
        DEBUG_SYNC();
        mem.free(levels[i].cells);
    }

    auto entries = mem.alloc<Entry>(total_cells);
    for (int i = 0, off = 0; i < num_levels; off += levels[i].num_cells, i++) {
        int num_cells = levels[i].num_cells;
        int next_level_off = off + num_cells;
        copy_entries<<<round_div(num_cells, 64), 64>>>(levels[i].entries, start_cell, entries + off, off, next_level_off, num_cells);
        DEBUG_SYNC();
        mem.free(levels[i].entries);
    }

    // Remap the cell indices in the references (which currently map to incorrect cells)
    remap_refs<<<round_div(total_refs, 64), 64>>>(cell_ids, start_cell, total_refs);
    DEBUG_SYNC();

    mem.free(start_cell);

    // Sort the references by cell (re-use old slots whenever possible)
    auto tmp_ref_ids  = mem.alloc<int>(total_refs);
    auto tmp_cell_ids = mem.alloc<int>(total_refs);
    auto new_ref_ids  = tmp_ref_ids;
    auto new_cell_ids = tmp_cell_ids;
    par.sort_pairs(cell_ids, ref_ids, new_cell_ids, new_ref_ids, total_refs, ilog2(new_total_cells));
    if (ref_ids  != new_ref_ids)  std::swap(ref_ids,  tmp_ref_ids);
    if (cell_ids != new_cell_ids) std::swap(cell_ids, tmp_cell_ids);
    mem.free(tmp_ref_ids);
    mem.free(tmp_cell_ids);

    // Compute the ranges of references for each cell
    compute_cell_ranges<<<round_div(total_refs, 64), 64>>>(cell_ids, cells, total_refs);
    DEBUG_SYNC();

    mem.free(cell_ids);

    grid.entries = entries;
    grid.ref_ids = ref_ids;
    grid.cells   = cells;
    grid.shift   = levels.size() - 1;
    grid.num_cells   = new_total_cells;
    grid.num_entries = total_cells;
    grid.num_refs    = total_refs;

    grid.offsets.resize(levels.size());
    for (int i = 0, off = 0; i < levels.size(); i++) {
        off += levels[i].num_cells;
        grid.offsets[i] = off;
    }
}

template <typename Primitive>
void build(MemManager& mem, const Primitive* prims, int num_prims, Grid& grid, float top_density, float snd_density) {
    Parallel par(mem);

    // Allocate a bounding box for each primitive + one for the global bounding box
    auto bboxes = mem.alloc<BBox>(num_prims + 1);

    compute_bboxes<<<round_div(num_prims, 64), 64>>>(prims, bboxes, num_prims);
    auto grid_bb = par.reduce(bboxes, num_prims, bboxes + num_prims,
        [] __device__ (BBox a, const BBox& b) { return a.extend(b); }, BBox::empty());
    auto dims = compute_grid_dims(grid_bb, num_prims, top_density);
    // Round to the next multiple of 2 on each dimension (in order to align the memory)
    dims.x = dims.x % 2 ? dims.x + 1 : dims.x;
    dims.y = dims.y % 2 ? dims.y + 1 : dims.y;
    dims.z = dims.z % 2 ? dims.z + 1 : dims.z;

    // Slightly enlarge the bounding box of the grid
    auto extents = grid_bb.extents();
    grid_bb.min -= extents * 0.001f;
    grid_bb.max += extents * 0.001f;

    set_global(hagrid::grid_dims, dims);
    set_global(hagrid::grid_bbox, grid_bb);

    int* log_dims = nullptr;
    int grid_shift = 0;
    std::vector<Level> levels;

    // Build top level
    first_build_iter(mem, snd_density, prims, num_prims, bboxes, grid_bb, dims, log_dims, grid_shift, levels);

    mem.free(bboxes);

    int iter = 1;
    while (build_iter(mem, prims, num_prims, dims, log_dims, levels)) iter++;

    concat_levels(mem, levels, grid);
    grid.small_cells = nullptr;
    grid.dims  = dims;
    grid.bbox  = grid_bb;
}

void build_grid(MemManager& mem, const Tri* tris, int num_tris, Grid& grid, float top_density, float snd_density) { build(mem, tris, num_tris, grid, top_density, snd_density); }

} // namespace hagrid

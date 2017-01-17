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
static __constant__ int   level_shift;

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
template <bool first_iter>
__global__ void count_new_refs(const Entry* __restrict__ entries,
                               const Cell*  __restrict__ cells,
                               const BBox*  __restrict__ bboxes,
                               const int*  __restrict__ ref_ids,
                               const int*  __restrict__ cell_ids,
                               int*        __restrict__ counts,
                               int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    auto ref = first_iter ? id : ref_ids[id];
    if (!first_iter && ref < 0) {
        // Deal with invalid references here
        counts[id] = 0;
        return;
    }

    BBox bbox;
    ivec3 dims;
    if (!first_iter) {
        auto cell_id = cell_ids[id];
        auto entry = entries[cell_id];
        auto cell = load_cell(cells + cell_id);
        bbox = BBox(grid_bbox.min + vec3(cell.min) * cell_size,
                    grid_bbox.min + vec3(cell.max) * cell_size);
        dims = ivec3(1 << entry.log_dim);
    } else {
        dims = grid_dims;
        bbox = grid_bbox;
    }

    auto ref_bb = load_bbox(bboxes + ref);
    auto range  = compute_range(dims, bbox, ref_bb);
    counts[id]  = max(0, range.size());
}

/// Emit the new references by inserting existing ones into the sub-levels
template <bool first_iter, typename Primitive>
__global__ void emit_new_refs(const Entry* __restrict__ entries,
                              const Cell*  __restrict__ cells,
                              const Primitive* __restrict__ prims,
                              const BBox* __restrict__ bboxes,
                              const int* __restrict__ start_emit,
                              const int* __restrict__ ref_ids,
                              const int* __restrict__ cell_ids,
                              int* __restrict__ new_ref_ids,
                              int* __restrict__ new_cell_ids,
                              int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    auto ref = first_iter ? id : ref_ids[id];
    if (!first_iter && ref < 0) return;

    BBox bbox;
    ivec3 dims;
    int cur_cell;
    if (!first_iter) {
        auto cell_id = cell_ids[id];
        auto entry = entries[cell_id];
        auto cell = load_cell(cells + cell_id);
        cur_cell = entry.begin;
        bbox = BBox(grid_bbox.min + vec3(cell.min) * cell_size,
                    grid_bbox.min + vec3(cell.max) * cell_size);
        dims = ivec3(1 << entry.log_dim);
    } else {
        cur_cell = 0;
        dims = grid_dims;
        bbox = grid_bbox;
    }

    // Emit references for intersected cells
    auto ref_bb = load_bbox(bboxes + ref);
    auto range  = compute_range(dims, bbox, ref_bb);
    auto sub_size = bbox.extents() / vec3(dims);
    auto prim = load_prim(prims + ref);
    auto start = start_emit[id + 0];
    auto end   = start_emit[id + 1];
    int x = range.lx;
    int y = range.ly;
    int z = range.lz;
    while (start < end) {
        auto sub_bb = BBox(bbox.min + sub_size * vec3(x + 0, y + 0, z + 0),
                           bbox.min + sub_size * vec3(x + 1, y + 1, z + 1));
        bool intersect = intersect_prim_box(prim, sub_bb);
        new_ref_ids [start] = !intersect ? -1 : ref;
        new_cell_ids[start] = !intersect ? -1 : cur_cell + x + dims.x * (y + dims.y * z);
        start++;
        x++;
        if (x > range.hx) { x = range.lx; y++; }
        if (y > range.hy) { y = range.ly; z++; }
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

    log_dims[id] = max(0, log_dims[id] - level_shift);
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

    entries[cell_id] = Entry(min(log_dim, level_shift), 0);
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

/// Generate new cells based on the previous level
template <bool first_iter>
__global__ void emit_new_cells(const Entry* __restrict__ entries,
                               const Cell* __restrict__ cells,
                               Cell* __restrict__ new_cells,
                               int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    if (first_iter) {
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
    } else {
        auto entry = entries[id];
        auto log_dim = entry.log_dim;
        if (log_dim == 0) return;

        auto start = entry.begin;
        auto n = 1 << (3 * log_dim);

        auto cell = load_cell(cells + id);
        int min_x = cell.min.x;
        int min_y = cell.min.y;
        int min_z = cell.min.z;
        int inc = (cell.max.x - cell.min.x) >> log_dim;
        int mask = (1 << log_dim) - 1;

        for (int i = 0; i < n; i++) {
            int x = min_x + (i & mask) * inc;
            int y = min_y + ((i >> log_dim) & mask) * inc;
            int z = min_z + (i >> (2 * log_dim)) * inc;

            cell.min = ivec3(x, y, z);
            cell.max = ivec3(x + inc, y + inc, z + inc);
            cell.begin = 0;
            cell.end   = 0;
            store_cell(new_cells + start + i, cell);
        }
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

/// Sets the cell ranges after run length encoding on references
__global__ void compute_cell_ranges(const int* __restrict__ cell_ids,
                                    Cell* cells,
                                    int num_refs) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_refs) return;

    bool last = id >= num_refs - 1;
    int cell_id = cell_ids[id + 0];
    int next_id = last ? -1 : cell_ids[id + 1];

    if (cell_id != next_id) cells[cell_id].end   = id;
    if (!last)              cells[next_id].begin = id;
}

template <bool first_iter, typename Primitive>
bool build_iter(MemManager& mem, const BuildParams& params,
                const Primitive* prims, int num_prims,
                const BBox* bboxes, const BBox& grid_bb, const ivec3& dims,
                int*& log_dims, int& grid_shift, std::vector<Level>& levels) {
    Parallel par(mem);

    int* cell_ids  = first_iter ? nullptr : levels.back().cell_ids;
    int* ref_ids   = first_iter ? nullptr : levels.back().ref_ids;
    Cell* cells    = first_iter ? nullptr : levels.back().cells;
    Entry* entries = first_iter ? nullptr : levels.back().entries;

    int num_top_cells = dims.x * dims.y * dims.z;
    int num_refs  = first_iter ? num_prims     : levels.back().num_refs;
    int num_cells = first_iter ? num_top_cells : levels.back().num_cells;

    int cur_level = levels.size();
    int prev_level = cur_level - 1;

    int num_new_cells = num_top_cells;
    int* start_cell = nullptr;
    int num_kept  = 0;
    if (!first_iter) {
        auto kept_flags = mem.alloc<int>(Slot::KEPT_FLAGS, num_refs);

        // Find out which cell will be split based on whether it is empty or not and the maximum depth
        compute_dims<<<round_div(num_refs, 64), 64>>>(cell_ids, cells, log_dims, entries, num_refs);
        DEBUG_SYNC();
        update_log_dims<<<round_div(num_top_cells, 64), 64>>>(log_dims, num_top_cells);
        DEBUG_SYNC();
        mark_kept_refs<<<round_div(num_refs, 64), 64>>>(cell_ids, entries, kept_flags, num_refs);
        DEBUG_SYNC();

        // Store the sub-cells starting index in the entries
        start_cell = mem.alloc<int>(Slot::START_CELL, num_cells + 1);
        num_new_cells = par.scan(par.transform(entries, [] __device__ (Entry e) {
            return e.log_dim == 0 ? 0 : 1 << (e.log_dim * 3);
        }), num_cells + 1, start_cell);
        update_entries<<<round_div(num_cells, 64), 64>>>(start_cell, entries, num_cells);
        DEBUG_SYNC();

        mem.free(Slot::START_CELL);

        // Partition the set of cells into the sets of those which will be split and those which won't
        auto tmp_ref_ids  = mem.alloc<int>(Slot::ref_array(cur_level), num_refs * 2);
        auto tmp_cell_ids = tmp_ref_ids + num_refs;
        int num_sel_refs  = par.partition(ref_ids,  tmp_ref_ids,  num_refs, kept_flags);
        int num_sel_cells = par.partition(cell_ids, tmp_cell_ids, num_refs, kept_flags);

        assert(num_sel_refs == num_sel_cells);
        mem.free(Slot::KEPT_FLAGS);

        std::swap(tmp_ref_ids,  ref_ids);
        std::swap(tmp_cell_ids, cell_ids);
        mem.swap(Slot::ref_array(cur_level), Slot::ref_array(prev_level));
        mem.free(Slot::ref_array(cur_level));

        num_kept = num_sel_refs;
        levels.back().ref_ids  = ref_ids;
        levels.back().cell_ids = cell_ids;
        levels.back().num_kept = num_kept;
    }

    int num_split = num_refs - num_kept;

    // Emission of the references in 3 passes: count new refs + scan + emission 
    auto start_emit     = mem.alloc<int>(Slot::START_EMIT,     num_split + 1);
    auto new_ref_counts = mem.alloc<int>(Slot::NEW_REF_COUNTS, num_split + 1);
    count_new_refs<first_iter><<<round_div(num_refs, 64), 64>>>(entries, cells, bboxes, ref_ids + num_kept, cell_ids + num_kept, new_ref_counts, num_split);
    DEBUG_SYNC();

    int num_new_refs = par.scan(new_ref_counts, num_split + 1, start_emit);
    mem.free(Slot::NEW_REF_COUNTS);

    if (num_new_refs == 0) {
        // Exit here because no new reference will be emitted
        mem.free(Slot::START_EMIT);
        mem.free(Slot::LOG_DIMS);
        log_dims = nullptr;
        return false;
    }

    auto new_ref_ids  = mem.alloc<int>(Slot::ref_array(cur_level), 2 * num_new_refs);
    auto new_cell_ids = new_ref_ids + num_new_refs;
    emit_new_refs<first_iter><<<round_div(num_split, 64), 64>>>(entries, cells,
        prims, bboxes,
        start_emit,
        ref_ids + num_kept, cell_ids + num_kept,
        new_ref_ids, new_cell_ids,
        num_split);
    DEBUG_SYNC();

    mem.free(Slot::START_EMIT);

    if (first_iter) {
        auto refs_per_cell = mem.alloc<int>(Slot::REFS_PER_CELL, num_top_cells);
        mem.zero(refs_per_cell, num_top_cells);
        count_refs_per_cell<<<round_div(num_new_refs, 64), 64>>>(new_cell_ids, refs_per_cell, num_new_refs);
        DEBUG_SYNC();

        // Compute an independent resolution in each of the top-level cells
        log_dims = mem.alloc<int>(Slot::LOG_DIMS, num_top_cells + 1);
        compute_log_dims<<<round_div(num_top_cells, 64), 64>>>(refs_per_cell, log_dims, params.snd_density, num_top_cells);
        DEBUG_SYNC();
        mem.free(Slot::REFS_PER_CELL);

        // Find the maximum resolution
        grid_shift = par.reduce(log_dims, num_top_cells, log_dims + num_top_cells, [] __device__ (int a, int b) { return max(a, b); });
        auto cell_size = grid_bb.extents() / vec3(dims << grid_shift);

        set_global(hagrid::grid_shift, &grid_shift);
        set_global(hagrid::cell_size,  &cell_size);
    }

    // Emission of the new cells
    auto new_cells   = mem.alloc<Cell >(Slot::cell_array(cur_level),  num_new_cells + 0);
    auto new_entries = mem.alloc<Entry>(Slot::entry_array(cur_level), num_new_cells + 1);
    emit_new_cells<first_iter><<<round_div(num_cells, 64), 64>>>(entries, cells, new_cells, num_cells);
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
    auto ref_ids = mem.alloc<int>(Slot::ref_array(num_levels + 0), total_refs);
    for (int i = 0, off = 0; i < levels.size(); off += levels[i].num_kept, i++) {
        mem.copy<Copy::DEV_TO_DEV>(ref_ids + off, levels[i].ref_ids, levels[i].num_kept);
    }
    // Copy the cell indices with an offset
    auto cell_ids = mem.alloc<int>(Slot::ref_array(num_levels + 1), total_refs);
    for (int i = 0, off = 0, cell_off = 0; i < levels.size(); off += levels[i].num_kept, cell_off += levels[i].num_cells, i++) {
        int num_kept = levels[i].num_kept;
        copy_refs<<<round_div(num_kept, 64), 64>>>(levels[i].cell_ids, cell_ids + off, cell_off, num_kept);
        DEBUG_SYNC();
        mem.free(Slot::ref_array(i));
    }

    // Mark the cells at the leaves of the structure as kept
    auto kept_cells = mem.alloc<int>(Slot::KEPT_FLAGS, total_cells + 1);
    for (int i = 0, cell_off = 0; i < levels.size(); cell_off += levels[i].num_cells, i++) {
        int num_cells = levels[i].num_cells;
        mark_kept_cells<<<round_div(num_cells, 64), 64>>>(levels[i].entries, kept_cells + cell_off, num_cells);
        DEBUG_SYNC();
    }

    // Compute the insertion position of each cell
    auto start_cell = mem.alloc<int>(Slot::START_CELL, total_cells + 1);
    int new_total_cells = par.scan(kept_cells, total_cells + 1, start_cell);
    mem.free(Slot::KEPT_FLAGS);

    // Allocate new cells, and copy only the cells that are kept
    auto cells = mem.alloc<Cell>(Slot::cell_array(num_levels), new_total_cells);
    for (int i = 0, cell_off = 0; i < levels.size(); cell_off += levels[i].num_cells, i++) {
        int num_cells = levels[i].num_cells;
        copy_cells<<<round_div(num_cells, 64), 64>>>(levels[i].cells, start_cell, cells, cell_off, num_cells);
        DEBUG_SYNC();
        mem.free(Slot::cell_array(i));
    }

    auto entries = mem.alloc<Entry>(Slot::entry_array(num_levels), total_cells);
    for (int i = 0, off = 0; i < levels.size(); off += levels[i].num_cells, i++) {
        int num_cells = levels[i].num_cells;
        int next_level_off = off + num_cells;
        copy_entries<<<round_div(num_cells, 64), 64>>>(levels[i].entries, start_cell, entries + off, off, next_level_off, num_cells);
        DEBUG_SYNC();
        mem.free(Slot::entry_array(i));
    }

    // Remap the cell indices in the references (which currently map to incorrect cells)
    remap_refs<<<round_div(total_refs, 64), 64>>>(cell_ids, start_cell, total_refs);
    DEBUG_SYNC();

    mem.free(Slot::START_CELL);

    // Sort the references by cell
    auto tmp_ref_ids  = mem.alloc<int>(Slot::ref_array(num_levels + 2), total_refs);
    auto tmp_cell_ids = mem.alloc<int>(Slot::ref_array(num_levels + 3), total_refs); 
    par.sort_pairs(ref_ids, cell_ids, tmp_ref_ids, tmp_cell_ids, total_refs);
    if (ref_ids != tmp_ref_ids) {
        std::swap(tmp_ref_ids, ref_ids);
        mem.swap(Slot::ref_array(num_levels + 2), Slot::ref_array(num_levels + 0));
    }
    if (cell_ids != tmp_cell_ids) {
        std::swap(tmp_cell_ids, cell_ids);
        mem.swap(Slot::ref_array(num_levels + 3), Slot::ref_array(num_levels + 1));
    }
    mem.free(Slot::ref_array(num_levels + 2));
    mem.free(Slot::ref_array(num_levels + 3));

    // Compute the ranges of references for each cell
    compute_cell_ranges<<<round_div(total_refs, 64), 64>>>(cell_ids, cells, total_refs);
    DEBUG_SYNC();

    mem.free(Slot::ref_array(num_levels + 1));

    grid.entries = entries;
    grid.ref_ids = ref_ids;
    grid.cells   = cells;
    grid.num_levels  = levels.size();
    grid.num_cells   = new_total_cells;
    grid.num_entries = total_cells;
    grid.num_refs    = total_refs;
}

template <typename Primitive>
void build(MemManager& mem, const BuildParams& params, const Primitive* prims, int num_prims, Grid& grid) {
    assert(params.valid());
    Parallel par(mem);

    // Allocate a bounding box for each primitive + one for the global bounding box
    auto bboxes = mem.alloc<BBox>(Slot::BBOXES, num_prims + 1);

    compute_bboxes<<<round_div(num_prims, 64), 64>>>(prims, bboxes, num_prims);
    auto grid_bb = par.reduce(bboxes, num_prims, bboxes + num_prims,
        [] __device__ (BBox a, const BBox& b) { return a.extend(b); }, BBox::empty());
    auto dims = compute_grid_dims(grid_bb, num_prims, params.top_density);
    int level_shift = params.level_shift;

    // Slightly enlarge the bounding box of the grid
    auto extents = grid_bb.extents();
    grid_bb.min -= extents * 0.001f;
    grid_bb.max += extents * 0.001f;

    set_global(hagrid::level_shift, &level_shift);
    set_global(hagrid::grid_dims, &dims);
    set_global(hagrid::grid_bbox, &grid_bb);

    int* log_dims = nullptr;
    int grid_shift = 0;
    std::vector<Level> levels;

    // Build top level
    build_iter<true>(mem, params, prims, num_prims, bboxes, grid_bb, dims, log_dims, grid_shift, levels);

    int iter = 1;
    while (build_iter<false>(mem, params, prims, num_prims, bboxes, grid_bb, dims, log_dims, grid_shift, levels)) iter++;
    mem.free(Slot::BBOXES);

    concat_levels(mem, levels, grid);
    grid.top_dims = dims;
    grid.shift    = grid_shift;
    grid.bbox     = grid_bb;
}

void build_grid(MemManager& mem, const BuildParams& params, const Tri* tris, int num_tris, Grid& grid) { build(mem, params, tris, num_tris, grid); }

} // namespace hagrid

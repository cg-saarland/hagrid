#include "parallel.cuh"
#include "build.h"

namespace hagrid {

__global__ void count_sentinel_refs(const Cell* cells, int* ref_counts, int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto cell = load_cell(cells + id);
    auto count = cell.end - cell.begin;
    ref_counts[id] = count > 0 ? count + 1 : 0;
}

__global__ void emit_small_cells(const Cell* cells,
                                 SmallCell* small_cells,
                                 int* __restrict__ refs,
                                 int* __restrict__ ref_scan,
                                 int* __restrict__ sentinel_refs,
                                 int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    auto cell = load_cell(cells + id);
    int first = ref_scan[id];
    int count = cell.end - cell.begin;

    SmallCell small_cell(usvec3(cell.min), usvec3(cell.max), count > 0 ? first : -1);
    store_cell(small_cells + id, small_cell);
   
    if (count > 0) {
        for (int i = 0; i < count; i++)
            sentinel_refs[first + i] = refs[cell.begin + i];
        sentinel_refs[first + count] = -1;
    }
}

bool compress_grid(MemManager& mem, Grid& grid) {
    auto dims = grid.dims << grid.shift;
    // Compression cannot work if the dimensions cannot fit into 16-bit indices
    if (dims.x >= (1 << 16) ||
        dims.y >= (1 << 16) ||
        dims.z >= (1 << 16))
        return false;

    Parallel par(mem);
    auto ref_counts  = mem.alloc<int>(grid.num_cells + 1);
    auto ref_scan    = mem.alloc<int>(grid.num_cells + 1);
    auto small_cells = mem.alloc<SmallCell>(grid.num_cells);
    count_sentinel_refs<<<round_div(grid.num_cells, 64), 64>>>(grid.cells, ref_counts, grid.num_cells);
    auto num_sentinel_refs = par.scan(ref_counts, grid.num_cells + 1, ref_scan);
    auto sentinel_refs = mem.alloc<int>(num_sentinel_refs);
    emit_small_cells<<<round_div(grid.num_cells, 64), 64>>>(grid.cells, small_cells, grid.ref_ids, ref_scan, sentinel_refs, grid.num_cells);
    grid.small_cells = small_cells;
    mem.free(grid.cells);
    mem.free(grid.ref_ids);
    mem.free(ref_counts);
    mem.free(ref_scan);
    grid.cells = nullptr;
    grid.ref_ids = sentinel_refs;
    grid.num_refs = num_sentinel_refs;
    return true;
}

} // namespace hagrid

#include "build.h"

namespace hagrid {

static constexpr int flat_levels = Entry::LOG_DIM_BITS + 1;
static __device__ int collapsed_count;

/// Collapses sub-entries that map to the same cell/sub-sub-entry
__global__ void collapse_entries(Entry* entries, int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    auto entry = entries[id];
    if (entry.log_dim) {
        auto ptr = (int4*)(entries + entry.begin);
        auto ptr0 = ptr[0];
        if (ptr0.x == ptr0.y &&
            ptr0.x == ptr0.z &&
            ptr0.x == ptr0.w) {
            auto ptr1 = ptr[1];
            if (ptr0.x == ptr1.x &&
                ptr1.x == ptr1.y &&
                ptr1.x == ptr1.z &&
                ptr1.x == ptr1.w) {
                entries[id] = entries[entry.begin];
                atomicAdd(&collapsed_count, 1);
            }
        }
    }
}

/// Computes the depth of each entry
__global__ void compute_depths(Entry* entries, int* depths, int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    // Traverse all sub-entries to get the depth
    auto entry = entries[id];
    int max_depth = 0;

    if (entry.log_dim) {
        Entry stack[flat_levels + 1];
        Entry top = entry;
        int morton = 0;
        int depth = 0;
        while (true) {
            if (top.log_dim) {
                int child = morton & 7;
                stack[depth++] = child < 7 ? top : make_entry(0, 0);
                morton = (morton + (child < 7 ? 1 : 0)) << 3;
                top = entries[top.begin + child];
                max_depth = max(max_depth, depth);
            } else {
                if (depth == 0) break;
                top = stack[--depth];
                morton = morton >> 3;
            }
        }
    }

    depths[id] = max_depth;
}

void flatten_grid(MemManager& mem, Grid& grid) {
    // Collapse voxel map entries when possible
    do {
        set_global(hagrid::collapsed_count, 0);
        collapse_entries<<<round_div(grid.num_entries, 64), 64>>>(grid.entries, grid.num_entries);
    } while (get_global(hagrid::collapsed_count) > 0);

    // Flatten the voxel map
    /*for (int i = max(0, grid.shift - flat_levels); i >= 0; i -= 1000) {//flat_levels) {
        int first = i > 0 ? grid.offsets[i - 1] : 0;
        int last  = grid.offsets[i];
        int num_flattened = last - first;
        auto depths = mem.alloc<int>(num_flattened);
        compute_depths<<<round_div(num_flattened, 64), 64>>>(grid.entries + first, depths, num_flattened);
        mem.free(depths);
    }*/
}

} // namespace hagrid

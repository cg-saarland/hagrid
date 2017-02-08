#include "build.h"

namespace hagrid {

static constexpr int flat_levels = Entry::LOG_DIM_BITS + 1;

/// Collapses sub-entries that map to the same cell/sub-sub-entry
__global__ void collapse_entries(Entry* entries, int first, int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    auto entry = entries[first + id];
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
                entries[first + id] = as<Entry>(ptr0);
            }
        }
    }
}

/// Computes the depth of each entry
__global__ void compute_depths(Entry* entries, int* depth, int first, int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    auto entry = entries[first + id];
    int d = 0;
    if (entry.log_dim) {
        auto ptr = (const int4*)(depth + entry.begin);
        auto d0 = ptr[0];
        auto d1 = ptr[1];
        d = 1 + max(max(max(d0.x, d1.x), max(d0.y, d1.y)),
                    max(max(d0.z, d1.z), max(d0.w, d1.w)));
    }
    depth[first + id] = d;
}

/// Flattens several voxel map levels into one larger level
__global__ void flatten_level(const Entry* __restrict__ entries,
                              Entry* __restrict__ new_entries,
                              int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    
}

void flatten_grid(MemManager& mem, Grid& grid) {
    auto depth = mem.alloc<int>(grid.num_entries);

    // Flatten the voxel map
    for (int i = grid.shift; i >= 0; i--) {
        int first = i > 0 ? grid.offsets[i - 1] : 0;
        int last  = grid.offsets[i];
        int num_entries = last - first;
        // Collapse voxel map entries when possible
        collapse_entries<<<round_div(num_entries, 64), 64>>>(grid.entries, first, num_entries);
        DEBUG_SYNC();
        compute_depths<<<round_div(num_entries, 64), 64>>>(grid.entries, depth, first, num_entries);
        DEBUG_SYNC();
    }  

    mem.free(depth);
}

} // namespace hagrid

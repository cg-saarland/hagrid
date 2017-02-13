#include "build.h"
#include "parallel.cuh"

namespace hagrid {

static constexpr int flat_levels = (1 << Entry::LOG_DIM_BITS) - 1;

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
__global__ void compute_depths(Entry* entries, int* depths, int first, int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    auto entry = entries[first + id];
    int d = 0;
    if (entry.log_dim) {
        auto ptr = (const int4*)(depths + entry.begin);
        auto d0 = ptr[0];
        auto d1 = ptr[1];
        d = 1 + max(max(max(d0.x, d1.x), max(d0.y, d1.y)),
                    max(max(d0.z, d1.z), max(d0.w, d1.w)));
    }
    depths[first + id] = d;
}

/// Copies the top-level entries and change their depth & start index
__global__ void copy_top_level(const Entry* __restrict__ entries,
                               const int* __restrict__ start_entries,
                               const int* __restrict__ depths,
                               Entry* __restrict__ new_entries,
                               int num_entries) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_entries) return;

    auto entry = entries[id];
    if (entry.log_dim) {
        entry = make_entry(min(depths[id], flat_levels), num_entries + start_entries[id]);
    }
    new_entries[id] = entry;
}

/// Flattens several voxel map levels into one larger level
__global__ void flatten_level(const Entry* __restrict__ entries,
                              const int* __restrict__ start_entries,
                              const int* __restrict__ depths,
                              Entry* __restrict__ new_entries,
                              int first_entry,
                              int offset, int next_offset,
                              int num_entries) {
    int id = blockIdx.x;

    int d = min(depths[id + first_entry], flat_levels);
    int num_sub_entries = d == 0 ? 0 : 1 << (3 * d);
    if (num_sub_entries <= 0) return;

    int start = offset + start_entries[id + first_entry];
    auto root = entries[id + first_entry];

    for (int i = threadIdx.x; i < num_sub_entries; i += blockDim.x) {
        // Treat i as a morton code
        int cur_d = d;
        int x = 0, y = 0, z = 0;
        int next_id = id;
        auto entry = root;
        while (cur_d > 0) {
            cur_d--;

            int pos = i >> (cur_d * 3);
            x += (pos & 1) ? (1 << cur_d) : 0;
            y += (pos & 2) ? (1 << cur_d) : 0;
            z += (pos & 4) ? (1 << cur_d) : 0;

            if (entry.log_dim) {
                next_id = entry.begin + (pos & 7);
                entry = entries[next_id];
            }
        }

        if (entry.log_dim) {
            entry = make_entry(min(depths[next_id], flat_levels), next_offset + start_entries[next_id]);
        }

        new_entries[start + x + ((y + (z << d)) << d)] = entry;
    }
}

void flatten_grid(MemManager& mem, Grid& grid) {
    Parallel par(mem);

    auto depths = mem.alloc<int>(grid.num_entries + 1);

    // Flatten the voxel map
    for (int i = grid.shift; i >= 0; i--) {
        int first = i > 0 ? grid.offsets[i - 1] : 0;
        int last  = grid.offsets[i];
        int num_entries = last - first;
        // Collapse voxel map entries when possible
        collapse_entries<<<round_div(num_entries, 64), 64>>>(grid.entries, first, num_entries);
        DEBUG_SYNC();
        compute_depths<<<round_div(num_entries, 64), 64>>>(grid.entries, depths, first, num_entries);
        DEBUG_SYNC();
    }

    // Compute the insertion position of each flattened level, and the total new number of entries
    auto start_entries = mem.alloc<int>(grid.num_entries + 1);
    std::vector<int> level_offsets(grid.shift);
    int total_entries = grid.offsets[0];
    for (int i = 0; i < grid.shift; i += flat_levels) {
        int first = i > 0 ? grid.offsets[i - 1] : 0;
        int last  = grid.offsets[i];
        int num_entries = last - first;

        // CUDA 8 bug: decltype(f(...)) is considered as a call to f (which forces to use __host__ here)
        int num_new_entries = par.scan(par.transform(depths + first, [] __host__ __device__ (int d) {
            return d > 0 ? 1 << (min(d, flat_levels) * 3) : 0;
        }), num_entries + 1, start_entries + first);
        level_offsets[i] = total_entries;
        total_entries += num_new_entries;
    }

    // Flatten the voxel map, by concatenating consecutive several levels together
    auto new_entries = mem.alloc<Entry>(total_entries);
    std::vector<int> new_offsets;

    copy_top_level<<<round_div(grid.offsets[0], 64), 64>>>(grid.entries, start_entries, depths, new_entries, grid.offsets[0]);
    for (int i = 0; i < grid.shift; i += flat_levels) {
        int first = i > 0 ? grid.offsets[i - 1] : 0;
        int last  = grid.offsets[i];
        int num_entries = last - first;

        int next_offset = i + flat_levels < grid.shift ? level_offsets[i + flat_levels] : 0;
        flatten_level<<<num_entries, 64>>>(grid.entries,
                                           start_entries,
                                           depths,
                                           new_entries,
                                           first,
                                           level_offsets[i],
                                           next_offset,
                                           num_entries);
        DEBUG_SYNC();

        new_offsets.emplace_back(level_offsets[i]);
    }
    new_offsets.emplace_back(total_entries);

    std::swap(new_entries, grid.entries);
    std::swap(new_offsets, grid.offsets);
    mem.free(new_entries);
    grid.num_entries = total_entries;

    mem.free(depths);
    mem.free(start_entries);
}

} // namespace hagrid

#ifndef BUILD_H
#define BUILD_H

#include "mem_manager.h"
#include "prims.h"
#include "grid.h"

namespace hagrid {

/// Builds an initial irregular grid.
/// The building process starts by creating a uniform grid of density 'top_density',
/// and then proceeds to compute an independent resolution in each of its cells
/// (using the second-level density 'snd_density').
/// In each cell, an octree depth is computed from these independent resolutions
/// and the primitive references are split until every cell has reached its maximum depth.
/// The voxel map follows the octree structure.
void build_grid(MemManager& mem, const Tri* tris, int num_tris, Grid& grid, float top_density, float snd_density);

/// Performs the neighbor merging optimization (merging cells according to the SAH).
void merge_grid(MemManager& mem, Grid& grid, float alpha);

/// Flattens the voxel map to speed up queries.
/// Once this optimization is performed, the voxel map no longer follows an octree structure.
/// Each inner node of the voxel map now may have up to 1 << (3 * (1 << Entry::LOG_DIM_BITS - 1)) children.
void flatten_grid(MemManager& mem, Grid& grid);

/// Performs the cell expansion optimization (expands cells over neighbors that share the same set of primitives).
void expand_grid(MemManager& mem, Grid& grid, const Tri* tris, int iters);

} // namespace hagrid

#endif // BUILD_H

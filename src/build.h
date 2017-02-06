#ifndef BUILD_H
#define BUILD_H

#include "mem_manager.h"
#include "prims.h"
#include "grid.h"

namespace hagrid {

void build_grid(MemManager& mem, const Tri* tris, int num_tris, Grid& grid, float top_density, float snd_density);
void merge_grid(MemManager& mem, Grid& grid, float alpha);
void flatten_grid(MemManager& mem, Grid& grid);
void expand_grid(MemManager& mem, Grid& grid, const Tri* tris, int iters);

} // namespace hagrid

#endif // BUILD_H
